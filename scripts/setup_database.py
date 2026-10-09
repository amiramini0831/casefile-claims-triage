"""Create the Casefile domain database, its users and its schema.

Run on a fresh machine, and again whenever the privileges below change::

    python scripts/setup_database.py --ask-admin-password

When only db/schema/ changed and no new table needs privileges, no admin
account is needed::

    python scripts/setup_database.py --schema-only

Five accounts are created, each limited to what its job requires:

* loader    creates the tables and loads data; used by setup and load scripts
* agent     reads policies, prior claims and the repair cost reference, and
            nothing else. It cannot read the current claims, which hold the
            values the agents must take from the documents.
* writer    the system's own account for the human gate: it stores
            recommendations and records write-backs, but cannot approve
* approver  the person's account: it records approvals and rejections
* graph     stores the checkpoints of each run in a database of its own and
            cannot see the domain data

Because the writer cannot insert an approval, the automated side of the
system has no credentials with which to approve its own recommendation.

Passwords are generated on the first run and stored in .env, which git
ignores. They are never printed. The admin password is read from the
CASEFILE_DB_ADMIN_PASSWORD environment variable or prompted for with
--ask-admin-password; it is never stored.
"""

from __future__ import annotations

import argparse
import getpass
import os
import re
import secrets
import sys
from collections.abc import Mapping, Sequence
from contextlib import closing
from dataclasses import dataclass, field
from pathlib import Path

import pymysql
from dotenv import dotenv_values, set_key
from pymysql.constants import CLIENT, ER

PROJECT_ROOT = Path(__file__).resolve().parents[1]
ENV_PATH = PROJECT_ROOT / ".env"
SCHEMA_DIR = PROJECT_ROOT / "db" / "schema"

# Accounts are usable from this machine only.
ACCOUNT_HOST = "localhost"
IDENTIFIER_PATTERN = re.compile(r"[a-z][a-z0-9_]{0,31}")

DEFAULTS: dict[str, str] = {
    "CASEFILE_DB_HOST": "127.0.0.1",
    "CASEFILE_DB_PORT": "3306",
    "CASEFILE_DOMAIN_DB": "casefile_domain",
    "CASEFILE_GRAPH_DB": "casefile_graph",
}


@dataclass(frozen=True)
class Role:
    """What one account may do. Privileges are per table unless stated.

    An account works in the domain database, except the one for the run
    checkpoints, which has a database of its own and no access to the domain.
    """

    database_privileges: str | None = None
    table_privileges: Mapping[str, str] = field(default_factory=dict)
    uses_graph_database: bool = False


ROLES: dict[str, Role] = {
    "loader": Role(
        database_privileges=(
            "SELECT, INSERT, UPDATE, DELETE, CREATE, ALTER, DROP, INDEX, REFERENCES"
        )
    ),
    "agent": Role(
        table_privileges={
            "policies": "SELECT",
            "prior_claims": "SELECT",
            "repair_cost_reference": "SELECT",
        }
    ),
    "writer": Role(
        table_privileges={
            "recommendations": "SELECT, INSERT",
            "approvals": "SELECT",
            "write_backs": "SELECT, INSERT",
        }
    ),
    "approver": Role(
        table_privileges={
            "recommendations": "SELECT",
            "approvals": "SELECT, INSERT",
            "write_backs": "SELECT",
        }
    ),
    # Stores the checkpoints of each run. The checkpoint library creates and
    # migrates its own tables, hence the schema privileges.
    "graph": Role(
        database_privileges="SELECT, INSERT, UPDATE, DELETE, CREATE, ALTER, INDEX",
        uses_graph_database=True,
    ),
}

# Statements each account must be refused. Every one is harmless even if a
# grant were wrong and it ran: it reads one row or matches no row at all.
# {domain} and {graph} stand for the two database names.
FORBIDDEN: dict[str, tuple[str, ...]] = {
    "agent": (
        "SELECT 1 FROM claims LIMIT 1",
        "SELECT 1 FROM approvals LIMIT 1",
        "DELETE FROM policies WHERE 1 = 0",
        "SELECT 1 FROM {graph}.checkpoints LIMIT 1",
    ),
    "graph": (
        "SELECT 1 FROM {domain}.policies LIMIT 1",
        "SELECT 1 FROM {domain}.claims LIMIT 1",
        "SELECT 1 FROM {domain}.approvals LIMIT 1",
    ),
    "writer": (
        "INSERT INTO approvals "
        "(recommendation_id, claim_number, decision, chosen_decision, approver) "
        "SELECT 0, '', 'approved', 'rule', '' FROM DUAL WHERE 1 = 0",
        "DELETE FROM write_backs WHERE 1 = 0",
        "SELECT 1 FROM claims LIMIT 1",
    ),
    "approver": (
        "INSERT INTO write_backs (approval_id, claim_number) "
        "SELECT 0, '' FROM DUAL WHERE 1 = 0",
        "INSERT INTO recommendations "
        "(run_id, case_id, claim_number, policy_number, rule_outcome, rule_reason, "
        "rule_explanation, model_outcome, model_reason, model_explanation) "
        "SELECT '', '', '', 0, 'recommend_deny', '', '', 'recommend_deny', '', '' "
        "FROM DUAL WHERE 1 = 0",
        "DELETE FROM approvals WHERE 1 = 0",
    ),
}


@dataclass(frozen=True)
class Account:
    """A MySQL account and its role."""

    name: str
    user: str
    password: str = field(repr=False)
    role: Role


@dataclass(frozen=True)
class Settings:
    """Connection settings resolved from .env."""

    host: str
    port: int
    database: str
    graph_database: str
    accounts: Mapping[str, Account]

    def database_of(self, account: Account) -> str:
        """Return the database ``account`` works in."""
        if account.role.uses_graph_database:
            return self.graph_database
        return self.database


def validate_identifier(name: str) -> str:
    """Return ``name`` if it is safe to use as a MySQL identifier.

    Database and table names cannot be sent as query parameters, so the
    identifiers this script interpolates into SQL are held to a strict pattern.
    """
    if not IDENTIFIER_PATTERN.fullmatch(name):
        raise ValueError(f"Unsafe MySQL identifier: {name!r}")
    return name


def load_settings() -> tuple[Settings, dict[str, str]]:
    """Read settings from .env, filling gaps with defaults and new passwords.

    Returns the settings together with the key/value pairs that are not in
    .env yet, so the caller can write them once the server is reachable.
    """
    stored = dotenv_values(ENV_PATH) if ENV_PATH.exists() else {}
    missing: dict[str, str] = {}

    def resolve(key: str, fallback: str) -> str:
        value = stored.get(key)
        if not value:
            value = fallback
            missing[key] = value
        return value

    values = {key: resolve(key, default) for key, default in DEFAULTS.items()}
    accounts = {}
    for name, role in ROLES.items():
        prefix = f"CASEFILE_{name.upper()}"
        accounts[name] = Account(
            name=name,
            user=validate_identifier(resolve(f"{prefix}_USER", f"casefile_{name}")),
            password=resolve(f"{prefix}_PASSWORD", secrets.token_urlsafe(32)),
            role=role,
        )

    settings = Settings(
        host=values["CASEFILE_DB_HOST"],
        port=int(values["CASEFILE_DB_PORT"]),
        database=validate_identifier(values["CASEFILE_DOMAIN_DB"]),
        graph_database=validate_identifier(values["CASEFILE_GRAPH_DB"]),
        accounts=accounts,
    )
    return settings, missing


def persist(missing: dict[str, str]) -> None:
    """Write newly resolved settings to .env."""
    if not missing:
        return
    ENV_PATH.touch(exist_ok=True)
    for key, value in missing.items():
        set_key(ENV_PATH, key, value, quote_mode="never")


def connect(
    settings: Settings,
    user: str,
    password: str,
    *,
    database: str | None = None,
    client_flag: int = 0,
) -> pymysql.connections.Connection:
    """Open an autocommit connection to the configured server."""
    return pymysql.connect(
        host=settings.host,
        port=settings.port,
        user=user,
        password=password,
        database=database,
        charset="utf8mb4",
        autocommit=True,
        client_flag=client_flag,
    )


def connect_as(
    settings: Settings, name: str, **options: int
) -> pymysql.connections.Connection:
    """Open a connection as one of the accounts, to the database it works in."""
    account = settings.accounts[name]
    return connect(
        settings,
        account.user,
        account.password,
        database=settings.database_of(account),
        **options,
    )


def create_database_and_users(
    admin: pymysql.connections.Connection, settings: Settings
) -> None:
    """Create the databases and accounts, leaving only database-wide grants.

    Every privilege is revoked first, so re-running the script after a role
    was narrowed really removes what was dropped. Table privileges are granted
    later, once the schema exists.
    """
    with admin.cursor() as cursor:
        for database in (settings.database, settings.graph_database):
            cursor.execute(
                f"CREATE DATABASE IF NOT EXISTS `{database}` "
                "CHARACTER SET utf8mb4 COLLATE utf8mb4_0900_ai_ci"
            )
        for account in settings.accounts.values():
            identity = (account.user, ACCOUNT_HOST)
            cursor.execute(
                "CREATE USER IF NOT EXISTS %s@%s IDENTIFIED BY %s",
                (*identity, account.password),
            )
            # .env is the source of truth for the password.
            cursor.execute(
                "ALTER USER %s@%s IDENTIFIED BY %s", (*identity, account.password)
            )
            cursor.execute("REVOKE ALL PRIVILEGES, GRANT OPTION FROM %s@%s", identity)
            privileges = account.role.database_privileges
            if privileges:
                database = settings.database_of(account)
                cursor.execute(
                    f"GRANT {privileges} ON `{database}`.* TO %s@%s", identity
                )


def grant_table_privileges(
    admin: pymysql.connections.Connection, settings: Settings
) -> None:
    """Grant each account its privileges on individual tables."""
    database = settings.database
    with admin.cursor() as cursor:
        for account in settings.accounts.values():
            for table, privileges in account.role.table_privileges.items():
                target = f"`{database}`.`{validate_identifier(table)}`"
                cursor.execute(
                    f"GRANT {privileges} ON {target} TO %s@%s",
                    (account.user, ACCOUNT_HOST),
                )


def apply_schema(settings: Settings) -> list[str]:
    """Run every SQL file in db/schema/ in name order as the loader account."""
    paths = sorted(SCHEMA_DIR.glob("*.sql"))
    connection = connect_as(settings, "loader", client_flag=CLIENT.MULTI_STATEMENTS)
    with closing(connection), connection.cursor() as cursor:
        for path in paths:
            cursor.execute(path.read_text(encoding="utf-8"))
            # Errors in later statements only surface when their result is read.
            while cursor.nextset():
                pass
    return [path.name for path in paths]


def verify_privileges(settings: Settings) -> None:
    """Prove that every limited account can do its job and nothing beyond it.

    Raises if an account cannot read a table it was granted, or if a
    statement it must be refused is accepted.
    """
    names = {
        "domain": f"`{settings.database}`",
        "graph": f"`{settings.graph_database}`",
    }
    denied = (ER.TABLEACCESS_DENIED_ERROR, ER.DBACCESS_DENIED_ERROR)
    for name, statements in FORBIDDEN.items():
        connection = connect_as(settings, name)
        with closing(connection), connection.cursor() as cursor:
            for table in settings.accounts[name].role.table_privileges:
                # The table name is a constant of this script and is validated.
                readable = f"`{validate_identifier(table)}`"
                cursor.execute(f"SELECT 1 FROM {readable} LIMIT 1")  # noqa: S608
                cursor.fetchall()
            for template in statements:
                statement = template.format(**names)
                try:
                    cursor.execute(statement)
                except pymysql.err.OperationalError as error:
                    if error.args[0] not in denied:
                        raise
                else:
                    raise RuntimeError(f"{name} was allowed to run: {statement}")
        print(
            f"Account {name!r}: allowed tables readable, forbidden statements refused."
        )


def parse_args(argv: Sequence[str] | None) -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument(
        "--admin-user",
        default="root",
        help="MySQL account allowed to create databases and users (default: root)",
    )
    parser.add_argument(
        "--ask-admin-password",
        action="store_true",
        help="prompt for the admin password instead of reading "
        "CASEFILE_DB_ADMIN_PASSWORD",
    )
    parser.add_argument(
        "--schema-only",
        action="store_true",
        help="skip the admin steps and only apply db/schema/ as the loader account",
    )
    return parser.parse_args(argv)


def connect_as_admin(
    args: argparse.Namespace, settings: Settings
) -> pymysql.connections.Connection | None:
    """Open the admin connection, or return None after explaining why not."""
    if args.ask_admin_password:
        admin_password = getpass.getpass(f"MySQL password for {args.admin_user}: ")
    else:
        admin_password = os.environ.get("CASEFILE_DB_ADMIN_PASSWORD", "")
    try:
        admin = connect(settings, args.admin_user, admin_password)
    except pymysql.err.OperationalError as error:
        print(
            f"Cannot connect to MySQL at {settings.host}:{settings.port} "
            f"as {args.admin_user!r}: {error.args[1]}",
            file=sys.stderr,
        )
        return None
    if not admin_password:
        print(
            f"WARNING: {args.admin_user!r} has an empty password. "
            "Set one before exposing this server to a network.",
            file=sys.stderr,
        )
    return admin


def main(argv: Sequence[str] | None = None) -> int:
    """Set up the database and report what was done. Returns an exit code."""
    args = parse_args(argv)
    settings, missing = load_settings()

    if args.schema_only:
        # Applying the schema only needs the loader account.
        needed = [
            key
            for key in missing
            if key.startswith(("CASEFILE_DB_", "CASEFILE_DOMAIN_", "CASEFILE_LOADER_"))
        ]
        if needed:
            print(
                f".env lacks {', '.join(needed)}. "
                "Run this script once without --schema-only.",
                file=sys.stderr,
            )
            return 1
        applied = apply_schema(settings)
        print(f"Applied schema files: {', '.join(applied) or 'none'}")
        print("Privileges were not changed; a new table needs a full run.")
        return 0

    admin = connect_as_admin(args, settings)
    if admin is None:
        return 1
    # Only now, so a failed connection does not leave unused passwords behind.
    persist(missing)
    with closing(admin):
        create_database_and_users(admin, settings)
        print(f"Database {settings.database!r} and accounts are in place.")
        applied = apply_schema(settings)
        print(f"Applied schema files: {', '.join(applied) or 'none'}")
        grant_table_privileges(admin, settings)

    verify_privileges(settings)
    return 0


if __name__ == "__main__":
    sys.exit(main())
