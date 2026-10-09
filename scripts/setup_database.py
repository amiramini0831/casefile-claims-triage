"""Create the Casefile domain database, its users and its schema.

Run once on a fresh machine::

    python scripts/setup_database.py

and whenever db/schema/ changes, which needs no admin account::

    python scripts/setup_database.py --schema-only

The script connects three times, each time with less privilege:

1. as a MySQL admin, to create the database and the two application users;
2. as the loader user, to apply the SQL files in db/schema/;
3. as the agent user, to prove that it can read but cannot write.

Application passwords are generated on the first run and stored in .env,
which git ignores. They are never printed. The admin password is read from
the CASEFILE_DB_ADMIN_PASSWORD environment variable or prompted for with
--ask-admin-password; it is never stored.
"""

from __future__ import annotations

import argparse
import getpass
import os
import re
import secrets
import sys
from collections.abc import Callable, Sequence
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
LOADER_PRIVILEGES = (
    "SELECT, INSERT, UPDATE, DELETE, CREATE, ALTER, DROP, INDEX, REFERENCES"
)
AGENT_PRIVILEGES = "SELECT"

DEFAULTS: dict[str, str] = {
    "CASEFILE_DB_HOST": "127.0.0.1",
    "CASEFILE_DB_PORT": "3306",
    "CASEFILE_DOMAIN_DB": "casefile_domain",
    "CASEFILE_LOADER_USER": "casefile_loader",
    "CASEFILE_AGENT_USER": "casefile_agent",
}
PASSWORD_KEYS = ("CASEFILE_LOADER_PASSWORD", "CASEFILE_AGENT_PASSWORD")
IDENTIFIER_PATTERN = re.compile(r"[a-z][a-z0-9_]{0,31}")


@dataclass(frozen=True)
class Account:
    """A MySQL account and the privileges it gets on the domain database."""

    user: str
    password: str = field(repr=False)
    privileges: str


@dataclass(frozen=True)
class Settings:
    """Connection settings resolved from .env."""

    host: str
    port: int
    database: str
    loader: Account
    agent: Account


def validate_identifier(name: str) -> str:
    """Return ``name`` if it is safe to use as a MySQL identifier.

    Database names cannot be sent as query parameters, so the one identifier
    this script interpolates into SQL is restricted to a strict pattern.
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

    def resolve(key: str, fallback: Callable[[], str]) -> str:
        value = stored.get(key)
        if not value:
            value = fallback()
            missing[key] = value
        return value

    values = {
        key: resolve(key, lambda d=default: d) for key, default in DEFAULTS.items()
    }
    passwords = {
        key: resolve(key, lambda: secrets.token_urlsafe(32)) for key in PASSWORD_KEYS
    }

    settings = Settings(
        host=values["CASEFILE_DB_HOST"],
        port=int(values["CASEFILE_DB_PORT"]),
        database=validate_identifier(values["CASEFILE_DOMAIN_DB"]),
        loader=Account(
            user=validate_identifier(values["CASEFILE_LOADER_USER"]),
            password=passwords["CASEFILE_LOADER_PASSWORD"],
            privileges=LOADER_PRIVILEGES,
        ),
        agent=Account(
            user=validate_identifier(values["CASEFILE_AGENT_USER"]),
            password=passwords["CASEFILE_AGENT_PASSWORD"],
            privileges=AGENT_PRIVILEGES,
        ),
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


def create_database_and_users(
    admin: pymysql.connections.Connection, settings: Settings
) -> None:
    """Create the domain database and reset both accounts to exact grants.

    Privileges are revoked before they are granted, so re-running the script
    after narrowing a privilege list really removes the dropped privileges.
    """
    database = settings.database
    with admin.cursor() as cursor:
        cursor.execute(
            f"CREATE DATABASE IF NOT EXISTS `{database}` "
            "CHARACTER SET utf8mb4 COLLATE utf8mb4_0900_ai_ci"
        )
        for account in (settings.loader, settings.agent):
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
            cursor.execute(
                f"GRANT {account.privileges} ON `{database}`.* TO %s@%s", identity
            )


def apply_schema(settings: Settings) -> list[str]:
    """Run every SQL file in db/schema/ in name order as the loader user."""
    paths = sorted(SCHEMA_DIR.glob("*.sql"))
    connection = connect(
        settings,
        settings.loader.user,
        settings.loader.password,
        database=settings.database,
        client_flag=CLIENT.MULTI_STATEMENTS,
    )
    with closing(connection), connection.cursor() as cursor:
        for path in paths:
            cursor.execute(path.read_text(encoding="utf-8"))
            # Errors in later statements only surface when their result is read.
            while cursor.nextset():
                pass
    return [path.name for path in paths]


def verify_agent_is_read_only(settings: Settings) -> list[str]:
    """Check that the agent user can list the tables but cannot modify them.

    Returns the table names the agent can see. Raises if a write is accepted.
    """
    connection = connect(
        settings,
        settings.agent.user,
        settings.agent.password,
        database=settings.database,
    )
    with closing(connection), connection.cursor() as cursor:
        cursor.execute("SHOW TABLES")
        tables = sorted(row[0] for row in cursor.fetchall())
        if not tables:
            raise RuntimeError("The agent user cannot see any table.")
        try:
            # Matches no row, so it is harmless even if the grant were wrong.
            cursor.execute("DELETE FROM policies WHERE 1 = 0")
        except pymysql.err.OperationalError as error:
            if error.args[0] != ER.TABLEACCESS_DENIED_ERROR:
                raise
        else:
            raise RuntimeError("The agent user was allowed to run DELETE.")
    return tables


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
        help="skip the admin step and only apply db/schema/ as the loader user",
    )
    return parser.parse_args(argv)


def create_as_admin(
    args: argparse.Namespace, settings: Settings, missing: dict[str, str]
) -> bool:
    """Run the admin step. Returns False if the admin cannot connect."""
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
        return False

    # Only now, so a failed connection does not leave unused passwords behind.
    persist(missing)
    with closing(admin):
        create_database_and_users(admin, settings)
    print(f"Database {settings.database!r} and users are in place.")

    if not admin_password:
        print(
            f"WARNING: {args.admin_user!r} has an empty password. "
            "Set one before exposing this server to a network.",
            file=sys.stderr,
        )
    return True


def main(argv: Sequence[str] | None = None) -> int:
    """Set up the database and report what was done. Returns an exit code."""
    args = parse_args(argv)
    settings, missing = load_settings()

    if args.schema_only:
        if missing:
            print(
                f".env lacks {', '.join(missing)}. "
                "Run this script once without --schema-only.",
                file=sys.stderr,
            )
            return 1
    elif not create_as_admin(args, settings, missing):
        return 1

    applied = apply_schema(settings)
    print(f"Applied schema files: {', '.join(applied) or 'none'}")

    tables = verify_agent_is_read_only(settings)
    print(f"Agent user is read-only and sees: {', '.join(tables)}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
