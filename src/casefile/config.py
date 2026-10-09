"""Runtime configuration, read from the process environment and .env."""

from __future__ import annotations

import os
from pathlib import Path
from typing import Literal, get_args

from dotenv import dotenv_values
from pydantic import BaseModel, ConfigDict, SecretStr

PROJECT_ROOT = Path(__file__).resolve().parents[2]
ENV_PATH = PROJECT_ROOT / ".env"
RAW_DATA_DIR = PROJECT_ROOT / "data" / "raw"
REPORTS_DIR = PROJECT_ROOT / "reports"

# loader:   creates tables and loads data
# agent:    reads policies, prior claims and the repair cost reference only
# writer:   stores recommendations and write-backs; cannot approve
# approver: records a person's approvals and rejections
# graph:    stores run checkpoints, in a separate database
DatabaseRole = Literal["loader", "agent", "writer", "approver", "graph"]


class ConfigurationError(RuntimeError):
    """A required setting is missing."""


class DatabaseAccount(BaseModel):
    """The credentials of one database role."""

    model_config = ConfigDict(frozen=True)

    user: str
    # SecretStr keeps the password out of logs, reprs and tracebacks.
    password: SecretStr


class DatabaseSettings(BaseModel):
    """Connection settings for the domain database."""

    model_config = ConfigDict(frozen=True)

    host: str
    port: int
    domain_database: str
    # Where run checkpoints live; ``None`` until the setup script created it.
    graph_database: str | None = None
    accounts: dict[DatabaseRole, DatabaseAccount]

    def credentials(self, role: DatabaseRole) -> tuple[str, SecretStr]:
        """Return the user name and password of ``role``."""
        account = self.accounts.get(role)
        if account is None:
            raise ConfigurationError(
                f"No credentials for the {role!r} role. "
                "Run scripts/setup_database.py to create them."
            )
        return account.user, account.password


def load_database_settings(env_path: Path = ENV_PATH) -> DatabaseSettings:
    """Build the database settings.

    Environment variables take precedence over the .env file, so a deployment
    can inject secrets without writing them to disk.
    """
    file_values = dotenv_values(env_path) if env_path.exists() else {}
    missing: list[str] = []

    def read(key: str) -> str:
        value = os.environ.get(key) or file_values.get(key)
        if not value:
            missing.append(key)
        return value or ""

    host = read("CASEFILE_DB_HOST")
    port = read("CASEFILE_DB_PORT")
    database = read("CASEFILE_DOMAIN_DB")
    if missing:
        raise ConfigurationError(
            f"Missing settings: {', '.join(missing)}. "
            "Run scripts/setup_database.py to create them."
        )
    # A role whose credentials are not there yet is left out rather than
    # failing everything: asking for it later raises a clear error.
    accounts = {}
    for role in get_args(DatabaseRole):
        prefix = f"CASEFILE_{role.upper()}"
        user = os.environ.get(f"{prefix}_USER") or file_values.get(f"{prefix}_USER")
        password = os.environ.get(f"{prefix}_PASSWORD") or file_values.get(
            f"{prefix}_PASSWORD"
        )
        if user and password:
            accounts[role] = {"user": user, "password": password}
    graph_database = os.environ.get("CASEFILE_GRAPH_DB") or file_values.get(
        "CASEFILE_GRAPH_DB"
    )
    return DatabaseSettings(
        host=host,
        port=port,
        domain_database=database,
        graph_database=graph_database,
        accounts=accounts,
    )
