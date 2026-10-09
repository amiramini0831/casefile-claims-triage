"""Runtime configuration, read from the process environment and .env."""

from __future__ import annotations

import os
from pathlib import Path
from typing import Literal

from dotenv import dotenv_values
from pydantic import BaseModel, ConfigDict, SecretStr

PROJECT_ROOT = Path(__file__).resolve().parents[2]
ENV_PATH = PROJECT_ROOT / ".env"
RAW_DATA_DIR = PROJECT_ROOT / "data" / "raw"
REPORTS_DIR = PROJECT_ROOT / "reports"

DatabaseRole = Literal["loader", "agent"]

_ENV_KEYS: dict[str, str] = {
    "host": "CASEFILE_DB_HOST",
    "port": "CASEFILE_DB_PORT",
    "domain_database": "CASEFILE_DOMAIN_DB",
    "loader_user": "CASEFILE_LOADER_USER",
    "loader_password": "CASEFILE_LOADER_PASSWORD",
    "agent_user": "CASEFILE_AGENT_USER",
    "agent_password": "CASEFILE_AGENT_PASSWORD",
}


class ConfigurationError(RuntimeError):
    """A required setting is missing."""


class DatabaseSettings(BaseModel):
    """Connection settings for the domain database.

    Passwords are ``SecretStr`` so they do not show up in logs, reprs or
    tracebacks.
    """

    model_config = ConfigDict(frozen=True)

    host: str
    port: int
    domain_database: str
    loader_user: str
    loader_password: SecretStr
    agent_user: str
    agent_password: SecretStr

    def credentials(self, role: DatabaseRole) -> tuple[str, SecretStr]:
        """Return the user name and password of ``role``."""
        if role == "loader":
            return self.loader_user, self.loader_password
        return self.agent_user, self.agent_password


def load_database_settings(env_path: Path = ENV_PATH) -> DatabaseSettings:
    """Build the database settings.

    Environment variables take precedence over the .env file, so a deployment
    can inject secrets without writing them to disk.
    """
    file_values = dotenv_values(env_path) if env_path.exists() else {}
    values: dict[str, str] = {}
    missing: list[str] = []
    for field_name, key in _ENV_KEYS.items():
        value = os.environ.get(key) or file_values.get(key)
        if value:
            values[field_name] = value
        else:
            missing.append(key)
    if missing:
        raise ConfigurationError(
            f"Missing settings: {', '.join(missing)}. "
            "Run scripts/setup_database.py to create them."
        )
    return DatabaseSettings(**values)
