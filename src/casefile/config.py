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
# Recorded model answers, so that a run can be repeated without the model.
RECORDINGS_DIR = PROJECT_ROOT / "recordings"

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


class LLMSettings(BaseModel):
    """How to reach the language model: any OpenAI-compatible chat API."""

    model_config = ConfigDict(frozen=True)

    base_url: str
    model: str
    api_key: SecretStr
    # Sent with every request when set. GLM-5.3 reasons at its maximum unless
    # told otherwise, which is slow and can use up the whole output allowance.
    reasoning_effort: str | None = "low"
    timeout_seconds: float = 120.0


def load_llm_settings(env_path: Path = ENV_PATH) -> LLMSettings:
    """Build the model settings from the environment and .env."""
    file_values = dotenv_values(env_path) if env_path.exists() else {}

    def read(key: str) -> str | None:
        return os.environ.get(key) or file_values.get(key)

    required = {
        "base_url": "CASEFILE_LLM_BASE_URL",
        "model": "CASEFILE_LLM_MODEL",
        "api_key": "CASEFILE_LLM_API_KEY",
    }
    values = {name: read(key) for name, key in required.items()}
    missing = [required[name] for name, value in values.items() if not value]
    if missing:
        raise ConfigurationError(
            f"Missing settings: {', '.join(missing)}. Set them in .env; "
            ".env.example shows the names."
        )
    effort = read("CASEFILE_LLM_REASONING_EFFORT")
    if effort:
        values["reasoning_effort"] = None if effort.lower() == "none" else effort
    return LLMSettings(**values)


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
