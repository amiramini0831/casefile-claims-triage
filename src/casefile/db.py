"""Engines for the domain database."""

from __future__ import annotations

from sqlalchemy import Engine, create_engine
from sqlalchemy.engine import URL

from casefile.config import DatabaseRole, DatabaseSettings, load_database_settings


def create_domain_engine(
    role: DatabaseRole, settings: DatabaseSettings | None = None
) -> Engine:
    """Create an engine that connects to the domain database as ``role``.

    Use ``"agent"`` for anything that only reads: that account has SELECT and
    nothing else, so a bug or a prompt injection cannot change domain data.
    """
    settings = settings or load_database_settings()
    user, password = settings.credentials(role)
    url = URL.create(
        "mysql+pymysql",
        username=user,
        password=password.get_secret_value(),
        host=settings.host,
        port=settings.port,
        database=settings.domain_database,
        query={"charset": "utf8mb4"},
    )
    return create_engine(url)
