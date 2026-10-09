"""Shared fixtures.

Tests that need MySQL are skipped when the database is not set up or not
running, so the unit tests stay runnable on any machine.
"""

from __future__ import annotations

from collections.abc import Iterator

import pytest
from sqlalchemy import Connection, MetaData, Table
from sqlalchemy.exc import OperationalError

from casefile.config import (
    ConfigurationError,
    DatabaseRole,
    DatabaseSettings,
    load_database_settings,
)
from casefile.db import create_domain_engine


@pytest.fixture(scope="session")
def database_settings() -> DatabaseSettings:
    try:
        return load_database_settings()
    except ConfigurationError as error:
        pytest.skip(f"database is not configured: {error}")


def _connection(role: DatabaseRole, settings: DatabaseSettings) -> Iterator[Connection]:
    """Yield a connection whose work is always rolled back."""
    engine = create_domain_engine(role, settings)
    try:
        connection = engine.connect()
    except OperationalError as error:
        engine.dispose()
        pytest.skip(f"MySQL is not reachable: {error.orig}")
    transaction = connection.begin()
    try:
        yield connection
    finally:
        transaction.rollback()
        connection.close()
        engine.dispose()


@pytest.fixture
def loader_connection(database_settings: DatabaseSettings) -> Iterator[Connection]:
    yield from _connection("loader", database_settings)


@pytest.fixture
def agent_connection(database_settings: DatabaseSettings) -> Iterator[Connection]:
    yield from _connection("agent", database_settings)


@pytest.fixture
def domain_tables(loader_connection: Connection) -> dict[str, Table]:
    metadata = MetaData()
    metadata.reflect(
        bind=loader_connection, only=("policies", "claims", "prior_claims")
    )
    return dict(metadata.tables)
