"""Where the checkpoints of a run are stored.

LangGraph has no official MySQL checkpointer, so this uses the community
package ``langgraph-checkpoint-mysql``. Checkpoints go to their own database,
through an account that cannot see the domain data.
"""

from __future__ import annotations

from collections.abc import Iterator
from contextlib import contextmanager

import pymysql
from langgraph.checkpoint.mysql.pymysql import PyMySQLSaver

from casefile.config import (
    ConfigurationError,
    DatabaseSettings,
    load_database_settings,
)


@contextmanager
def open_checkpointer(
    settings: DatabaseSettings | None = None,
) -> Iterator[PyMySQLSaver]:
    """Open the MySQL checkpointer, creating its tables on first use."""
    settings = settings or load_database_settings()
    user, password = settings.credentials("graph")
    if not settings.graph_database:
        raise ConfigurationError(
            "CASEFILE_GRAPH_DB is not set. Run scripts/setup_database.py."
        )
    connection = pymysql.connect(
        host=settings.host,
        port=settings.port,
        user=user,
        password=password.get_secret_value(),
        database=settings.graph_database,
        charset="utf8mb4",
        # The checkpointer manages its own transactions.
        autocommit=True,
    )
    try:
        saver = PyMySQLSaver(connection)
        saver.setup()
        yield saver
    finally:
        connection.close()
