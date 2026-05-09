"""Smoke test: can we connect to local Postgres and is pgvector loaded?"""

from __future__ import annotations

import pytest

from src.database.connection import get_db_connection


@pytest.mark.db
def test_postgres_connection_and_pgvector() -> None:
    conn = get_db_connection()
    try:
        with conn.cursor() as cur:
            cur.execute("SELECT 1")
            assert cur.fetchone() == (1,)

            cur.execute("SELECT extname FROM pg_extension WHERE extname='vector'")
            row = cur.fetchone()
            assert row is not None, "pgvector extension is not installed"
            assert row[0] == "vector"
    finally:
        conn.close()
