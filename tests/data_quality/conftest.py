"""Shared fixtures for data-quality tests.

These tests check the contents of the loaded `cdramas` table against the
guarantees the cleaning notebook (notebooks/02_cleaning.ipynb) is supposed
to enforce. They run against the local Postgres + pgvector container, so
they're gated behind @pytest.mark.db like the retrieval suite.

A module-scoped fixture is used so each test file fetches its slice of
the table once, not once per test.
"""

from __future__ import annotations

import psycopg
import pytest

from src.database.connection import get_db_connection
from src.env import load_secrets


@pytest.fixture(scope="module")
def db_conn() -> psycopg.Connection:
    """psycopg connection for data-quality tests."""
    load_secrets()
    return get_db_connection()
