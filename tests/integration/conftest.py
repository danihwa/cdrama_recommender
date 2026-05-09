"""Shared fixtures for integration tests.

conftest.py is a special pytest file — any fixture defined here is
automatically available to every test in this directory (and below)
without needing an import.  pytest discovers it by name convention.
"""

import psycopg
import pytest

from src.database.connection import get_db_connection
from src.env import load_secrets


@pytest.fixture(scope="module")
def db_conn() -> psycopg.Connection:
    """psycopg connection for DB-hitting integration tests."""
    load_secrets()
    return get_db_connection()
