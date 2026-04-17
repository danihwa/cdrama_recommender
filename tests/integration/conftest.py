"""Shared fixtures for integration tests.

conftest.py is a special pytest file — any fixture defined here is
automatically available to every test in this directory (and below)
without needing an import.  pytest discovers it by name convention.
"""

import pytest
from openai import OpenAI
from supabase import Client

from src.database.connection import get_db_connection
from src.env import load_secrets


@pytest.fixture(scope="module")
def openai_client():
    """Create a single OpenAI client shared across all tests in a module.

    scope="module" means this fixture runs once per test *file*, not once
    per test function.  Since every test in test_parse_user_query.py calls
    the same API, reusing one client avoids re-creating the HTTP connection
    pool ~47 times.

    Scope options (from narrowest to widest):
      "function"  — new client per test       (safest, slowest)
      "class"     — new client per test class
      "module"    — new client per .py file    <-- we use this
      "session"   — one client for entire run  (fastest, shared across files)
    """
    load_secrets()
    return OpenAI()


@pytest.fixture(scope="module")
def supabase() -> Client:
    """Authenticated Supabase client for DB-hitting integration tests.

    Same module scope as openai_client — one connection per test file
    is enough. load_secrets() is idempotent, so calling it here alongside
    the openai_client fixture is safe.
    """
    load_secrets()
    return get_db_connection()
