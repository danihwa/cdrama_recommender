"""Shared fixtures for parser tests.

conftest.py is a special pytest file — any fixture defined here is
automatically available to every test in this directory (and below)
without needing an import.  pytest discovers it by name convention.
"""

import pytest
from openai import OpenAI

from src.env import load_secrets


@pytest.fixture(scope="module")
def openai_client():
    """Create a single OpenAI client shared across all tests in a module.

    scope="module" means this fixture runs once per test *file*, not once
    per test function.  Since every parser eval calls the same API, reusing
    one client avoids re-creating the HTTP connection pool for every case.
    """
    load_secrets()
    return OpenAI()
