"""Connection helper for local Postgres."""

from __future__ import annotations

import os

import psycopg
from pgvector.psycopg import register_vector

from src.env import load_secrets

load_secrets()


def get_db_connection() -> psycopg.Connection:
    """Return a psycopg connection with pgvector type adapters registered.

    Reads ``DATABASE_URL`` from the environment (loaded via dotenv).
    ``register_vector`` teaches psycopg to ser/de the ``embedding``
    column as a plain Python ``list[float]``, so callers don't have to
    parse the textual ``vector`` representation by hand.

    Raises:
        ValueError: If ``DATABASE_URL`` is missing.
    """
    url = os.environ.get("DATABASE_URL")
    if not url:
        raise ValueError("DATABASE_URL is missing from .env file!")

    try:
        conn = psycopg.connect(url, autocommit=True)
    except Exception as e:
        raise ConnectionError(f"Failed to connect to Postgres: {e}") from e

    register_vector(conn)
    return conn
