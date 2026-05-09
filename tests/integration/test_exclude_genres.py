"""End-to-end smoke tests for the exclude_genres SQL filter.

These tests hit the local Postgres DB to verify the filter actually works
at the database layer.  The parser-level tests in test_parse_user_query.py
only cover LLM output — they can't catch a broken RPC parameter or a
missing query clause.

Why two tests?
    exclude_genres lives in TWO code paths:
      - match_documents RPC          (reference + semantic modes)
      - psycopg plain SQL             (sql mode, via NOT && array overlap)
    Each test covers one path. Together they confirm both layers of the
    filter wiring are intact.

Gated behind @pytest.mark.db so CI without DB access skips:
    uv run pytest -m "not integration"
"""

from __future__ import annotations

import psycopg
import pytest

from src.recommender.models import QueryFilters
from src.recommender.search_reference import retrieve_reference_candidates
from src.recommender.search_sql import retrieve_sql_candidates


@pytest.mark.db
def test_sql_mode_excludes_genre(db_conn: psycopg.Connection) -> None:
    """SQL-mode query with exclude_genres returns zero rows with that genre.

    Uses a broad filter (recent year floor, no reference, no description)
    so the DB returns a meaningful batch where a broken filter would
    almost certainly leak a romance row through.
    """
    filters = QueryFilters(
        search_mode="sql",
        min_year=2020,
        exclude_genres=["Romance"],
    )
    rows = retrieve_sql_candidates(filters, db_conn, match_count=10)

    assert rows, "Expected at least one row — the DB shouldn't be empty"
    for row in rows:
        row_genres = {g.lower() for g in (row.get("genres") or [])}
        assert "romance" not in row_genres, (
            f"'{row['title']}' leaked through exclude_genres: {row['genres']}"
        )


@pytest.mark.db
def test_reference_mode_excludes_genre(db_conn: psycopg.Connection) -> None:
    """Reference-mode query with exclude_genres hits the match_documents RPC.

    If functions.sql hasn't been applied to the local DB (e.g., the
    Postgres volume was wiped without restarting the container), this
    fails with `psycopg.errors.UndefinedFunction` — which is a useful
    signal that the SQL migration step is pending.
    """
    filters = QueryFilters(
        search_mode="reference",
        reference_title="Nirvana in Fire",
        exclude_genres=["Romance"],
    )
    rows = retrieve_reference_candidates(filters, db_conn, match_count=10)

    assert rows, "Reference drama 'Nirvana in Fire' should resolve to candidates"
    for row in rows:
        row_genres = {g.lower() for g in (row.get("genres") or [])}
        assert "romance" not in row_genres, (
            f"'{row['title']}' leaked through exclude_genres: {row['genres']}"
        )
