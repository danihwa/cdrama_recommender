"""Helpers shared by all three retrieval strategies.

Every search module (reference / semantic / sql) needs to normalize
genres (done either directly in the sql path or indirectly via vector_search),
resolve exclusion titles to DB ids, or call the vector-search SQL function — so
they live here to avoid duplication.
"""

from __future__ import annotations

import re

import psycopg
from psycopg.rows import dict_row

from src.recommender.models import QueryFilters

# Cosine-similarity floor for the match_documents function. Rows below this are
# dropped server-side, so the reranker never sees obvious non-matches.
MATCH_THRESHOLD = 0.3


def normalize_genres(genres: list[str]) -> list[str]:
    """
    Genres are stored lowercase in the DB.
    The parser LLM may return genres in any case, and with extra whitespace ->
    normalize to avoid false mismatches.
    """
    return [g.lower().strip() for g in genres]


def lookup_drama_by_title(
    title: str, conn: psycopg.Connection, columns: str = "id"
) -> dict | None:
    """Look up a single drama by partial title match.

    Uses ``ILIKE '%...%'`` for a forgiving substring match. If the first
    attempt misses, retries once with punctuation stripped from the input —
    catches LLM parser variations like "How Dare You?!" vs the stored
    "How Dare You!?".

    ``columns`` is interpolated directly into the SELECT list — callers
    pass static column names ("id", "id, embedding, title"), never user
    input, so SQL injection is not in scope here.
    """
    patterns = [title]
    stripped = re.sub(r"[^\w\s]", "", title).strip()
    if stripped and stripped != title:
        patterns.append(stripped)

    sql = f"SELECT {columns} FROM cdramas WHERE title ILIKE %s LIMIT 1"
    with conn.cursor(row_factory=dict_row) as cur:
        for pattern in patterns:
            cur.execute(sql, (f"%{pattern}%",))
            row = cur.fetchone()
            if row is not None:
                return row
    return None


def find_exclude_ids(titles: list[str], conn: psycopg.Connection) -> list[int]:
    """Match drama titles to DB ids so they can be excluded from results."""
    ids: list[int] = []
    for title in titles:
        row = lookup_drama_by_title(title, conn)
        if row is not None:
            ids.append(row["id"])
        else:
            print(f"   Warning: '{title}' not found in DB — cannot exclude it")
    return ids


def vector_search(
    query_vector: list[float],
    filters: QueryFilters,
    exclude_ids: list[int],
    conn: psycopg.Connection,
    match_count: int,
) -> list[dict]:
    """Run cosine-similarity search via the ``match_documents`` SQL function.

    ``match_documents`` is a Postgres function defined in
    ``src/database/functions.sql``. It performs the cosine-similarity
    search against the ``embedding`` column *and* applies the year /
    score / genre / exclusion filters in the same query, which is much
    more efficient than fetching a large number of results and filtering
    in Python.

    Unset filters are passed as ``None`` — the function's ``IS NULL``
    check short-circuits each ``WHERE`` clause, so the filter is
    effectively disabled.
    """
    filter_exclude_ids = exclude_ids or None
    filter_genres = normalize_genres(filters.genres) if filters.genres else None
    filter_exclude_genres = (
        normalize_genres(filters.exclude_genres) if filters.exclude_genres else None
    )

    sql = """
        SELECT *
        FROM match_documents(
            %(query_embedding)s,
            %(match_threshold)s,
            %(match_count)s,
            %(filter_year)s,
            %(filter_score)s,
            %(exclude_ids)s,
            %(filter_genres)s,
            %(exclude_genres)s
        )
    """
    params = {
        "query_embedding": query_vector,
        "match_threshold": MATCH_THRESHOLD,
        "match_count": match_count,
        "filter_year": filters.min_year,
        "filter_score": filters.min_score,
        "exclude_ids": filter_exclude_ids,
        "filter_genres": filter_genres,
        "exclude_genres": filter_exclude_genres,
    }
    with conn.cursor(row_factory=dict_row) as cur:
        cur.execute(sql, params)
        return cur.fetchall()
