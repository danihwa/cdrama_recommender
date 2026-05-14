"""
SQL-mode retrieval: pure structured filters, no vector search.

Used when the user gives ONLY structured criteria ("romance from 2022
rated above 8") with no plot description and no reference drama. There's
nothing to embed here, so the vector layer is skipped entirely and just run
a plain SQL query against cdramas.

Note that the other two modes also apply these same filters.
The difference is that semantic/reference do so via the match_documents
Postgres function (which combines cosine search with the filters), while
this mode runs a plain SQL query — there is no embedding to join against.
"""


from __future__ import annotations

import psycopg
from psycopg.rows import dict_row

from src.recommender._shared import find_exclude_ids, normalize_genres, print_smoke_results
from src.recommender.models import QueryFilters


SELECT_COLUMNS = (
    "id, title, native_title, year, synopsis, "
    "mdl_score, genres, tags, watchers, mdl_url"
)


def retrieve_sql_candidates(
    filters: QueryFilters,
    conn: psycopg.Connection,
    match_count: int,
) -> list[dict]:
    """Pure filter query — no embedding, no similarity.

    Ordered by mdl_score DESC so the scorer (which will see similarity=0
    for every row) still gets a quality-first starting order.
    The ensemble score collapses to quality + popularity.

    Each filter is appended only when set, mirroring the IS NULL guards
    used by ``match_documents`` for the vector modes.
    """
    exclude_ids = find_exclude_ids(filters.exclude_titles, conn)

    where_clauses: list[str] = []
    params: list[object] = []

    if filters.min_year is not None:
        where_clauses.append("year >= %s")
        params.append(filters.min_year)
    if filters.min_score is not None:
        where_clauses.append("mdl_score >= %s")
        params.append(filters.min_score)
    # && is the Postgres array overlap operator: true if the drama's
    # genres share at least one element with the filter list. Using @>
    # (contains) instead would require the drama to have ALL listed genres.
    if filters.genres:
        where_clauses.append("genres && %s")
        params.append(normalize_genres(filters.genres))
    if filters.exclude_genres:
        where_clauses.append("NOT (genres && %s)")
        params.append(normalize_genres(filters.exclude_genres))
    if exclude_ids:
        where_clauses.append("NOT (id = ANY(%s))")
        params.append(exclude_ids)

    where_sql = f"WHERE {' AND '.join(where_clauses)}" if where_clauses else ""
    sql = (
        f"SELECT {SELECT_COLUMNS} FROM cdramas "
        f"{where_sql} "
        f"ORDER BY mdl_score DESC LIMIT %s"
    )
    params.append(match_count)

    with conn.cursor(row_factory=dict_row) as cur:
        cur.execute(sql, params)
        return cur.fetchall()


if __name__ == "__main__":
    # Smoke test: romance dramas from 2022+ rated 8.0+
    # to run: uv run src/recommender/search_sql.py
    from src.database.connection import get_db_connection

    filters = QueryFilters(
        search_mode="sql",
        genres=["romance"],
        min_year=2022,
        min_score=8.0,
    )
    rows = retrieve_sql_candidates(filters, get_db_connection(), match_count=5)
    print_smoke_results(rows)
