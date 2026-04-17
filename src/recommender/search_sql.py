"""
SQL-mode retrieval: pure structured filters, no vector search.

Used when the user gives ONLY structured criteria ("romance from 2022
rated above 8") with no plot description and no reference drama. There's
nothing to embed here, so the vector layer is skipped entirely and just run
a plain SQL query against cdramas.

Note that the other two modes also apply these same filters. 
The difference is that semantic/reference do so via the match_documents RPC
(Postgres function) while this mode uses the Supabase query builder
directly (no embedding to join against).
"""


from __future__ import annotations

from supabase import Client

from src.recommender._shared import find_exclude_ids, normalize_genres
from src.recommender.models import QueryFilters


SELECT_COLUMNS = (
    "id, title, native_title, year, synopsis, "
    "mdl_score, genres, tags, watchers, mdl_url"
)


def retrieve_sql_candidates(
    filters: QueryFilters,
    supabase: Client,
    match_count: int,
) -> list[dict]:
    """Pure filter query — no embedding, no similarity.

    Ordered by mdl_score DESC so the reranker (which will see similarity=0
    for every row) still gets a quality-first starting order.
    The ensemble score collapses to quality + popularity.
    """
    exclude_ids = find_exclude_ids(filters.exclude_titles, supabase)

    # Start with a base query and chain filters onto it conditionally.
    # The Supabase query builder is lazy — nothing hits the DB until
    # .execute() is called — so reassigning `query` just accumulates
    # WHERE clauses without triggering any network round-trips.
    query = supabase.table("cdramas").select(SELECT_COLUMNS)

    # gte = "greater than or equal". Only apply the clause if the filter is set
    if filters.min_year is not None:
        query = query.gte("year", filters.min_year)
    if filters.min_score is not None:
        query = query.gte("mdl_score", filters.min_score)
    # overlaps maps to Postgres array overlap (&&): true if the drama's
    # genres share at least one element with the filter 
    # contains would require the drama to have ALL the listed genres.
    if filters.genres:
        query = query.overlaps("genres", normalize_genres(filters.genres))
    if filters.exclude_genres:
        query = query.not_.overlaps(
            "genres", normalize_genres(filters.exclude_genres)
        )
    if exclude_ids:
        query = query.not_.in_("id", exclude_ids)

    result = (
        query.order("mdl_score", desc=True)
        .limit(match_count)
        .execute()
    )
    return result.data


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
    print(f"Got {len(rows)} rows:")
    for r in rows:
        print(f"  [{r['year']}] {r['title']} — {r['mdl_score']}")