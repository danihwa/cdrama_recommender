"""Helpers shared by all three retrieval strategies.

Every search module (reference / semantic / sql) needs to normalize
genres (done either directly in the sql path or indirectly via vector_search),
resolve exclusion titles to DB ids, or call the vector-search RPC — so
they live here to avoid duplication.
"""

from __future__ import annotations

import re

from supabase import Client

from src.recommender.models import QueryFilters

# Cosine-similarity floor for the match_documents RPC. Rows below this are
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
    title: str, supabase: Client, columns: str = "id"
) -> dict | None:
    """Look up a single drama by partial title match.

    Uses ``ILIKE '%...%'`` for a forgiving substring match. If the first
    attempt misses, retries once with punctuation stripped from the input —
    catches LLM parser variations like "How Dare You?!" vs the stored
    "How Dare You!?".

    ``columns`` is passed straight to ``.select()`` so callers can fetch
    just the id (for exclusions) or extra fields like ``embedding`` (for
    reference search).
    """
    # Build the list of patterns to try: original title first, then a
    # punctuation-stripped fallback if it differs. [^\w\s] matches any
    # char that's neither a word char (\w = [A-Za-z0-9_]) nor whitespace.
    patterns = [title]
    stripped = re.sub(r"[^\w\s]", "", title).strip()
    if stripped and stripped != title:
        patterns.append(stripped)

    for pattern in patterns:
        result = (
            supabase.table("cdramas")
            .select(columns)
            .ilike("title", f"%{pattern}%")
            .limit(1)
            .execute()
        )
        if result.data:
            return result.data[0]
    return None


def find_exclude_ids(titles: list[str], supabase: Client) -> list[int]:
    """Match drama titles to DB ids so they can be excluded from results.

    The downstream exclusion filter works on ids, not titles, so each
    title has to be looked up first.
    """
    ids: list[int] = []
    for title in titles:
        row = lookup_drama_by_title(title, supabase)
        if row is not None:
            ids.append(row["id"])
        else:
            print(f"   Warning: '{title}' not found in DB — cannot exclude it")
    return ids


def vector_search(
    query_vector: list[float],
    filters: QueryFilters,
    exclude_ids: list[int],
    supabase: Client,
    match_count: int,
) -> list[dict]:
    """Run cosine-similarity search via the ``match_documents`` RPC (Remote Procedure Call).

    ``match_documents`` is a Postgres function defined on Supabase (see
    ``src/database/functions.sql``). It performs the cosine-similarity search
    against the ``embedding`` column *and* applies the year / score / genre /
    exclusion filters in the same query, which is much more efficient
    than fetching a large number of results and filtering in Python.

    Unset filters are passed as ``None`` — the RPC's ``IS NULL`` check
    short-circuits each ``WHERE`` clause, so the filter is effectively
    disabled. ``QueryFilters`` already leaves ``min_year`` / ``min_score``
    as ``None`` when unset; empty list filters are explicitly coerced to
    ``None`` here to match that shape.
    """
    # Coerce empty lists to None so the RPC sees the same "unset" shape
    # for every optional filter.
    filter_exclude_ids = exclude_ids or None
    filter_genres = normalize_genres(filters.genres) if filters.genres else None
    filter_exclude_genres = (
        normalize_genres(filters.exclude_genres) if filters.exclude_genres else None
    )

    result = supabase.rpc(
        "match_documents",
        {
            "query_embedding": query_vector,
            "match_threshold": MATCH_THRESHOLD,
            "match_count": match_count,
            "filter_year": filters.min_year,
            "filter_score": filters.min_score,
            "exclude_ids": filter_exclude_ids,
            "filter_genres": filter_genres,
            "exclude_genres": filter_exclude_genres,
        },
    ).execute()
    return result.data
