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


def normalize_genres(genres: list[str]) -> list[str]:
    """
    Genres are stored lowercase in the DB.
    The parser LLM may return genres in any case, and with extra whitespace ->
    normalize to avoid false mismatches.
    """
    return [g.lower().strip() for g in genres]


def find_exclude_ids(titles: list[str], supabase: Client) -> list[int]:
    """
    Match drama titles to DB ids so they can be excluded from results.

    The downstream exclusion filter works on ids, not titles,
    so each title has to be looked up first. ``ILIKE '%...%'``
    gives a forgiving substring match.

    If the first attempt misses, we retry once with non-word characters
    stripped from the user's input. This catches punctuation mismatches
    like "How Dare You?!" against a stored "How Dare You!?".
    """

    def _lookup(query: str) -> int | None:
        result = (
            supabase.table("cdramas")
            .select("id")
            .ilike("title", f"%{query}%")
            .limit(1)
            .execute()
        )
        return result.data[0]["id"] if result.data else None

    ids: list[int] = []
    for title in titles:
        match_id = _lookup(title)
        if match_id is None:
            stripped = re.sub(r"[^\w\s]", "", title).strip()
            if stripped and stripped != title:
                match_id = _lookup(stripped)
        if match_id is not None:
            ids.append(match_id)
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

    The ``match_threshold`` of 0.3 is a hard floor — below that, results are
    effectively noise for our embedding model.

    Unset filters are passed as ``None`` — the RPC's ``IS NULL`` check
    short-circuits each ``WHERE`` clause, so the filter is effectively
    disabled. ``QueryFilters`` already leaves ``min_year`` / ``min_score``
    as ``None`` when unset; empty ``exclude_ids`` / ``filter_genres``
    lists are explicitly coerced to ``None`` to match that shape.
    """
    result = supabase.rpc(
        "match_documents",
        {
            "query_embedding": query_vector,
            "match_threshold": 0.3,
            "match_count": match_count,
            "filter_year": filters.min_year,
            "filter_score": filters.min_score,
            "exclude_ids": exclude_ids if exclude_ids else None,
            "filter_genres": (
                normalize_genres(filters.genres) if filters.genres else None
            ),
        },
    ).execute()
    return result.data
