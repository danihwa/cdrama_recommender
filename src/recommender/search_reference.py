"""
Reference-mode retrieval: "recommend something like <drama X>".

This is the primary recommender flow. The strategy is simple - every drama
in the DB already has an embedding vector stored alongside it 
(computed from its synopsis+tags during ingestion). To find dramas "similar to X":

  1. Look up drama X in the DB and read its embedding.
  2. Run a vector search using X's embedding as the query.
  3. The nearest neighbors in embedding space are our candidates.

No LLM call is needed at retrieval time — it's reusing the embeddings
computed at ingestion.
"""

from __future__ import annotations

import psycopg

from src.recommender._shared import (
    find_exclude_ids,
    lookup_drama_by_title,
    print_smoke_results,
    vector_search,
)
from src.recommender.models import QueryFilters


def get_reference_drama(
    title: str, conn: psycopg.Connection
) -> tuple[int, list[float]] | None:
    """Fetches the (id, embedding) pair for a reference drama by title.

    Returns ``None`` if no row matches — id and embedding always travel
    together, so a single sentinel keeps the caller from null-checking
    each field.
    """
    row = lookup_drama_by_title(title, conn, columns="id, embedding, title")
    if row is None:
        print(f"Warning: '{title}' not found in database")
        return None

    print(f"   Found reference drama: '{row['title']}'")
    return row["id"], row["embedding"]


def retrieve_reference_candidates(
    filters: QueryFilters,
    conn: psycopg.Connection,
    match_count: int,
) -> list[dict]:
    """Vector search anchored on an existing drama's embedding.

    Returns [] if the reference title is missing or cannot be resolved —
    the pipeline treats that as a no-results outcome rather than silently
    falling back. Deliberately no fallback to semantic search: if the user
    named a specific drama, silently switching strategies would be more
    confusing than admitting the miss.
    """
    # Narrow QueryFilters.reference_title (str | None) → str before
    # handing it to get_reference_drama, which only accepts str. Mirrors
    # the same guard in search_semantic.py for filters.description.
    title = (filters.reference_title or "").strip()
    if not title:
        print("Warning: reference search requested but no reference_title provided")
        return []

    print(f"\nReference drama: '{title}'")
    reference = get_reference_drama(title, conn)
    if reference is None:
        return []
    ref_id, query_vector = reference

    # An embedding has cosine similarity 1.0 with itself, so without this
    # the reference drama would always come back as its own top match.
    # Resolved after the reference check so exclude lookups don't run
    # when we're about to return [] anyway.
    exclude_ids = [ref_id] + find_exclude_ids(filters.exclude_titles, conn)
    return vector_search(query_vector, filters, exclude_ids, conn, match_count)


if __name__ == "__main__":
    # Smoke test: look up a reference drama and pull its nearest neighbors.
    # to run: uv run src/recommender/search_reference.py
    from src.database.connection import get_db_connection

    filters = QueryFilters(
        search_mode="reference",
        reference_title="Love between lines",
        min_score=8.0,
    )
    rows = retrieve_reference_candidates(
        filters,
        get_db_connection(),
        match_count=5,
    )
    print_smoke_results(rows)