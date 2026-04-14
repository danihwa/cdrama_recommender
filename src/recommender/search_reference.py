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

import re

from supabase import Client

from src.recommender._shared import find_exclude_ids, vector_search
from src.recommender.models import QueryFilters


def get_reference_drama(
    title: str, supabase: Client
) -> tuple[int | None, list[float] | None]:
    """Fetches the id and embedding for a reference drama by title.

    Falls back to a punctuation-stripped search if the exact title yields no
    results, to handle LLM parser variations like 'How dare you?!' vs
    'How Dare You!?'.
    """
    # Nested helper so the "try raw, retry stripped" pair below doesn't
    # duplicate the query. Kept inside because no other caller needs it —
    # nested defs are a lightweight way to scope helpers to one function.
    def search(pattern: str) -> list[dict]:
        return (
            supabase.table("cdramas")
            .select("id, embedding, title")
            .ilike("title", f"%{pattern}%")
            .limit(1)
            .execute()
        ).data

    data = search(title)

    # Retry with punctuation stripped. [^\w\s] matches any char that's
    # neither a word char (\w = [A-Za-z0-9_]) nor whitespace (\s) —
    # i.e. punctuation and symbols. The negation is what makes it concise.
    if not data:
        stripped = re.sub(r"[^\w\s]", "", title).strip()
        if stripped and stripped != title:
            data = search(stripped)

    if data:
        row = data[0]
        print(f"   Found reference drama: '{row['title']}'")
        return row["id"], row["embedding"]

    print(f"Warning: '{title}' not found in database")
    return None, None


def retrieve_reference_candidates(
    filters: QueryFilters,
    supabase: Client,
    match_count: int,
) -> list[dict]:
    """Vector search anchored on an existing drama's embedding.

    Returns [] if the reference title is missing or cannot be resolved —
    the pipeline treats that as a no-results outcome rather than silently
    falling back.
    """
    # Narrow QueryFilters.reference_title (str | None) → str before
    # handing it to get_reference_drama, which only accepts str. Mirrors
    # the same guard in search_semantic.py for filters.description.
    title = (filters.reference_title or "").strip()
    if not title:
        print("Warning: reference search requested but no reference_title provided")
        return []

    exclude_ids = find_exclude_ids(filters.exclude_titles, supabase)

    print(f"\nReference drama: '{title}'")
    ref_id, query_vector = get_reference_drama(title, supabase)

    # An embedding has cosine similarity 1.0 with itself, so without this
    # the reference drama would always come back as its own top match.
    if ref_id is not None:
        exclude_ids = [ref_id] + exclude_ids

    # Reference couldn't be resolved → return empty; the pipeline turns
    # this into a no-results message. Deliberately no fallback to semantic
    # search: if the user named a specific drama, silently switching
    # strategies would be more confusing than admitting the miss.
    if query_vector is None:
        return []

    return vector_search(query_vector, filters, exclude_ids, supabase, match_count)


if __name__ == "__main__":
    # Smoke test: look up a reference drama and pull its nearest neighbors.
    # to run: uv run src/recommender/search_reference.py
    from src.database.connection import get_db_connection

    filters = QueryFilters(
        search_mode="reference",
        reference_title="Nirvana in Fire",
        min_score=8.0,
    )
    rows = retrieve_reference_candidates(
        filters,
        get_db_connection(),
        match_count=5,
    )
    print(f"Got {len(rows)} rows:")
    for r in rows:
        print(
            f"  [{r['year']}] {r['title']} "
            f"— score {r['mdl_score']}, similarity {r['similarity']:.3f}"
        )