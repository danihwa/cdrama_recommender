"""
Semantic-mode retrieval: "I saw a drama where <plot description>".

Used when the user describes what they want but can't name a reference
drama. Contrast with reference mode:

  - reference: embedding comes from an existing DB row (no LLM call)
  - semantic:  OpenAI embeddings API is called on the user's own text,
               then search with that fresh vector

The search itself (comparing one vector against all drama embeddings) is
identical in both modes — only the source of the query vector differs.
"""

from __future__ import annotations

import psycopg
from openai import OpenAI

from src.recommender._shared import find_exclude_ids, vector_search
from src.recommender.models import QueryFilters

# CRITICAL: both constants MUST match what was used to embed the dramas at
# ingestion time. Vectors from different models — or the same model with
# different dimensions — live in different spaces, and cosine similarity
# between them is meaningless. 
EMBEDDING_MODEL = "text-embedding-3-large"
EMBEDDING_DIMENSIONS = 3072


def embed_query(text: str, openai: OpenAI) -> list[float]:
    """Embeds a free-form query for vector search against drama embeddings."""
    response = openai.embeddings.create(
        model=EMBEDDING_MODEL,
        dimensions=EMBEDDING_DIMENSIONS,
        input=text,
    )
    return response.data[0].embedding


def retrieve_semantic_candidates(
    filters: QueryFilters,
    conn: psycopg.Connection,
    openai: OpenAI,
    match_count: int,
    fallback_query: str = "",
) -> list[dict]:
    """Vector search using an embedding of the user's own description.

    Ideally the parser LLM isolated a clean description ("heroine had amnesia,
    enemies-to-lovers") into ``filters.description``. If it didn't, we embed
    the raw user query as a last resort — worse signal because it includes
    filter words like "rated above 8", but better than returning nothing.
    """
    query_text = (filters.description or fallback_query).strip()
    if not query_text:
        print("Warning: semantic search requested but no description available")
        return []

    print(f"\nSemantic query: '{query_text}'")
    query_vector = embed_query(query_text, openai)

    exclude_ids = find_exclude_ids(filters.exclude_titles, conn)
    return vector_search(query_vector, filters, exclude_ids, conn, match_count)


if __name__ == "__main__":
    # Smoke test: embed a plot description and pull top matches.
    # to run: uv run src/recommender/search_semantic.py
    from src.database.connection import get_db_connection

    filters = QueryFilters(
        search_mode="semantic",
        description="heroine loses her memory and falls for her former enemy",
        min_score=8.0,
    )
    rows = retrieve_semantic_candidates(
        filters,
        get_db_connection(),
        OpenAI(),
        match_count=5,
    )
    print(f"Got {len(rows)} rows:")
    for r in rows:
        print(
            f"  [{r['year']}] {r['title']} "
            f"— score {r['mdl_score']}, similarity {r['similarity']:.3f}"
        )