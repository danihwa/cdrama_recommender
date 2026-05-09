"""End-to-end RAG pipeline for the drama recommender.

This module is the orchestrator — it's the only file in `recommender/`
that knows about the whole flow. The other files each handle one piece:

    models.py            → schema (QueryFilters)
    _shared.py           → primitives used by every search mode
    search_reference.py  → "similar to X" retrieval
    search_semantic.py   → plot-description retrieval
    search_sql.py        → pure filter retrieval

The pipeline here is the classic RAG shape: Parse → Route → Retrieve →
Rerank → Generate. Each function below corresponds to one stage, and
`run_rag` is the entry point that chains them together.
"""

from __future__ import annotations

import math
from typing import Iterator

import psycopg
from openai import OpenAI

from src.database.connection import get_db_connection
from src.env import load_secrets
from src.recommender.models import QueryFilters
from src.recommender.search_reference import retrieve_reference_candidates
from src.recommender.search_semantic import retrieve_semantic_candidates
from src.recommender.search_sql import retrieve_sql_candidates


MATCH_COUNT = 10  # candidates to fetch before reranking
TOP_N = 5  # candidates passed to the generator after reranking
MAX_RESPONSE_TOKENS = 500  # generator output budget

# History window shared by parser and generator (6 messages ≈ last 3 turns).
HISTORY_MESSAGES = 6


MODEL = "gpt-4o-mini"

# Off-topic / harmful / injection queries are refused via the SearchMode
# Literal in QueryFilters — the schema enum is enough; adding a refusal
# section to this prompt regresses semantic mode (see notebooks/04a).
PARSE_SYSTEM_PROMPT = """\
You are a Chinese drama expert. Extract search parameters from the user's query.

First decide search_mode — one of:
- "reference": user names a specific drama as an anchor ("similar to X", "like X", "more like X"). Put X in reference_title.
- "semantic":  user describes plot / characters / vibe but cannot name a title. Put the description in the description field — capture plot/character cues only, NOT filters like year or rating.
- "sql":       user gives only structured filters — genre, year, rating — with no plot description and no reference title.

Field rules:
- reference_title: only set in reference mode. NEVER put a drama title in genres.
- description: only set in semantic mode. Leave null/empty if user did not describe a plot.
- genres: only explicit genre words (romance, historical, wuxia, fantasy, mystery, etc). Ignore vague words like 'good' or 'fun'.
- exclude_genres: genre names the user wants AVOIDED ("no romance", "not wuxia", "avoid fantasy"). Put them here, NEVER in genres. A genre must never appear in both lists.
- exclude_titles: any drama the user says they watched, finished, didn't enjoy, or wants skipped.
- min_year: 'after 20XX' = 20XX+1, 'from 20XX onwards' = 20XX, 'no older than 20XX' = 20XX.
- min_score: 'rating above X' = X, 'rating at least X' = X, 'highly rated'/'top rated' = 8.5, 'good rating' = 8.0.

Leave a field null/empty if the user did not specify it — do not guess.\
"""

RECOMMEND_SYSTEM_PROMPT = """\
You are a warm, enthusiastic Chinese drama recommender.
Recommend ONLY dramas from the provided context — never invent titles.
Pick the 3 best matches. For each, write one paragraph explaining specifically why it fits the user's request.
Mention the MDL score as a quality signal.
Do not repeat yourself. Each recommendation is exactly one paragraph.
If you find yourself repeating content, stop.
If the request is not about drama recommendations, decline politely and suggest the user ask about a genre, mood, or drama they enjoyed.\
"""


def parse_user_query(
    user_query: str, openai: OpenAI, history: list[dict] | None = None
) -> QueryFilters:
    """
    Parses a natural language query into a structured QueryFilters object.

    Uses openai.chat.completions.parse - OpenAI's "structured outputs"
    endpoint: Pydantic model is passed as response_format and the API
    guarantees the response validates against it.

    Temperature set to 0 keeps the parser as deterministic as possible.
    Passes the last HISTORY_MESSAGES messages so follow-ups like "something
    older" can be resolved against prior context.
    """
    completion = openai.chat.completions.parse(
        model=MODEL,
        temperature=0,
        messages=[
            {"role": "system", "content": PARSE_SYSTEM_PROMPT},
            *(history or [])[-HISTORY_MESSAGES:],
            {"role": "user", "content": user_query},
        ],
        response_format=QueryFilters,
    )
    parsed = completion.choices[0].message.parsed
    if parsed is None:
        raise ValueError("Parser LLM returned no structured output")
    return parsed


def retrieve_candidates(
    user_query: str,
    filters: QueryFilters,
    conn: psycopg.Connection,
    openai: OpenAI,
    match_count: int = MATCH_COUNT,
) -> list[dict]:
    """Routes to the right retrieval strategy based on filters.search_mode."""
    if filters.search_mode == "reference":
        return retrieve_reference_candidates(filters, conn, match_count)

    if filters.search_mode == "semantic":
        return retrieve_semantic_candidates(
            filters, conn, openai, match_count, fallback_query=user_query
        )

    if filters.search_mode == "sql":
        return retrieve_sql_candidates(filters, conn, match_count)

    # Keeps the type checker happy and would catch a
    # future bug if fourth mode was added and the router wasn't updated.
    raise ValueError(f"Unknown search_mode: {filters.search_mode}")


def rerank_candidates(
    candidates: list[dict],
    *,
    w_sim: float = 0.70,
    w_quality: float = 0.20,
    w_popularity: float = 0.10,
) -> list[dict]:
    """
    Ensemble reranker combining three signals:

      similarity  (w_sim)       — cosine similarity from match_documents, [0, 1].
      quality     (w_quality)   — MDL score / 10, range [0, 1].
      popularity  (w_popularity) — log-scaled watchers, normalised to the
                                   candidate-set max (not globally).

    The weights are nominal, not effective. In a top-10 set the similarity
    values are usually clustered in a narrow band (~0.1 spread), so in
    practice similarity dominates by roughly 2x rather than 7x — all three
    signals end up contributing comparable spread to the final ordering.

    Two runtime notes for when an ordering looks surprising:
    - SQL mode: every candidate has similarity=0, so the ranking collapses
      to quality+popularity — which matches the intent of a filter-only query.
    - min_score filter: match_documents has already pruned low-scoring rows
      in SQL, so the quality term can barely differentiate the survivors.
      It earns its weight mostly on queries without an explicit score filter.
    """
    if not candidates:
        return candidates

    # Log-scaling trick: popular shows can have 100x more watchers than
    # niche ones, and a linear scale would let a single blockbuster
    # dominate the popularity term. Taking log(watchers) compresses that
    # range, and dividing by log(max_watchers) normalises to [0, 1].
    # The max is per-batch, not global — the most-watched drama in this
    # candidate set always gets popularity=1.0, even if it's niche overall.
    max_watchers = max((d.get("watchers") or 1 for d in candidates), default=1)
    # log(1) = 0 would make every popularity score divide-by-zero; floor
    # the divisor at 1 for the rare batch where every drama has 1 watcher.
    log_max_watchers = math.log(max_watchers) if max_watchers > 1 else 1

    for drama in candidates:
        similarity = drama.get("similarity", 0.0)
        quality = (drama.get("mdl_score") or 0.0) / 10.0
        popularity = _popularity_score(drama, log_max_watchers)
        drama["ensemble_score"] = (
            w_sim * similarity + w_quality * quality + w_popularity * popularity
        )
    return sorted(candidates, key=lambda d: d["ensemble_score"], reverse=True)


def _popularity_score(drama: dict, log_max_watchers: float) -> float:
    """Log-scaled watcher count, normalised to [0, 1] against the batch max."""
    watchers = drama.get("watchers") or 1
    return math.log(watchers) / log_max_watchers


def _format_drama(d: dict) -> str:
    """Renders one drama as the four-line block the generator LLM expects."""
    genres = ", ".join(d.get("genres") or [])
    tags = ", ".join((d.get("tags") or [])[:5])
    return (
        f"Title: {d['title']} ({d['year']}) | MDL Score: {d['mdl_score']}\n"
        f"Genres: {genres}\n"
        f"Tags: {tags}\n"
        f"Synopsis: {d['synopsis']}"
    )


def build_context(dramas: list[dict]) -> str:
    """Formats candidate dramas into the text block the generator LLM sees."""
    return "\n\n".join(_format_drama(d) for d in dramas)


def generate_recommendation(
    user_query: str,
    dramas: list[dict],
    openai: OpenAI,
    history: list[dict] | None = None,
) -> str:
    """Generates a personalised recommendation grounded in the provided candidates."""
    recent_history = (history or [])[-HISTORY_MESSAGES:]
    response = openai.chat.completions.create(
        model=MODEL,
        max_tokens=MAX_RESPONSE_TOKENS,
        messages=[
            {"role": "system", "content": RECOMMEND_SYSTEM_PROMPT},
            *recent_history,
            {
                "role": "user",
                "content": f"""\
My request: {user_query}

Dramas to choose from:
{build_context(dramas)}\
""",
            },
        ],
    )
    content = response.choices[0].message.content
    if content is None:
        raise ValueError("Generator LLM returned no content")
    return content


def generate_recommendation_stream(
    user_query: str,
    dramas: list[dict],
    openai: OpenAI,
    history: list[dict] | None = None,
) -> Iterator[str]:
    """Streaming variant of generate_recommendation — yields tokens.

    Mirrors generate_recommendation but uses stream=True so callers can
    forward partial output (e.g., over SSE) instead of waiting for the
    whole response. Empty deltas (role-only or finish-reason chunks) are
    skipped so callers always see real text.
    """
    recent_history = (history or [])[-HISTORY_MESSAGES:]
    stream = openai.chat.completions.create(
        model=MODEL,
        max_tokens=MAX_RESPONSE_TOKENS,
        stream=True,
        messages=[
            {"role": "system", "content": RECOMMEND_SYSTEM_PROMPT},
            *recent_history,
            {
                "role": "user",
                "content": f"""\
My request: {user_query}

Dramas to choose from:
{build_context(dramas)}\
""",
            },
        ],
    )
    for chunk in stream:
        delta = chunk.choices[0].delta.content
        if delta:
            yield delta


NO_RESULTS_MESSAGE = (
    "Sorry, no dramas found matching those criteria. Try relaxing the filters!"
)

REFUSED_MESSAGE = (
    "I'm only able to help with Chinese drama recommendations! "
    "Feel free to ask me about a genre, mood, or a drama you've enjoyed."
)


def run_rag(
    user_query: str,
    conn: psycopg.Connection,
    openai: OpenAI,
    history: list[dict] | None = None,
) -> tuple[str, list[dict]]:
    """Full RAG pipeline: Parse → Route → Retrieve → Rerank → Generate."""
    history = history or []

    # Stage 1 — Parse: LLM turns natural language into QueryFilters.
    print("Parsing your request...")
    filters: QueryFilters = parse_user_query(user_query, openai, history)
    print(f"   Filters: {filters.model_dump()}")
    print(f"   Mode: {filters.search_mode}")

    if filters.search_mode == "refused":
        return REFUSED_MESSAGE, []

    # Stages 2+3 — Route + Retrieve: pick the right search strategy and pull candidates.
    candidates = retrieve_candidates(user_query, filters, conn, openai)

    if not candidates:
        return NO_RESULTS_MESSAGE, []

    # Stage 4 — Rerank: blend similarity with quality + popularity.
    print(f"\nReranking {len(candidates)} candidates...")
    reranked = rerank_candidates(candidates)

    for d in reranked[:TOP_N]:
        print(
            f"   {d['title']:<40} "
            f"ensemble: {d.get('ensemble_score', 0):.3f} | "
            f"sim: {d.get('similarity', 0):.3f} | "
            f"score: {d.get('mdl_score', 0)}"
        )

    # Stage 5 — Generate
    top = reranked[:TOP_N]
    print("\nGenerating recommendations...\n")
    response = generate_recommendation(user_query, top, openai, history)
    return response, top


if __name__ == "__main__":
    # to run: uv run src/recommender/pipeline.py
    load_secrets()
    openai_client = OpenAI()
    conn = get_db_connection()

    query = "Recommend me something similar to How Dare You!? I already saw Dream within a dream. The drama should be rated above 8"
    print("=" * 60)
    print(f"Query: {query}")
    print("=" * 60)
    response, _ = run_rag(query, conn, openai_client)
    print(response)
