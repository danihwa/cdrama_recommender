"""Pydantic schemas for the recommender pipeline.

Defines QueryFilters — the structured representation of a user's natural
language query that the parser LLM produces and every downstream stage
(routing, retrieval, reranking) consumes.
"""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel

# Literal restricts a type to a fixed set of allowed values. SearchMode
# can only ever be one of these three strings — type checkers and pydantic
# will reject anything else.
SearchMode = Literal["reference", "semantic", "sql"]


class QueryFilters(BaseModel):
    """Structured representation of the user's natural language query."""

    # Which retrieval strategy the pipeline should use:
    #   - "reference": vector search anchored on an existing drama's embedding
    #   - "semantic":  vector search on an embedding of the user's own query text
    #   - "sql":       pure filter query (genre/year/rating), no embedding
    #
    # Default is "reference" because it's the primary recommender flow
    # ("recommend something like X"). The parser LLM overrides this on
    # every real query, so the default only kicks in as a safety fallback
    search_mode: SearchMode = "reference"

    # Set when user says "similar to X" / "like X" / "recommend something like X".
    # Required for reference mode; ignored otherwise.
    reference_title: str | None = None

    # Free-form description of plot / characters / vibe when the user cannot
    # name a reference drama ("heroine had amnesia and was enemies with the
    # hero"). Used as the text to embed in semantic mode.
    description: str | None = None

    # Explicit genre names extracted from the query, e.g. ["Romance", "Historical"].
    # Used as a hard SQL filter (genres && ARRAY[...]) in every mode.
    genres: list[str] = []

    # Hard lower bounds for year and score — applied as SQL WHERE clauses.
    min_year: int | None = None
    min_score: float | None = None

    # Dramas the user says they already watched — excluded from results.
    # In reference mode, the reference drama is always excluded too.
    exclude_titles: list[str] = []
