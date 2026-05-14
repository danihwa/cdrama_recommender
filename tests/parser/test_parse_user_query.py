"""Parser eval suite for parse_user_query — real LLM calls against gpt-4o-mini.

These are golden-set evals: they call the real OpenAI API to verify that
the parser prompt + model combination produces correct structured output.
This is different from unit tests (pure logic, no I/O) and from
retrieval tests (which test DB-backed filter wiring, not LLM behaviour).

Why real API calls instead of mocks?
    The whole point is to catch prompt regressions — if the LLM starts
    misclassifying search_mode or dropping filters after a prompt edit,
    mocks wouldn't catch that.  The tradeoff is cost (~$0.01 per full run)
    and speed (~50s), which is why we gate them behind a marker.

Assertion design — two key ideas:

  1. PARTIAL assertions: each case only checks the fields it cares about.
     If a case tests min_year parsing, it doesn't assert on genres.  This
     means adding a new field to QueryFilters won't break unrelated tests.

  2. FLEXIBLE matching via callables: instead of exact equality, a case can
     use a lambda for fuzzy checks.  This is critical because temperature=0
     is near-deterministic but not perfectly so — the LLM might capitalise
     a title differently or add an extra genre.

     Plain value:   "search_mode": "reference"        -> exact match
     Lambda:        "reference_title": lambda t: ...   -> truthy check

On top of per-case checks, every case also runs a set of cross-mode
invariants (reference_title must be None outside reference mode, etc.)
and a title-not-in-genres guard that catches a known prompt regression.

Cases live in tests/fixtures/parser_cases.py so the notebook can share them.

Run all:   uv run pytest tests/parser/test_parse_user_query.py -v
Run one:   uv run pytest tests/parser/test_parse_user_query.py -v -k "punctuation_title"
Skip:      uv run pytest -m "not parser"
"""

from __future__ import annotations

import pytest
from openai import OpenAI

from src.recommender.models import QueryFilters
from src.recommender.pipeline import parse_user_query
from tests.fixtures.parser_cases import CASES, MODE_CASES

ALL_CASES = CASES + MODE_CASES


# ---------------------------------------------------------------------------
# Assertion helpers
# ---------------------------------------------------------------------------


def _assert_field(actual_value, expected):
    if callable(expected):
        assert expected(actual_value), (
            f"Custom assertion failed for value: {actual_value!r}"
        )
    else:
        assert actual_value == expected, (
            f"Expected {expected!r}, got {actual_value!r}"
        )


def _assert_invariants(result: QueryFilters, case: dict) -> None:
    """Invariants that must hold for every case regardless of what it tests.

    Catches "silent" regressions — e.g. reference_title leaking into semantic
    mode — that per-field partial assertions wouldn't notice.
    """
    mode = result.search_mode

    if mode != "reference":
        assert result.reference_title is None, (
            f"reference_title must be None in {mode} mode, "
            f"got {result.reference_title!r}"
        )
    if mode != "semantic":
        assert result.description is None, (
            f"description must be None in {mode} mode, "
            f"got {result.description!r}"
        )

    # "NEVER put a drama title in genres." — real regression we've seen.
    if result.reference_title:
        title_words = {w.lower() for w in result.reference_title.split() if len(w) > 2}
        genre_words = {g.lower() for g in result.genres}
        leaked = title_words & genre_words
        assert not leaked, (
            f"Title words leaked into genres: {leaked} "
            f"(title={result.reference_title!r}, genres={result.genres})"
        )

    # "A genre must never appear in both lists." — contradictory SQL filter.
    if result.genres and result.exclude_genres:
        overlap = (
            {g.lower() for g in result.genres}
            & {g.lower() for g in result.exclude_genres}
        )
        assert not overlap, (
            f"Genres appear in both include and exclude: {overlap} "
            f"(genres={result.genres}, exclude_genres={result.exclude_genres})"
        )


# ---------------------------------------------------------------------------
# Parametrised test
# ---------------------------------------------------------------------------


# @pytest.mark.parser lets us skip these in CI where there's no OpenAI key:
#   uv run pytest -m "not parser"
# The marker is registered in pyproject.toml so --strict-markers doesn't warn.
@pytest.mark.parser
@pytest.mark.parametrize("case", ALL_CASES, ids=[c["id"] for c in ALL_CASES])
def test_parse_user_query(case: dict, openai_client: OpenAI) -> None:
    result = parse_user_query(
        user_query=case["query"],
        openai=openai_client,
        history=case.get("history"),
    )

    for field, expected in case["expect"].items():
        actual = getattr(result, field)
        _assert_field(actual, expected)

    _assert_invariants(result, case)
