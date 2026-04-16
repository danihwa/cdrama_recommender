"""Integration tests for parse_user_query — real LLM calls against gpt-4o-mini.

These are "eval-style" integration tests: they call the real OpenAI API to
verify that the parser prompt + model combination produces correct structured
output.  This is different from unit tests (pure logic, no I/O) and from
mock-based integration tests (which test wiring, not LLM behaviour).

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

Run all:   uv run pytest tests/integration/test_parse_user_query.py -v
Run one:   uv run pytest tests/integration/test_parse_user_query.py -v -k "punctuation_title"
Skip:      uv run pytest -m "not integration"
"""

from __future__ import annotations

import pytest
from openai import OpenAI

from src.recommender.models import QueryFilters
from src.recommender.pipeline import parse_user_query


# ---------------------------------------------------------------------------
# Golden dataset
#
# Each entry is a dict with:
#   id      – human-readable name shown in pytest output
#   query   – the raw user string
#   history – optional list of prior messages (for multi-turn cases)
#   expect  – dict of fields we assert on (only what matters for this case)
#
# We do PARTIAL assertions: if a field isn't in `expect`, we don't care
# about it.  This keeps each test focused on the behaviour it's testing.
#
# Values in `expect` can be:
#   - a plain value    →  assert actual == expected
#   - a callable       →  assert expected(actual) is truthy
# ---------------------------------------------------------------------------

CASES: list[dict] = [
    # ------------------------------------------------------------------
    # 1. REFERENCE TITLE
    #
    # The parser sees "similar to X" / "like X" / "more like X" and must
    # extract X into reference_title.  These tests cover various phrasings
    # and check that the title survives punctuation, informal casing, and
    # non-ASCII input.
    # ------------------------------------------------------------------
    {
        "id": "reference_similar_to",
        "query": "Something similar to Nirvana in Fire",
        "expect": {
            "search_mode": "reference",
            # Plain string = exact equality.  Use this when you're confident
            # the LLM will return the title verbatim.
            "reference_title": "Nirvana in Fire",
            "genres": [],
        },
    },
    {
        "id": "reference_like",
        "query": "I want something like The Story of Ming Lan",
        "expect": {
            "search_mode": "reference",
            # Lambda = flexible check.  The LLM might return "Story of Ming Lan"
            # or "The Story of Ming Lan" — both are fine.  We just need the key
            # words to be present so the downstream DB ILIKE lookup will match.
            "reference_title": lambda t: "ming lan" in (t or "").lower(),
        },
    },
    {
        "id": "reference_punctuation_title",
        "query": "Recommend me something similar to How Dare You!?",
        "expect": {
            "search_mode": "reference",
            "reference_title": lambda t: "how dare you" in (t or "").lower(),
        },
        # Punctuation must survive — pipeline does the fuzzy DB lookup,
        # not the parser.  Parser should preserve the title as-is.
    },
    {
        "id": "reference_informal",
        "query": "anything like word of honor??",
        "expect": {
            "search_mode": "reference",
            "reference_title": lambda t: "word of honor" in (t or "").lower(),
        },
    },
    {
        "id": "reference_more_like",
        "query": "More like Story of Ming Lan",
        "expect": {
            "search_mode": "reference",
            "reference_title": lambda t: "ming lan" in (t or "").lower(),
        },
    },
    {
        "id": "reference_no_extras",
        # A "negative" test — when the user gives *only* a title with no
        # filters, every other field should stay at its default (None or []).
        # This catches over-eager parsing (e.g. inventing a genre from the
        # title or guessing a score).
        "query": "Recommend something like The Untamed",
        "expect": {
            "search_mode": "reference",
            "reference_title": lambda t: "untamed" in (t or "").lower(),
            "genres": [],
            "min_year": None,
            "min_score": None,
            "exclude_titles": [],
        },
    },
    {
        "id": "reference_chinese_title",
        # Real users may type drama names in Chinese.  We don't check the
        # exact title because the LLM might return it in Chinese, English,
        # or pinyin — we just need *something* non-empty so the DB lookup
        # has a chance to match.
        "query": "Recommend something similar to 琅琊榜",
        "expect": {
            "search_mode": "reference",
            "reference_title": lambda t: t is not None and len(t) > 0,
        },
    },

    # ------------------------------------------------------------------
    # 2. GENRE EXTRACTION
    #
    # Genres are used as hard SQL filters (WHERE genres && ARRAY[...]),
    # so false positives are worse than false negatives — an extra genre
    # narrows results, but a wrong genre eliminates good matches entirely.
    # ------------------------------------------------------------------
    {
        "id": "genres_single_with_reference",
        "query": "Something like Nirvana in Fire but in the romance genre",
        "expect": {
            "search_mode": "reference",
            "reference_title": lambda t: "nirvana" in (t or "").lower(),
            "genres": lambda g: "romance" in [x.lower() for x in g],
        },
    },
    {
        "id": "genres_multiple_with_reference",
        "query": "Similar to The Story of Ming Lan, historical romance mystery",
        "expect": {
            "search_mode": "reference",
            "reference_title": lambda t: "ming lan" in (t or "").lower(),
            # issubset() checks that our expected genres are all present,
            # but doesn't fail if the LLM adds extra genres.  This is the
            # right tradeoff: we care that "historical" and "romance" are
            # extracted, but we don't mind if "mystery" also appears.
            "genres": lambda g: {"historical", "romance"}.issubset(
                {x.lower() for x in g}
            ),
        },
    },
    {
        "id": "genres_wuxia_with_reference",
        "query": "Wuxia fantasy similar to The Untamed",
        "expect": {
            "search_mode": "reference",
            "reference_title": lambda t: "untamed" in (t or "").lower(),
            "genres": lambda g: "wuxia" in [x.lower() for x in g],
        },
    },
    {
        "id": "no_genres_when_only_reference",
        "query": "Something similar to Joy of Life",
        "expect": {
            "search_mode": "reference",
            "reference_title": lambda t: "joy of life" in (t or "").lower(),
            "genres": [],
        },
    },

    # ------------------------------------------------------------------
    # 3. MIN YEAR
    #
    # The system prompt defines specific interpretation rules:
    #   'no older than 2020' = 2020
    #   'after 2018'         = 2019   (the year *after* 2018)
    #   'from 2020 onwards'  = 2020
    #
    # These are exact-match assertions because the prompt gives the LLM
    # unambiguous rules.  If the LLM returns 2020 for "after 2018", that's
    # a real prompt regression worth catching.
    # ------------------------------------------------------------------
    {
        "id": "year_no_older_than",
        "query": "Romance dramas, nothing older than 2020",
        "expect": {"min_year": 2020},
    },
    {
        "id": "year_after",
        "query": "Only dramas after 2018",
        "expect": {"min_year": 2019},
    },
    {
        "id": "year_from_onwards",
        "query": "From 2022 onwards",
        "expect": {"min_year": 2022},
    },
    {
        "id": "year_not_set_when_absent",
        "query": "Romance drama with good ratings",
        "expect": {"min_year": None},
    },

    # ------------------------------------------------------------------
    # 4. MIN SCORE
    #
    # Explicit numbers ("above 8", "at least 8.5") get exact checks.
    # Vague language ("highly rated", "good rating") uses lambdas because
    # the LLM's interpretation can reasonably vary — the prompt gives
    # guidelines (8.5 for "highly rated", 8.0 for "good rating") but
    # we accept a range rather than a single number.
    # ------------------------------------------------------------------
    {
        "id": "score_above",
        "query": "Drama rated above 8",
        "expect": {"min_score": 8.0},
    },
    {
        "id": "score_at_least",
        "query": "Rating at least 8.5",
        "expect": {"min_score": 8.5},
    },
    {
        "id": "score_highly_rated",
        "query": "Only highly rated dramas",
        "expect": {"min_score": lambda s: s is not None and s >= 8.0},
    },
    {
        "id": "score_good_rating",
        "query": "Something with a good rating",
        "expect": {"min_score": lambda s: s is not None and s >= 7.5},
    },
    {
        "id": "score_not_set_when_absent",
        "query": "Historical romance drama",
        "expect": {"min_score": None},
    },

    # ------------------------------------------------------------------
    # 5. SEARCH MODE — SEMANTIC
    #
    # Semantic mode triggers when the user describes a plot, characters,
    # or vibe *without* naming a specific drama.  The parser must put the
    # description into the `description` field (which gets embedded) and
    # keep reference_title as None.
    # ------------------------------------------------------------------
    {
        "id": "semantic_plot_description",
        "query": (
            "I remember a drama where the heroine had amnesia and was "
            "enemies with the hero before falling in love"
        ),
        "expect": {
            "search_mode": "semantic",
            "reference_title": None,
            "description": lambda d: d is not None and "amnesia" in d.lower(),
        },
    },
    {
        "id": "semantic_vibe_description",
        "query": (
            "A dark revenge drama with a protagonist pretending to be "
            "someone else in a palace setting"
        ),
        "expect": {
            "search_mode": "semantic",
            "description": lambda d: d is not None and len(d) > 10,
        },
    },
    {
        "id": "semantic_ceo_vibe",
        "query": (
            "Something about a cold CEO who falls for a bubbly girl, "
            "set in the fashion industry"
        ),
        "expect": {
            "search_mode": "semantic",
            "description": lambda d: d is not None and "ceo" in d.lower(),
        },
    },
    {
        "id": "semantic_character_focus",
        "query": (
            "Looking for a drama where the female lead is a doctor "
            "and the male lead is cold but protective"
        ),
        "expect": {
            "search_mode": "semantic",
            "description": lambda d: d is not None and "doctor" in d.lower(),
        },
    },
    {
        "id": "semantic_with_genre_filter",
        "query": (
            "A story about time travel where the female lead goes back "
            "to ancient China, romance genre"
        ),
        "expect": {
            "search_mode": "semantic",
            "reference_title": None,
            "description": lambda d: d is not None and "time travel" in d.lower(),
            "genres": lambda g: "romance" in [x.lower() for x in g],
        },
    },
    {
        "id": "semantic_with_year_filter",
        "query": (
            "A drama about a cold CEO who falls for a bubbly girl, "
            "from 2021 onwards"
        ),
        "expect": {
            "search_mode": "semantic",
            "description": lambda d: d is not None and "ceo" in d.lower(),
            "min_year": 2021,
        },
    },
    {
        "id": "semantic_with_exclude",
        "query": (
            "Something about a contract marriage that turns real, "
            "I've already seen Well Intended Love"
        ),
        "expect": {
            "search_mode": "semantic",
            "description": lambda d: d is not None and "contract" in d.lower(),
            "exclude_titles": lambda t: any(
                "well intended love" in x.lower() for x in t
            ),
        },
    },
    {
        "id": "semantic_description_has_no_filters",
        "query": (
            "I'm looking for a drama where the main character travels back "
            "in time and gets caught up in palace schemes, "
            "from 2020 or later, rated above 8.5"
        ),
        "expect": {
            "search_mode": "semantic",
            "description": lambda d: (
                d is not None
                and "palace" in d.lower()
                # prompt says: "Capture plot/character cues (no filters like
                # year or rating)".  Description should not contain the numbers.
                and "2020" not in d
                and "8.5" not in d
            ),
            "min_year": 2020,
            "min_score": 8.5,
        },
    },

    # ------------------------------------------------------------------
    # 6. SEARCH MODE — SQL
    #
    # SQL mode triggers when the user gives *only* structured filters
    # (genre, year, rating) with no plot description and no reference
    # title.  No embedding is generated — the query goes straight to
    # a SQL WHERE clause.  These cases also check that reference_title
    # and description stay None (cross-mode invariants).
    # ------------------------------------------------------------------
    {
        "id": "sql_genre_only",
        "query": "Romance dramas",
        "expect": {
            "search_mode": "sql",
            "reference_title": None,
            "description": None,
            "genres": lambda g: "romance" in [x.lower() for x in g],
        },
    },
    {
        "id": "sql_genre_year_score",
        "query": "Historical romance from 2022 rated above 8",
        "expect": {
            "search_mode": "sql",
            "reference_title": None,
            "genres": lambda g: {"historical", "romance"}.issubset(
                {x.lower() for x in g}
            ),
            "min_year": 2022,
            "min_score": 8.0,
        },
    },
    {
        "id": "sql_year_and_score_only",
        "query": "Dramas from 2023 onwards with rating at least 8.5",
        "expect": {
            "search_mode": "sql",
            "min_year": 2023,
            "min_score": 8.5,
        },
    },
    {
        "id": "sql_year_after_rule",
        "query": "Mystery dramas after 2018 with a good rating",
        "expect": {
            "search_mode": "sql",
            "genres": lambda g: "mystery" in [x.lower() for x in g],
            "min_year": 2019,
            "min_score": lambda s: s is not None and s >= 7.5,
        },
    },
    {
        "id": "sql_highly_rated",
        "query": "Highly rated dramas from 2023",
        "expect": {
            "search_mode": "sql",
            "min_year": 2023,
            "min_score": lambda s: s is not None and s >= 8.0,
        },
    },
    {
        "id": "sql_wuxia",
        "query": "Good wuxia dramas",
        "expect": {
            "search_mode": "sql",
            "reference_title": None,
            "genres": lambda g: "wuxia" in [x.lower() for x in g],
        },
    },

    # ------------------------------------------------------------------
    # 7. EXCLUDE TITLES
    #
    # The parser must pick up dramas the user says they "already watched",
    # "just finished", or "don't want".  These get resolved to DB ids
    # downstream (via ILIKE), so the parser just needs to extract the
    # title text — it doesn't need to normalise or look anything up.
    # ------------------------------------------------------------------
    {
        "id": "exclude_just_finished",
        "query": (
            "Something similar to Nirvana in Fire, "
            "but I just finished Joy of Life so exclude that"
        ),
        "expect": {
            "search_mode": "reference",
            "reference_title": lambda t: "nirvana" in (t or "").lower(),
            "exclude_titles": lambda t: any(
                "joy of life" in x.lower() for x in t
            ),
        },
    },
    {
        "id": "exclude_already_saw",
        "query": "I already saw Dream Within a Dream, something else please",
        "expect": {
            "exclude_titles": lambda t: any("dream" in x.lower() for x in t),
        },
    },
    {
        "id": "exclude_multiple",
        "query": (
            "I've watched Nirvana in Fire and The Story of Ming Lan already"
        ),
        "expect": {
            "exclude_titles": lambda t: len(t) >= 2,
        },
    },

    # ------------------------------------------------------------------
    # 8. COMPOUND QUERIES (realistic multi-field)
    #
    # Real users don't ask one thing at a time.  These cases combine
    # multiple fields in a single query to test that the parser can
    # extract everything simultaneously without fields interfering
    # with each other.
    # ------------------------------------------------------------------
    {
        "id": "compound_full",
        "query": (
            "Recommend me something similar to How Dare You!? "
            "I already saw Dream Within a Dream. "
            "The drama should be rated above 8"
        ),
        "expect": {
            "search_mode": "reference",
            "reference_title": lambda t: "how dare you" in (t or "").lower(),
            "exclude_titles": lambda t: any("dream" in x.lower() for x in t),
            "min_score": 8.0,
        },
    },
    {
        "id": "compound_year_genre_score",
        "query": (
            "Something like The Story of Ming Lan, historical romance "
            "from 2020 onwards, rating at least 8.5"
        ),
        "expect": {
            "search_mode": "reference",
            "reference_title": lambda t: "ming lan" in (t or "").lower(),
            "genres": lambda g: {"historical", "romance"}.issubset(
                {x.lower() for x in g}
            ),
            "min_year": 2020,
            "min_score": 8.5,
        },
    },
    {
        "id": "compound_everything",
        "query": (
            "Like Nirvana in Fire but with comedy and romance, "
            "nothing before 2015, rating at least 8.5, "
            "and I've already seen Joy of Life and Rise of Phoenixes"
        ),
        "expect": {
            "search_mode": "reference",
            "reference_title": lambda t: "nirvana" in (t or "").lower(),
            "genres": lambda g: {"comedy", "romance"}.issubset(
                {x.lower() for x in g}
            ),
            "min_year": 2015,
            "min_score": 8.5,
            "exclude_titles": lambda t: (
                len(t) >= 2
                and any("joy of life" in x.lower() for x in t)
            ),
        },
    },
    {
        "id": "compound_semantic_with_filters",
        "query": (
            "A drama about a female general who disguises herself as a man "
            "to fight in a war, historical, from 2020 onwards, rating above 8"
        ),
        "expect": {
            "search_mode": "semantic",
            "description": lambda d: d is not None and "disguise" in d.lower(),
            "genres": lambda g: "historical" in [x.lower() for x in g],
            "min_year": 2020,
            "min_score": 8.0,
        },
    },

    # ------------------------------------------------------------------
    # 9. MULTI-TURN (history context)
    #
    # The pipeline passes the last HISTORY_MESSAGES (6) messages as
    # conversation history.  Follow-up queries like "something older"
    # only make sense if the parser can see what came before.
    #
    # These are the hardest cases because the LLM must infer context
    # from prior turns — "something older" means "lower min_year than
    # what we discussed", which requires reading the history.
    # ------------------------------------------------------------------
    {
        "id": "multiturn_followup_reference",
        "query": "something older",
        "history": [
            {
                "role": "user",
                "content": "Recommend something like Love Between Fairy and Devil",
            },
            {
                "role": "assistant",
                "content": (
                    "Here are some dramas similar to "
                    "Love Between Fairy and Devil (2022)..."
                ),
            },
        ],
        "expect": {
            "search_mode": "reference",
            "reference_title": lambda t: t is not None and len(t) > 0,
        },
    },
    {
        "id": "multiturn_refine_year",
        "query": "Actually, only from 2021 onwards",
        "history": [
            {"role": "user", "content": "Romance dramas with good ratings"},
            {
                "role": "assistant",
                "content": "Here are some great romance dramas...",
            },
        ],
        "expect": {"min_year": 2021},
    },
    {
        "id": "multiturn_add_exclusion",
        "query": "Exclude Joy of Life from the results",
        "history": [
            {
                "role": "user",
                "content": "Something similar to Nirvana in Fire",
            },
            {
                "role": "assistant",
                "content": "I recommend Joy of Life...",
            },
        ],
        "expect": {
            "exclude_titles": lambda t: any(
                "joy of life" in x.lower() for x in t
            ),
        },
    },
    {
        "id": "multiturn_long_history_windowed",
        "query": "something similar but more romance",
        "history": [
            # 8 messages — only the last 6 should be visible to the parser.
            # The first pair mentions "The Untamed" which should NOT leak
            # into the result if windowing works.
            {"role": "user", "content": "Recommend something like The Untamed"},
            {"role": "assistant", "content": "Here are some wuxia dramas..."},
            {"role": "user", "content": "Actually I'd like something like Hidden Love"},
            {"role": "assistant", "content": "Hidden Love (2023) is a great pick..."},
            {"role": "user", "content": "More like that please"},
            {"role": "assistant", "content": "Here are similar youth romance dramas..."},
            {"role": "user", "content": "Any with higher ratings?"},
            {"role": "assistant", "content": "These are all rated above 8.5..."},
        ],
        "expect": {
            "search_mode": "reference",
            # Should pick up Hidden Love (in the window), not The Untamed
            # (outside the window).
            "reference_title": lambda t: (
                t is not None
                and "untamed" not in t.lower()
            ),
            "genres": lambda g: "romance" in [x.lower() for x in g],
        },
    },

    # ------------------------------------------------------------------
    # 10. EDGE CASES
    #
    # These don't test *correct* parsing — they test that the parser
    # doesn't crash.  Garbage input should still return a valid
    # QueryFilters object (OpenAI's structured output guarantees this,
    # but it's worth verifying).
    # ------------------------------------------------------------------
    {
        "id": "edge_nonsense_input",
        "query": "asdfghjkl zxcvbnm",
        "expect": {
            # Should still return a valid QueryFilters without crashing.
            # We don't care which mode — just that it parsed.
            "search_mode": lambda m: m in ("reference", "semantic", "sql"),
        },
    },
    {
        "id": "edge_empty_query",
        "query": "   ",
        "expect": {
            "search_mode": lambda m: m in ("reference", "semantic", "sql"),
        },
    },
]


# ---------------------------------------------------------------------------
# Assertion helpers
#
# These are called by the test function below, not by pytest directly.
# Splitting assertions into helpers keeps the test function short and
# makes error messages more specific when something fails.
# ---------------------------------------------------------------------------


def _assert_field(actual_value, expected):
    """Check one field against its expectation.

    The dual plain-value / callable pattern is the core of the flexible
    assertion design:

        "min_year": 2020                          # exact match
        "genres": lambda g: "romance" in [...]    # truthy check

    When the LLM's output is deterministic (year, score from explicit
    numbers), use a plain value.  When it can reasonably vary (title
    casing, extra genres), use a lambda.
    """
    if callable(expected):
        assert expected(actual_value), (
            f"Custom assertion failed for value: {actual_value!r}"
        )
    else:
        assert actual_value == expected, (
            f"Expected {expected!r}, got {actual_value!r}"
        )


def _assert_invariants(result: QueryFilters, case: dict) -> None:
    """Invariants that must hold for EVERY case, not just the ones that test them.

    These catch "silent" regressions — a case that tests min_year wouldn't
    notice if the LLM also started leaking reference_title into semantic
    mode.  Running these globally means every case is also an implicit
    cross-mode test.
    """
    mode = result.search_mode

    # Cross-mode field exclusivity (from system prompt):
    #   "reference_title: only set in reference mode"
    #   "description: only set in semantic mode"
    # If the LLM puts a title in a semantic result, the pipeline would
    # try to do a reference lookup AND an embedding search — wasting
    # tokens and returning confusing results.
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

    # System prompt: "NEVER put a drama title in genres."
    # This is a real regression we've seen — the LLM sometimes puts
    # "Nirvana in Fire" or parts of it into the genres list.  The check
    # splits the title into words (>2 chars to skip "in", "of") and
    # looks for overlap with genres.
    if result.reference_title:
        title_words = {w.lower() for w in result.reference_title.split() if len(w) > 2}
        genre_words = {g.lower() for g in result.genres}
        leaked = title_words & genre_words
        assert not leaked, (
            f"Title words leaked into genres: {leaked} "
            f"(title={result.reference_title!r}, genres={result.genres})"
        )


# ---------------------------------------------------------------------------
# Parametrised test
#
# Why one function with parametrize instead of 47 separate test functions?
#
#   pytest.mark.parametrize turns each dict in CASES into an independent
#   test.  If case 3 fails, cases 4-47 still run — you see ALL failures
#   at once, not just the first.  The `ids` argument gives readable names
#   in the output (e.g. test_parse_user_query[year_after]) instead of
#   test_parse_user_query[0], test_parse_user_query[1], etc.
#
#   Adding a new test case is just appending a dict to CASES — no new
#   function, no new boilerplate.
# ---------------------------------------------------------------------------


# @pytest.mark.integration lets us skip these in CI where there's no API key:
#   uv run pytest -m "not integration"
# The marker is registered in pyproject.toml so --strict-markers doesn't warn.
@pytest.mark.integration
@pytest.mark.parametrize("case", CASES, ids=[c["id"] for c in CASES])
def test_parse_user_query(case: dict, openai_client: OpenAI) -> None:
    # openai_client comes from conftest.py — pytest injects it automatically
    # because the parameter name matches the fixture name.
    result = parse_user_query(
        user_query=case["query"],
        openai=openai_client,
        history=case.get("history"),
    )

    # Per-case partial assertions — only check the fields this case cares about
    for field, expected in case["expect"].items():
        actual = getattr(result, field)
        _assert_field(actual, expected)

    # Global invariants — these run on EVERY case regardless of what it tests
    _assert_invariants(result, case)
