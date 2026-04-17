"""Shared golden-set cases for parse_user_query.

Used by:
  - tests/integration/test_parse_user_query.py  (pytest regression suite)
  - notebooks/04a_parser_lab.ipynb              (prompt dev / model comparison)

Each case is a dict with:
  id      – human-readable name shown in pytest output
  query   – raw user string
  history – optional list of prior messages (multi-turn cases)
  expect  – fields to assert on; plain value = exact match, callable = truthy check
"""

from __future__ import annotations

CASES: list[dict] = [
    # ------------------------------------------------------------------
    # 1. REFERENCE TITLE
    # ------------------------------------------------------------------
    {
        "id": "reference_similar_to",
        "query": "Something similar to Nirvana in Fire",
        "expect": {
            "search_mode": "reference",
            "reference_title": "Nirvana in Fire",
            "genres": [],
        },
    },
    {
        "id": "reference_like",
        "query": "I want something like The Story of Ming Lan",
        "expect": {
            "search_mode": "reference",
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
        "query": "Recommend something similar to 琅琊榜",
        "expect": {
            "search_mode": "reference",
            "reference_title": lambda t: t is not None and len(t) > 0,
        },
    },

    # ------------------------------------------------------------------
    # 2. GENRE EXTRACTION
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
                and "2020" not in d
                and "8.5" not in d
            ),
            "min_year": 2020,
            "min_score": 8.5,
        },
    },

    # ------------------------------------------------------------------
    # 6. SEARCH MODE — SQL
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
        "query": "I've watched Nirvana in Fire and The Story of Ming Lan already",
        "expect": {
            "exclude_titles": lambda t: len(t) >= 2,
        },
    },

    # ------------------------------------------------------------------
    # 7b. EXCLUDE GENRES
    # ------------------------------------------------------------------
    {
        "id": "exclude_genre_no_x",
        "query": "Historical dramas, no romance please",
        "expect": {
            "genres": lambda g: "historical" in [x.lower() for x in g],
            "exclude_genres": lambda g: "romance" in [x.lower() for x in g],
        },
    },
    {
        "id": "exclude_genre_but_not",
        "query": "Wuxia but not fantasy",
        "expect": {
            "genres": lambda g: "wuxia" in [x.lower() for x in g],
            "exclude_genres": lambda g: "fantasy" in [x.lower() for x in g],
        },
    },
    {
        "id": "exclude_genre_with_reference",
        "query": "Something like Nirvana in Fire, avoid romance",
        "expect": {
            "search_mode": "reference",
            "reference_title": lambda t: "nirvana" in (t or "").lower(),
            "exclude_genres": lambda g: "romance" in [x.lower() for x in g],
        },
    },

    # ------------------------------------------------------------------
    # 8. COMPOUND QUERIES
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
            "reference_title": lambda t: (
                t is not None
                and "untamed" not in t.lower()
            ),
            "genres": lambda g: "romance" in [x.lower() for x in g],
        },
    },

    # ------------------------------------------------------------------
    # 10. EDGE CASES
    # ------------------------------------------------------------------
    {
        "id": "edge_nonsense_input",
        "query": "asdfghjkl zxcvbnm",
        "expect": {
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
