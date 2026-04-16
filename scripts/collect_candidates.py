"""Collect raw candidate sets from all three retrieval modes.

HOW THIS FILE FITS IN
=====================
This is a one-shot data collection script — you run it once (or whenever
the database changes significantly) to snapshot what the retrieval layer
actually returns for a set of representative queries.

The output (tests/evals/fixtures/candidate_sets.json) is then used by
the weight calibration eval (tests/evals/test_weight_calibration.py)
to test whether the reranker's weights produce good orderings.

The flow is:

    1. THIS SCRIPT  →  runs 13 queries against the real DB
                    →  saves raw candidates to candidate_sets.json

    2. YOU (human)  →  look at the candidates and write "golden preferences"
                       (e.g. "Joy of Life should rank above Song of Glory")

    3. THE EVAL     →  loads candidate_sets.json + your preferences
                    →  tries many weight combos to find the best one

So this script is the data-gathering step. It doesn't decide anything
about weights — it just captures the raw material.

WHY WE NEED REAL DATA
=====================
The reranker's formula looks simple:

    score = 0.70 * similarity + 0.20 * quality + 0.10 * popularity

But the *actual* influence of each signal depends on how spread out
its values are in a real candidate set. If similarity ranges from
0.80 to 0.82 (tight cluster) while quality ranges from 7.0 to 9.5
(wide spread), then quality ends up driving more of the ordering
despite having a smaller weight. You can't see this from the formula
alone — you need real data.

Usage: uv run scripts/collect_candidates.py
"""

from __future__ import annotations

import json
from pathlib import Path

from openai import OpenAI
from supabase import Client

from src.database.connection import get_db_connection
from src.env import load_secrets
from src.recommender.models import QueryFilters
from src.recommender.search_reference import retrieve_reference_candidates
from src.recommender.search_semantic import retrieve_semantic_candidates
from src.recommender.search_sql import retrieve_sql_candidates

MATCH_COUNT = 10
OUTPUT_PATH = Path("tests/evals/fixtures/candidate_sets.json")

# Representative queries covering all three modes.
#
# We want diversity here: different genres, different score ranges,
# popular vs. niche anchor dramas. The more variety, the better our
# golden preferences will generalise.
#
# Each entry has:
#   label   — a short name used to reference this set in preferences
#   mode    — which retrieval function to call
#   filters — the QueryFilters fields (same shape as the parser LLM output)
QUERIES: list[dict] = [
    # --- Reference mode (5 queries) ---
    # "Similar to X" — the retrieval looks up X's embedding and finds
    # nearest neighbours. These candidates will have high similarity
    # values (typically 0.7–0.85) in a tight cluster.
    {
        "label": "ref_nirvana_in_fire",
        "mode": "reference",
        "filters": {"search_mode": "reference", "reference_title": "Nirvana in Fire"},
    },
    {
        "label": "ref_love_o2o",
        "mode": "reference",
        "filters": {"search_mode": "reference", "reference_title": "Love O2O"},
    },
    {
        "label": "ref_story_of_minglan",
        "mode": "reference",
        "filters": {"search_mode": "reference", "reference_title": "The Story of Ming Lan"},
    },
    {
        "label": "ref_reset",
        "mode": "reference",
        "filters": {"search_mode": "reference", "reference_title": "Reset"},
    },
    {
        "label": "ref_hidden_love",
        "mode": "reference",
        "filters": {
            "search_mode": "reference",
            "reference_title": "Hidden Love",
            "min_score": 8.0,
        },
    },
    # --- Semantic mode (5 queries) ---
    # User describes a vibe/plot — we embed their description and search.
    # Similarity values are typically lower (0.5–0.65) because a
    # free-text description is less precise than a known drama's embedding.
    {
        "label": "sem_amnesia_enemies",
        "mode": "semantic",
        "filters": {
            "search_mode": "semantic",
            "description": "heroine loses her memory and falls for her former enemy",
        },
    },
    {
        "label": "sem_doctor_cold_ml",
        "mode": "semantic",
        "filters": {
            "search_mode": "semantic",
            "description": "female lead is a doctor and the male lead is cold and aloof",
        },
    },
    {
        "label": "sem_political_intrigue",
        "mode": "semantic",
        "filters": {
            "search_mode": "semantic",
            "description": "political intrigue and court scheming in ancient China",
            "genres": ["historical"],
        },
    },
    {
        "label": "sem_time_travel_romance",
        "mode": "semantic",
        "filters": {
            "search_mode": "semantic",
            "description": "time travel romance where modern person goes back to ancient times",
        },
    },
    {
        "label": "sem_campus_romance",
        "mode": "semantic",
        "filters": {
            "search_mode": "semantic",
            "description": "sweet campus romance between classmates who grew up together",
            "min_score": 7.5,
        },
    },
    # --- SQL mode (3 queries) ---
    # Pure filters, no embedding. Every candidate gets similarity = 0.
    # The reranker ranks by quality + popularity only.
    {
        "label": "sql_romance_2023_high",
        "mode": "sql",
        "filters": {
            "search_mode": "sql",
            "genres": ["romance"],
            "min_year": 2023,
            "min_score": 8.0,
        },
    },
    {
        "label": "sql_historical_broad",
        "mode": "sql",
        "filters": {
            "search_mode": "sql",
            "genres": ["historical"],
        },
    },
    {
        "label": "sql_mystery_thriller",
        "mode": "sql",
        "filters": {
            "search_mode": "sql",
            "genres": ["mystery", "thriller"],
            "min_score": 7.5,
        },
    },
]


def collect_all(supabase: Client, openai: OpenAI) -> list[dict]:
    """Run each query and return labelled candidate sets."""
    results = []
    for q in QUERIES:
        label = q["label"]
        mode = q["mode"]
        filters = QueryFilters(**q["filters"])

        print(f"\n{'='*60}")
        print(f"Collecting: {label} (mode={mode})")

        if mode == "reference":
            candidates = retrieve_reference_candidates(filters, supabase, MATCH_COUNT)
        elif mode == "semantic":
            candidates = retrieve_semantic_candidates(
                filters, supabase, openai, MATCH_COUNT
            )
        elif mode == "sql":
            candidates = retrieve_sql_candidates(filters, supabase, MATCH_COUNT)
        else:
            raise ValueError(f"Unknown mode: {mode}")

        print(f"  → {len(candidates)} candidates")
        for c in candidates[:3]:
            print(
                f"    {c['title']:<35} "
                f"sim={c.get('similarity', 0):.3f}  "
                f"score={c.get('mdl_score', 0)}  "
                f"watchers={c.get('watchers', 0)}"
            )
        if len(candidates) > 3:
            print(f"    ... and {len(candidates) - 3} more")

        results.append(
            {
                "label": label,
                "mode": mode,
                "filters": q["filters"],
                "candidates": candidates,
            }
        )
    return results


def main() -> None:
    load_secrets()
    supabase = get_db_connection()
    openai = OpenAI()

    results = collect_all(supabase, openai)

    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    OUTPUT_PATH.write_text(json.dumps(results, indent=2, default=str))
    print(f"\n\nSaved {len(results)} candidate sets to {OUTPUT_PATH}")


if __name__ == "__main__":
    main()
