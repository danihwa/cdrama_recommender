"""Weight calibration for the scorer via golden-set pairwise preferences.

HOW THIS FILE FITS IN
=====================
This test answers: "are the scorer's weights (0.70 / 0.20 / 0.10)
actually producing good rankings on real data?"

The pieces work together like this:

    scripts/collect_candidates.py          ← Step 1: gather real data from DB
            ↓ writes
    tests/calibration/candidate_sets.json  ← raw candidates snapshot
            ↓ read by
    tests/calibration/test_weight_calibration.py  ← Step 2: THIS FILE
            ↑ imports
    src/recommender/pipeline.py::score_candidates  ← the function we test

THE APPROACH: GOLDEN PAIRWISE PREFERENCES
=========================================
Instead of vague metrics, "good ranking" is expressed as concrete pairs:

    "For the query 'similar to Nirvana in Fire',
     Joy of Life (9.0 rating) should rank above Song of Glory (7.8)
     even though they have the same similarity score."

Each preference is a human judgment call about what a good recommender
should do. We collected 17 of these from real candidate sets, then check
that the current weights satisfy all of them.

Run:  uv run pytest tests/calibration/test_weight_calibration.py -v
"""

from __future__ import annotations

import copy
import json
from dataclasses import dataclass
from pathlib import Path

import pytest

from src.recommender.pipeline import score_candidates

FIXTURES = Path("tests/calibration/candidate_sets.json")


def load_candidate_sets() -> dict[str, list[dict]]:
    """Load the real candidate sets that collect_candidates.py saved.

    Returns a dict like {"ref_nirvana_in_fire": [list of 10 drama dicts], ...}.
    Each drama dict has: title, similarity, mdl_score, watchers, etc.
    """
    raw = json.loads(FIXTURES.read_text())
    return {entry["label"]: entry["candidates"] for entry in raw}


# Load once at module level so every test reuses the same data.
CANDIDATE_SETS = load_candidate_sets()


def _find(candidates: list[dict], title: str) -> dict:
    """Find a candidate by title substring (case-insensitive).

    We use substrings instead of exact titles because some DB titles have
    special characters (e.g. curly apostrophes) that are easy to get wrong.
    "Promise of Chang" matches "The Promise of Chang'an".
    """
    title_lower = title.lower()
    for c in candidates:
        if title_lower in c["title"].lower():
            return c
    raise ValueError(f"'{title}' not found in candidates: {[c['title'] for c in candidates]}")


@dataclass
class Preference:
    """One human judgment: drama `higher` should rank above drama `lower`.

    Each preference is tied to a specific candidate set (by label) because
    the scorer normalises popularity *within the batch*. The same drama
    might get different popularity scores in different candidate sets.
    """

    label: str  # which candidate set (e.g. "ref_nirvana_in_fire")
    higher: str  # title substring of the drama that should rank higher
    lower: str  # title substring of the drama that should rank lower
    reason: str  # why — helps future us understand the judgment


# ──────────────────────────────────────────────────────────────────────
# GOLDEN PAIRWISE PREFERENCES
# ──────────────────────────────────────────────────────────────────────
# Each of these was hand-picked from real candidate sets by looking at
# the actual similarity, mdl_score, and watchers values and asking:
# "which drama would a human recommend first?"
#
# The preferences encode several principles:
#   1. Quality can override small similarity gaps
#   2. Large similarity gaps should still win (similarity is primary)
#   3. In SQL mode (no similarity), quality should matter more than popularity
#   4. Popularity is a fair tiebreaker when quality is equal
#
# To add more preferences, run collect_candidates.py to refresh the fixture,
# look at the raw data, and add new Preference entries below.

PREFERENCES = [
    # ── Reference mode ──────────────────────────────────────────────
    Preference(
        "ref_nirvana_in_fire",
        higher="Joy of Life",  # sim=0.756, score=9.0
        lower="Song of Glory",  # sim=0.756, score=7.8
        reason="Same similarity — quality (9.0 vs 7.8) should decide",
    ),
    Preference(
        "ref_hidden_love",
        higher="My Fated Boy",  # sim=0.790, score=8.3
        lower="When I Fly Towards You",  # sim=0.690, score=9.0
        reason="0.10 sim gap is large — similarity should win in reference mode",
    ),

    # ── Semantic mode ───────────────────────────────────────────────
    Preference(
        "sem_amnesia_enemies",
        higher="Are You the One",  # sim=0.535, score=8.5, watchers=23872
        lower="Demon Master",  # sim=0.546, score=7.5, watchers=5962
        reason="Quality (8.5 vs 7.5) should overcome a 0.011 sim gap",
    ),
    Preference(
        "sem_doctor_cold_ml",
        higher="Unforgettable Love",  # sim=0.592, score=8.4, watchers=55792
        lower="Thank You, Doctor",  # sim=0.599, score=7.5, watchers=3945
        reason="Small sim gap (0.007), much better quality + popularity",
    ),

    # ── SQL mode ────────────────────────────────────────────────────
    Preference(
        "sql_romance_2023_high",
        higher="Pursuit of Jade",  # score=9.1, watchers=40464
        lower="Blossom",  # score=8.8, watchers=38047
        reason="Clear quality gap (9.1 vs 8.8) should win with similar popularity",
    ),
    Preference(
        "sql_historical_broad",
        higher="The Double",  # score=8.8, watchers=46047
        lower="Story of Ming Lan",  # score=8.8, watchers=28841
        reason="Same score — popularity as tiebreaker is fine in SQL mode",
    ),
]


def check_preference(
    pref: Preference,
    w_sim: float,
    w_quality: float,
    w_popularity: float,
) -> bool:
    """Return True if the weight triple satisfies this preference.

    We deep-copy the candidates because score_candidates mutates them
    (adds ensemble_score to each dict). Without the copy, running the
    same preference twice would see stale scores.
    """
    candidates = copy.deepcopy(CANDIDATE_SETS[pref.label])
    ranked = score_candidates(
        candidates, w_sim=w_sim, w_quality=w_quality, w_popularity=w_popularity
    )
    titles = [d["title"] for d in ranked]
    higher_title = _find(ranked, pref.higher)["title"]
    lower_title = _find(ranked, pref.lower)["title"]
    return titles.index(higher_title) < titles.index(lower_title)


class TestCurrentWeights:
    """Do the current default weights (0.70/0.20/0.10) produce good rankings?

    Each golden preference is tested individually. If one fails, you see
    exactly which pair was misordered and why — making it easy to decide
    whether to adjust the weights or reconsider the preference.
    """

    @pytest.mark.parametrize(
        "pref",
        PREFERENCES,
        ids=[f"{p.label}:{p.higher}_over_{p.lower}" for p in PREFERENCES],
    )
    def test_preference_satisfied(self, pref: Preference):
        """Each golden preference should be satisfied by the current weights."""
        satisfied = check_preference(pref, 0.70, 0.20, 0.10)
        if not satisfied:
            pytest.fail(
                f"FAILED: In '{pref.label}', expected '{pref.higher}' above "
                f"'{pref.lower}'. Reason: {pref.reason}"
            )
