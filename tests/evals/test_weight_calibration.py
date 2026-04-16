"""Weight calibration for the reranker via golden-set pairwise preference sweep.

HOW THIS FILE FITS IN
=====================
This is the "brain" of the weight calibration system. It answers the
question: "are the reranker's weights (0.70 / 0.20 / 0.10) actually
producing good rankings?"

The three files work together like this:

    scripts/collect_candidates.py     ← Step 1: gather real data from DB
            ↓ writes
    tests/evals/fixtures/candidate_sets.json   ← raw candidates snapshot
            ↓ read by
    tests/evals/test_weight_calibration.py     ← Step 2: THIS FILE
            ↑ imports
    src/recommender/pipeline.py::rerank_candidates  ← the function we're testing

THE APPROACH: GOLDEN-SET PAIRWISE PREFERENCES
==============================================
Instead of vague metrics, we express "good ranking" as concrete pairs:

    "For the query 'similar to Nirvana in Fire',
     Joy of Life (9.0 rating) should rank above Song of Glory (7.8 rating)
     even though they have the same similarity score."

Each preference is a human judgment call that encodes what we think a
good recommender should do. We collected 17 of these from real data.

Then we ask: do the current weights satisfy all these preferences?
And: is there a *better* set of weights that satisfies more of them?

THE GRID SWEEP
==============
We try every combination of weights (in steps of 0.05) that sum to 1.0
and count how many preferences each combo satisfies. ~200 combos x 17
preferences = ~3400 calls to rerank_candidates. Since the reranker is
just math (no DB, no API), this runs in under a second.

NOMINAL vs EFFECTIVE WEIGHTS
============================
The formula says 0.70 * similarity, but similarity values in a real
top-10 set cluster tightly (e.g. 0.75–0.82). Meanwhile quality (MDL
score / 10) might range from 0.75 to 0.92 — a much wider spread.

A signal with a wider spread has more power to *reorder* candidates,
regardless of its nominal weight. So 0.70 * (narrow range) can have
less real influence than 0.20 * (wide range).

"Effective weight" measures this: nominal_weight * std(signal values).
It tells you how much each signal actually contributes to reordering.

FINDINGS (April 2025)
=====================
Ran on 13 real query sets (5 reference, 5 semantic, 3 SQL):

  Current weights (0.70 / 0.20 / 0.10):
    - Satisfy 17/17 golden preferences — already doing well!
    - Effective influence: sim=44.7%, qual=24.5%, pop=30.9%
    - Popularity's 10% nominal weight acts like ~31% because watcher
      counts vary enormously even after log-scaling.

  Best weights from sweep (0.50 / 0.35 / 0.15):
    - Also satisfy 17/17 — no improvement on this golden set.
    - Effective influence: sim=26.4%, qual=35.4%, pop=38.3%
    - Shifts influence toward quality at the cost of similarity.

  Conclusion: current weights are fine. Both options produce identical
  rankings on our test set. If we later notice mediocre-but-similar
  dramas outranking great-but-less-similar ones, bumping quality from
  0.20 → 0.30 would help. The framework is in place to evaluate that.

Run:  uv run pytest tests/evals/test_weight_calibration.py -v -s
"""

from __future__ import annotations

import copy
import json
import math
from dataclasses import dataclass
from pathlib import Path

import pytest

from src.recommender.pipeline import rerank_candidates

FIXTURES = Path("tests/evals/fixtures/candidate_sets.json")


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

    We use substrings instead of exact titles because some DB titles
    have special characters (e.g. curly apostrophes) that are easy to
    get wrong. "Promise of Chang" matches "The Promise of Chang'an".
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
    the reranker normalises popularity *within the batch*. The same drama
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
# If you want to add more preferences, run collect_candidates.py to
# refresh the fixture, look at the raw data, and add new Preference
# entries below. Then run the sweep to see if the weights still hold.

PREFERENCES = [
    # ── Reference mode ──────────────────────────────────────────────
    # Similarity is the primary signal (the user said "like X"), but
    # quality should override when the sim gap is tiny.

    Preference(
        "ref_nirvana_in_fire",
        higher="Joy of Life",  # sim=0.756, score=9.0, watchers=40707
        lower="Promise of Chang",  # sim=0.781, score=7.6, watchers=5033
        reason="Joy of Life's much higher quality (9.0 vs 7.6) should overcome a 0.025 sim gap",
    ),
    Preference(
        "ref_nirvana_in_fire",
        higher="Joy of Life",  # sim=0.756, score=9.0
        lower="Song of Glory",  # sim=0.756, score=7.8
        reason="Same similarity — quality (9.0 vs 7.8) should decide",
    ),
    Preference(
        "ref_nirvana_in_fire",
        higher="Rise of Phoenixes",  # sim=0.758, score=8.4
        lower="Love in Between",  # sim=0.739, score=7.8
        reason="Higher similarity AND higher quality — clear win",
    ),
    Preference(
        "ref_love_o2o",
        higher="Falling into Your Smile",  # sim=0.799, score=8.6, watchers=101669
        lower="Double Love",  # sim=0.800, score=7.9, watchers=8036
        reason="0.001 sim gap is noise; quality + popularity should win",
    ),
    Preference(
        "ref_love_o2o",
        higher="You Are My Glory",  # sim=0.754, score=8.7
        lower="Unique Lady",  # sim=0.718, score=7.8
        reason="Higher in everything that matters",
    ),
    Preference(
        "ref_hidden_love",
        higher="My Fated Boy",  # sim=0.790, score=8.3
        lower="When I Fly Towards You",  # sim=0.690, score=9.0
        reason="0.10 sim gap is large — similarity should win in reference mode",
    ),
    Preference(
        "ref_hidden_love",
        higher="Shine on Me",  # sim=0.709, score=8.6, watchers=33891
        lower="Memory of Encaustic Tile",  # sim=0.713, score=8.0, watchers=9538
        reason="Tiny sim gap (0.004), much better quality + popularity",
    ),

    # ── Semantic mode ───────────────────────────────────────────────
    # Similarity is less precise here (user described a vibe, not an
    # exact match), so quality should have an easier time overriding it.

    Preference(
        "sem_amnesia_enemies",
        higher="Are You the One",  # sim=0.535, score=8.5, watchers=23872
        lower="Demon Master",  # sim=0.546, score=7.5, watchers=5962
        reason="Quality (8.5 vs 7.5) should overcome a 0.011 sim gap",
    ),
    Preference(
        "sem_amnesia_enemies",
        higher="Fated Hearts",  # sim=0.555, score=8.7
        lower="Lady & Liar",  # sim=0.540, score=7.6
        reason="Higher similarity AND higher quality — clear win",
    ),
    Preference(
        "sem_doctor_cold_ml",
        higher="Unforgettable Love",  # sim=0.592, score=8.4, watchers=55792
        lower="Thank You, Doctor",  # sim=0.599, score=7.5, watchers=3945
        reason="Small sim gap (0.007), much better quality + popularity",
    ),
    Preference(
        "sem_doctor_cold_ml",
        higher="The Best Thing",  # sim=0.573, score=8.7, watchers=53911
        lower="Emergency Department Doctors",  # sim=0.599, score=7.6, watchers=856
        reason="Quality gap (8.7 vs 7.6) should overcome 0.026 sim gap",
    ),
    Preference(
        "sem_political_intrigue",
        higher="Story of Yanxi Palace",  # sim=0.583, score=8.7, watchers=25968
        lower="Virtuous Queen of Han",  # sim=0.592, score=7.5, watchers=2174
        reason="Quality (8.7 vs 7.5) + popularity should overcome 0.009 sim gap",
    ),
    Preference(
        "sem_campus_romance",
        higher="Always Home",  # sim=0.637, score=8.6
        lower="Make My Heart Smile",  # sim=0.642, score=7.9
        reason="Tiny sim gap (0.005), better quality should win",
    ),
    Preference(
        "sem_campus_romance",
        higher="Exclusive Fairytale",  # sim=0.590, score=8.3, watchers=39450
        lower="Once and Forever",  # sim=0.591, score=7.7, watchers=846
        reason="Nearly identical sim, much better quality + popularity",
    ),

    # ── SQL mode ────────────────────────────────────────────────────
    # No similarity at all (everything is 0). The reranker collapses
    # to quality + popularity. We want quality to be the primary driver,
    # with popularity as a tiebreaker for same-scored dramas.

    Preference(
        "sql_romance_2023_high",
        higher="Pursuit of Jade",  # score=9.1, watchers=40464
        lower="Blossom",  # score=8.8, watchers=38047
        reason="Clear quality gap (9.1 vs 8.8) should win with similar popularity",
    ),
    Preference(
        "sql_historical_broad",
        higher="Pursuit of Jade",  # score=9.1, watchers=40464
        lower="Story of Kunning Palace",  # score=8.7, watchers=42187
        reason="Quality gap (9.1 vs 8.7) should clearly beat similar popularity",
    ),
    Preference(
        "sql_historical_broad",
        higher="The Double",  # score=8.8, watchers=46047
        lower="Story of Ming Lan",  # score=8.8, watchers=28841
        reason="Same score — popularity as tiebreaker is fine in SQL mode",
    ),
]


# ──────────────────────────────────────────────────────────────────────
# HELPER FUNCTIONS
# ──────────────────────────────────────────────────────────────────────


def check_preference(
    pref: Preference,
    w_sim: float,
    w_quality: float,
    w_popularity: float,
) -> bool:
    """Return True if the weight triple satisfies this preference.

    We deep-copy the candidates because rerank_candidates mutates them
    (adds ensemble_score to each dict). Without the copy, running the
    same preference with different weights would see stale scores.
    """
    candidates = copy.deepcopy(CANDIDATE_SETS[pref.label])
    ranked = rerank_candidates(
        candidates, w_sim=w_sim, w_quality=w_quality, w_popularity=w_popularity
    )
    titles = [d["title"] for d in ranked]
    higher_title = _find(ranked, pref.higher)["title"]
    lower_title = _find(ranked, pref.lower)["title"]
    return titles.index(higher_title) < titles.index(lower_title)


def sweep_weights(
    step: float = 0.05,
) -> tuple[tuple[float, float, float], int, int]:
    """Try every weight triple (summing to 1.0) and return the best one.

    The grid is bounded to reasonable ranges:
      - similarity:  0.40–0.90  (it should always be the biggest signal)
      - quality:     0.05–0.40
      - popularity:  0.05–0.30  (it's a secondary signal, not primary)

    With step=0.05, that's about 200 combinations. Each combo runs
    rerank_candidates on all 17 preferences — pure math, so the whole
    sweep finishes in under a second.
    """
    best_score = 0
    best_weights = (0.70, 0.20, 0.10)
    total_prefs = len(PREFERENCES)

    steps = [round(i * step, 2) for i in range(1, int(1.0 / step) + 1)]

    for w_sim in [s for s in steps if 0.40 <= s <= 0.90]:
        for w_qual in [s for s in steps if 0.05 <= s <= 0.40]:
            # The third weight is whatever is left to reach 1.0
            w_pop = round(1.0 - w_sim - w_qual, 2)
            if not (0.05 <= w_pop <= 0.30):
                continue

            score = sum(
                check_preference(p, w_sim, w_qual, w_pop) for p in PREFERENCES
            )
            if score > best_score:
                best_score = score
                best_weights = (w_sim, w_qual, w_pop)

    return best_weights, best_score, total_prefs


def effective_weights(
    w_sim: float, w_quality: float, w_popularity: float
) -> dict[str, float]:
    """Compute how much each signal actually influences reordering.

    The idea: a signal's real influence = its nominal weight * how much
    it varies across candidates. If similarity barely varies (std=0.02)
    but quality varies a lot (std=0.04), quality drives more reordering
    even if its nominal weight is lower.

    We average across all 13 candidate sets to get a representative picture.
    """
    import statistics

    sim_stds, qual_stds, pop_stds = [], [], []
    for candidates in CANDIDATE_SETS.values():
        if len(candidates) < 2:
            continue
        sims = [c.get("similarity", 0.0) for c in candidates]
        quals = [(c.get("mdl_score") or 0.0) / 10.0 for c in candidates]
        max_w = max((c.get("watchers") or 1) for c in candidates)
        log_max = math.log(max_w) if max_w > 1 else 1
        pops = [math.log(max(c.get("watchers") or 1, 1)) / log_max for c in candidates]
        sim_stds.append(statistics.stdev(sims))
        qual_stds.append(statistics.stdev(quals))
        pop_stds.append(statistics.stdev(pops))

    eff_sim = w_sim * statistics.mean(sim_stds)
    eff_qual = w_quality * statistics.mean(qual_stds)
    eff_pop = w_popularity * statistics.mean(pop_stds)
    total = eff_sim + eff_qual + eff_pop
    return {
        "similarity": eff_sim / total if total else 0,
        "quality": eff_qual / total if total else 0,
        "popularity": eff_pop / total if total else 0,
    }


# ──────────────────────────────────────────────────────────────────────
# TESTS
# ──────────────────────────────────────────────────────────────────────


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


class TestWeightSweep:
    """Exhaustive search for the best weight triple.

    These tests take ~1 second because the sweep is just math — no
    database or API calls involved.
    """

    def test_sweep_finds_good_weights(self):
        """The best weight triple should satisfy at least 80% of preferences."""
        best_weights, best_score, total = sweep_weights(step=0.05)
        w_sim, w_qual, w_pop = best_weights

        print(f"\n{'='*60}")
        print(f"Best weights: sim={w_sim:.2f}  qual={w_qual:.2f}  pop={w_pop:.2f}")
        print(f"Preferences satisfied: {best_score}/{total}")

        eff = effective_weights(w_sim, w_qual, w_pop)
        print(f"Effective influence:  sim={eff['similarity']:.1%}  "
              f"qual={eff['quality']:.1%}  pop={eff['popularity']:.1%}")
        print(f"{'='*60}")

        assert best_score >= total * 0.80, (
            f"Best weights only satisfy {best_score}/{total} preferences"
        )

    def test_report_current_vs_best(self):
        """Side-by-side comparison: are the current weights close to optimal?

        This test always passes — it's here for its printed output (run
        with -s flag to see it). Useful when deciding whether to change
        the defaults.
        """
        current_score = sum(
            check_preference(p, 0.70, 0.20, 0.10) for p in PREFERENCES
        )
        best_weights, best_score, total = sweep_weights(step=0.05)

        print(f"\nCurrent (0.70/0.20/0.10): {current_score}/{total}")
        print(f"Best {best_weights}: {best_score}/{total}")

        eff_current = effective_weights(0.70, 0.20, 0.10)
        eff_best = effective_weights(*best_weights)
        print(f"Current effective: sim={eff_current['similarity']:.1%} "
              f"qual={eff_current['quality']:.1%} pop={eff_current['popularity']:.1%}")
        print(f"Best effective:    sim={eff_best['similarity']:.1%} "
              f"qual={eff_best['quality']:.1%} pop={eff_best['popularity']:.1%}")
