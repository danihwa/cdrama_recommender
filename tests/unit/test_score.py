"""Unit tests for score_candidates().

The scorer blends three signals into one score:
    ensemble_score = 0.70 * similarity + 0.20 * (mdl_score/10) + 0.10 * popularity
where popularity = log(watchers) / log(max_watchers_in_batch).
"""

from __future__ import annotations

import pytest

from src.recommender.pipeline import score_candidates


def drama(
    title: str = "Test Drama",
    similarity: float = 0.0,
    mdl_score: float = 0.0,
    watchers: int = 1,
) -> dict:
    return {"title": title, "similarity": similarity, "mdl_score": mdl_score, "watchers": watchers}


def test_empty_list_returns_empty():
    assert score_candidates([]) == []


@pytest.mark.parametrize(
    "candidate",
    [
        {"title": "A", "mdl_score": 8.0, "watchers": 100},          # similarity missing
        {"title": "A", "similarity": 0.5, "watchers": 100},          # mdl_score missing
        {"title": "A", "similarity": 0.5, "mdl_score": None, "watchers": 100},
        {"title": "A", "similarity": 0.5, "mdl_score": 8.0, "watchers": 0},
        {"title": "A", "similarity": 0.5, "mdl_score": 8.0, "watchers": None},
    ],
)
def test_missing_or_zero_fields_do_not_crash(candidate):
    """Missing keys, None values, and watchers=0 must coerce to safe defaults."""
    result = score_candidates([candidate])
    assert result[0]["ensemble_score"] >= 0


def test_ensemble_score_formula():
    """Spot-check the exact formula on a single candidate (popularity = 1.0)."""
    c = drama("A", similarity=0.8, mdl_score=9.0, watchers=5000)
    result = score_candidates([c])
    expected = 0.70 * 0.8 + 0.20 * 0.9 + 0.10 * 1.0
    assert result[0]["ensemble_score"] == pytest.approx(expected)


def test_log_scaling_compresses_popularity():
    """A 1,000,000x watcher gap contributes exactly 0.10 to the score."""
    niche = drama("Niche", similarity=0.85, mdl_score=8.0, watchers=1)
    blockbuster = drama("Blockbuster", similarity=0.85, mdl_score=8.0, watchers=1_000_000)
    result = score_candidates([niche, blockbuster])
    n = next(r for r in result if r["title"] == "Niche")
    b = next(r for r in result if r["title"] == "Blockbuster")
    assert b["ensemble_score"] - n["ensemble_score"] == pytest.approx(0.10)


def test_sorted_descending_by_ensemble_score():
    candidates = [
        drama("C", similarity=0.5),
        drama("A", similarity=0.9),
        drama("B", similarity=0.7),
    ]
    result = score_candidates(candidates)
    scores = [r["ensemble_score"] for r in result]
    assert scores == sorted(scores, reverse=True)
    assert result[0]["title"] == "A"
