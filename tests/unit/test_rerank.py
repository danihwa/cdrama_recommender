"""
tests/unit/test_rerank.py
-------------------------
Unit tests for rerank_candidates().

THE BIG PICTURE
===============
The reranker (src/recommender/pipeline.py) is the step in our RAG pipeline
that takes ~10 candidate dramas retrieved from the database and decides
which order to show them in. It blends three signals into one score:

    ensemble_score = 0.70 * similarity + 0.20 * quality + 0.10 * popularity

Where:
  - similarity: cosine similarity from the vector search (0 to 1).
                Higher = the drama's embedding is closer to the query.
  - quality:    MDL score divided by 10, so it's on a 0–1 scale too.
  - popularity: log(watchers) normalised to the batch max, also 0–1.

Because the function is *pure* (no database calls, no API calls — just
math on a list of dicts), we can test it thoroughly without any mocking
or network access. That's what this file does.

No API calls, no DB — pure Python. Fast enough to run on every commit.

Run:
    uv run pytest tests/unit/test_rerank.py -v
"""

from __future__ import annotations

import math

import pytest

from src.recommender.pipeline import rerank_candidates


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def drama(
    title: str = "Test Drama",
    similarity: float = 0.0,
    mdl_score: float = 0.0,
    watchers: int = 1,
) -> dict:
    """Factory that builds a fake drama dict with only the fields the reranker reads.

    The reranker only looks at similarity, mdl_score, and watchers — it
    ignores genres, tags, synopsis, etc. Keeping the factory minimal makes
    each test honest about the real interface.
    """
    return {"title": title, "similarity": similarity, "mdl_score": mdl_score, "watchers": watchers}


# ---------------------------------------------------------------------------
# Edge cases
# ---------------------------------------------------------------------------


def test_empty_list_returns_empty():
    """The reranker should handle an empty candidate list gracefully."""
    assert rerank_candidates([]) == []


def test_single_candidate_returned():
    """With one drama, it's automatically the most popular in its "batch".

    The reranker normalises popularity to the batch max, so a single
    drama always gets popularity = log(watchers)/log(watchers) = 1.0.
    """
    result = rerank_candidates([drama("A", similarity=0.9, mdl_score=8.0, watchers=1000)])
    assert len(result) == 1


def test_missing_similarity_defaults_to_zero():
    """If the dict has no 'similarity' key at all, treat it as 0.

    This happens in SQL mode where the query builder doesn't include
    a similarity column — the key simply isn't in the dict.
    """
    candidate = {"title": "A", "mdl_score": 8.0, "watchers": 100}
    result = rerank_candidates([candidate])
    # similarity=0, quality=0.8, popularity=1.0 (only one drama -> max_watchers=its own)
    assert result[0]["ensemble_score"] == pytest.approx(0.0 * 0.70 + 0.8 * 0.20 + 1.0 * 0.10)


def test_missing_mdl_score_defaults_to_zero():
    """If 'mdl_score' key is missing, quality contribution should be 0."""
    candidate = {"title": "A", "similarity": 0.8, "watchers": 100}
    result = rerank_candidates([candidate])
    assert result[0]["ensemble_score"] == pytest.approx(0.8 * 0.70 + 0.0 * 0.20 + 1.0 * 0.10)


def test_none_mdl_score_treated_as_zero():
    """If mdl_score is explicitly None (DB returned null), same as missing.

    This is a separate case from "missing key" because Python's
    `dict.get("mdl_score")` returns None (not KeyError), and the
    reranker uses `(x or 0.0)` to coerce None -> 0.0.
    """
    candidate = {"title": "A", "similarity": 0.5, "mdl_score": None, "watchers": 100}
    result = rerank_candidates([candidate])
    assert result[0]["ensemble_score"] == pytest.approx(0.5 * 0.70 + 0.0 * 0.20 + 1.0 * 0.10)


def test_zero_watchers_does_not_crash():
    """watchers=0 is coerced to 1 via `or 1` — must not cause log(0).

    Without the guard, math.log(0) would raise ValueError. The reranker
    uses `max(watchers, 1)` so log never sees values < 1.
    """
    candidate = {"title": "A", "similarity": 0.5, "mdl_score": 8.0, "watchers": 0}
    result = rerank_candidates([candidate])
    assert "ensemble_score" in result[0]
    assert result[0]["ensemble_score"] >= 0


def test_none_watchers_does_not_crash():
    """watchers=None (DB null) should be coerced to 1, not crash.

    When there's only one candidate and its watchers is None, the
    max_watchers for the batch is also 1, which triggers the
    `log_max_watchers = 1` fallback (to avoid dividing by log(1)=0).
    So popularity = log(1)/1 = 0.0.
    """
    candidate = {"title": "A", "similarity": 0.5, "mdl_score": 8.0, "watchers": None}
    result = rerank_candidates([candidate])
    assert "ensemble_score" in result[0]


# ---------------------------------------------------------------------------
# Score calculation
# ---------------------------------------------------------------------------


def test_ensemble_score_weights():
    """Spot-check the exact formula: 0.70*sim + 0.20*quality + 0.10*popularity."""
    # Single candidate -> popularity = log(watchers) / log(watchers) = 1.0
    c = drama("A", similarity=0.8, mdl_score=9.0, watchers=5000)
    result = rerank_candidates([c])

    expected = 0.70 * 0.8 + 0.20 * (9.0 / 10.0) + 0.10 * 1.0
    assert result[0]["ensemble_score"] == pytest.approx(expected)


def test_popularity_normalised_relative_to_max():
    """The drama with most watchers gets popularity=1.0; others are scaled down.

    This is per-batch normalisation — the most-watched drama in this
    candidate set always gets popularity=1.0, even if it's niche globally.
    """
    low = drama("Low", similarity=0.5, mdl_score=7.0, watchers=100)
    high = drama("High", similarity=0.5, mdl_score=7.0, watchers=10_000)

    result = rerank_candidates([low, high])

    high_result = next(r for r in result if r["title"] == "High")
    low_result = next(r for r in result if r["title"] == "Low")

    assert high_result["ensemble_score"] == pytest.approx(
        0.70 * 0.5 + 0.20 * 0.7 + 0.10 * 1.0
    )
    expected_low_pop = math.log(100) / math.log(10_000)
    assert low_result["ensemble_score"] == pytest.approx(
        0.70 * 0.5 + 0.20 * 0.7 + 0.10 * expected_low_pop
    )


def test_log_scaling_compresses_popularity():
    """Log-scaling means a 1,000,000x watcher gap only contributes 0.10 to the score.

    Without log-scaling, a single blockbuster could dominate the
    popularity term. Log compresses the range so popularity acts more
    like a tiebreaker than a dominant signal.

    Here both dramas have identical similarity and quality, so the
    ONLY difference in their ensemble_score comes from popularity.
    watchers=1 -> log(1)/log(1M) = 0.0, watchers=1M -> log(1M)/log(1M) = 1.0.
    Score diff = 0.10 * (1.0 - 0.0) = 0.10.
    """
    niche = drama("Niche", similarity=0.85, mdl_score=8.0, watchers=1)
    blockbuster = drama("Blockbuster", similarity=0.85, mdl_score=8.0, watchers=1_000_000)

    result = rerank_candidates([niche, blockbuster])

    niche_result = next(r for r in result if r["title"] == "Niche")
    blockbuster_result = next(r for r in result if r["title"] == "Blockbuster")
    assert blockbuster_result["ensemble_score"] - niche_result["ensemble_score"] == pytest.approx(0.10)


def test_sql_mode_zero_similarity():
    """When all similarities are 0, ranking = quality + popularity only.

    In SQL mode the user just gave filters ("romance from 2022 rated 8+")
    with no plot description and no reference drama. There's nothing to
    embed, so every candidate comes back with similarity = 0. The
    reranker's similarity term drops out — which is exactly what you'd
    want for a filter query.
    """
    candidates = [
        drama("A", similarity=0.0, mdl_score=9.5, watchers=50000),
        drama("B", similarity=0.0, mdl_score=8.0, watchers=80000),
        drama("C", similarity=0.0, mdl_score=7.0, watchers=1000),
    ]
    result = rerank_candidates(candidates)
    # C should be last — lowest quality AND lowest popularity
    assert result[-1]["title"] == "C"
    # Verify scores only use quality + popularity (max_watchers = 80000)
    for d in result:
        assert d["ensemble_score"] == pytest.approx(
            0.20 * (d["mdl_score"] / 10.0)
            + 0.10 * (math.log(max(d["watchers"], 1)) / math.log(80000))
        )


def test_numerical_correctness():
    """Hand-computed scores verified with pytest.approx.

    This is the "golden calculation" test — if someone refactors the
    reranker, this catches any subtle math changes. We compute each
    intermediate value by hand and compare.

    Candidate A: sim=0.92, mdl=8.5, watchers=25000 (the batch max)
    Candidate B: sim=0.88, mdl=9.2, watchers=5000

    B has higher quality but lower similarity. With the default weights,
    A wins because the similarity gap (0.04) weighted at 0.70 outweighs
    B's quality advantage (0.07) weighted at 0.20.
    """
    candidates = [
        drama("A", similarity=0.92, mdl_score=8.5, watchers=25000),
        drama("B", similarity=0.88, mdl_score=9.2, watchers=5000),
    ]
    log_max = math.log(25000)

    pop_a = math.log(25000) / log_max  # = 1.0 (batch max always gets 1.0)
    score_a = 0.70 * 0.92 + 0.20 * 0.85 + 0.10 * pop_a

    pop_b = math.log(5000) / log_max  # ~= 0.84
    score_b = 0.70 * 0.88 + 0.20 * 0.92 + 0.10 * pop_b

    result = rerank_candidates(candidates)
    assert result[0]["title"] == "A"
    assert result[0]["ensemble_score"] == pytest.approx(score_a)
    assert result[1]["title"] == "B"
    assert result[1]["ensemble_score"] == pytest.approx(score_b)


# ---------------------------------------------------------------------------
# Sort order
# ---------------------------------------------------------------------------


def test_sorted_descending_by_ensemble_score():
    """Results should come back sorted highest ensemble_score first."""
    candidates = [
        drama("C", similarity=0.5),
        drama("A", similarity=0.9),
        drama("B", similarity=0.7),
    ]
    result = rerank_candidates(candidates)
    scores = [r["ensemble_score"] for r in result]
    assert scores == sorted(scores, reverse=True)


def test_all_same_scores():
    """Identical candidates should all get the same ensemble_score.

    We don't assert order among ties — Python's sort is stable so it
    preserves input order, but that's an implementation detail we
    shouldn't depend on.
    """
    candidates = [
        drama("A", similarity=0.8, mdl_score=8.0, watchers=500),
        drama("B", similarity=0.8, mdl_score=8.0, watchers=500),
        drama("C", similarity=0.8, mdl_score=8.0, watchers=500),
    ]
    result = rerank_candidates(candidates)
    assert len(result) == 3
    scores = [d["ensemble_score"] for d in result]
    assert scores[0] == pytest.approx(scores[1])
    assert scores[1] == pytest.approx(scores[2])


def test_similarity_dominates_when_scores_differ_greatly():
    """A drama with much higher similarity should rank first despite lower quality.

    This is the core design choice: similarity is weighted 0.70, so a
    big sim advantage (0.95 vs 0.50) can't be overcome by quality alone.
    """
    high_sim = drama("HighSim", similarity=0.95, mdl_score=6.0, watchers=100)
    high_qual = drama("HighQual", similarity=0.50, mdl_score=9.5, watchers=100)

    result = rerank_candidates([high_sim, high_qual])
    assert result[0]["title"] == "HighSim"


def test_quality_breaks_tie_when_similarity_equal():
    """Equal similarity -> higher MDL score should win."""
    low_q = drama("LowQ", similarity=0.8, mdl_score=6.0, watchers=100)
    high_q = drama("HighQ", similarity=0.8, mdl_score=9.0, watchers=100)

    result = rerank_candidates([low_q, high_q])
    assert result[0]["title"] == "HighQ"


def test_original_list_not_mutated_in_order():
    """rerank_candidates returns a new sorted list; input order is preserved.

    Note: the dicts themselves ARE mutated (ensemble_score is added),
    but the original list's ordering stays intact. sorted() creates a
    new list rather than sorting in-place.
    """
    candidates = [
        drama("First", similarity=0.3),
        drama("Second", similarity=0.9),
    ]
    result = rerank_candidates(candidates)
    # Result is sorted — "Second" should come first
    assert result[0]["title"] == "Second"
    # But the original list still starts with "First"
    assert candidates[0]["title"] == "First"
