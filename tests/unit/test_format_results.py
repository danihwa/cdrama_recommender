"""Unit tests for format_results — the numbered candidate table."""

from __future__ import annotations

from src.recommender.pipeline import format_results


def _drama(
    title: str, year: int, mdl_score: float, similarity: float = 0.0
) -> dict:
    return {
        "title": title,
        "year": year,
        "mdl_score": mdl_score,
        "similarity": similarity,
    }


def test_header_and_seven_numbered_rows() -> None:
    rows = [
        _drama("Drama A", 2023, 9.2, similarity=0.847),
        _drama("Drama B", 2024, 8.8, similarity=0.821),
        _drama("Drama C", 2022, 8.6, similarity=0.802),
        _drama("Drama D", 2023, 8.5, similarity=0.798),
        _drama("Drama E", 2024, 8.7, similarity=0.781),
        _drama("Drama F", 2022, 8.4, similarity=0.764),
        _drama("Drama G", 2023, 8.3, similarity=0.751),
    ]
    out = format_results(rows)
    assert "Top 7 candidates:" in out
    assert "============================================================" in out
    for i in range(1, 8):
        assert f" {i}. " in out
    assert "[2023] Drama A" in out
    assert "score 9.2" in out
    assert "similarity 0.847" in out


def test_sql_mode_em_dash_for_zero_similarity() -> None:
    rows = [
        _drama("SQL One", 2022, 8.5, similarity=0.0),
        _drama("SQL Two", 2023, 8.2, similarity=0.0),
    ]
    out = format_results(rows)
    # Every row has similarity 0 -> SQL mode -> em dash, no 0.000.
    assert "0.000" not in out
    assert "similarity   —" in out


def test_reference_mode_keeps_decimal_when_any_similarity_is_nonzero() -> None:
    rows = [
        _drama("R One", 2022, 8.5, similarity=0.812),
        _drama("R Two", 2023, 8.2, similarity=0.0),  # legitimately exactly 0
    ]
    out = format_results(rows)
    # At least one nonzero -> not SQL mode -> all rows print numeric similarity.
    assert "0.812" in out
    assert "0.000" in out
    assert "—" not in out  # em dash only used for SQL mode


def test_empty_list_returns_empty_string_or_no_table() -> None:
    out = format_results([])
    assert "Top 7 candidates:" not in out
