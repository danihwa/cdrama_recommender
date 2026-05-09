"""Unit tests for format_parsed_filters — the 'Parsed:' block in the CLI."""

from __future__ import annotations

from src.recommender.models import QueryFilters
from src.recommender.pipeline import format_parsed_filters


def test_minimal_reference_query_shows_only_set_fields() -> None:
    filters = QueryFilters(
        search_mode="reference",
        reference_title="Nirvana in Fire",
        min_score=8.0,
    )
    out = format_parsed_filters(filters)
    assert "mode             = reference" in out
    assert "reference_title  = Nirvana in Fire" in out
    assert "min_score        = 8.0" in out
    # Empty defaults are hidden.
    assert "genres" not in out
    assert "exclude_genres" not in out
    assert "exclude_titles" not in out
    assert "description" not in out
    assert "min_year" not in out


def test_exclude_genres_displayed_when_present() -> None:
    filters = QueryFilters(
        search_mode="reference",
        reference_title="Nirvana in Fire",
        exclude_genres=["wuxia", "fantasy"],
    )
    out = format_parsed_filters(filters)
    assert "exclude_genres" in out
    assert "wuxia" in out
    assert "fantasy" in out


def test_mode_always_shown_even_at_default() -> None:
    filters = QueryFilters()  # all defaults; search_mode = "reference"
    out = format_parsed_filters(filters)
    assert "mode             = reference" in out


def test_starts_with_parsed_header() -> None:
    filters = QueryFilters(search_mode="sql", min_year=2022)
    out = format_parsed_filters(filters)
    assert out.startswith("Parsed:\n")
