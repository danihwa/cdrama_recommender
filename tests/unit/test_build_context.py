from __future__ import annotations

from src.recommender.pipeline import build_context


def drama(**overrides) -> dict:
    """Minimal drama dict with all fields build_context reads."""
    base = {
        "title": "Test Drama",
        "year": 2023,
        "mdl_score": 8.5,
        "synopsis": "A test synopsis.",
        "genres": ["Romance"],
        "tags": ["Tag1", "Tag2"],
    }
    return {**base, **overrides}


# ---------------------------------------------------------------------------
# Edge cases
# ---------------------------------------------------------------------------


def test_empty_list_returns_empty_string():
    assert build_context([]) == ""


# ---------------------------------------------------------------------------
# Format
# ---------------------------------------------------------------------------


def test_title_line_format():
    output = build_context([drama(title="Nirvana in Fire", year=2015, mdl_score=9.0)])
    assert "Title: Nirvana in Fire (2015) | MDL Score: 9.0" in output


def test_synopsis_appears_in_output():
    output = build_context([drama(synopsis="A general clears his name.")])
    assert "Synopsis: A general clears his name." in output


def test_genres_joined_by_comma():
    output = build_context([drama(genres=["Historical", "Political"])])
    assert "Genres: Historical, Political" in output


def test_none_genres_does_not_crash():
    """genres=None (DB null) should produce an empty Genres line, not a TypeError."""
    output = build_context([drama(genres=None)])
    assert "Genres: " in output


def test_none_tags_does_not_crash():
    """tags=None (DB null) should produce an empty Tags line, not a TypeError."""
    output = build_context([drama(tags=None)])
    assert "Tags: " in output


def test_tags_truncated_to_five():
    """Only the first 5 tags appear — the 6th is silently dropped."""
    output = build_context([drama(tags=["T1", "T2", "T3", "T4", "T5", "T6"])])
    assert "T5" in output
    assert "T6" not in output


def test_multiple_dramas_separated_by_double_newline():
    """Two drama blocks are separated by a blank line so the LLM sees clear boundaries."""
    output = build_context([drama(title="Drama A"), drama(title="Drama B")])
    assert "Drama A" in output
    assert "Drama B" in output
    assert "\n\nTitle:" in output
