"""Tests for generate_recommendation_stream — the streaming variant of
generate_recommendation. We mock OpenAI here because the real client is
external; the test verifies our token-extraction loop, not OpenAI itself.
"""

from __future__ import annotations

from unittest.mock import MagicMock

from src.recommender.pipeline import generate_recommendation_stream


def _chunk(text: str | None):
    """Build a fake ChatCompletionChunk with the shape openai 2.x returns."""
    chunk = MagicMock()
    chunk.choices = [MagicMock()]
    chunk.choices[0].delta.content = text
    return chunk


def _drama() -> dict:
    """Minimal drama dict matching the fields build_context reads."""
    return {
        "title": "X",
        "year": 2024,
        "mdl_score": 8.0,
        "synopsis": "",
        "genres": [],
        "tags": [],
    }


def test_yields_token_text_in_order():
    fake_openai = MagicMock()
    fake_openai.chat.completions.create.return_value = iter(
        [_chunk("Hello "), _chunk("world"), _chunk("!")]
    )

    tokens = list(
        generate_recommendation_stream(
            user_query="anything", dramas=[_drama()], openai=fake_openai
        )
    )
    assert tokens == ["Hello ", "world", "!"]


def test_skips_chunks_with_no_content():
    """OpenAI streams sometimes emit empty chunks (role-only delta, finish_reason)."""
    fake_openai = MagicMock()
    fake_openai.chat.completions.create.return_value = iter(
        [_chunk(None), _chunk("hi"), _chunk(None)]
    )

    tokens = list(
        generate_recommendation_stream(
            user_query="anything", dramas=[_drama()], openai=fake_openai
        )
    )
    assert tokens == ["hi"]


def test_passes_stream_true_to_openai():
    fake_openai = MagicMock()
    fake_openai.chat.completions.create.return_value = iter([])

    list(
        generate_recommendation_stream(
            user_query="x", dramas=[_drama()], openai=fake_openai
        )
    )
    _, kwargs = fake_openai.chat.completions.create.call_args
    assert kwargs.get("stream") is True
