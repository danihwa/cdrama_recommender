"""Validation tests for the API request schemas."""

from __future__ import annotations

import pytest
from pydantic import ValidationError

from src.api.schemas import RecommendRequest


def test_minimal_valid_request():
    req = RecommendRequest(message="hello")
    assert req.message == "hello"
    assert req.history == []


def test_history_with_valid_roles_accepted():
    req = RecommendRequest(
        message="hello",
        history=[
            {"role": "user", "content": "hi"},
            {"role": "assistant", "content": "hey back"},
        ],
    )
    assert len(req.history) == 2
    assert req.history[0].role == "user"


def test_empty_message_rejected():
    with pytest.raises(ValidationError):
        RecommendRequest(message="")


def test_unknown_role_rejected():
    with pytest.raises(ValidationError):
        RecommendRequest(
            message="x", history=[{"role": "system", "content": "bad"}]
        )


def test_history_missing_content_rejected():
    with pytest.raises(ValidationError):
        RecommendRequest(message="x", history=[{"role": "user"}])
