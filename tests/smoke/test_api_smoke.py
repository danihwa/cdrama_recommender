"""End-to-end smoke test: real Supabase + real OpenAI through the API.

Marked with `db` and `parser` so the default `pytest -m "not db and not
parser"` invocation skips it. Run explicitly with:
    uv run pytest tests/smoke/test_api_smoke.py -v
"""

from __future__ import annotations

import json

import pytest
from fastapi.testclient import TestClient

from src.api.main import app


@pytest.mark.db
@pytest.mark.parser
def test_recommend_streams_end_to_end():
    """The full pipeline produces at least one candidate and one token."""
    with TestClient(app) as client:  # `with` triggers lifespan -> real clients
        response = client.post(
            "/recommend",
            json={"message": "A cozy historical romance", "history": []},
        )
        assert response.status_code == 200

    seen_candidates = False
    seen_token = False
    seen_done = False
    for line in response.text.splitlines():
        if line.startswith("event: candidates"):
            seen_candidates = True
        elif line.startswith("event: token"):
            seen_token = True
        elif line.startswith("event: done"):
            seen_done = True

    assert seen_candidates, "expected at least one candidates event"
    assert seen_token, "expected at least one token event"
    assert seen_done, "expected a done event"


@pytest.mark.db
def test_dramas_returns_real_titles():
    with TestClient(app) as client:
        response = client.get("/dramas")
        assert response.status_code == 200
        body = response.json()
        assert isinstance(body["titles"], list)
        assert len(body["titles"]) > 0
        assert all(isinstance(t, str) for t in body["titles"])
