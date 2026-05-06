"""TestClient-based unit tests for the API routes.

We bypass the real lifespan by NOT using `with TestClient(...)` — that
keeps the tests offline. Each test sets up the bits of `app.state` it
needs and monkeypatches the pipeline functions imported by routes.
"""

from __future__ import annotations

import json

import pytest
from fastapi.testclient import TestClient

import src.api.routes as routes_module
from src.api.main import app
from src.recommender.models import QueryFilters


def _parse_sse(text: str) -> list[dict]:
    """Parse an SSE response body into a list of {event, data} dicts.

    Each frame is "event: NAME\\ndata: PAYLOAD\\n\\n". sse-starlette also
    emits a leading ping comment ("ping - ...") and a trailing newline,
    which this helper ignores. Comment lines (starting with ":") and
    blank lines are skipped; only event/data pairs are returned.
    """
    events: list[dict] = []
    current: dict = {}
    for raw in text.splitlines():
        line = raw.rstrip("\r")
        if not line:
            if current:
                events.append(current)
                current = {}
            continue
        if line.startswith(":"):
            continue
        if line.startswith("event:"):
            current["event"] = line.split(":", 1)[1].strip()
        elif line.startswith("data:"):
            current["data"] = line.split(":", 1)[1].strip()
    if current:
        events.append(current)
    return events


@pytest.fixture
def client():
    """A TestClient with stubbed app.state — no real Supabase/OpenAI."""
    app.state.supabase = object()
    app.state.openai = object()
    app.state.drama_titles = ["A Drama", "B Drama", "C Drama"]
    return TestClient(app)


def test_get_dramas_returns_titles(client):
    response = client.get("/dramas")
    assert response.status_code == 200
    assert response.json() == {"titles": ["A Drama", "B Drama", "C Drama"]}


def _stub_pipeline(
    monkeypatch,
    *,
    filters: QueryFilters,
    candidates: list[dict],
    tokens: list[str],
    raise_in_generator: Exception | None = None,
):
    """Replace the three pipeline functions imported into routes.py."""
    monkeypatch.setattr(
        routes_module, "parse_user_query", lambda *a, **kw: filters
    )
    monkeypatch.setattr(
        routes_module, "retrieve_candidates", lambda *a, **kw: candidates
    )

    def fake_stream(*a, **kw):
        if raise_in_generator is not None:
            raise raise_in_generator
        yield from tokens

    monkeypatch.setattr(
        routes_module, "generate_recommendation_stream", fake_stream
    )


def _candidate(title: str = "Drama A") -> dict:
    return {
        "title": title,
        "year": 2024,
        "mdl_score": 8.5,
        "mdl_url": f"https://example.com/{title}",
        "synopsis": "Test synopsis.",
        "similarity": 0.9,
        "watchers": 1000,
        "genres": ["romance"],
    }


def test_recommend_happy_path_emits_candidates_then_tokens_then_done(client, monkeypatch):
    _stub_pipeline(
        monkeypatch,
        filters=QueryFilters(search_mode="semantic", description="test"),
        candidates=[_candidate("A"), _candidate("B")],
        tokens=["Hello ", "world"],
    )

    response = client.post("/recommend", json={"message": "give me something fun"})
    assert response.status_code == 200

    events = _parse_sse(response.text)
    names = [e["event"] for e in events]
    assert names == ["candidates", "token", "token", "done"]
    assert json.loads(events[1]["data"]) == {"text": "Hello "}
    assert json.loads(events[2]["data"]) == {"text": "world"}

    candidates_payload = json.loads(events[0]["data"])
    assert [c["title"] for c in candidates_payload["candidates"]] == ["A", "B"]


def test_recommend_refusal_emits_info_then_done_no_candidates(client, monkeypatch):
    _stub_pipeline(
        monkeypatch,
        filters=QueryFilters(search_mode="refused"),
        candidates=[],
        tokens=[],
    )

    response = client.post("/recommend", json={"message": "off-topic"})
    events = _parse_sse(response.text)

    names = [e["event"] for e in events]
    assert names == ["info", "done"]
    assert "only able to help with" in json.loads(events[0]["data"])["message"]


def test_recommend_empty_results_emits_info_then_done(client, monkeypatch):
    _stub_pipeline(
        monkeypatch,
        filters=QueryFilters(search_mode="sql", min_year=2099),
        candidates=[],
        tokens=[],
    )

    response = client.post("/recommend", json={"message": "anything from 2099"})
    events = _parse_sse(response.text)

    names = [e["event"] for e in events]
    assert names == ["info", "done"]
    assert "no dramas found" in json.loads(events[0]["data"])["message"]


def test_recommend_mid_stream_exception_emits_error_then_done(client, monkeypatch):
    _stub_pipeline(
        monkeypatch,
        filters=QueryFilters(search_mode="semantic", description="test"),
        candidates=[_candidate("A")],
        tokens=[],
        raise_in_generator=RuntimeError("boom"),
    )

    response = client.post("/recommend", json={"message": "trigger an error"})
    assert response.status_code == 200

    events = _parse_sse(response.text)
    names = [e["event"] for e in events]
    assert "error" in names
    assert names[-1] == "done"
