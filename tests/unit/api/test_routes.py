"""TestClient-based unit tests for the API routes.

We bypass the real lifespan by NOT using `with TestClient(...)` — that
keeps the tests offline. Each test sets up the bits of `app.state` it
needs and monkeypatches the pipeline functions imported by routes.
"""

from __future__ import annotations

import json

import pytest
from fastapi.testclient import TestClient

from src.api.main import app


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
