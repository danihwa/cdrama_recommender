# FastAPI Backend Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Move the recommender pipeline behind an HTTP API so `app.py` becomes a pure frontend that talks to FastAPI over SSE.

**Architecture:** New `src/api/` package exposes two endpoints: `GET /dramas` (sidebar dropdown) and `POST /recommend` (SSE stream emitting `candidates`, `token`, `info`, `error`, `done` events). `pipeline.py` gets a new `generate_recommendation_stream` that yields tokens; the existing `run_rag` stays for the CLI. `app.py` swaps its in-process pipeline call for `httpx.stream` against the API.

**Tech Stack:** FastAPI, sse-starlette, httpx, OpenAI Python SDK 2.x (streaming), Supabase, Streamlit, pytest.

**Spec reference:** `docs/superpowers/specs/2026-05-06-fastapi-backend-design.md`

---

## File Structure

**Create:**
- `src/api/__init__.py` — empty
- `src/api/schemas.py` — `ChatMessage`, `RecommendRequest` Pydantic models
- `src/api/main.py` — FastAPI instance + lifespan
- `src/api/routes.py` — `/dramas` and `/recommend` handlers + SSE generator
- `tests/unit/api/__init__.py` — empty
- `tests/unit/api/test_schemas.py` — Pydantic validation tests
- `tests/unit/api/test_routes.py` — TestClient-based route tests with mocked pipeline
- `tests/unit/test_generate_stream.py` — `generate_recommendation_stream` unit tests
- `tests/smoke/test_api_smoke.py` — end-to-end `/recommend` against real Supabase + OpenAI

**Modify:**
- `pyproject.toml` — add fastapi, uvicorn, sse-starlette, httpx
- `src/recommender/pipeline.py` — add `generate_recommendation_stream` (existing functions untouched)
- `app.py` — remove direct pipeline + Supabase imports, consume API over HTTP+SSE
- `README.md` — update "Running it locally" to two-terminal flow

---

## Task 1: Add FastAPI / SSE / httpx dependencies

**Files:**
- Modify: `pyproject.toml`

- [ ] **Step 1: Update pyproject.toml dependencies**

Open `pyproject.toml` and replace the `dependencies` array with:

```toml
dependencies = [
    "bs4>=0.0.2",
    "dotenv>=0.9.9",
    "fastapi>=0.115",
    "httpx>=0.27",
    "matplotlib>=3.10.8",
    "openai>=2.31.0",
    "pandas>=3.0.2",
    "pyarrow>=23.0.1",
    "pytest>=9.0.3",
    "requests>=2.33.1",
    "sse-starlette>=2.1",
    "streamlit>=1.56.0",
    "supabase>=2.28.3",
    "uvicorn[standard]>=0.30",
]
```

- [ ] **Step 2: Sync the lockfile**

Run: `uv sync`
Expected: completes with no errors. `uv.lock` should now mention the four new packages.

- [ ] **Step 3: Smoke-check the imports**

Run: `uv run python -c "import fastapi, sse_starlette, httpx, uvicorn; print('ok')"`
Expected: `ok`

- [ ] **Step 4: Commit**

```bash
git add pyproject.toml uv.lock
git commit -m "chore: add fastapi, sse-starlette, httpx, uvicorn deps"
```

---

## Task 2: Add `generate_recommendation_stream` to pipeline.py

**Files:**
- Modify: `src/recommender/pipeline.py` — add new function next to `generate_recommendation`
- Create: `tests/unit/test_generate_stream.py`

The existing `generate_recommendation` calls OpenAI without streaming and returns a single string. This task adds a peer function that calls OpenAI with `stream=True` and yields the text delta from each chunk. The existing function is **not** modified — `run_rag` and the CLI keep working unchanged.

- [ ] **Step 1: Write failing tests**

Create `tests/unit/test_generate_stream.py`:

```python
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
```

- [ ] **Step 2: Run the tests to confirm they fail**

Run: `uv run pytest tests/unit/test_generate_stream.py -v`
Expected: 3 ImportError-style failures with `cannot import name 'generate_recommendation_stream'`.

- [ ] **Step 3: Implement the function**

Open `src/recommender/pipeline.py`. Add this **directly below** the existing `generate_recommendation` function (around line 232), keeping all other code unchanged:

```python
def generate_recommendation_stream(
    user_query: str,
    dramas: list[dict],
    openai: OpenAI,
    history: list[dict] | None = None,
):
    """Streaming variant of generate_recommendation — yields tokens.

    Mirrors generate_recommendation but uses stream=True so callers can
    forward partial output (e.g., over SSE) instead of waiting for the
    whole response. Empty deltas (role-only or finish-reason chunks) are
    skipped so callers always see real text.
    """
    recent_history = (history or [])[-HISTORY_MESSAGES:]
    stream = openai.chat.completions.create(
        model=MODEL,
        max_tokens=MAX_RESPONSE_TOKENS,
        stream=True,
        messages=[
            {"role": "system", "content": RECOMMEND_SYSTEM_PROMPT},
            *recent_history,
            {
                "role": "user",
                "content": f"""\
My request: {user_query}

Dramas to choose from:
{build_context(dramas)}\
""",
            },
        ],
    )
    for chunk in stream:
        delta = chunk.choices[0].delta.content
        if delta:
            yield delta
```

- [ ] **Step 4: Run the tests to confirm they pass**

Run: `uv run pytest tests/unit/test_generate_stream.py -v`
Expected: 3 passed.

- [ ] **Step 5: Run the full unit suite to confirm no regressions**

Run: `uv run pytest tests/unit -v`
Expected: all green; existing rerank/build_context tests still pass.

- [ ] **Step 6: Commit**

```bash
git add src/recommender/pipeline.py tests/unit/test_generate_stream.py
git commit -m "feat: add generate_recommendation_stream for SSE-friendly token output"
```

---

## Task 3: Add API request schemas

**Files:**
- Create: `src/api/__init__.py`
- Create: `src/api/schemas.py`
- Create: `tests/unit/api/__init__.py`
- Create: `tests/unit/api/test_schemas.py`

- [ ] **Step 1: Create the empty package init files**

Create `src/api/__init__.py` (empty file).
Create `tests/unit/api/__init__.py` (empty file).

- [ ] **Step 2: Write failing schema tests**

Create `tests/unit/api/test_schemas.py`:

```python
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
```

- [ ] **Step 3: Run the tests to confirm they fail**

Run: `uv run pytest tests/unit/api/test_schemas.py -v`
Expected: ImportError-style failures with `cannot import name 'RecommendRequest'`.

- [ ] **Step 4: Implement the schemas**

Create `src/api/schemas.py`:

```python
"""Pydantic models for the cdrama-recommender HTTP API."""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, Field


class ChatMessage(BaseModel):
    """One message in the conversation history.

    Roles match what OpenAI's chat completion expects, restricted to the
    two we ever pass through (Streamlit only stores user/assistant turns).
    """

    role: Literal["user", "assistant"]
    content: str


class RecommendRequest(BaseModel):
    """Body for POST /recommend.

    `message` is the user's latest query (with the sidebar filter hint
    already glued on by the client). `history` is the windowed
    conversation — the API does no further trimming.
    """

    message: str = Field(..., min_length=1)
    history: list[ChatMessage] = Field(default_factory=list)
```

- [ ] **Step 5: Run the tests to confirm they pass**

Run: `uv run pytest tests/unit/api/test_schemas.py -v`
Expected: 5 passed.

- [ ] **Step 6: Commit**

```bash
git add src/api/__init__.py src/api/schemas.py tests/unit/api/__init__.py tests/unit/api/test_schemas.py
git commit -m "feat: add RecommendRequest and ChatMessage API schemas"
```

---

## Task 4: Bootstrap FastAPI app + `GET /dramas`

**Files:**
- Create: `src/api/main.py`
- Create: `src/api/routes.py`
- Create: `tests/unit/api/test_routes.py`

The route module gets created here with just `/dramas`; `/recommend` is added in Task 5. This task also sets up the lifespan + `app.state` pattern that `/recommend` will reuse.

- [ ] **Step 1: Write the failing test for `GET /dramas`**

Create `tests/unit/api/test_routes.py`:

```python
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
```

- [ ] **Step 2: Run the test to confirm it fails**

Run: `uv run pytest tests/unit/api/test_routes.py -v`
Expected: ImportError on `from src.api.main import app`.

- [ ] **Step 3: Create the routes module with `/dramas`**

Create `src/api/routes.py`:

```python
"""HTTP endpoints for the cdrama-recommender API."""

from __future__ import annotations

from fastapi import APIRouter, Request

router = APIRouter()


@router.get("/dramas")
def list_dramas(request: Request) -> dict:
    """Return the alphabetised drama titles for Streamlit's sidebar dropdown.

    The list is built once at startup (see main.py:lifespan) and stored
    on app.state, so this handler is just a read.
    """
    return {"titles": request.app.state.drama_titles}
```

- [ ] **Step 4: Create the FastAPI app with lifespan**

Create `src/api/main.py`:

```python
"""FastAPI app entry point for the cdrama-recommender API.

Run locally with:
    uv run uvicorn src.api.main:app --reload
"""

from __future__ import annotations

from contextlib import asynccontextmanager

from fastapi import FastAPI
from openai import OpenAI

from src.database.connection import get_db_connection
from src.env import load_secrets

from .routes import router


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Build shared clients once at startup and load drama titles for /dramas."""
    load_secrets()
    app.state.supabase = get_db_connection()
    app.state.openai = OpenAI()

    titles_response = (
        app.state.supabase.table("cdramas")
        .select("title")
        .order("title")
        .execute()
    )
    app.state.drama_titles = [row["title"] for row in titles_response.data]

    yield


app = FastAPI(title="cdrama-recommender API", lifespan=lifespan)
app.include_router(router)
```

- [ ] **Step 5: Run the test to confirm it passes**

Run: `uv run pytest tests/unit/api/test_routes.py -v`
Expected: 1 passed.

- [ ] **Step 6: Manually smoke-test that the server starts**

In a terminal: `uv run uvicorn src.api.main:app --port 8000`
Then in a second terminal: `curl http://127.0.0.1:8000/dramas | head -c 200`
Expected: JSON `{"titles": [...]}` with real titles. Stop the server with Ctrl-C.

(If `curl` isn't on hand, opening `http://127.0.0.1:8000/docs` in a browser and clicking "Try it out" on `/dramas` is fine.)

- [ ] **Step 7: Commit**

```bash
git add src/api/main.py src/api/routes.py tests/unit/api/test_routes.py
git commit -m "feat: bootstrap FastAPI app with /dramas endpoint"
```

---

## Task 5: Add `POST /recommend` (happy + refusal + empty + error paths)

**Files:**
- Modify: `src/api/routes.py` — add the streaming `/recommend` handler
- Modify: `tests/unit/api/test_routes.py` — add four new tests

All four code paths share one generator function, so they're implemented together. The tests are written first (one per path) to drive the shape.

- [ ] **Step 1: Add an SSE parsing helper to the test file**

In `tests/unit/api/test_routes.py`, add this above the existing fixture:

```python
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
```

- [ ] **Step 2: Add the four failing tests for `/recommend`**

First, add two imports to the top of `tests/unit/api/test_routes.py`, alongside the existing `import json` / `from fastapi.testclient...` block:

```python
import src.api.routes as routes_module
from src.recommender.models import QueryFilters
```

Then append the helpers and tests to the bottom of the same file:

```python
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
```

- [ ] **Step 3: Run the new tests to confirm they fail**

Run: `uv run pytest tests/unit/api/test_routes.py -v`
Expected: 4 failures (404 from `/recommend`); the existing `/dramas` test still passes.

- [ ] **Step 4: Implement `/recommend` in routes.py**

Replace the **entire contents** of `src/api/routes.py` with:

```python
"""HTTP endpoints for the cdrama-recommender API."""

from __future__ import annotations

import json
import logging

from fastapi import APIRouter, Request
from sse_starlette import EventSourceResponse

from src.recommender.pipeline import (
    NO_RESULTS_MESSAGE,
    REFUSED_MESSAGE,
    TOP_N,
    generate_recommendation_stream,
    parse_user_query,
    rerank_candidates,
    retrieve_candidates,
)

from .schemas import RecommendRequest

logger = logging.getLogger(__name__)

router = APIRouter()


@router.get("/dramas")
def list_dramas(request: Request) -> dict:
    """Return the alphabetised drama titles for Streamlit's sidebar dropdown."""
    return {"titles": request.app.state.drama_titles}


def _serialize_candidate(d: dict) -> dict:
    """Project a candidate dict down to fields Streamlit's render_candidate uses.

    Pruning here keeps embeddings and other heavy DB columns out of the
    SSE payload. The numeric defaults (0.0) matter because SQL-mode rows
    have no similarity score; sending None instead would crash the
    f-string formatter on the client side.
    """
    return {
        "title": d.get("title"),
        "year": d.get("year"),
        "mdl_score": d.get("mdl_score"),
        "mdl_url": d.get("mdl_url"),
        "similarity": d.get("similarity") or 0.0,
        "ensemble_score": d.get("ensemble_score") or 0.0,
        "watchers": d.get("watchers"),
        "genres": d.get("genres") or [],
        "synopsis": d.get("synopsis", ""),
    }


def _sse(event: str, payload: dict) -> dict:
    """Build the dict shape sse-starlette's EventSourceResponse expects."""
    return {"event": event, "data": json.dumps(payload)}


@router.post("/recommend")
def recommend(request: Request, body: RecommendRequest):
    """Stream a recommendation as SSE: candidates → token* → done.

    Refusals and empty-result paths emit a single `info` event instead of
    candidates+tokens. Mid-stream failures emit `error` and end cleanly
    with `done`. Pre-stream validation errors are handled by FastAPI as
    standard 422 responses.
    """
    supabase = request.app.state.supabase
    openai = request.app.state.openai
    history = [m.model_dump() for m in body.history]

    def event_stream():
        try:
            filters = parse_user_query(body.message, openai, history)

            if filters.search_mode == "refused":
                yield _sse("info", {"message": REFUSED_MESSAGE})
                yield _sse("done", {})
                return

            candidates = retrieve_candidates(body.message, filters, supabase, openai)
            if not candidates:
                yield _sse("info", {"message": NO_RESULTS_MESSAGE})
                yield _sse("done", {})
                return

            top = rerank_candidates(candidates)[:TOP_N]
            yield _sse(
                "candidates",
                {"candidates": [_serialize_candidate(c) for c in top]},
            )

            for token in generate_recommendation_stream(
                body.message, top, openai, history
            ):
                yield _sse("token", {"text": token})

            yield _sse("done", {})

        except Exception:
            logger.exception("recommend pipeline failed")
            yield _sse(
                "error",
                {
                    "message": (
                        "Something went wrong on our end. Please try again."
                    )
                },
            )
            yield _sse("done", {})

    return EventSourceResponse(event_stream())
```

- [ ] **Step 5: Run the tests to confirm they pass**

Run: `uv run pytest tests/unit/api/test_routes.py -v`
Expected: 5 passed (the original `/dramas` test plus 4 new `/recommend` tests).

- [ ] **Step 6: Run the full unit suite for regressions**

Run: `uv run pytest tests/unit -v`
Expected: all green.

- [ ] **Step 7: Commit**

```bash
git add src/api/routes.py tests/unit/api/test_routes.py
git commit -m "feat: add streaming /recommend SSE endpoint"
```

---

## Task 6: Refactor `app.py` to consume the API over HTTP+SSE

**Files:**
- Modify: `app.py` — drop direct `run_rag` / Supabase imports, talk to API instead

`app.py` keeps the same UI shape (sidebar filters, chat box, candidate preview). Only the *source* of data and recommendations changes: `httpx.stream` against `/recommend` instead of an in-process call, and `httpx.get` against `/dramas` instead of a direct Supabase query.

- [ ] **Step 1: Replace the imports and helpers section**

In `app.py`, replace **lines 1–47** (everything from the module docstring through the existing `if "messages" not in st.session_state` block) with:

```python
"""Streamlit chat UI for the cdrama recommender.

Pure frontend — talks to the FastAPI backend over HTTP+SSE. The API URL
is read from the API_URL env var, defaulting to localhost.

Run with:
    uv run uvicorn src.api.main:app --reload    # in one terminal
    uv run streamlit run app.py                 # in another
"""

from __future__ import annotations

import json
import os

import httpx
import streamlit as st


API_URL = os.environ.get("API_URL", "http://127.0.0.1:8000")

st.set_page_config(
    page_title="C-Drama Recommender",
    page_icon="🎬",
    layout="wide",
)


@st.cache_data
def load_drama_titles() -> list[str]:
    """Fetch the catalogue's drama titles from the API for the sidebar dropdown."""
    response = httpx.get(f"{API_URL}/dramas", timeout=10.0)
    response.raise_for_status()
    return response.json()["titles"]


def _iter_sse(response: httpx.Response):
    """Parse an SSE stream into (event_name, data_dict) tuples.

    Frames are separated by blank lines. Only `event:` and `data:` lines
    are tracked; comments (starting with `:`) and other field types are
    ignored. `data:` payloads are decoded as JSON.
    """
    event = None
    data = None
    for raw in response.iter_lines():
        line = raw.rstrip("\r")
        if not line:
            if event is not None:
                yield event, json.loads(data) if data else {}
            event = None
            data = None
            continue
        if line.startswith(":"):
            continue
        if line.startswith("event:"):
            event = line.split(":", 1)[1].strip()
        elif line.startswith("data:"):
            data = line.split(":", 1)[1].strip()
    if event is not None:
        yield event, json.loads(data) if data else {}


drama_titles = load_drama_titles()

if "messages" not in st.session_state:
    st.session_state.messages = []
if "candidates" not in st.session_state:
    st.session_state.candidates = {}
if "prompt_input" not in st.session_state:
    st.session_state.prompt_input = ""
```

- [ ] **Step 2: Replace the search-button handler**

In `app.py`, find the block starting `if search:` (around the original line 266) and replace its body — from `raw_prompt = prompt.strip()` through `st.rerun()` — with:

```python
        raw_prompt = prompt.strip()
        filter_hint = build_filter_hint(
            min_score,
            min_year,
            include_genres,
            exclude_genres,
            exclude_titles,
        )
        if not raw_prompt and not filter_hint:
            st.warning("Enter a query or select at least one filter.")
        else:
            user_prompt = raw_prompt or "Recommend dramas matching my selected filters."
            request_text = (
                f"{user_prompt}\n\n{filter_hint}" if filter_hint else user_prompt
            )

            st.session_state.messages.append({"role": "user", "content": user_prompt})
            with st.chat_message("user"):
                st.markdown(user_prompt)

            history = [
                {"role": m["role"], "content": m["content"]}
                for m in st.session_state.messages[:-1][-HISTORY_WINDOW:]
            ]

            with st.chat_message("assistant"):
                placeholder = st.empty()
                response_text = ""
                top: list[dict] = []
                try:
                    with httpx.stream(
                        "POST",
                        f"{API_URL}/recommend",
                        json={"message": request_text, "history": history},
                        timeout=httpx.Timeout(30.0, read=None),
                    ) as r:
                        r.raise_for_status()
                        for event, data in _iter_sse(r):
                            if event == "candidates":
                                top = data["candidates"]
                            elif event == "token":
                                response_text += data["text"]
                                placeholder.markdown(response_text)
                            elif event == "info":
                                response_text = data["message"]
                                placeholder.markdown(response_text)
                            elif event == "error":
                                response_text = data["message"]
                                placeholder.warning(response_text)
                            elif event == "done":
                                break
                except Exception as e:
                    response_text = f"Something went wrong: `{e}`"
                    placeholder.markdown(response_text)

            assistant_idx = len(st.session_state.messages)
            st.session_state.messages.append(
                {"role": "assistant", "content": response_text}
            )
            if top:
                st.session_state.candidates[assistant_idx] = top
            st.rerun()
```

- [ ] **Step 3: Verify the file still parses**

Run: `uv run python -c "import ast; ast.parse(open('app.py').read())"`
Expected: no output (clean parse).

- [ ] **Step 4: Manually verify end to end**

Terminal 1: `uv run uvicorn src.api.main:app --reload`
Wait until uvicorn prints `Application startup complete`.

Terminal 2: `uv run streamlit run app.py`
Open the browser tab Streamlit prints (usually `http://localhost:8501`).

In the UI:
1. Click an example button (e.g. "Like Nirvana in Fire, but shorter") and Search. Expected: tokens stream in, candidate preview populates on the right.
2. In the sidebar, set "Minimum rating" to `9.5` and submit. Expected: either a small set of candidates or the no-results info message.
3. Type something off-topic ("what's the weather?") and submit. Expected: the refusal info message renders cleanly with no candidates.

If any of those don't work, debug before committing.

- [ ] **Step 5: Commit**

```bash
git add app.py
git commit -m "refactor: stream recommendations from FastAPI backend"
```

---

## Task 7: Smoke test + README update

**Files:**
- Create: `tests/smoke/test_api_smoke.py`
- Modify: `README.md`

- [ ] **Step 1: Write the smoke test**

Create `tests/smoke/test_api_smoke.py`:

```python
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
```

- [ ] **Step 2: Run the smoke test**

Run: `uv run pytest tests/smoke/test_api_smoke.py -v`
Expected: 2 passed (this hits real services and may take ~10s).

- [ ] **Step 3: Confirm the default invocation still skips it**

Run: `uv run pytest -m "not db and not parser" -v`
Expected: smoke tests deselected; unit tests pass.

- [ ] **Step 4: Update the README**

In `README.md`, find the "Running it locally" section and replace its body (the part after the secrets snippet, starting with `Then:`) with:

````markdown
Then start the API and the UI in two separate terminals:

```bash
# Terminal 1 — the FastAPI backend
uv run uvicorn src.api.main:app --reload

# Terminal 2 — the Streamlit frontend
uv run streamlit run app.py
```

The chat UI opens in your browser and talks to the API at
`http://127.0.0.1:8000`. Override that with the `API_URL` env var if you
run the backend on a different host or port.

There's also a quick CLI run of the pipeline (no API involved):

```bash
uv run src/recommender/pipeline.py
```
````

- [ ] **Step 5: Commit**

```bash
git add tests/smoke/test_api_smoke.py README.md
git commit -m "test: add API smoke test and update README for two-terminal flow"
```

---

## Final verification

- [ ] **Run the full test suite (unit only):**

Run: `uv run pytest -m "not db and not parser" -v`
Expected: all unit tests pass; smoke tests skipped.

- [ ] **Run the smoke suite:**

Run: `uv run pytest tests/smoke -v`
Expected: all smoke tests pass.

- [ ] **End-to-end manual check:** start the API and Streamlit, run one
  reference, one semantic, and one filter-only query. Confirm
  candidates appear, text streams in, and the sidebar dropdown
  populates.
