# FastAPI Backend for cdrama_recommender — Design

**Date:** 2026-05-06
**Status:** Spec — awaiting implementation plan

## Goal

Move the recommender pipeline behind an HTTP API so the Streamlit app
becomes a pure frontend. The API owns all access to Supabase and OpenAI;
Streamlit only knows the API URL.

This is portfolio-shaped work: the value is demonstrating clean
frontend/backend separation and a streaming SSE endpoint, not building
a public service.

## Non-goals

These are explicitly out of scope for this spec:

- Auth, rate limiting, or any deployment configuration. Local-only.
- A decomposed pipeline API (`/parse`, `/retrieve`, `/generate` as
  separate endpoints). One `/recommend` endpoint covers the only client.
- A structured `filters` field on the request. Sidebar filters keep
  flowing through the existing `build_filter_hint` text-glue path so the
  parser LLM remains the single source of filter truth.
- Streamlit-side tests. The repo doesn't have any today and the
  streaming consumer is small enough to verify by running it.

## Architecture

```
┌─────────────┐  HTTP+SSE   ┌──────────────┐
│  Streamlit  │ ──────────► │   FastAPI    │
│  (app.py)   │ ◄────────── │  (src/api/)  │
└─────────────┘             └──────┬───────┘
                                   │
                       ┌───────────┼─────────────┐
                       ▼           ▼             ▼
                   Supabase     OpenAI    src/recommender/*
```

After this change, `app.py` no longer imports from `src.recommender` or
`src.database` and no longer reads `SUPABASE_URL` / `SUPABASE_KEY`.

## Components

A new package mirroring the existing `src/recommender/`,
`src/scraper/`, `src/database/` layout:

```
src/api/
  __init__.py
  main.py       # FastAPI() instance + lifespan that builds Supabase/OpenAI clients once
  routes.py     # POST /recommend (SSE), GET /dramas
  schemas.py    # RecommendRequest, ChatMessage Pydantic models
```

### Refactor inside `src/recommender/pipeline.py`

Add a `generate_recommendation_stream` function that yields tokens,
sitting next to the existing batch `generate_recommendation`. The batch
function stays untouched so the CLI `__main__` and any tests that use
`run_rag` keep working. The streaming variant uses
`openai.chat.completions.create(..., stream=True)` and yields the text
delta from each chunk.

`run_rag` itself is not modified. The API route orchestrates the stages
directly (`parse_user_query → retrieve_candidates → rerank_candidates →
generate_recommendation_stream`) because it needs to emit the
`candidates` SSE event between rerank and generation.

### `app.py`

- Remove `from src.recommender.pipeline import run_rag`.
- Remove `from src.database.connection import get_db_connection`.
- Remove the `OpenAI` import.
- Replace `load_drama_titles` with a call to `GET /dramas`.
- Replace the inline `run_rag(...)` call with a streaming `httpx.stream`
  call against `POST /recommend`, consuming SSE events and updating an
  `st.empty()` placeholder as `token` events arrive.
- `API_URL` env var, defaulting to `http://127.0.0.1:8000`.

### Run command

Two terminals:

```
uv run uvicorn src.api.main:app --reload      # terminal 1
uv run streamlit run app.py                    # terminal 2
```

### README

The "Running it locally" section is updated to show the two-terminal
flow. The `~/secrets/.env` example stays the same (the API process
still reads it via `load_secrets`); only the run commands change.

## API contract

### `POST /recommend`

Request body:

```json
{
  "message": "Like Nirvana in Fire but no older than 2025",
  "history": [
    {"role": "user", "content": "..."},
    {"role": "assistant", "content": "..."}
  ]
}
```

`history` is the last 6 messages — Streamlit owns this window, same as
today. Filters from the sidebar are still glued onto `message` by the
existing `build_filter_hint` in `app.py`; the API does not see filters
as a structured concept.

Response: `text/event-stream` via `sse-starlette`'s
`EventSourceResponse`. Four event types in total.

**Happy path:**

```
event: candidates
data: {"candidates": [{title, year, mdl_score, mdl_url, similarity, ensemble_score, watchers, genres, synopsis}, ...]}

event: token
data: {"text": "Here"}

event: token
data: {"text": "'s a great"}

... many tokens ...

event: done
data: {}
```

The `candidates` payload mirrors exactly what Streamlit's
`render_candidate` already reads from `top` — same field names, same
shapes. The top N (currently 5) is preserved.

**Refusal or empty-results path** — used when the parser returns
`search_mode: "refused"` OR retrieval returns `[]`:

```
event: info
data: {"message": "<REFUSED_MESSAGE or NO_RESULTS_MESSAGE>"}
event: done
data: {}
```

The existing `REFUSED_MESSAGE` and `NO_RESULTS_MESSAGE` constants in
`pipeline.py` are reused. Streamlit treats `info` as the assistant
message text and skips the candidates expander.

**Mid-stream error path** — Supabase/OpenAI failures, parser returning
no structured output, etc.:

```
event: error
data: {"message": "Something went wrong on our end. Please try again."}
event: done
data: {}
```

HTTP status stays `200` because headers are already flushed; same
pattern OpenAI's own SDKs use. The actual exception is logged
server-side, not exposed to the client (no stack traces in the response).

### `GET /dramas`

Response:

```json
{"titles": ["...", "...", "..."]}
```

Alphabetised. Cached server-side at app startup since the catalogue
doesn't change at request time.

### Pre-stream errors

Request validation errors (malformed JSON, missing fields, bad history
shapes) return standard FastAPI `422` / `400` responses. Streamlit's
existing `try/except` around the call surfaces these as
"Something went wrong: ...".

## Streamlit-side SSE consumer

A small inline helper in `app.py` (~20 lines) parses SSE frames from
`httpx.iter_lines()`. No `sseclient-py` dep — the parser is small enough
to keep the dep list shorter:

```python
with httpx.stream("POST", f"{API_URL}/recommend", json=...) as r:
    placeholder = st.empty()
    text = ""
    for sse in iter_sse(r):
        if sse.event == "candidates":
            top = sse.data["candidates"]
        elif sse.event == "token":
            text += sse.data["text"]
            placeholder.markdown(text)
        elif sse.event == "info":
            text = sse.data["message"]
            placeholder.markdown(text)
        elif sse.event == "error":
            text = sse.data["message"]
            placeholder.warning(text)
        elif sse.event == "done":
            break
```

## Testing

Three layers, mirroring the existing `tests/unit/`, `tests/integration/`,
`tests/smoke/` split:

### `tests/unit/api/test_routes.py`

Uses `fastapi.testclient.TestClient` with monkeypatched
`parse_user_query`, `retrieve_candidates`, and
`generate_recommendation_stream`. Assertions:

- Happy path emits `candidates`, then ≥1 `token`, then `done`, in that
  order.
- Refusal path (parser returns `search_mode: "refused"`) emits `info` +
  `done`, no `candidates` event.
- Empty-results path (retrieval returns `[]`) emits `info` + `done`,
  no `candidates` event.
- Mid-stream exception emits `error` + `done`, status code stays 200.
- `GET /dramas` returns alphabetised titles from a mocked Supabase
  response.

### `tests/unit/api/test_schemas.py`

Pydantic validation for `RecommendRequest`:

- Rejects history items missing `role` or `content`.
- Rejects unknown roles.
- Accepts empty history.

### `tests/smoke/test_api_smoke.py`

One end-to-end test that hits the real services through `TestClient`.
Marked `@pytest.mark.db` and `@pytest.mark.parser` so the existing
`pytest -m "not db and not parser"` invocation continues to skip it.

## Dependencies

To add to `pyproject.toml` (exact pins decided at implementation time
after a context7 check):

- `fastapi`
- `uvicorn[standard]`
- `sse-starlette` — handles SSE framing and client-disconnect detection.
- `httpx` — explicit dep for Streamlit (currently transitive via
  `supabase`).

## Open questions

None. All scope decisions are locked in the Non-goals section above.
