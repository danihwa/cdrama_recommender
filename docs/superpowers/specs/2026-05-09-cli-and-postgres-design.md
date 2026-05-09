# CLI + Local Postgres Refactor — Design

**Date:** 2026-05-09
**Status:** Approved (pending user review of this doc)

## Goal

Strip the project down from a Streamlit + FastAPI + Supabase RAG chatbot
to a pure CLI tool that prints a fixed-shape result table, backed by a
local Postgres in Docker. Keep the parser LLM and all three search modes;
drop the LLM-written recommendation prose.

## Motivation

- The Streamlit UI is unfinished and not the part of the project the
  author wants to keep iterating on.
- Supabase free-tier projects get paused after a stretch of inactivity,
  which has been a friction point. Local Postgres in Docker removes that
  dependency.
- For a portfolio piece, the value is the retrieval / rerank pipeline
  and the data work behind it — not the chat surface.

## Architecture

### Before

```
Streamlit (app.py)  →  FastAPI (src/api/)  →  pipeline.run_rag  →  Supabase
                                              ├── parse (LLM)
                                              ├── retrieve (3 modes)
                                              ├── rerank
                                              └── generate (LLM)
```

### After

```
CLI REPL (pipeline.py __main__)  →  pipeline.run_rag  →  Local Postgres (Docker)
                                    ├── parse (LLM)
                                    ├── retrieve (3 modes)
                                    └── rerank
```

Stages: **Parse → Route → Retrieve → Rerank**. The Generate stage and
everything that fed it (`RECOMMEND_SYSTEM_PROMPT`,
`generate_recommendation`, `generate_recommendation_stream`,
`_format_drama`, `build_context`, `MAX_RESPONSE_TOKENS`, the `Iterator`
import) are removed.

## CLI behaviour

Single entry point: `uv run src/recommender/pipeline.py`.

Interactive REPL: prompt for a query, run the pipeline, print the parsed
filters + result table, prompt again. Conversation history is held in a
list inside the loop and threaded into `parse_user_query` so follow-ups
like *"something older"* still work. Type `exit` (or send EOF / Ctrl-C)
to quit.

### Sample session

```
> recommend something like Nirvana in Fire, rated above 8

Parsed:
  mode             = reference
  reference_title  = Nirvana in Fire
  min_score        = 8.0

Searching...

============================================================
Top 7 candidates:
============================================================
 1. [2023] Nirvana in Fire                  — score 9.2, similarity 0.847
 2. [2024] The Untamed                      — score 8.8, similarity 0.821
 3. [2022] Joy of Life                      — score 8.6, similarity 0.802
 4. [2023] Story of Yanxi Palace            — score 8.5, similarity 0.798
 5. [2024] Word of Honor                    — score 8.7, similarity 0.781
 6. [2022] Love Like the Galaxy             — score 8.4, similarity 0.764
 7. [2023] Story of Kunning Palace          — score 8.3, similarity 0.751

> something older

Parsed:
  mode             = reference
  reference_title  = Nirvana in Fire
  min_score        = 8.0
  ...

> exit
```

### Output rules

- **Parsed block** always prints `mode` (renamed from `search_mode` for
  the display). The remaining `QueryFilters` fields are printed only
  when non-empty and non-None — empty lists and `None` values are
  hidden so the block stays compact.
- **Results table** is fixed-width: index (right-padded), `[year]`,
  title (left-padded to a fixed width), `score`, `similarity` to three
  decimals.
- **SQL-mode rows** print `similarity   —` (em dash) instead of
  `0.000`, so the user can see similarity wasn't computed rather than
  every drama being equally bad.
- **Refusals** (search_mode=`refused`) print the existing
  `REFUSED_MESSAGE` and skip the table. Same for empty-result paths
  (`NO_RESULTS_MESSAGE`).

### Constants

- `MATCH_COUNT`: 10 → **14** (fetch 14, rerank, show top 7)
- `TOP_N`:        5 → **7**
- `MAX_RESPONSE_TOKENS`: removed (no generator)

## Local Postgres in Docker

### Container

- Image: `pgvector/pgvector:pg16` (official Postgres image with the
  `vector` extension preinstalled).
- Compose file at project root with one service (`db`), one named
  volume (`pgdata`) for persistence across reboots, and a port mapping
  `5432:5432`.
- `src/database/schema.sql` and `src/database/functions.sql` are
  bind-mounted read-only into `/docker-entrypoint-initdb.d/` so the
  container creates the `cdramas` table and the `match_documents`
  function on first start.

### Env

`~/secrets/.env` switches from two Supabase variables to one:

```
DATABASE_URL=postgresql://postgres:postgres@localhost:5432/cdramas
```

### Workflow

```
docker compose up -d                     # one-time: starts Postgres + initdb
uv run src/database/loader.py            # one-time: loads parquet → DB
uv run src/recommender/pipeline.py       # CLI REPL
```

`docker compose down` stops the container; `docker compose down -v`
wipes the named volume for a clean re-load.

## Code changes

### Python client

Replace the `supabase` Python client with `psycopg[binary]>=3.2` and
`pgvector>=0.3`. Why these:
- **psycopg3** is the modern, well-maintained driver. We have ~3
  distinct queries; SQLAlchemy is overkill.
- **pgvector** Python package adds `register_vector(conn)` so the
  `embedding` column ser/de as plain Python `list[float]` — same shape
  the rest of the code already produces.

### File-by-file

**`src/database/connection.py`** — `get_db_connection()` returns a
`psycopg.Connection`, reads `DATABASE_URL`, calls
`register_vector(conn)` once at construction.

**`src/recommender/_shared.py`** — same function signatures and behaviour,
internals rewritten:
- `lookup_drama_by_title`: `supabase.table(...).ilike(...)` →
  `SELECT {columns} FROM cdramas WHERE title ILIKE %s LIMIT 1`.
  Two-attempt punctuation fallback preserved.
- `vector_search`: `supabase.rpc("match_documents", {...})` →
  `SELECT * FROM match_documents(%(query_embedding)s, %(match_threshold)s, ...)`.
  Same parameter set; the SQL function itself is unchanged.
- `find_exclude_ids` and `normalize_genres` need no logic changes,
  only the `Client` → `Connection` type hint where they call helpers.

**`src/recommender/search_reference.py`, `search_semantic.py`, `search_sql.py`**
— change `supabase: Client` parameters to `conn: psycopg.Connection`
throughout. Bodies stay the same; they delegate to `_shared.py`.

**`src/recommender/pipeline.py`** — major edits:
- Remove `RECOMMEND_SYSTEM_PROMPT`, `MAX_RESPONSE_TOKENS`,
  `_format_drama`, `build_context`, `generate_recommendation`,
  `generate_recommendation_stream`, the `Iterator` import.
- Update `MATCH_COUNT`, `TOP_N`.
- `run_rag` returns just the formatted result string (no more
  `(text, top_dramas)` tuple). Callers updated.
- New `format_parsed_filters(filters: QueryFilters) -> str` — produces
  the "Parsed:" block, hiding empty/None fields.
- New `format_results(reranked: list[dict]) -> str` — produces the
  fixed-width numbered table. Detects SQL mode (similarity all zero)
  and prints `—` for similarity.
- Replace the hard-coded `__main__` smoke query with a REPL loop that
  reads stdin in a `while True`, threads history, exits on `exit` /
  EOF / `KeyboardInterrupt`.

**`src/database/loader.py`** — `supabase.table(...).upsert(...)` →
psycopg `INSERT INTO cdramas (...) VALUES (...) ON CONFLICT (mdl_id) DO UPDATE SET ...`
via `executemany`. Batch size unchanged.

**`tests/integration/conftest.py`, `tests/integration/test_exclude_genres.py`,
`scripts/collect_candidates.py`** — swap Supabase imports/fixtures for
the new psycopg connection.

**`pyproject.toml`** — remove `fastapi`, `streamlit`, `sse-starlette`,
`uvicorn`, `httpx`, `supabase`; add `psycopg[binary]>=3.2`,
`pgvector>=0.3`. Update the `db` marker docstring from "Supabase" to
"local Postgres".

**`README.md`** — rewrite the "Running it locally" section: drop the
two-terminal Streamlit + FastAPI flow, add `docker compose up -d`,
change env vars from `SUPABASE_*` to `DATABASE_URL`, remove the
Streamlit-related description from the top of the file. Mention that
results are printed as a CLI table, not LLM-written paragraphs.

### New files

- `docker-compose.yml` at project root.
- `tests/smoke/test_postgres_connection.py` — minimal "can connect and
  SELECT 1, and the `vector` extension is present" smoke test, replaces
  the deleted Supabase variant.

### Deletions

- `app.py`
- `src/api/` (entire package)
- `tests/smoke/test_api_smoke.py`
- `tests/smoke/test_supabase_connection.py`
- `tests/unit/api/` (entire directory)
- `tests/unit/test_build_context.py`
- `tests/unit/test_generate_stream.py`
- `tests/evals/llm_judge.py` — depended on the generator's text output
- `notebooks/04b_generator_lab.ipynb`

### Kept untouched

- `tests/evals/test_weight_calibration.py` — exercises only the
  reranker, no generator dependency.
- `tests/evals/conftest.py`, `tests/evals/test_parse_user_query.py`,
  `tests/fixtures/parser_cases.py` — parser evals are unaffected.
- `tests/unit/test_rerank.py` — reranker unit tests stand.
- `notebooks/01_explore.ipynb`, `02_cleaning.ipynb`,
  `03_embeddings.ipynb`, `04a_parser_lab.ipynb` — no generator code.
- `src/scraper/` — entirely untouched.

## Out of scope

- Re-running the scrape or re-embedding. The existing
  `data/cleaned/dramas_with_vectors.parquet` is the source of truth for
  the new local DB.
- Any UI replacement (web, TUI, etc.). The CLI is the final surface.
- Backwards compatibility with the Supabase deployment. After this
  refactor the project no longer talks to Supabase at all.

## Risks and trade-offs

- **Anyone running this needs Docker Desktop.** Acceptable for a
  portfolio project; mentioned explicitly in the README.
- **Loader on a fresh DB takes a few minutes** ingesting ~thousands of
  1536-dim vectors. One-time cost, acceptable.
- **Losing the LLM-written recommendation reduces "wow factor"** for a
  reader skimming the demo. Mitigated by leaving the
  `04b_generator_lab.ipynb` removal in commit history so anyone curious
  can find it; the README still describes the parser LLM's role so the
  RAG-ish nature of the project is not hidden.
