# CLI + Local Postgres Refactor Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Replace the Streamlit + FastAPI + Supabase chatbot with a pure CLI REPL that prints a result table, backed by local Postgres in Docker. Keep the parser LLM and all three search modes; drop the generator LLM.

**Architecture:** Pipeline stages collapse to **Parse → Route → Retrieve → Rerank**. The DB client swaps from `supabase` to `psycopg[binary]` + `pgvector`. The new entry point is a REPL inside `src/recommender/pipeline.py`'s `__main__` block. Schema and `match_documents()` SQL are unchanged — we just stop calling them through Supabase RPC.

**Tech Stack:** Python 3.12+, uv, psycopg3 (`psycopg[binary]>=3.2`), pgvector Python package (`pgvector>=0.3`), Postgres 16 + pgvector via the `pgvector/pgvector:pg16` Docker image, OpenAI Python SDK (already used by parser), pytest.

**Spec:** [docs/superpowers/specs/2026-05-09-cli-and-postgres-design.md](../specs/2026-05-09-cli-and-postgres-design.md)

---

## Task 1: Stand up local Postgres in Docker

**Files:**
- Create: `docker-compose.yml`
- Modify: `~/secrets/.env` (user's local file — instructions only, not committed)

- [ ] **Step 1: Create the compose file**

Create `docker-compose.yml` at the project root:

```yaml
services:
  db:
    image: pgvector/pgvector:pg16
    container_name: cdrama_postgres
    restart: unless-stopped
    environment:
      POSTGRES_USER: postgres
      POSTGRES_PASSWORD: postgres
      POSTGRES_DB: cdramas
    ports:
      - "5432:5432"
    volumes:
      - pgdata:/var/lib/postgresql/data
      - ./src/database/schema.sql:/docker-entrypoint-initdb.d/01_schema.sql:ro
      - ./src/database/functions.sql:/docker-entrypoint-initdb.d/02_functions.sql:ro
    healthcheck:
      test: ["CMD-SHELL", "pg_isready -U postgres -d cdramas"]
      interval: 5s
      timeout: 3s
      retries: 10

volumes:
  pgdata:
```

The bind-mounted SQL files in `/docker-entrypoint-initdb.d/` only run on the **first** start of an empty volume — Postgres' initdb mechanism. Numbering them (`01_`, `02_`) guarantees schema runs before functions.

- [ ] **Step 2: Update `~/secrets/.env` (user-local, do not commit)**

Replace the two Supabase variables with one:

```
DATABASE_URL=postgresql://postgres:postgres@localhost:5432/cdramas
```

`OPENAI_API_KEY=...` stays. `SUPABASE_URL` / `SUPABASE_SECRET_KEY` can be removed.

- [ ] **Step 3: Bring up the container**

Run:
```
docker compose up -d
```
Expected: `Container cdrama_postgres  Started`. Wait for healthcheck to pass:
```
docker compose ps
```
Expected: `STATUS` column shows `Up ... (healthy)`.

- [ ] **Step 4: Verify schema and function loaded**

Run:
```
docker exec -it cdrama_postgres psql -U postgres -d cdramas -c "\dt"
docker exec -it cdrama_postgres psql -U postgres -d cdramas -c "\df match_documents"
docker exec -it cdrama_postgres psql -U postgres -d cdramas -c "SELECT extname FROM pg_extension WHERE extname='vector';"
```
Expected: `cdramas` table listed, `match_documents` function listed, `vector` extension present.

- [ ] **Step 5: Commit**

```
git add docker-compose.yml
git commit -m "feat: add docker compose for local postgres + pgvector"
```

---

## Task 2: Swap dependencies and rewrite connection layer

**Files:**
- Modify: `pyproject.toml`
- Modify: `src/database/connection.py`
- Create: `tests/smoke/test_postgres_connection.py`
- Delete: `tests/smoke/test_supabase_connection.py`

- [ ] **Step 1: Edit `pyproject.toml` dependencies**

Replace the `dependencies` list in `pyproject.toml`. Current list:

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

Replace with:

```toml
dependencies = [
    "bs4>=0.0.2",
    "dotenv>=0.9.9",
    "matplotlib>=3.10.8",
    "openai>=2.31.0",
    "pandas>=3.0.2",
    "pgvector>=0.3",
    "psycopg[binary]>=3.2",
    "pyarrow>=23.0.1",
    "pytest>=9.0.3",
    "requests>=2.33.1",
]
```

Also update the `db` marker docstring in the same file:

```toml
[tool.pytest.ini_options]
markers = [
    "db: tests that hit local Postgres (deselect with -m 'not db')",
    "parser: tests that call the OpenAI API via parse_user_query (deselect with -m 'not parser')",
]
```

- [ ] **Step 2: Sync the new dependency set**

Run:
```
uv sync
```
Expected: removes `fastapi`, `streamlit`, `sse-starlette`, `uvicorn`, `httpx`, `supabase`; adds `psycopg`, `pgvector`. No errors.

- [ ] **Step 3: Rewrite `src/database/connection.py`**

Replace the entire file:

```python
"""Connection helper for local Postgres."""

from __future__ import annotations

import os

import psycopg
from pgvector.psycopg import register_vector

from src.env import load_secrets

load_secrets()


def get_db_connection() -> psycopg.Connection:
    """Return a psycopg connection with pgvector type adapters registered.

    Reads ``DATABASE_URL`` from the environment (loaded via dotenv).
    ``register_vector`` teaches psycopg to ser/de the ``embedding``
    column as a plain Python ``list[float]``, so callers don't have to
    parse the textual ``vector`` representation by hand.

    Raises:
        ValueError: If ``DATABASE_URL`` is missing.
    """
    url = os.environ.get("DATABASE_URL")
    if not url:
        raise ValueError("DATABASE_URL is missing from .env file!")

    try:
        conn = psycopg.connect(url, autocommit=True)
    except Exception as e:
        raise ConnectionError(f"Failed to connect to Postgres: {e}") from e

    register_vector(conn)
    return conn
```

`autocommit=True` matches Supabase's behaviour (every `.execute()` was its own transaction) and keeps callers from needing to commit explicitly.

- [ ] **Step 4: Write the failing smoke test**

Create `tests/smoke/test_postgres_connection.py`:

```python
"""Smoke test: can we connect to local Postgres and is pgvector loaded?"""

from __future__ import annotations

import pytest

from src.database.connection import get_db_connection


@pytest.mark.db
def test_postgres_connection_and_pgvector() -> None:
    conn = get_db_connection()
    try:
        with conn.cursor() as cur:
            cur.execute("SELECT 1")
            assert cur.fetchone() == (1,)

            cur.execute("SELECT extname FROM pg_extension WHERE extname='vector'")
            row = cur.fetchone()
            assert row is not None, "pgvector extension is not installed"
            assert row[0] == "vector"
    finally:
        conn.close()
```

- [ ] **Step 5: Delete the obsolete Supabase smoke test**

Run:
```
git rm tests/smoke/test_supabase_connection.py
```

- [ ] **Step 6: Run the smoke test**

Run:
```
uv run pytest tests/smoke/test_postgres_connection.py -v
```
Expected: PASS. (Postgres container from Task 1 must be running.)

- [ ] **Step 7: Commit**

```
git add pyproject.toml uv.lock src/database/connection.py tests/smoke/test_postgres_connection.py
git commit -m "feat: swap supabase client for psycopg + pgvector"
```

---

## Task 3: Rewrite `_shared.py` for psycopg

**Files:**
- Modify: `src/recommender/_shared.py`
- Modify: `tests/integration/conftest.py`
- Modify: `tests/integration/test_exclude_genres.py`

- [ ] **Step 1: Update the integration test fixture**

Replace `tests/integration/conftest.py` with:

```python
"""Shared fixtures for integration tests.

conftest.py is a special pytest file — any fixture defined here is
automatically available to every test in this directory (and below)
without needing an import.  pytest discovers it by name convention.
"""

import psycopg
import pytest

from src.database.connection import get_db_connection
from src.env import load_secrets


@pytest.fixture(scope="module")
def db_conn() -> psycopg.Connection:
    """psycopg connection for DB-hitting integration tests."""
    load_secrets()
    return get_db_connection()
```

- [ ] **Step 2: Update the integration test that uses the fixture**

Open `tests/integration/test_exclude_genres.py` and rename every `supabase` parameter to `db_conn` (function arguments only — the fixture name change drives everything else). Run:

```
uv run pytest tests/integration/test_exclude_genres.py --collect-only
```
Expected: tests collect without error (you'll see them listed; they may fail at runtime — that's fine, we fix the underlying calls in Step 3).

- [ ] **Step 3: Rewrite `src/recommender/_shared.py`**

Replace the entire file:

```python
"""Helpers shared by all three retrieval strategies.

Every search module (reference / semantic / sql) needs to normalize
genres (done either directly in the sql path or indirectly via vector_search),
resolve exclusion titles to DB ids, or call the vector-search SQL function — so
they live here to avoid duplication.
"""

from __future__ import annotations

import re

import psycopg
from psycopg.rows import dict_row

from src.recommender.models import QueryFilters

# Cosine-similarity floor for the match_documents function. Rows below this are
# dropped server-side, so the reranker never sees obvious non-matches.
MATCH_THRESHOLD = 0.3


def normalize_genres(genres: list[str]) -> list[str]:
    """
    Genres are stored lowercase in the DB.
    The parser LLM may return genres in any case, and with extra whitespace ->
    normalize to avoid false mismatches.
    """
    return [g.lower().strip() for g in genres]


def lookup_drama_by_title(
    title: str, conn: psycopg.Connection, columns: str = "id"
) -> dict | None:
    """Look up a single drama by partial title match.

    Uses ``ILIKE '%...%'`` for a forgiving substring match. If the first
    attempt misses, retries once with punctuation stripped from the input —
    catches LLM parser variations like "How Dare You?!" vs the stored
    "How Dare You!?".

    ``columns`` is interpolated directly into the SELECT list — callers
    pass static column names ("id", "id, embedding, title"), never user
    input, so SQL injection is not in scope here.
    """
    patterns = [title]
    stripped = re.sub(r"[^\w\s]", "", title).strip()
    if stripped and stripped != title:
        patterns.append(stripped)

    sql = f"SELECT {columns} FROM cdramas WHERE title ILIKE %s LIMIT 1"
    with conn.cursor(row_factory=dict_row) as cur:
        for pattern in patterns:
            cur.execute(sql, (f"%{pattern}%",))
            row = cur.fetchone()
            if row is not None:
                return row
    return None


def find_exclude_ids(titles: list[str], conn: psycopg.Connection) -> list[int]:
    """Match drama titles to DB ids so they can be excluded from results."""
    ids: list[int] = []
    for title in titles:
        row = lookup_drama_by_title(title, conn)
        if row is not None:
            ids.append(row["id"])
        else:
            print(f"   Warning: '{title}' not found in DB — cannot exclude it")
    return ids


def vector_search(
    query_vector: list[float],
    filters: QueryFilters,
    exclude_ids: list[int],
    conn: psycopg.Connection,
    match_count: int,
) -> list[dict]:
    """Run cosine-similarity search via the ``match_documents`` SQL function.

    ``match_documents`` is a Postgres function defined in
    ``src/database/functions.sql``. It performs the cosine-similarity
    search against the ``embedding`` column *and* applies the year /
    score / genre / exclusion filters in the same query, which is much
    more efficient than fetching a large number of results and filtering
    in Python.

    Unset filters are passed as ``None`` — the function's ``IS NULL``
    check short-circuits each ``WHERE`` clause, so the filter is
    effectively disabled.
    """
    filter_exclude_ids = exclude_ids or None
    filter_genres = normalize_genres(filters.genres) if filters.genres else None
    filter_exclude_genres = (
        normalize_genres(filters.exclude_genres) if filters.exclude_genres else None
    )

    sql = """
        SELECT *
        FROM match_documents(
            %(query_embedding)s,
            %(match_threshold)s,
            %(match_count)s,
            %(filter_year)s,
            %(filter_score)s,
            %(exclude_ids)s,
            %(filter_genres)s,
            %(exclude_genres)s
        )
    """
    params = {
        "query_embedding": query_vector,
        "match_threshold": MATCH_THRESHOLD,
        "match_count": match_count,
        "filter_year": filters.min_year,
        "filter_score": filters.min_score,
        "exclude_ids": filter_exclude_ids,
        "filter_genres": filter_genres,
        "exclude_genres": filter_exclude_genres,
    }
    with conn.cursor(row_factory=dict_row) as cur:
        cur.execute(sql, params)
        return cur.fetchall()
```

- [ ] **Step 4: Run the integration test**

Run:
```
uv run pytest tests/integration/test_exclude_genres.py -v
```
Expected: PASS. (Requires Postgres container running and data loaded — but data isn't loaded yet, so this test will likely fail "no rows". That's fine for now; we'll re-run after Task 4. Confirm only that the test collects and runs without import / type errors.)

If the test errors before reaching the assertion (e.g. `psycopg.OperationalError` or `AttributeError`), fix the underlying issue. If it reaches an assertion failure due to empty data, move on — Task 4 fixes it.

- [ ] **Step 5: Commit**

```
git add src/recommender/_shared.py tests/integration/conftest.py tests/integration/test_exclude_genres.py
git commit -m "refactor: rewrite _shared.py for psycopg"
```

---

## Task 4: Rewrite the loader and repopulate the DB

**Files:**
- Modify: `src/database/loader.py`

- [ ] **Step 1: Rewrite `src/database/loader.py`**

Replace the entire file:

```python
"""Load embedded dramas from a parquet file into local Postgres."""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from src.database.connection import get_db_connection


UPSERT_SQL = """
    INSERT INTO cdramas (
        mdl_id, mdl_url, title, native_title, synopsis,
        episodes, year, genres, tags, mdl_score, watchers, embedding
    ) VALUES (
        %(mdl_id)s, %(mdl_url)s, %(title)s, %(native_title)s, %(synopsis)s,
        %(episodes)s, %(year)s, %(genres)s, %(tags)s, %(mdl_score)s, %(watchers)s, %(embedding)s
    )
    ON CONFLICT (mdl_id) DO UPDATE SET
        mdl_url      = EXCLUDED.mdl_url,
        title        = EXCLUDED.title,
        native_title = EXCLUDED.native_title,
        synopsis     = EXCLUDED.synopsis,
        episodes     = EXCLUDED.episodes,
        year         = EXCLUDED.year,
        genres       = EXCLUDED.genres,
        tags         = EXCLUDED.tags,
        mdl_score    = EXCLUDED.mdl_score,
        watchers     = EXCLUDED.watchers,
        embedding    = EXCLUDED.embedding,
        updated_at   = CURRENT_TIMESTAMP
"""


def prepare_record(row: dict) -> dict:
    """Coerce numpy/pandas containers to the plain Python lists psycopg + pgvector expect."""
    return {
        "mdl_id": int(row["mdl_id"]),
        "mdl_url": row["mdl_url"],
        "title": row["title"],
        "native_title": row.get("native_title"),
        "synopsis": row.get("synopsis"),
        "episodes": int(row["episodes"]) if row.get("episodes") is not None else None,
        "year": int(row["year"]) if row.get("year") is not None else None,
        "genres": list(row["genres"]),
        "tags": list(row["tags"]),
        "mdl_score": float(row["mdl_score"]) if row.get("mdl_score") is not None else None,
        "watchers": int(row["watchers"]) if row.get("watchers") is not None else None,
        "embedding": list(row["embedding"]),
    }


def insert_dramas(parquet_path: str | Path, batch_size: int = 100) -> None:
    """
    Read dramas_with_vectors.parquet and upsert into cdramas in batches.

    Uses ON CONFLICT (mdl_id) so re-runs are safe — if a drama already
    exists, the row gets updated instead of throwing. Batching keeps
    memory bounded when uploading thousands of 3072-dim vectors.
    """
    conn = get_db_connection()

    print(f"Loading data from {parquet_path}...")
    df = pd.read_parquet(parquet_path)

    records = [prepare_record(row) for row in df.to_dict(orient="records")]
    total = len(records)
    print(f"Found {total} dramas. Starting upsert...")

    success, failed = 0, 0

    with conn.cursor() as cur:
        for i in range(0, total, batch_size):
            batch = records[i : i + batch_size]
            batch_num = i // batch_size + 1
            total_batches = (total + batch_size - 1) // batch_size

            try:
                cur.executemany(UPSERT_SQL, batch)
                success += len(batch)
                print(f"  Batch {batch_num}/{total_batches} OK ({success}/{total})")
            except Exception as e:
                failed += len(batch)
                print(f"  Batch {batch_num}/{total_batches} FAIL — {e}")

    conn.close()
    print(f"\nDone! {success} upserted, {failed} failed.")


if __name__ == "__main__":
    # to run: uv run src/database/loader.py
    DATA_FILE = Path("data/cleaned/dramas_with_vectors.parquet")
    insert_dramas(DATA_FILE, batch_size=100)
```

- [ ] **Step 2: Run the loader against local Postgres**

Run:
```
uv run src/database/loader.py
```
Expected: progress lines like `Batch 1/N OK (100/total)`, ending with `Done! N upserted, 0 failed.` (Several minutes for the full set of 3072-dim vectors.)

- [ ] **Step 3: Verify rows landed**

Run:
```
docker exec -it cdrama_postgres psql -U postgres -d cdramas -c "SELECT count(*) FROM cdramas;"
```
Expected: row count matches the parquet (compare against `df.shape[0]` from the loader's earlier print).

Run:
```
docker exec -it cdrama_postgres psql -U postgres -d cdramas -c "SELECT title, year, mdl_score FROM cdramas LIMIT 3;"
```
Expected: three real rows printed, with a non-null `mdl_score`.

- [ ] **Step 4: Re-run the integration test**

Run:
```
uv run pytest tests/integration/test_exclude_genres.py -v
```
Expected: PASS now that the table has data.

- [ ] **Step 5: Commit**

```
git add src/database/loader.py
git commit -m "refactor: rewrite loader for psycopg upsert"
```

---

## Task 5: Update remaining DB consumers

**Files:**
- Modify: `src/recommender/search_reference.py`
- Modify: `src/recommender/search_semantic.py`
- Modify: `src/recommender/search_sql.py`
- Modify: `scripts/collect_candidates.py`

- [ ] **Step 1: Update `search_reference.py`**

In `src/recommender/search_reference.py`, change the imports and replace every `supabase: Client` parameter with `conn: psycopg.Connection`. Specifically:

Replace:
```python
from supabase import Client
```
with:
```python
import psycopg
```

Then in the file, replace every occurrence of:
- `supabase: Client` → `conn: psycopg.Connection`
- inside function bodies, `supabase` → `conn` for any pass-through to `_shared.py` helpers

The smoke test in `__main__` (`get_db_connection()`) still works without changes because `get_db_connection()` now returns a `psycopg.Connection`.

- [ ] **Step 2: Update `search_semantic.py`**

Apply the same `Client` → `psycopg.Connection` rename throughout `src/recommender/search_semantic.py`. Same import swap (`from supabase import Client` → `import psycopg`).

- [ ] **Step 3: Update `search_sql.py`**

Apply the same rename throughout `src/recommender/search_sql.py`. If the file currently uses `supabase.table(...)` directly (rather than going through `_shared.py`), rewrite those calls as `cur.execute(...)` against the `cdramas` table — read the file first to see its current shape, then mirror the patterns established in `_shared.py`.

- [ ] **Step 4: Update `scripts/collect_candidates.py`**

Apply the same rename throughout `scripts/collect_candidates.py`: import `psycopg` instead of `supabase`, swap `Client` for `psycopg.Connection`.

- [ ] **Step 5: Run all DB-marked tests**

Run:
```
uv run pytest -m db -v
```
Expected: PASS for every test. If any fails with an import error or `AttributeError` referencing `supabase`, fix the missing rename.

- [ ] **Step 6: Run the search_reference smoke**

Run:
```
uv run src/recommender/search_reference.py
```
Expected: prints "Found reference drama: ..." then 5 results with `[year] title — score X, similarity Y`. (This script still uses `match_count=5`; we change it to 7 in Task 9.)

- [ ] **Step 7: Commit**

```
git add src/recommender/search_reference.py src/recommender/search_semantic.py src/recommender/search_sql.py scripts/collect_candidates.py
git commit -m "refactor: switch retrieval modules from supabase to psycopg"
```

---

## Task 6: Strip generator from `pipeline.py`

**Files:**
- Modify: `src/recommender/pipeline.py`
- Delete: `tests/unit/test_build_context.py`
- Delete: `tests/unit/test_generate_stream.py`
- Delete: `tests/evals/llm_judge.py`
- Delete: `notebooks/04b_generator_lab.ipynb`

- [ ] **Step 1: Remove generator constants and functions**

Open `src/recommender/pipeline.py` and delete:
- The `RECOMMEND_SYSTEM_PROMPT` constant
- The `MAX_RESPONSE_TOKENS` constant
- The `_format_drama` function
- The `build_context` function
- The `generate_recommendation` function
- The `generate_recommendation_stream` function
- `from typing import Iterator` (no longer used)

Also remove the generator call at the bottom of `run_rag` — for now, replace its body's last lines with a temporary `return reranked[:TOP_N]` so the file still parses. Tasks 8 and 9 finish the new shape.

- [ ] **Step 2: Run unit tests to confirm `_format_drama` / `build_context` callers are gone**

Run:
```
uv run pytest tests/unit/test_build_context.py tests/unit/test_generate_stream.py
```
Expected: tests fail to even import (`ImportError: cannot import name '_format_drama'`). That's the signal these test files are now obsolete and safe to delete.

- [ ] **Step 3: Delete the obsolete generator-related files**

Run:
```
git rm tests/unit/test_build_context.py
git rm tests/unit/test_generate_stream.py
git rm tests/evals/llm_judge.py
git rm notebooks/04b_generator_lab.ipynb
```

- [ ] **Step 4: Run the rest of the unit tests**

Run:
```
uv run pytest tests/unit -v
```
Expected: PASS. (`test_rerank.py` and any other unit test should still work — none of them touched the generator.)

- [ ] **Step 5: Commit**

```
git add src/recommender/pipeline.py
git commit -m "refactor: strip generator stage from pipeline"
```

---

## Task 7: Add `format_parsed_filters` (TDD)

**Files:**
- Create: `tests/unit/test_format_parsed_filters.py`
- Modify: `src/recommender/pipeline.py`

- [ ] **Step 1: Write the failing test**

Create `tests/unit/test_format_parsed_filters.py`:

```python
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
```

- [ ] **Step 2: Run the failing test**

Run:
```
uv run pytest tests/unit/test_format_parsed_filters.py -v
```
Expected: FAIL with `ImportError: cannot import name 'format_parsed_filters'`.

- [ ] **Step 3: Implement `format_parsed_filters`**

Add to `src/recommender/pipeline.py` (place it near the other formatting utilities — anywhere above `run_rag` is fine):

```python
def format_parsed_filters(filters: QueryFilters) -> str:
    """Render the 'Parsed:' block for the REPL.

    ``mode`` is always shown. Other fields are printed only when non-empty
    and non-None — empty lists, ``None``, and the default 'reference'
    mode-vs-explicit-reference distinction don't add information, so they
    stay hidden to keep the block compact.

    The two-space indent and 16-character left padding on the field name
    match the column alignment in the spec sample.
    """
    lines = ["Parsed:", f"  {'mode':<16} = {filters.search_mode}"]

    optional_fields: list[tuple[str, object]] = [
        ("reference_title", filters.reference_title),
        ("description", filters.description),
        ("genres", filters.genres),
        ("exclude_genres", filters.exclude_genres),
        ("exclude_titles", filters.exclude_titles),
        ("min_year", filters.min_year),
        ("min_score", filters.min_score),
    ]
    for name, value in optional_fields:
        if value is None:
            continue
        if isinstance(value, list) and not value:
            continue
        lines.append(f"  {name:<16} = {value}")
    return "\n".join(lines)
```

- [ ] **Step 4: Run the test to confirm it passes**

Run:
```
uv run pytest tests/unit/test_format_parsed_filters.py -v
```
Expected: PASS for all four tests.

- [ ] **Step 5: Commit**

```
git add tests/unit/test_format_parsed_filters.py src/recommender/pipeline.py
git commit -m "feat: add format_parsed_filters for CLI display"
```

---

## Task 8: Add `format_results` (TDD)

**Files:**
- Create: `tests/unit/test_format_results.py`
- Modify: `src/recommender/pipeline.py`

- [ ] **Step 1: Write the failing test**

Create `tests/unit/test_format_results.py`:

```python
"""Unit tests for format_results — the numbered candidate table."""

from __future__ import annotations

from src.recommender.pipeline import format_results


def _drama(
    title: str, year: int, mdl_score: float, similarity: float = 0.0
) -> dict:
    return {
        "title": title,
        "year": year,
        "mdl_score": mdl_score,
        "similarity": similarity,
    }


def test_header_and_seven_numbered_rows() -> None:
    rows = [
        _drama("Drama A", 2023, 9.2, similarity=0.847),
        _drama("Drama B", 2024, 8.8, similarity=0.821),
        _drama("Drama C", 2022, 8.6, similarity=0.802),
        _drama("Drama D", 2023, 8.5, similarity=0.798),
        _drama("Drama E", 2024, 8.7, similarity=0.781),
        _drama("Drama F", 2022, 8.4, similarity=0.764),
        _drama("Drama G", 2023, 8.3, similarity=0.751),
    ]
    out = format_results(rows)
    assert "Top 7 candidates:" in out
    assert "============================================================" in out
    for i in range(1, 8):
        assert f" {i}. " in out
    assert "[2023] Drama A" in out
    assert "score 9.2" in out
    assert "similarity 0.847" in out


def test_sql_mode_em_dash_for_zero_similarity() -> None:
    rows = [
        _drama("SQL One", 2022, 8.5, similarity=0.0),
        _drama("SQL Two", 2023, 8.2, similarity=0.0),
    ]
    out = format_results(rows)
    # Every row has similarity 0 -> SQL mode -> em dash, no 0.000.
    assert "0.000" not in out
    assert "similarity   —" in out


def test_reference_mode_keeps_decimal_when_any_similarity_is_nonzero() -> None:
    rows = [
        _drama("R One", 2022, 8.5, similarity=0.812),
        _drama("R Two", 2023, 8.2, similarity=0.0),  # legitimately exactly 0
    ]
    out = format_results(rows)
    # At least one nonzero -> not SQL mode -> all rows print numeric similarity.
    assert "0.812" in out
    assert "0.000" in out
    assert "—" not in out  # em dash only used for SQL mode


def test_empty_list_returns_empty_string_or_no_table() -> None:
    out = format_results([])
    assert "Top 7 candidates:" not in out
```

- [ ] **Step 2: Run the failing test**

Run:
```
uv run pytest tests/unit/test_format_results.py -v
```
Expected: FAIL with `ImportError: cannot import name 'format_results'`.

- [ ] **Step 3: Implement `format_results`**

Add to `src/recommender/pipeline.py` near `format_parsed_filters`:

```python
RESULTS_HEADER = "=" * 60
TITLE_WIDTH = 38  # left-pad title to this width so columns line up


def format_results(reranked: list[dict]) -> str:
    """Render the top reranked candidates as a numbered fixed-width table.

    SQL-mode candidates have similarity 0 across the board (no vector
    search ran). Detecting that — every row's similarity is exactly 0 —
    lets us print an em dash instead of a wall of ``0.000``s, which
    otherwise looks like every drama is equally bad.
    """
    if not reranked:
        return ""

    sql_mode = all((d.get("similarity") or 0.0) == 0.0 for d in reranked)

    lines = [
        RESULTS_HEADER,
        f"Top {len(reranked)} candidates:",
        RESULTS_HEADER,
    ]
    for i, d in enumerate(reranked, start=1):
        title = d["title"]
        year = d["year"]
        score = d["mdl_score"]
        similarity_field = (
            "similarity   —"
            if sql_mode
            else f"similarity {(d.get('similarity') or 0.0):.3f}"
        )
        lines.append(
            f"{i:>2}. [{year}] {title:<{TITLE_WIDTH}} — score {score}, {similarity_field}"
        )
    return "\n".join(lines)
```

- [ ] **Step 4: Run the test to confirm it passes**

Run:
```
uv run pytest tests/unit/test_format_results.py -v
```
Expected: PASS for all four tests.

- [ ] **Step 5: Commit**

```
git add tests/unit/test_format_results.py src/recommender/pipeline.py
git commit -m "feat: add format_results for CLI candidate table"
```

---

## Task 9: Update constants, simplify `run_rag`, add REPL loop

**Files:**
- Modify: `src/recommender/pipeline.py`

- [ ] **Step 1: Update constants**

In `src/recommender/pipeline.py`, change:
```python
MATCH_COUNT = 10  # candidates to fetch before reranking
TOP_N = 5  # candidates passed to the generator after reranking
```
to:
```python
MATCH_COUNT = 14  # candidates to fetch before reranking
TOP_N = 7         # candidates shown after reranking
```

- [ ] **Step 2: Simplify `run_rag` to return a single formatted string**

Replace the existing `run_rag` body. The new version returns a `str` (no more tuple), uses `format_parsed_filters` and `format_results`, and emits `NO_RESULTS_MESSAGE` / `REFUSED_MESSAGE` as plain strings:

```python
def run_rag(
    user_query: str,
    conn: psycopg.Connection,
    openai: OpenAI,
    history: list[dict] | None = None,
) -> str:
    """Full pipeline: Parse → Route → Retrieve → Rerank → Format.

    Returns a single human-readable string. Refusal and no-result cases
    return their canned messages; otherwise returns the parsed-filters
    block followed by the formatted results table.
    """
    history = history or []

    print("Parsing your request...")
    filters: QueryFilters = parse_user_query(user_query, openai, history)

    if filters.search_mode == "refused":
        return REFUSED_MESSAGE

    print(format_parsed_filters(filters))
    print("\nSearching...\n")

    candidates = retrieve_candidates(user_query, filters, conn, openai)
    if not candidates:
        return NO_RESULTS_MESSAGE

    reranked = rerank_candidates(candidates)
    return format_results(reranked[:TOP_N])
```

Also: at the top of the file, change `from supabase import Client` to `import psycopg` if it's still there, and update `retrieve_candidates`'s parameter type to `conn: psycopg.Connection`. The current signature uses `supabase: Client` and forwards it — rename in lockstep.

- [ ] **Step 3: Replace `__main__` with a REPL loop**

Replace the bottom `if __name__ == "__main__":` block with:

```python
if __name__ == "__main__":
    # to run: uv run src/recommender/pipeline.py
    load_secrets()
    openai_client = OpenAI()
    db_conn = get_db_connection()

    print("C-Drama recommender — type your query, or 'exit' to quit.\n")
    history: list[dict] = []
    try:
        while True:
            try:
                user_query = input("> ").strip()
            except EOFError:
                print()
                break
            if not user_query:
                continue
            if user_query.lower() in {"exit", "quit"}:
                break

            output = run_rag(user_query, db_conn, openai_client, history)
            print(output)
            print()

            history.append({"role": "user", "content": user_query})
            history.append({"role": "assistant", "content": output})
    except KeyboardInterrupt:
        print()
    finally:
        db_conn.close()
```

`load_secrets`, `OpenAI`, and `get_db_connection` should already be imported at the top of the file from the existing pipeline; if any are missing, add them to the imports.

- [ ] **Step 4: Smoke-run the REPL**

Run:
```
uv run src/recommender/pipeline.py
```

In the prompt, type:
```
Recommend something like Nirvana in Fire, rated above 8
```
Expected:
- A `Parsed:` block with `mode = reference`, `reference_title`, `min_score`.
- A `Top 7 candidates:` table with seven numbered rows.
- New `>` prompt.

Type `exit` to quit. Confirm the program exits cleanly.

Try a follow-up:
```
> a romance from 2022 above 8
> something older
```
The `something older` query should reuse the prior context via `history` (the parser sees it).

Try a SQL-mode query:
```
> a fantasy from 2023 above 8.5
```
Expected: similarity column shows `—`, not `0.000`.

- [ ] **Step 5: Commit**

```
git add src/recommender/pipeline.py
git commit -m "feat: convert pipeline.py to interactive CLI repl"
```

---

## Task 10: Delete the Streamlit + FastAPI surface

**Files:**
- Delete: `app.py`
- Delete: `src/api/` (entire directory)
- Delete: `tests/smoke/test_api_smoke.py`
- Delete: `tests/unit/api/` (entire directory)

- [ ] **Step 1: Delete the files**

Run:
```
git rm app.py
git rm -r src/api
git rm tests/smoke/test_api_smoke.py
git rm -r tests/unit/api
```

- [ ] **Step 2: Verify no stragglers reference the deleted modules**

Run:
```
uv run pytest --collect-only -q
```
Expected: collection succeeds with no `ImportError`. If anything still imports `src.api`, fix it (search with the Grep tool: pattern `from src.api`, then update or delete).

- [ ] **Step 3: Run the full test suite (excluding network-marked tests)**

Run:
```
uv run pytest -m "not parser" -v
```
Expected: every test passes. (Skip `parser`-marked tests because they need OpenAI; skip nothing else.)

- [ ] **Step 4: Commit**

```
git commit -m "refactor: remove streamlit ui and fastapi backend"
```

---

## Task 11: Rewrite the README

**Files:**
- Modify: `README.md`

- [ ] **Step 1: Replace `README.md`**

Overwrite the file with the following content:

````markdown
# C-Drama Recommender

A small CLI tool that helps you find your next Chinese drama to watch.

You describe what you're in the mood for in plain English — a vibe, a
plot fragment you half-remember, or a drama you already loved — and the
tool prints the seven best matches from a curated catalogue.

## What it can do

Three different ways of asking, all at the same prompt:

- **"Something like *Nirvana in Fire*, but no older than 2025"** — anchor on a drama
  you already know and find others with a similar feel.
- **"A heroine with amnesia who's enemies with the male lead"** — describe
  the plot or mood when you can't remember the title.
- **"A romance from 2022 rated above 8"** — just give filters: genre,
  year, rating.

You can also tell it what to *avoid* ("no fantasy", "exclude wuxia") or
list dramas you've already finished so they don't show up again.
Follow-ups like *"something older"* work because the REPL keeps
conversation history within the session.

## How it works (the short version)

A small RAG-style pipeline runs on every query:

1. **Parse** — an LLM reads your message and turns it into a small
   structured object: search mode, genres, year, rating, things to
   include or exclude.
2. **Route** — depending on what you asked, the pipeline picks one of
   three search strategies (reference drama, plot description, or
   filters only).
3. **Retrieve** — it pulls about fourteen candidate dramas from local
   Postgres. For the first two modes that means vector search on
   embeddings; for filters it's plain SQL.
4. **Rerank** — the candidates get a final score that mixes similarity,
   the drama's MyDramaList rating, and how popular it is. The top seven
   are printed.

The catalogue itself was scraped from MyDramaList, cleaned up, embedded,
and loaded into a local Postgres + pgvector container.

## Running it locally

You'll need Python 3.12+, [uv](https://docs.astral.sh/uv/), Docker
Desktop, and an OpenAI API key.

Put your secrets in `~/secrets/.env`:

```
OPENAI_API_KEY=...
DATABASE_URL=postgresql://postgres:postgres@localhost:5432/cdramas
```

Start Postgres and load the catalogue (one-time):

```bash
docker compose up -d                     # start Postgres + pgvector
uv run src/database/loader.py            # load embedded dramas from parquet
```

Then run the recommender:

```bash
uv run src/recommender/pipeline.py
```

You'll get a `> ` prompt; type a query, hit enter, and the seven best
matches print as a table. Type `exit` to quit.

Stop the container with `docker compose down`. Add `-v` to wipe the
named volume and start fresh on the next `up`.

## Tests

```bash
uv run pytest                              # everything (needs DB + OpenAI)
uv run pytest tests/unit                   # fast, no network
uv run pytest -m "not db and not parser"   # skip anything that hits a real service
```

DB-marked tests need `docker compose up -d` first.

## Project layout

```
src/recommender/   # the pipeline (parse, route, retrieve, rerank)
src/scraper/       # one-off MyDramaList scraper
src/database/      # connection, schema.sql, functions.sql, parquet loader
notebooks/         # exploration, cleaning, embeddings, parser experiments
tests/             # unit, integration, smoke, and eval tests
docker-compose.yml # local Postgres + pgvector container
```

## Why I built it

A portfolio project to learn the moving parts of a real RAG-flavoured
application end to end — scraping and cleaning data, embeddings and
vector search, prompt engineering, evaluation. The domain is just an
excuse: I watch a lot of C-dramas and the existing recommendation tools
never quite get the vibe right.

## A note on the data

The catalogue was scraped from MyDramaList for educational and portfolio
use only, in a single one-off run to seed the database (Kaggle's existing
C-drama datasets were too out of date). The scraped data is not
redistributed and is not committed to this repo (see `.gitignore`). I
checked MyDramaList's `robots.txt` and only scraped paths it allows
(`/search` and `/id/...`); the disallowed admin and write endpoints are
not touched. The run was single-threaded with a 1.5-second delay between
requests, fully resumable so retries didn't re-hit pages, and only
collected publicly visible information. The original scrape used a
generic browser User-Agent; the scraper in this repo now sends a
bot-style identifier (`Mozilla/5.0 (compatible; cdrama-recommender/1.0)`)
— if you re-run it, please add a `+mailto:` contact of your own so the
site owner can reach you. If you're from MyDramaList and would like the project
taken down or modified, please open a GitHub issue and I'll act on it
the same day.
````

- [ ] **Step 2: Sanity-check the README**

Open `README.md` in the IDE. Confirm:
- No mention of Streamlit, FastAPI, `app.py`, `uvicorn`, two-terminal setup.
- No mention of `SUPABASE_URL` / `SUPABASE_SECRET_KEY`.
- The `Running it locally` section reads top-to-bottom as something a fresh reader could follow.

- [ ] **Step 3: Final test run**

Run:
```
uv run pytest -m "not parser" -v
```
Expected: every test passes.

- [ ] **Step 4: Commit**

```
git add README.md
git commit -m "docs: rewrite readme for cli + local postgres workflow"
```

---

## Self-Review

Spec sections vs. tasks:
- "Strip generator stage" → Task 6.
- "REPL behaviour & sample session" → Task 9 (REPL), Tasks 7–8 (formatting).
- "MATCH_COUNT 10→14, TOP_N 5→7" → Task 9 step 1.
- "Local Postgres in Docker (compose, schema bind-mount, env)" → Task 1.
- "Python client: psycopg + pgvector" → Task 2.
- "File-by-file edits (connection, _shared, search_*, loader)" → Tasks 2, 3, 4, 5.
- "New `tests/smoke/test_postgres_connection.py`" → Task 2.
- "Deletions list (app.py, src/api/, generator tests, llm_judge, 04b notebook)" → Tasks 6 (generator) and 10 (UI/API).
- "README rewrite" → Task 11.

Coverage: complete. No spec requirement is unimplemented.

Type/name consistency check:
- `psycopg.Connection` used consistently as the type hint everywhere `Client` was used before.
- `db_conn` is the consistent local variable / fixture name in the REPL and integration tests.
- `format_parsed_filters` and `format_results` names match between definitions, tests, and call sites.
- `MATCH_COUNT = 14`, `TOP_N = 7` set in one place (Task 9) and never contradicted.

No placeholders detected.
