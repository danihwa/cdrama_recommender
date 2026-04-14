# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project

`cdrama-recommender` — a Chinese drama recommender built on scraped MyDramaList (MDL) data, Supabase + pgvector, and OpenAI embeddings. Python 3.12, managed with `uv`.

## Important context
Current date: April 2026. Always use up-to-date libraries and 
modern Python practices. If unsure about the latest version of 
a library, prefer checking before assuming. Avoid deprecated 
approaches — for example use modern OpenAI client syntax, 
current Supabase Python SDK patterns etc.

## Working principles

<!-- Adapted from forrestchang/andrej-karpathy-skills (MIT). -->

- **Surface uncertainty before coding.** State assumptions explicitly; if multiple interpretations exist, ask instead of picking silently.
- **Define verifiable success before looping.** Turn vague asks into concrete checks ("fix the bug" → "write a failing test, then make it pass") so you can self-verify without constant clarification.

## Commands

All Python entry points are run through `uv` so they use the project venv and lockfile:

```bash
uv sync                                    # install/refresh deps from uv.lock
uv run src/scraper/run_scrape.py           # full resumable scrape → data/raw/dramas.jsonl
uv run src/database/loader.py              # upsert data/cleaned/dramas_with_vectors.parquet into Supabase
uv run src/database/connection.py          # smoke-test Supabase connectivity
uv run pytest                              # run test suite (tests/unit, tests/integration, tests/evals)
uv run pytest tests/unit/test_foo.py::test_bar   # run a single test
```

There is no configured linter or formatter; don't invent one.

## Git Commits
Keep commit messages short and concise. Use a single line following conventional commits format (e.g. `feat: add drama search endpoint`). No bullet points, no body, no lengthy descriptions.

## Dev environment

- `docker-compose.yml` builds a Debian + uv + Claude Code container, mounts the repo at `/workspace`, and loads secrets from the host (`C:/Users/danic/secrets/.env`). The `.venv` lives in a named volume so host/container architectures don't clash.
- `src/env.py::load_secrets()` looks for `~/secrets/.env` first, then falls back to walking up from cwd. Any module that needs env vars should call it (see `src/database/connection.py`).
- Required env vars: `SUPABASE_URL`, `SUPABASE_SECRET_KEY` (service-role key — used to bypass RLS), and OpenAI credentials for embeddings.

## Architecture

The repo is a linear data pipeline that feeds a query-time recommender. The stages are split across `src/` modules and Jupyter notebooks:

1. **Scrape** (`src/scraper/`) — `run_scrape.py` orchestrates `drama_urls.py` (collects URLs from MDL search pages) and `drama_info.py` (parses each detail page). Output is appended line-by-line to `data/raw/dramas.jsonl` so the run is resumable: `load_already_scraped()` keys on `mdl_url` to skip completed entries. Failures are isolated per URL and logged to `data/failed_urls.txt`. `_http.py` centralises the User-Agent + `requests.get` wrapper.
2. **Clean** (`notebooks/02_cleaning.ipynb`) — reads the JSONL, normalises fields, writes parquet.
3. **Embed** (`notebooks/03_embeddings.ipynb`) — calls OpenAI's embedding model (dim **3072**) to produce `data/cleaned/dramas_with_vectors.parquet`. The 3072 dimension is load-bearing: it must match `schema.sql` (`vector(3072)`) and `functions.sql` (`query_embedding vector(3072)`). Changing the embedding model requires updating both.
4. **Load** (`src/database/loader.py`) — batched `upsert` into the `cdramas` table keyed on `mdl_id`, so re-runs are safe. Embeddings/genres/tags must be plain Python lists (not numpy) before upsert.
5. **Recommend** (`src/recommender/`) — query-time layer. `models.py` defines `QueryFilters`, the Pydantic schema the parser LLM emits and every downstream stage consumes. `search_mode` is one of three strategies:
   - `reference` — vector search anchored on an existing drama's embedding ("like X")
   - `semantic` — vector search on an embedding of the user's free-form description
   - `sql` — pure filter query (genre/year/rating), no embedding

### Database layer (non-obvious)

- The `cdramas` table schema lives in `src/database/schema.sql`; the similarity-search RPC lives in `src/database/functions.sql`. These are **applied manually in the Supabase SQL editor**, not via migrations — if you change them, you must apply them there yourself.
- `match_documents` is the single entry point for similarity search. It pushes **all** hard filters (year, score, genres, excluded ids) into SQL alongside the cosine-distance computation so Postgres prunes before scoring. `src/recommender/_shared.py::vector_search` is the only Python caller — do not bypass it by filtering in Python after a generic vector query.
- Filter sentinels in `vector_search`: `None` min_year/min_score get replaced with `1900`/`0.0` (permissive defaults). Empty `exclude_ids`/`genres` are passed as `None` so the RPC treats them as "no filter".
- Genres in the DB are lowercase — always run user input through `_shared.normalize_genres()` before comparison.
- Exclusion titles are resolved to ids via `ILIKE '%title%'` in `_shared.find_exclude_ids`; a missed title is a soft warning, not an error.

### Notebooks vs modules

The cleaning and embedding stages intentionally live in `notebooks/` (exploratory, run by hand), while scraping, loading, and the recommender runtime live in `src/` (reusable, importable). When a notebook stabilises, its reusable bits should migrate into `src/`, not the other way round.
