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
src/recommender/   # the pipeline (parse, route, retrieve, score)
src/scraper/       # one-off MyDramaList scraper
src/database/      # connection, schema.sql, functions.sql, parquet loader
notebooks/         # exploration, cleaning, embeddings, parser experiments
tests/             # unit, integration, eval, and data-quality tests
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
