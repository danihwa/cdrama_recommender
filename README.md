# C-Drama Recommender

> **Work in progress** — the RAG pipeline (parse → retrieve → rerank →
> generate) works end to end, but the Streamlit UI in `app.py` is still
> being polished.

A small chat app that helps you find your next Chinese drama to watch.

You can describe what you're in the mood for in plain English — a vibe, a
plot fragment you half-remember, or a drama you already loved — and the
app will pull a handful of candidates from a curated catalogue and write
you a short, personal recommendation explaining why each one fits.

## What it can do

Three different ways of asking, all in the same chat box:

- **"Something like *Nirvana in Fire*, but no older than 2025"** — anchor on a drama
  you already know and find others with a similar feel.
- **"A heroine with amnesia who's enemies with the male lead"** — describe
  the plot or mood when you can't remember the title.
- **"A romance from 2022 rated above 8"** — just give filters: genre,
  year, rating.

You can also tell it what to *avoid* ("no fantasy", "exclude wuxia") or
list dramas you've already finished so they don't show up again. The
sidebar has the same filters as sliders and dropdowns if you'd rather
click than type.

## How it works (the short version)

Under the hood it's a retrieval-augmented chatbot. Each message goes
through five stages:

1. **Parse** — an LLM reads your message and turns it into a small
   structured object: search mode, genres, year, rating, things to
   include or exclude.
2. **Route** — depending on what you asked, the app picks one of three
   search strategies (reference drama, plot description, or filters only).
3. **Retrieve** — it pulls about ten candidate dramas from a Postgres
   database. For the first two modes that means vector search on
   embeddings; for filters it's plain SQL.
4. **Rerank** — the candidates get a final score that mixes similarity,
   the drama's MyDramaList rating, and how popular it is.
5. **Generate** — a second LLM call writes the actual recommendation
   paragraphs, grounded in the top candidates so it can't hallucinate
   titles.

The catalogue itself was scraped from MyDramaList, cleaned up, embedded,
and loaded into Supabase with pgvector.

## Running it locally

You'll need Python 3.12+, [uv](https://docs.astral.sh/uv/), an OpenAI API
key, and a Supabase project with the catalogue loaded.

Put your secrets in `~/secrets/.env`:

```
OPENAI_API_KEY=...
SUPABASE_URL=...
SUPABASE_KEY=...
```

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

## Tests

```bash
uv run pytest                              # everything
uv run pytest tests/unit                   # fast, no network
uv run pytest -m "not db and not parser"   # skip anything that hits a real service
```

## Project layout

```
src/recommender/   # the RAG pipeline (parse, route, retrieve, rerank, generate)
src/scraper/       # one-off MyDramaList scraper
src/database/      # Supabase connection + data loader
notebooks/         # exploration, cleaning, embeddings, prompt experiments
tests/             # unit, integration, smoke, and eval tests
app.py             # Streamlit chat UI
```

## Why I built it

A portfolio project to learn the moving parts of a real RAG application
end to end — scraping and cleaning data, embeddings and vector search,
prompt engineering, evaluation, and wrapping a friendly UI around it.
The domain is just an excuse: I watch a lot of C-dramas and the existing
recommendation tools never quite get the vibe right.

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
