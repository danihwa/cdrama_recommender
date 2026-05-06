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
