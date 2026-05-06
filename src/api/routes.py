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
    """Stream a recommendation as SSE: candidates -> token* -> done.

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
