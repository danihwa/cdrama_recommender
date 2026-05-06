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
