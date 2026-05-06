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
