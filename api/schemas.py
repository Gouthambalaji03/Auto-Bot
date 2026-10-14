"""
api/schemas.py — Pydantic request and response models for the AutoBot REST API.

All models are independent of the Gradio UI layer and mirror the contracts
already established by agents/automotive_agent.py (AutoBotStreamUpdate,
chat_with_autobot return tuple) without modifying them.
"""

from __future__ import annotations

from typing import Optional
from pydantic import BaseModel, Field


# ─────────────────────────────────────────────
# Chat
# ─────────────────────────────────────────────

class ChatRequest(BaseModel):
    """Body for both POST /chat (SSE) and POST /chat/sync."""
    message: str = Field(..., min_length=1, description="User's message to AutoBot")
    history: list[dict] = Field(
        default_factory=list,
        description="Prior conversation turns in {role, content} format"
    )
    user_id: Optional[int] = Field(
        default=None,
        description="Authenticated user ID — enables DB persistence"
    )
    session_id: Optional[str] = Field(
        default=None,
        description="Existing session UUID — continues a prior conversation"
    )


class ChatSyncResponse(BaseModel):
    """Response body for POST /chat/sync (non-streaming)."""
    output: str = Field(..., description="Full markdown response from AutoBot")
    intent: str = Field(..., description="Comma-joined intent labels, e.g. 'buying,finance'")
    model_name: str = Field(..., description="Model identifier string")
    elapsed_seconds: float = Field(..., description="Wall-clock time for the full run")
    session_id: Optional[str] = Field(
        default=None,
        description="Session UUID used or created during this request"
    )


# ─────────────────────────────────────────────
# Auth
# ─────────────────────────────────────────────

class SignupRequest(BaseModel):
    """Body for POST /auth/signup."""
    username: str = Field(..., min_length=3, description="Desired username (min 3 chars)")
    email: str = Field(..., description="User email address")
    password: str = Field(..., min_length=6, description="Password (min 6 chars)")


class SignupResponse(BaseModel):
    """Response body for POST /auth/signup."""
    success: bool
    message: str
    user: Optional[dict] = Field(
        default=None,
        description="{id, username, email, created_at} on success, null on failure"
    )


class LoginRequest(BaseModel):
    """Body for POST /auth/login."""
    username_or_email: str = Field(..., description="Username or email address")
    password: str = Field(..., description="Account password")


class LoginResponse(BaseModel):
    """Response body for POST /auth/login."""
    success: bool
    message: str
    user: Optional[dict] = Field(
        default=None,
        description="{id, username, email} on success, null on failure"
    )


# ─────────────────────────────────────────────
# Health
# ─────────────────────────────────────────────

class HealthResponse(BaseModel):
    """Response body for GET /health."""
    status: str = Field(..., description="'ok' or 'degraded'")
    db: str = Field(..., description="'connected' or error message")
    agent: str = Field(..., description="'ready' or 'not_initialised'")
    model: str = Field(default="gemini-3.6-flash")


# ─────────────────────────────────────────────
# Sessions
# ─────────────────────────────────────────────

class MessageRecord(BaseModel):
    """A single chat message record from the database."""
    role: str
    content: str
    intent: Optional[str] = None
    created_at: Optional[str] = None
