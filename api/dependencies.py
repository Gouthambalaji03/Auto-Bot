"""
api/dependencies.py — Async wrappers for synchronous DB calls + shared FastAPI dependencies.

All db/ functions use synchronous psycopg2. Calling them directly inside an
async FastAPI handler would block the event loop. Every function here wraps its
synchronous counterpart with asyncio.to_thread() so DB work runs in a
thread-pool thread while the event loop stays free for other requests.

Nothing in db/ is modified — these are purely additive thin wrappers.
"""

from __future__ import annotations

import asyncio
from typing import Optional

from db.auth import db_create_user, db_authenticate_user
from db.queries import (
    db_get_conversation_messages,
    db_create_conversation,
    db_save_chat_message,
)
from db.connection import get_pool


# ─────────────────────────────────────────────
# Auth wrappers
# ─────────────────────────────────────────────

async def async_create_user(
    username: str,
    email: str,
    password: str,
) -> tuple[bool, str, Optional[dict]]:
    """Async wrapper around db_create_user (psycopg2 → thread pool)."""
    return await asyncio.to_thread(db_create_user, username, email, password)


async def async_authenticate_user(
    username_or_email: str,
    password: str,
) -> tuple[bool, str, Optional[dict]]:
    """Async wrapper around db_authenticate_user (psycopg2 → thread pool)."""
    return await asyncio.to_thread(db_authenticate_user, username_or_email, password)


# ─────────────────────────────────────────────
# Session / history wrappers
# ─────────────────────────────────────────────

async def async_get_conversation_messages(
    session_id: str,
    user_id: int,
) -> list[dict]:
    """Async wrapper around db_get_conversation_messages."""
    return await asyncio.to_thread(db_get_conversation_messages, session_id, user_id)


async def async_create_conversation(
    user_id: int,
    session_id: str,
    title: str,
) -> dict:
    """Async wrapper around db_create_conversation."""
    return await asyncio.to_thread(db_create_conversation, user_id, session_id, title)


async def async_save_chat_message(
    user_id: int,
    session_id: str,
    role: str,
    content: str,
    intent: str = "",
) -> dict:
    """Async wrapper around db_save_chat_message."""
    return await asyncio.to_thread(
        db_save_chat_message, user_id, session_id, role, content, intent
    )


# ─────────────────────────────────────────────
# DB health check
# ─────────────────────────────────────────────

async def check_db_connection() -> str:
    """
    Attempts a lightweight SELECT 1 query to verify DB reachability.
    Returns 'connected' on success or an error message string on failure.
    """
    def _ping() -> str:
        try:
            pool = get_pool()
            conn = pool.getconn()
            try:
                with conn.cursor() as cur:
                    cur.execute("SELECT 1")
            finally:
                pool.putconn(conn)
            return "connected"
        except Exception as exc:
            return f"error: {type(exc).__name__}: {exc}"

    return await asyncio.to_thread(_ping)
