"""
api/routers/sessions.py — GET /sessions/{session_id}/history
"""

from __future__ import annotations

import logging

from fastapi import APIRouter, HTTPException, Query

from api.dependencies import async_get_conversation_messages
from api.schemas import MessageRecord

logger = logging.getLogger(__name__)
router = APIRouter(prefix="/sessions", tags=["Sessions"])


@router.get(
    "/{session_id}/history",
    response_model=list[MessageRecord],
    summary="Load all messages in a past conversation",
    description=(
        "Returns the full ordered message list for a session owned by the given user. "
        "Pass the user_id returned by POST /auth/login as a query parameter."
    ),
)
async def get_session_history(
    session_id: str,
    user_id: int = Query(..., description="The authenticated user's ID"),
) -> list[MessageRecord]:
    logger.info("History request — user_id=%s session_id=%s", user_id, session_id)

    try:
        messages = await async_get_conversation_messages(session_id, user_id)
    except Exception as exc:
        logger.error(
            "History DB error — user_id=%s session_id=%s error=%s: %s",
            user_id, session_id, type(exc).__name__, exc,
        )
        raise HTTPException(status_code=503, detail=f"Database error: {exc}") from exc

    if not messages:
        logger.warning(
            "History NOT FOUND — user_id=%s session_id=%s",
            user_id, session_id,
        )
        raise HTTPException(
            status_code=404,
            detail=f"No conversation found for session '{session_id}' and user {user_id}.",
        )

    logger.info(
        "History loaded — user_id=%s session_id=%s messages=%d",
        user_id, session_id, len(messages),
    )
    return [
        MessageRecord(
            role=m["role"],
            content=m["content"],
            intent=m.get("intent") or None,
            created_at=str(m["created_at"]) if m.get("created_at") else None,
        )
        for m in messages
    ]
