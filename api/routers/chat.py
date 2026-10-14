"""
api/routers/chat.py — POST /chat (SSE streaming)  |  POST /chat/sync

Both endpoints reuse the existing stream_chat_with_autobot and chat_with_autobot
coroutines from agents/automotive_agent.py without modification.

POST /chat   → Server-Sent Events stream; each event is one AutoBotStreamUpdate
              serialised as JSON. Mirrors the field names of AutoBotStreamUpdate
              exactly so any existing client of the Gradio stream can adapt easily.

POST /chat/sync → Awaits the full response and returns a single JSON object.
                  Useful for simple CLI clients, cURL tests, or non-streaming
                  integrations.

Error handling:
  - 429 Gemini rate-limit  → HTTP 429 (sync) or SSE error event (streaming)
  - Missing API key        → HTTP 503
  - DB errors              → WARNING logged + graceful degradation (chat works)
  - Other exceptions       → HTTP 500 (sync) or SSE error event (streaming)
"""

from __future__ import annotations

import json
import logging
import time
import uuid
import asyncio
from typing import AsyncIterator

from fastapi import APIRouter
from fastapi.responses import StreamingResponse
from fastapi import HTTPException

from agents.automotive_agent import (
    stream_chat_with_autobot,
    chat_with_autobot,
    AutoBotStreamUpdate,
)
from api.dependencies import (
    async_create_conversation,
    async_save_chat_message,
)
from api.schemas import ChatRequest, ChatSyncResponse

logger = logging.getLogger(__name__)
router = APIRouter(prefix="/chat", tags=["Chat"])


# ─────────────────────────────────────────────
# Internal helpers
# ─────────────────────────────────────────────

def _is_rate_limit(exc: Exception) -> bool:
    """Detect Gemini 429 / quota errors."""
    msg = str(exc)
    return "429" in msg or "quota" in msg.lower() or "RESOURCE_EXHAUSTED" in msg


async def _persist_turn(
    user_id: int,
    session_id: str,
    user_message: str,
    assistant_response: str,
    intent: str,
    first_message: bool,
) -> None:
    """
    Silently persists a conversation turn to PostgreSQL.
    Failures are logged as WARNING — never bubble up to the caller.
    """
    try:
        if first_message:
            title = user_message.strip()[:50] or "Chat via API"
            await async_create_conversation(user_id, session_id, title)
            logger.debug(
                "Conversation created — user_id=%s session_id=%s title=%r",
                user_id, session_id, title,
            )
        await async_save_chat_message(user_id, session_id, "user", user_message)
        await async_save_chat_message(user_id, session_id, "assistant", assistant_response, intent)
        logger.debug(
            "Turn persisted — user_id=%s session_id=%s intent=%r",
            user_id, session_id, intent,
        )
    except Exception as exc:
        logger.warning(
            "DB persistence failed (non-fatal) — user_id=%s session_id=%s error=%s: %s",
            user_id, session_id, type(exc).__name__, exc,
        )


def _sse_event(data: dict, event: str = "update") -> str:
    """Format a dict as a Server-Sent Event string."""
    return f"event: {event}\ndata: {json.dumps(data)}\n\n"


# ─────────────────────────────────────────────
# POST /chat — SSE streaming
# ─────────────────────────────────────────────

@router.post(
    "",
    summary="Stream AutoBot's response via Server-Sent Events",
    description=(
        "Streams the agent's response token-by-token as Server-Sent Events. "
        "Each event carries a JSON object with fields: content, complete, "
        "intents, tool_calls, elapsed_seconds — mirroring AutoBotStreamUpdate. "
        "The stream closes after the event where complete=true. "
        "On error, a final 'error' event is emitted before the stream closes."
    ),
    response_class=StreamingResponse,
    responses={
        200: {"description": "SSE stream of AutoBotStreamUpdate events"},
        422: {"description": "Invalid request body"},
    },
)
async def chat_stream(body: ChatRequest) -> StreamingResponse:
    """POST /chat — returns a Server-Sent Events stream."""

    async def event_generator() -> AsyncIterator[str]:
        session_id = body.session_id
        is_first_message = session_id is None
        if body.user_id and not session_id:
            session_id = str(uuid.uuid4())

        start = time.perf_counter()
        logger.info(
            "SSE stream START — user_id=%s session_id=%s msg_len=%d",
            body.user_id, session_id, len(body.message),
        )

        final_content = ""
        final_intent = "unclassified"
        update: Optional[AutoBotStreamUpdate] = None

        try:
            async for update in stream_chat_with_autobot(
                user_message=body.message,
                history=body.history,
                user_id=body.user_id,
                session_id=session_id,
            ):
                update: AutoBotStreamUpdate
                payload = {
                    "content": update.content,
                    "complete": update.complete,
                    "intents": list(update.intents),
                    "tool_calls": list(update.tool_calls),
                    "elapsed_seconds": update.elapsed_seconds,
                }
                if session_id:
                    payload["session_id"] = session_id

                yield _sse_event(payload)

                if update.complete:
                    final_content = update.content
                    final_intent = ",".join(update.intents) if update.intents else "unclassified"

            elapsed = time.perf_counter() - start
            logger.info(
                "SSE stream COMPLETE — user_id=%s intent=%r tools=%s elapsed=%.2fs",
                body.user_id, final_intent,
                list(update.tool_calls) if update else [],
                elapsed,
            )

            # Persist to DB after stream closes (non-blocking)
            if body.user_id and session_id and final_content:
                asyncio.create_task(
                    _persist_turn(
                        user_id=body.user_id,
                        session_id=session_id,
                        user_message=body.message,
                        assistant_response=final_content,
                        intent=final_intent,
                        first_message=is_first_message,
                    )
                )

        except Exception as exc:
            elapsed = time.perf_counter() - start
            if _is_rate_limit(exc):
                logger.warning(
                    "SSE stream RATE LIMITED — user_id=%s elapsed=%.2fs error=%s",
                    body.user_id, elapsed, exc,
                )
                yield _sse_event(
                    {"error": "Gemini API rate limit. Please retry in 30 seconds.", "code": 429},
                    event="error",
                )
            else:
                logger.error(
                    "SSE stream ERROR — user_id=%s elapsed=%.2fs",
                    body.user_id, elapsed,
                    exc_info=True,
                )
                yield _sse_event(
                    {"error": f"Internal error: {type(exc).__name__}", "code": 500},
                    event="error",
                )

    return StreamingResponse(
        event_generator(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "X-Accel-Buffering": "no",
            "Connection": "keep-alive",
        },
    )


# ─────────────────────────────────────────────
# POST /chat/sync — non-streaming
# ─────────────────────────────────────────────

@router.post(
    "/sync",
    response_model=ChatSyncResponse,
    summary="Get AutoBot's full response synchronously",
    description=(
        "Awaits the complete agent run and returns a single JSON response. "
        "Suitable for CLI tools, cURL tests, and clients that do not support SSE. "
        "Returns HTTP 429 on Gemini rate limits, 503 on LLM configuration errors."
    ),
)
async def chat_sync(body: ChatRequest) -> ChatSyncResponse:
    """POST /chat/sync — waits for the full response and returns it as JSON."""
    session_id = body.session_id
    is_first_message = session_id is None
    if body.user_id and not session_id:
        session_id = str(uuid.uuid4())

    start = time.perf_counter()
    logger.info(
        "Sync chat START — user_id=%s session_id=%s msg_len=%d",
        body.user_id, session_id, len(body.message),
    )

    try:
        output, intent, _ok, model_name, elapsed = await chat_with_autobot(
            user_message=body.message,
            history=body.history,
            user_id=body.user_id,
            session_id=session_id,
        )
    except Exception as exc:
        total_elapsed = time.perf_counter() - start
        if _is_rate_limit(exc):
            logger.warning(
                "Sync chat RATE LIMITED — user_id=%s elapsed=%.2fs",
                body.user_id, total_elapsed,
            )
            raise HTTPException(
                status_code=429,
                detail="Gemini API rate limit. Please retry in 30 seconds.",
                headers={"Retry-After": "30"},
            ) from exc
        if "GEMINI_API_KEY" in str(exc) or "GOOGLE_API_KEY" in str(exc):
            logger.error(
                "Sync chat LLM NOT CONFIGURED — user_id=%s elapsed=%.2fs",
                body.user_id, total_elapsed,
            )
            raise HTTPException(
                status_code=503,
                detail="LLM not configured. Set GEMINI_API_KEY in your .env file.",
            ) from exc
        logger.error(
            "Sync chat ERROR — user_id=%s elapsed=%.2fs",
            body.user_id, total_elapsed,
            exc_info=True,
        )
        raise HTTPException(
            status_code=500, detail=f"Internal error: {type(exc).__name__}"
        ) from exc

    total_elapsed = time.perf_counter() - start
    logger.info(
        "Sync chat COMPLETE — user_id=%s intent=%r model=%s elapsed=%.2fs",
        body.user_id, intent, model_name, total_elapsed,
    )

    # Persist conversation turn in the background (non-blocking)
    if body.user_id and session_id and output:
        asyncio.create_task(
            _persist_turn(
                user_id=body.user_id,
                session_id=session_id,
                user_message=body.message,
                assistant_response=output,
                intent=intent,
                first_message=is_first_message,
            )
        )

    return ChatSyncResponse(
        output=output,
        intent=intent,
        model_name=model_name,
        elapsed_seconds=elapsed,
        session_id=session_id,
    )
