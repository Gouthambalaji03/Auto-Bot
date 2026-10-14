"""
api/routers/health.py — GET /health
"""

from __future__ import annotations

import logging

from fastapi import APIRouter
from fastapi.responses import JSONResponse

from agents.automotive_agent import _cached_agent
from api.dependencies import check_db_connection
from api.schemas import HealthResponse

logger = logging.getLogger(__name__)
router = APIRouter(tags=["Health"])


@router.get(
    "/health",
    response_model=HealthResponse,
    summary="Liveness and readiness check",
    description=(
        "Returns HTTP 200 with status='ok' when the API, database, and LLM agent "
        "are all operational. Returns HTTP 503 with status='degraded' if the database "
        "is unreachable."
    ),
)
async def health_check() -> JSONResponse:
    db_status = await check_db_connection()
    try:
        from agents.automotive_agent import get_automotive_agent
        get_automotive_agent()
        agent_status = "ready"
    except Exception as exc:
        agent_status = f"unconfigured: {exc}"
    overall_ok = db_status == "connected"
    status_code = 200 if overall_ok else 503

    # Use DEBUG so /health polling doesn't flood INFO logs in production
    log_fn = logger.debug if overall_ok else logger.warning
    log_fn(
        "Health check — db=%s agent=%s → HTTP %d",
        db_status, agent_status, status_code,
    )

    import os
    model_name = os.getenv("GEMINI_MODEL", "gemini-3.6-flash")
    payload = HealthResponse(
        status="ok" if overall_ok else "degraded",
        db=db_status,
        agent=agent_status,
        model=model_name,
    )
    return JSONResponse(content=payload.model_dump(), status_code=status_code)
