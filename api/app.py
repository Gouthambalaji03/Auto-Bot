"""
api/app.py — FastAPI application factory for AutoBot REST API.

Creates and configures the FastAPI application instance:
  - Request timing middleware (logs every HTTP request with method, path, status, duration)
  - CORS middleware (configurable via CORS_ORIGINS env var)
  - Global exception handlers (DB pool errors, unhandled exceptions)
  - Startup warm-up: initialises the Pydantic AI agent singleton so the
    first real request isn't penalised by cold-start initialisation time
  - All routers registered under their respective prefixes

This module does NOT start a server — import `app` from here in api_server.py
and let uvicorn handle the lifecycle.

The existing Gradio entrypoint (main.py) is completely unaffected.
"""

from __future__ import annotations

import logging
import os
import time

from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

from api.logger import configure_logging
from api.routers import health, chat, auth, sessions

# Ensure logging is configured even when the module is imported directly
# (e.g. during tests). api_server.py also calls this before uvicorn starts.
configure_logging()

logger = logging.getLogger(__name__)


from contextlib import asynccontextmanager


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Lifespan context manager for warm-up and graceful shutdown."""
    import asyncio
    from db.connection import validate_db_config

    logger.info("AutoBot API starting up...")
    try:
        validate_db_config()
        logger.info("Database config validated — DATABASE_URL is set")
    except Exception as exc:
        logger.warning("Database config validation warning: %s", exc)

    def _warm_agent():
        try:
            from agents.automotive_agent import get_automotive_agent
            get_automotive_agent()
            logger.info("Pydantic AI agent singleton ready")
        except Exception as exc:
            logger.warning("Agent warm-up skipped: %s", exc)

    await asyncio.to_thread(_warm_agent)
    logger.info("AutoBot API startup complete ✅")
    yield
    logger.info("AutoBot API shutdown complete")


# ─────────────────────────────────────────────
# Application factory
# ─────────────────────────────────────────────

def create_app() -> FastAPI:
    app = FastAPI(
        title="AutoBot API",
        description=(
            "REST + Server-Sent Events API for AutoBot — the AI automobile assistant "
            "for the Indian market. Exposes the same Pydantic AI agent used by the "
            "Gradio UI, allowing any HTTP client to query AutoBot programmatically."
        ),
        version="1.0.0",
        docs_url="/docs",
        redoc_url="/redoc",
        openapi_url="/openapi.json",
        lifespan=lifespan,
    )

    # ── Request timing middleware ──────────────────────────────────────────
    # Logs every HTTP request: method, path, status code, and duration.
    # This is the industry-standard "access log" at application level.
    # Example:
    #   2026-08-13 23:19:45  INFO  api.app  POST /auth/signup → 200  (142ms)

    @app.middleware("http")
    async def log_requests(request: Request, call_next):
        start = time.perf_counter()
        response = await call_next(request)
        elapsed_ms = (time.perf_counter() - start) * 1000

        # Skip logging for favicon and openapi schema (noise)
        if request.url.path not in ("/favicon.ico", "/openapi.json"):
            logger.info(
                "%s %s → %d  (%.0fms)",
                request.method,
                request.url.path,
                response.status_code,
                elapsed_ms,
            )
        return response

    # ── CORS ──────────────────────────────────────────────────────────────
    cors_origins_raw = os.getenv("CORS_ORIGINS", "*")
    cors_origins = (
        ["*"] if cors_origins_raw.strip() == "*"
        else [o.strip() for o in cors_origins_raw.split(",") if o.strip()]
    )
    app.add_middleware(
        CORSMiddleware,
        allow_origins=cors_origins,
        allow_credentials=cors_origins != ["*"],
        allow_methods=["*"],
        allow_headers=["*"],
    )

    # ── Global exception handler ───────────────────────────────────────────

    @app.exception_handler(Exception)
    async def unhandled_exception_handler(request: Request, exc: Exception) -> JSONResponse:
        logger.error(
            "Unhandled exception on %s %s — %s: %s",
            request.method, request.url.path,
            type(exc).__name__, exc,
            exc_info=True,   # includes full traceback in logs
        )
        return JSONResponse(
            status_code=500,
            content={"detail": "Internal server error. Check API logs for details."},
        )

    # ── Routers ───────────────────────────────────────────────────────────

    app.include_router(health.router)
    app.include_router(chat.router)
    app.include_router(auth.router)
    app.include_router(sessions.router)

    # ── Root route — live status page ─────────────────────────────────────

    @app.get(
        "/",
        tags=["Status"],
        summary="AutoBot API — root status",
        description="Returns live DB and agent status. HTTP 200 = everything operational.",
    )
    async def root() -> JSONResponse:
        from api.dependencies import check_db_connection

        db_status = await check_db_connection()
        try:
            from agents.automotive_agent import get_automotive_agent
            get_automotive_agent()
            agent_status = "ready"
            agent_ok = True
        except Exception as exc:
            agent_status = f"unconfigured: {exc}"
            agent_ok = False

        db_ok = db_status == "connected"

        logger.info(
            "Status check — DB: %s | Agent: %s",
            "✅ connected" if db_ok else f"❌ {db_status}",
            "✅ ready" if agent_ok else f"❌ {agent_status}",
        )

        overall = "ok" if (db_ok and agent_ok) else "degraded"
        model_name = os.getenv("GEMINI_MODEL", "gemini-3.6-flash")

        return JSONResponse(
            status_code=200 if overall == "ok" else 503,
            content={
                "api": "AutoBot REST API",
                "version": "1.0.0",
                "status": overall,
                "services": {
                    "database": {
                        "status": "connected" if db_ok else "error",
                        "detail": db_status,
                        "icon": "✅" if db_ok else "❌",
                    },
                    "agent": {
                        "status": agent_status,
                        "model": model_name,
                        "icon": "✅" if agent_ok else "❌",
                    },
                },
                "docs": "http://localhost:8000/docs",
                "endpoints": {
                    "health":      "GET  /health",
                    "chat_stream": "POST /chat",
                    "chat_sync":   "POST /chat/sync",
                    "signup":      "POST /auth/signup",
                    "login":       "POST /auth/login",
                    "history":     "GET  /sessions/{session_id}/history",
                },
            },
        )

    # ── Suppress favicon 404 noise ────────────────────────────────────────

    @app.get("/favicon.ico", include_in_schema=False)
    async def favicon():
        from fastapi.responses import Response
        return Response(status_code=204)

    return app


# Module-level app instance — used by uvicorn in api_server.py
app = create_app()
