"""
api/logger.py — Centralised logging configuration for the AutoBot API.

Single source of truth for log format, level, and handler setup.
Call configure_logging() once at process startup (api_server.py does this).

Usage in any api/ module:
    import logging
    logger = logging.getLogger(__name__)
    logger.info("...")

Log level is controlled by the LOG_LEVEL environment variable (default: INFO).
Valid values: DEBUG, INFO, WARNING, ERROR, CRITICAL
"""

from __future__ import annotations

import logging
import os
import sys


# ── Format ────────────────────────────────────────────────────────────────────
# Example output:
# 2026-08-13 23:19:45  INFO      api.routers.auth              Signup SUCCESS — user_id=7 elapsed=142ms
_FORMAT = "%(asctime)s  %(levelname)-8s  %(name)-32s  %(message)s"
_DATE   = "%Y-%m-%d %H:%M:%S"


def configure_logging() -> None:
    """
    Configure root logging for the entire AutoBot API process.

    - Writes to stdout (captured by uvicorn / Docker / systemd).
    - Level is controlled by the LOG_LEVEL env var (default INFO).
    - Idempotent: safe to call multiple times.
    - Quiets noisy third-party packages (httpx, google-genai) to WARNING
      so they don't drown out application logs.
    """
    level_name = os.getenv("LOG_LEVEL", "INFO").upper()
    level = getattr(logging, level_name, logging.INFO)

    handler = logging.StreamHandler(sys.stdout)
    handler.setFormatter(logging.Formatter(_FORMAT, datefmt=_DATE))

    root = logging.getLogger()
    root.setLevel(level)
    # Clear any handlers already attached (e.g. if called twice or by uvicorn)
    root.handlers.clear()
    root.addHandler(handler)

    # ── Quiet noisy third-party packages ──────────────────────────────────
    # httpx / httpcore: used by pydantic-ai for Gemini calls — very chatty at DEBUG
    logging.getLogger("httpx").setLevel(logging.WARNING)
    logging.getLogger("httpcore").setLevel(logging.WARNING)
    # google-genai SDK internal debug noise
    logging.getLogger("google").setLevel(logging.WARNING)
    # pydantic-ai internals
    logging.getLogger("pydantic_ai").setLevel(logging.WARNING)
    # Keep uvicorn's access log at INFO so HTTP requests are still visible
    logging.getLogger("uvicorn.access").setLevel(logging.INFO)
    logging.getLogger("uvicorn.error").setLevel(logging.INFO)

    root.info(
        "autobot.api.logger  Logging configured — level=%s",
        level_name,
    )
