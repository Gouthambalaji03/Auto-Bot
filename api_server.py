"""
api_server.py — AutoBot FastAPI server entrypoint.

Run with:
    python api_server.py

The API starts on http://localhost:8000 by default.
Interactive docs: http://localhost:8000/docs

Configuration (via .env or environment variables):
    API_PORT=8000          Port to listen on (default: 8000)
    API_HOST=0.0.0.0       Host to bind to (default: 0.0.0.0)
    CORS_ORIGINS=*         Comma-separated allowed origins (default: *)
    DATABASE_URL=...       PostgreSQL connection string (required)
    GEMINI_API_KEY=...     Google Gemini API key (required)
"""

import sys
import io

# Windows UTF-8 stdout fix (matches main.py)
if sys.platform == "win32" and hasattr(sys.stdout, "buffer"):
    try:
        sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")
    except Exception:
        pass

import os
import uvicorn
from dotenv import load_dotenv
from api.logger import configure_logging

load_dotenv(override=True)


if __name__ == "__main__":
    # Configure logging before anything else — this sets format and level for
    # the entire process, including all modules imported by uvicorn workers.
    configure_logging()

    host = os.getenv("API_HOST", "0.0.0.0")
    port = int(os.getenv("API_PORT", "8000"))

    import logging
    log = logging.getLogger("autobot.api.server")
    log.info("Launching AutoBot API on http://%s:%d", host, port)
    log.info("Interactive docs → http://localhost:%d/docs", port)

    uvicorn.run(
        "api.app:app",
        host=host,
        port=port,
        reload=False,
        log_level="info",
    )

