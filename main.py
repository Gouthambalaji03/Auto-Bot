"""
AutoBot — AI Automobile Assistant
Entry point: builds the Gradio UI and launches the server.

Architecture:
    main.py           ← launch entry point
    ui/
        app.py        ← Gradio UI layout & event wiring
    agents/
        automotive_agent.py ← Single LLM-Led Pydantic AI Automotive Agent
    models/
        schemas.py    ← Pydantic response schemas
    tools/
        car_tools.py  ← Database query helpers & EMI math
    db/
        connection.py ← PostgreSQL pool + TTL cache
        queries.py    ← DB query helpers
        fuzzy_queries.py ← fuzzy car/issue matching
"""

import sys
import io

if sys.platform == "win32" and hasattr(sys.stdout, "buffer"):
    try:
        sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")
    except Exception:
        pass

from dotenv import load_dotenv
load_dotenv(override=True)

from db.connection import validate_db_config
validate_db_config()

import gradio as gr
import uvicorn
from fastapi import FastAPI
from fastapi.responses import RedirectResponse
from ui.app import build_ui, PREMIUM_CSS, FORCED_LIGHT_THEME, AUTO_ROUTER_JS


def create_application() -> FastAPI:
    app = FastAPI(title="AutoBot UI")

    @app.middleware("http")
    async def spa_redirect_middleware(request, call_next):
        path = request.url.path
        if request.method in ("GET", "HEAD") and (
            path in ("/login", "/signup", "/chat", "/chat/new")
            or (path.startswith("/chat/") and not path.startswith("/chat/queue"))
        ):
            return RedirectResponse(url=f"/?route={path}", status_code=307)
        return await call_next(request)

    demo = build_ui()
    app = gr.mount_gradio_app(
        app,
        demo,
        path="/",
        theme=FORCED_LIGHT_THEME,
        css=PREMIUM_CSS,
        head=f"<style>{PREMIUM_CSS}</style><script>{AUTO_ROUTER_JS}</script>",
        show_error=True,
    )
    return app


if __name__ == "__main__":
    app = create_application()
    for port in range(7860, 7870):
        try:
            print(f"[AUTOBOT] [STARTUP] Launching server on port {port}...", flush=True)
            uvicorn.run(app, host="0.0.0.0", port=port, log_level="warning")
            break
        except OSError:
            if port == 7869:
                raise
            print(f"[AUTOBOT] [STARTUP] Port {port} is occupied; trying the next port...", flush=True)
