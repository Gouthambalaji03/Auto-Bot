# ─────────────────────────────────────────────────────────────────
# AutoBot — Production Dockerfile
# Multi-stage / slim Python container for AutoBot
# ─────────────────────────────────────────────────────────────────

FROM python:3.11-slim AS base

# Prevent Python from writing .pyc files and buffer stdout/stderr
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    GRADIO_SERVER_NAME="0.0.0.0" \
    GRADIO_SERVER_PORT=7860

WORKDIR /app

# Install OS utilities (curl for healthchecks)
RUN apt-get update && apt-get install -y --no-install-recommends \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Install Python dependencies first for layer caching
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy application source code
COPY . .

# Ensure entrypoint and scripts have execution rights
RUN chmod +x scripts/docker-entrypoint.sh

# Expose Gradio UI (7860) and FastAPI server (8000)
EXPOSE 7860 8000

# Healthcheck targeting root HTTP endpoint
HEALTHCHECK --interval=30s --timeout=10s --start-period=15s --retries=3 \
    CMD curl -f http://localhost:7860/ || exit 1

ENTRYPOINT ["scripts/docker-entrypoint.sh"]

# Default to running the unified Gradio UI server
CMD ["python", "main.py"]
