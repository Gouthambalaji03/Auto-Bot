#!/usr/bin/env bash
set -e

echo "=================================================="
echo "  AutoBot Container Starting"
echo "=================================================="

# Run database migrations and seeding if DATABASE_URL is provided
if [ -n "${DATABASE_URL:-}" ]; then
    echo "[AUTOBOT] Checking & provisioning database schema..."
    python db/migrate.py || {
        echo "[AUTOBOT] [WARN] db/migrate.py exited with an error; continuing startup..."
    }
else
    echo "[AUTOBOT] [INFO] DATABASE_URL not set; skipping database migration."
fi

echo "[AUTOBOT] Executing application command: $@"
exec "$@"
