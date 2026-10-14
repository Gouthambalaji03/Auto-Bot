"""
db/connection.py — PostgreSQL connection pool + in-memory TTL cache

Architecture:
    ┌─────────────────────────────────────────────────────────┐
    │  Tool call (e.g. get_all_cars)                          │
    │       ↓                                                 │
    │  TTLCache.get(key)                                      │
    │       ↓ HIT → return instantly (0 DB call, ~0ms)        │
    │       ↓ MISS                                            │
    │  ThreadedConnectionPool.getconn()                       │
    │       ↓                                                 │
    │  Execute SQL query                                      │
    │       ↓                                                 │
    │  pool.putconn(conn)  ← return conn to pool             │
    │       ↓                                                 │
    │  TTLCache.set(key, result, ttl=300)  ← cache it        │
    │       ↓                                                 │
    │  return result                                          │
    └─────────────────────────────────────────────────────────┘

Pool sizing:
    minconn=2  → 2 connections always warm and ready
    maxconn=10 → scales up to 10 under concurrent load

TTL cache:
    Default TTL = 300 seconds (5 minutes)
    Car data rarely changes at runtime, so 5-min cache is ideal.
    Force-refresh any key by calling cache.delete(key).
"""

import os
import time
import threading
from typing import Any, Optional
from contextlib import contextmanager

from dotenv import load_dotenv
load_dotenv(override=True)

# Lazy import — only fails if psycopg2-binary not installed
try:
    import psycopg2
    import psycopg2.pool
    import psycopg2.extras   # for RealDictCursor (rows as dicts)
    PSYCOPG2_AVAILABLE = True
except ImportError:
    PSYCOPG2_AVAILABLE = False


# ─────────────────────────────────────────────
# TTL In-Memory Cache
# ─────────────────────────────────────────────

class TTLCache:
    """
    Thread-safe in-memory cache with per-key TTL expiry.

    Usage:
        cache = TTLCache(default_ttl=300)
        cache.set("all_cars", data)          # store with 5-min TTL
        cache.get("all_cars")                # returns data or None if expired
        cache.delete("all_cars")             # force-invalidate
        cache.clear()                        # flush everything
    """

    def __init__(self, default_ttl: int = 300):
        self._store: dict[str, tuple[Any, float]] = {}   # key → (value, expire_at)
        self._lock = threading.RLock()
        self.default_ttl = default_ttl

    def get(self, key: str) -> Optional[Any]:
        with self._lock:
            entry = self._store.get(key)
            if entry is None:
                return None
            value, expire_at = entry
            if time.monotonic() > expire_at:
                del self._store[key]   # expired — evict
                return None
            return value

    def set(self, key: str, value: Any, ttl: Optional[int] = None) -> None:
        ttl = ttl if ttl is not None else self.default_ttl
        expire_at = time.monotonic() + ttl
        with self._lock:
            self._store[key] = (value, expire_at)

    def delete(self, key: str) -> None:
        with self._lock:
            self._store.pop(key, None)

    def clear(self) -> None:
        with self._lock:
            self._store.clear()

    @property
    def size(self) -> int:
        with self._lock:
            return len(self._store)


# ─────────────────────────────────────────────
# PostgreSQL Connection Pool Singleton
# ─────────────────────────────────────────────

_pool: Optional["psycopg2.pool.ThreadedConnectionPool"] = None
_pool_lock = threading.Lock()

_cache: Optional[TTLCache] = None
_cache_lock = threading.Lock()


def validate_db_config() -> None:
    """Call at app startup — fails fast if DB is not configured."""
    if not os.getenv("DATABASE_URL"):
        raise RuntimeError(
            "DATABASE_URL is required. Set it in your .env file.\n"
            "Run 'python db/migrate.py' to set up the database."
        )


def get_pool() -> "psycopg2.pool.ThreadedConnectionPool":
    """
    Returns the singleton ThreadedConnectionPool.
    Creates it on first call (lazy init).

    ThreadedConnectionPool is safe for multi-threaded access.
    Each thread borrows a connection, uses it, and returns it.

    Raises:
        RuntimeError: if DATABASE_URL is not set or psycopg2 not installed
        psycopg2.OperationalError: if PostgreSQL is unreachable
    """
    global _pool
    if _pool is not None:
        return _pool

    with _pool_lock:
        if _pool is not None:   # double-checked locking
            return _pool

        if not PSYCOPG2_AVAILABLE:
            raise RuntimeError(
                "psycopg2-binary is not installed. "
                "Run: pip install psycopg2-binary"
            )

        database_url = os.getenv("DATABASE_URL")
        if not database_url:
            raise RuntimeError(
                "DATABASE_URL not set in environment.\n"
                "Add to your .env file:\n"
                "  DATABASE_URL=postgresql://user:password@localhost:5432/autobot_db\n"
                "\nFree cloud options:\n"
                "  • Supabase: https://supabase.com  (500MB free)\n"
                "  • Neon:     https://neon.tech      (512MB free)\n"
                "  • Render:   https://render.com     (90 days free)"
            )

        _pool = psycopg2.pool.ThreadedConnectionPool(
            minconn=2,          # always keep 2 connections warm
            maxconn=10,         # scale up to 10 under load
            dsn=database_url,
            cursor_factory=psycopg2.extras.RealDictCursor,   # rows → dicts
        )
        print(f"✅ [DB POOL] PostgreSQL connection pool created (min=2, max=10)")
        return _pool


def get_cache() -> TTLCache:
    """
    Returns the singleton TTLCache (5-minute TTL by default).
    """
    global _cache
    if _cache is not None:
        return _cache
    with _cache_lock:
        if _cache is None:
            _cache = TTLCache(default_ttl=300)
    return _cache


@contextmanager
def get_conn():
    """
    Context manager that borrows a connection from the pool,
    yields it, and returns it when done.

    Usage:
        with get_conn() as conn:
            with conn.cursor() as cur:
                cur.execute("SELECT ...")
                rows = cur.fetchall()

    Automatically handles:
        - Returning connection to pool on success
        - Rolling back + returning on exception
    """
    pool = get_pool()
    conn = pool.getconn()
    try:
        yield conn
        conn.commit()
    except Exception:
        conn.rollback()
        raise
    finally:
        pool.putconn(conn)


import atexit

def close_pool() -> None:
    """Cleanly shut down the pool (call on app exit)."""
    global _pool
    if _pool is not None:
        _pool.closeall()
        _pool = None
        print("🔌 [DB POOL] PostgreSQL connection pool closed")

# Register cleanup on process exit
atexit.register(close_pool)
