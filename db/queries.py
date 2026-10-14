"""
db/queries.py — All SQL query functions for AutoBot

Each function:
  1. Checks the TTL cache first (fast path, ~0ms)
  2. On cache miss: queries PostgreSQL via the connection pool
  3. Normalises the result into the same dict format used by the old JSON DB
  4. Stores result in cache before returning

This means tools/car_tools.py functions keep the EXACT same signatures —
the agents don't know or care whether data comes from JSON or PostgreSQL.
"""

from __future__ import annotations
from typing import Optional
from .connection import get_conn, get_cache


# ─────────────────────────────────────────────
# Cache key constants
# ─────────────────────────────────────────────
_CACHE_ALL_CARS       = "db:all_cars"
_CACHE_SERVICE_INT    = "db:service_intervals"
_CACHE_ISSUES_PREFIX  = "db:issue:"
_CACHE_ALL_ISSUES     = "db:all_issues"
_CACHE_CAR_NAME_PFX   = "db:car_name:"
_CACHE_BUDGET_PFX     = "db:budget:"
_CACHE_FUEL_PFX       = "db:fuel:"
_CACHE_SEGMENT_PFX    = "db:segment:"


# ─────────────────────────────────────────────
# Internal: row → dict normaliser
# Converts a DB row into the same dict structure
# that the old JSON file returned.
# ─────────────────────────────────────────────
def _row_to_car_dict(row: dict) -> dict:
    """
    Converts a PostgreSQL RealDictRow → plain dict matching old JSON structure.

    Old JSON format:
        {
          "id": 1, "name": "...", "brand": "...", "segment": "...",
          "price_lakh": {"min": 6.49, "max": 9.64},
          "fuel_type": ["Petrol", "CNG"],
          "mileage_kmpl": {"petrol": 23.76, "cng": 30.90},
          ...
        }
    """
    return {
        "id":                   row["id"],
        "name":                 row["name"],
        "brand":                row["brand"],
        "segment":              row["segment"],
        "price_lakh": {
            "min": float(row["price_min_lakh"]),
            "max": float(row["price_max_lakh"]),
        },
        "fuel_type":            list(row["fuel_types"] or []),
        "mileage_kmpl":         dict(row["mileage_data"] or {}),
        "engine_cc":            row["engine_cc"],
        "seating":              row["seating"],
        "transmission":         list(row["transmission"] or []),
        "features":             list(row["features"] or []),
        "service_interval_km":  row["service_interval_km"],
        "popular_in":           list(row["popular_in"] or []),
        "best_for":             list(row["best_for"] or []),
    }


# ─────────────────────────────────────────────
# Query: Get All Cars
# ─────────────────────────────────────────────
def db_get_all_cars() -> list[dict]:
    """
    Returns all cars from PostgreSQL.
    Cache key: 'db:all_cars', TTL: 5 min.
    """
    cache = get_cache()
    cached = cache.get(_CACHE_ALL_CARS)
    if cached is not None:
        return cached

    with get_conn() as conn:
        with conn.cursor() as cur:
            cur.execute("SELECT * FROM cars ORDER BY id")
            rows = cur.fetchall()

    result = [_row_to_car_dict(dict(r)) for r in rows]
    cache.set(_CACHE_ALL_CARS, result)
    return result


# ─────────────────────────────────────────────
# Query: Filter by Budget
# ─────────────────────────────────────────────
def db_filter_by_budget(max_budget_lakh: float) -> list[dict]:
    """
    Returns cars where price_min_lakh <= max_budget_lakh.
    Cache key: 'db:budget:{max_budget_lakh}'.
    """
    cache = get_cache()
    key = f"{_CACHE_BUDGET_PFX}{max_budget_lakh}"
    cached = cache.get(key)
    if cached is not None:
        return cached

    with get_conn() as conn:
        with conn.cursor() as cur:
            cur.execute(
                "SELECT * FROM cars WHERE price_min_lakh <= %s ORDER BY price_min_lakh",
                (max_budget_lakh,)
            )
            rows = cur.fetchall()

    result = [_row_to_car_dict(dict(r)) for r in rows]
    cache.set(key, result)
    return result


# ─────────────────────────────────────────────
# Query: Filter by Fuel Type
# ─────────────────────────────────────────────
def db_filter_by_fuel(fuel_type: str) -> list[dict]:
    """
    Returns cars that support the given fuel type (case-insensitive).
    Uses PostgreSQL array operator: %s = ANY(fuel_types).
    Cache key: 'db:fuel:{fuel_type}'.
    """
    cache = get_cache()
    fuel_normalised = fuel_type.title()
    key = f"{_CACHE_FUEL_PFX}{fuel_normalised}"
    cached = cache.get(key)
    if cached is not None:
        return cached

    with get_conn() as conn:
        with conn.cursor() as cur:
            cur.execute(
                "SELECT * FROM cars WHERE %s = ANY(fuel_types) ORDER BY id",
                (fuel_normalised,)
            )
            rows = cur.fetchall()

    result = [_row_to_car_dict(dict(r)) for r in rows]
    cache.set(key, result)
    return result


# ─────────────────────────────────────────────
# Query: Filter by Segment
# ─────────────────────────────────────────────
def db_filter_by_segment(segment: str) -> list[dict]:
    """
    Returns cars matching the segment (case-insensitive ILIKE).
    Cache key: 'db:segment:{segment}'.
    """
    cache = get_cache()
    key = f"{_CACHE_SEGMENT_PFX}{segment.lower()}"
    cached = cache.get(key)
    if cached is not None:
        return cached

    with get_conn() as conn:
        with conn.cursor() as cur:
            cur.execute(
                "SELECT * FROM cars WHERE segment ILIKE %s ORDER BY id",
                (f"%{segment}%",)
            )
            rows = cur.fetchall()

    result = [_row_to_car_dict(dict(r)) for r in rows]
    cache.set(key, result)
    return result


# ─────────────────────────────────────────────
# Query: Get Car by Name
# ─────────────────────────────────────────────
def db_get_car_by_name(name: str) -> Optional[dict]:
    """
    Returns the first car whose name contains the search term (ILIKE).
    Cache key: 'db:car_name:{name.lower()}'.
    """
    cache = get_cache()
    key = f"{_CACHE_CAR_NAME_PFX}{name.lower()}"
    cached = cache.get(key)
    if cached is not None:
        return cached if cached != "__NONE__" else None

    with get_conn() as conn:
        with conn.cursor() as cur:
            cur.execute(
                "SELECT * FROM cars WHERE name ILIKE %s LIMIT 1",
                (f"%{name}%",)
            )
            row = cur.fetchone()

    if row is None:
        cache.set(key, "__NONE__")   # cache the miss too
        return None

    result = _row_to_car_dict(dict(row))
    cache.set(key, result)
    return result


# ─────────────────────────────────────────────
# Query: Get Common Issue Info
# ─────────────────────────────────────────────
def db_get_common_issue(keyword: str) -> Optional[dict]:
    """
    Finds a common issue by keyword (tries exact match, then partial ILIKE).
    Returns dict matching old JSON structure, or None if not found.
    Cache key: 'db:issue:{keyword.lower()}'.
    """
    cache = get_cache()
    key = f"{_CACHE_ISSUES_PREFIX}{keyword.lower()}"
    cached = cache.get(key)
    if cached is not None:
        return cached if cached != "__NONE__" else None

    normalised = keyword.lower().replace(" ", "_")

    with get_conn() as conn:
        with conn.cursor() as cur:
            # 1. Exact match on issue_key
            cur.execute(
                "SELECT * FROM common_issues WHERE issue_key = %s",
                (normalised,)
            )
            row = cur.fetchone()

            if row is None:
                # 2. Partial ILIKE match
                cur.execute(
                    "SELECT * FROM common_issues WHERE issue_key ILIKE %s LIMIT 1",
                    (f"%{normalised}%",)
                )
                row = cur.fetchone()

    if row is None:
        cache.set(key, "__NONE__")
        return None

    row = dict(row)
    # Normalise to match old JSON shape expected by car_tools.py
    result = {
        "possible_causes": list(row.get("causes") or []),
        "most_likely_cause": row.get("most_likely", ""),
        "severity": row.get("severity", "Medium"),
        "estimated_cost_inr": _parse_cost_range(row.get("cost_range", "")),
    }
    cache.set(key, result)
    return result


def _parse_cost_range(cost_str: str) -> dict:
    """
    Parses '₹300 - ₹1500' → {"min": 300, "max": 1500}.
    Falls back to {"min": 1000, "max": 5000} on parse failure.
    """
    import re
    nums = re.findall(r'\d+', cost_str.replace(",", ""))
    if len(nums) >= 2:
        return {"min": int(nums[0]), "max": int(nums[1])}
    return {"min": 1000, "max": 5000}


def db_get_all_issues() -> dict:
    """
    Returns all common issues as a dict keyed by issue_key.
    Cache key: 'db:all_issues', TTL: 5 min.
    """
    cache = get_cache()
    cached = cache.get(_CACHE_ALL_ISSUES)
    if cached is not None:
        return cached

    with get_conn() as conn:
        with conn.cursor() as cur:
            cur.execute("SELECT * FROM common_issues ORDER BY id")
            rows = cur.fetchall()

    result = {
        row["issue_key"]: {
            "causes": list(row.get("causes") or []),
            "most_likely": row.get("most_likely", ""),
            "severity": row.get("severity", "Medium"),
            "cost_range": row.get("cost_range", ""),
        }
        for row in rows
    }
    cache.set(_CACHE_ALL_ISSUES, result)
    return result


# ─────────────────────────────────────────────
# Query: Get Service Intervals
# ─────────────────────────────────────────────
def db_get_service_intervals() -> dict:
    """
    Returns all service intervals as a dict keyed by item_key.
    Cache key: 'db:service_intervals', TTL: 5 min.

    Return format (same as old JSON):
        {
          "engine_oil": {"interval_km": 10000, "cost": "₹1500 - ₹3000"},
          "air_filter": {"interval_km": 20000, "cost": "₹300 - ₹800"},
          ...
        }
    """
    cache = get_cache()
    cached = cache.get(_CACHE_SERVICE_INT)
    if cached is not None:
        return cached

    with get_conn() as conn:
        with conn.cursor() as cur:
            cur.execute("SELECT item_key, interval_km, cost FROM service_intervals ORDER BY id")
            rows = cur.fetchall()

    result = {
        row["item_key"]: {
            "interval_km": row["interval_km"],
            "cost": row["cost"],
        }
        for row in rows
    }
    cache.set(_CACHE_SERVICE_INT, result)
    return result


# ─────────────────────────────────────────────
# Queries: Conversation & Chat History Persistence
# ─────────────────────────────────────────────

def db_create_conversation(user_id: int, session_id: str, title: str) -> dict:
    """Creates a new conversation entry for a user or updates an existing title."""
    title_clean = title.strip()[:200] if title.strip() else "New Conversation"
    with get_conn() as conn:
        with conn.cursor() as cur:
            cur.execute(
                """
                INSERT INTO conversations (session_id, user_id, title)
                VALUES (%s, %s, %s)
                ON CONFLICT (session_id) DO UPDATE
                SET title = EXCLUDED.title, updated_at = NOW()
                RETURNING id, session_id, user_id, title, created_at, updated_at
                """,
                (session_id, user_id, title_clean)
            )
            row = cur.fetchone()
            return dict(row)


def db_save_chat_message(user_id: int, session_id: str, role: str, content: str, intent: str = "") -> dict:
    """Saves a user or assistant message to the chat_messages table."""
    with get_conn() as conn:
        with conn.cursor() as cur:
            cur.execute(
                """
                INSERT INTO chat_messages (session_id, user_id, role, content, intent)
                VALUES (%s, %s, %s, %s, %s)
                RETURNING id, session_id, user_id, role, content, intent, created_at
                """,
                (session_id, user_id, role, content, intent)
            )
            row = cur.fetchone()
            cur.execute(
                "UPDATE conversations SET updated_at = NOW() WHERE session_id = %s",
                (session_id,)
            )
            return dict(row)


def db_get_user_conversations(user_id: int) -> list[dict]:
    """Returns all chat conversations for a specific user ordered by newest first."""
    with get_conn() as conn:
        with conn.cursor() as cur:
            cur.execute(
                """
                SELECT c.session_id, c.title, c.created_at, c.updated_at,
                       COUNT(m.id) AS message_count
                FROM conversations c
                LEFT JOIN chat_messages m ON c.session_id = m.session_id
                WHERE c.user_id = %s
                GROUP BY c.id, c.session_id, c.title, c.created_at, c.updated_at
                ORDER BY c.updated_at DESC
                """,
                (user_id,)
            )
            rows = cur.fetchall()
            return [dict(r) for r in rows]


def db_get_conversation_messages(session_id: str, user_id: int) -> list[dict]:
    """Returns all messages in a conversation for a specific user ordered chronologically."""
    with get_conn() as conn:
        with conn.cursor() as cur:
            cur.execute(
                """
                SELECT role, content, intent, created_at
                FROM chat_messages
                WHERE session_id = %s AND user_id = %s
                ORDER BY id ASC
                """,
                (session_id, user_id)
            )
            rows = cur.fetchall()
            return [dict(r) for r in rows]


def db_delete_conversation(session_id: str, user_id: int) -> bool:
    """Deletes a conversation and its messages for a user."""
    with get_conn() as conn:
        with conn.cursor() as cur:
            cur.execute(
                "DELETE FROM conversations WHERE session_id = %s AND user_id = %s",
                (session_id, user_id)
            )
            return cur.rowcount > 0

