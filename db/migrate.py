"""
db/migrate.py — One-Time Migration: Seed Data → PostgreSQL

Usage:
    python db/migrate.py

What it does:
    1. Reads DATABASE_URL from .env
    2. Creates 3 tables with indexes and timestamps (idempotent — safe to re-run):
       • cars
       • common_issues
       • service_intervals
    3. Loads data/seed.json
    4. Inserts all seed rows (ON CONFLICT DO NOTHING — no duplicates)
    5. Updates sequence generators and prints a summary
"""

import json
import os
import sys
from pathlib import Path

# Make sure we can import dotenv from the project root
sys.path.insert(0, str(Path(__file__).parent.parent))

from dotenv import load_dotenv
load_dotenv()

import psycopg2
import psycopg2.extras


# ─────────────────────────────────────────────
# SQL DDL — Create Tables & Indexes
# ─────────────────────────────────────────────

CREATE_CARS_TABLE = """
CREATE TABLE IF NOT EXISTS cars (
    id                  SERIAL PRIMARY KEY,
    name                VARCHAR(200) NOT NULL,
    brand               VARCHAR(100) NOT NULL,
    segment             VARCHAR(100),
    price_min_lakh      DECIMAL(7, 2),
    price_max_lakh      DECIMAL(7, 2),
    fuel_types          TEXT[],
    mileage_data        JSONB,
    engine_cc           INTEGER,
    seating             INTEGER,
    transmission        TEXT[],
    features            TEXT[],
    service_interval_km INTEGER,
    popular_in          TEXT[],
    best_for            TEXT[],
    image_url           TEXT,
    created_at          TIMESTAMPTZ DEFAULT NOW(),
    updated_at          TIMESTAMPTZ DEFAULT NOW()
);

CREATE INDEX IF NOT EXISTS idx_cars_price_min ON cars (price_min_lakh);
CREATE INDEX IF NOT EXISTS idx_cars_price_max ON cars (price_max_lakh);
CREATE INDEX IF NOT EXISTS idx_cars_fuel_gin  ON cars USING GIN (fuel_types);
CREATE INDEX IF NOT EXISTS idx_cars_brand     ON cars (brand);
CREATE INDEX IF NOT EXISTS idx_cars_segment   ON cars (segment);
"""

CREATE_ISSUES_TABLE = """
CREATE TABLE IF NOT EXISTS common_issues (
    id          SERIAL PRIMARY KEY,
    issue_key   VARCHAR(120) UNIQUE NOT NULL,
    causes      TEXT[],
    most_likely VARCHAR(250),
    severity    VARCHAR(20),
    cost_range  VARCHAR(60),
    created_at  TIMESTAMPTZ DEFAULT NOW(),
    updated_at  TIMESTAMPTZ DEFAULT NOW()
);
"""

CREATE_SERVICE_TABLE = """
CREATE TABLE IF NOT EXISTS service_intervals (
    id          SERIAL PRIMARY KEY,
    item_key    VARCHAR(120) UNIQUE NOT NULL,
    interval_km INTEGER,
    cost        VARCHAR(60),
    created_at  TIMESTAMPTZ DEFAULT NOW(),
    updated_at  TIMESTAMPTZ DEFAULT NOW()
);
"""

CREATE_USERS_TABLE = """
CREATE TABLE IF NOT EXISTS users (
    id            SERIAL PRIMARY KEY,
    username      VARCHAR(80) UNIQUE NOT NULL,
    email         VARCHAR(255) UNIQUE NOT NULL,
    password_hash VARCHAR(255) NOT NULL,
    created_at    TIMESTAMPTZ DEFAULT NOW(),
    updated_at    TIMESTAMPTZ DEFAULT NOW()
);

CREATE INDEX IF NOT EXISTS idx_users_username ON users (username);
CREATE INDEX IF NOT EXISTS idx_users_email ON users (email);
"""

CREATE_CONVERSATIONS_TABLE = """
CREATE TABLE IF NOT EXISTS conversations (
    id          SERIAL PRIMARY KEY,
    session_id  VARCHAR(64) UNIQUE NOT NULL,
    user_id     INTEGER NOT NULL REFERENCES users(id) ON DELETE CASCADE,
    title       VARCHAR(255) NOT NULL,
    created_at  TIMESTAMPTZ DEFAULT NOW(),
    updated_at  TIMESTAMPTZ DEFAULT NOW()
);

CREATE INDEX IF NOT EXISTS idx_conversations_user_id ON conversations (user_id);
"""

CREATE_MESSAGES_TABLE = """
CREATE TABLE IF NOT EXISTS chat_messages (
    id          SERIAL PRIMARY KEY,
    session_id  VARCHAR(64) NOT NULL REFERENCES conversations(session_id) ON DELETE CASCADE,
    user_id     INTEGER NOT NULL REFERENCES users(id) ON DELETE CASCADE,
    role        VARCHAR(20) NOT NULL,
    content     TEXT NOT NULL,
    intent      VARCHAR(50),
    created_at  TIMESTAMPTZ DEFAULT NOW()
);

CREATE INDEX IF NOT EXISTS idx_chat_messages_session_id ON chat_messages (session_id);
"""


# ─────────────────────────────────────────────
# SQL DML — Insert Data
# ─────────────────────────────────────────────

INSERT_CAR = """
INSERT INTO cars (
    id, name, brand, segment,
    price_min_lakh, price_max_lakh,
    fuel_types, mileage_data, engine_cc, seating,
    transmission, features, service_interval_km,
    popular_in, best_for
)
VALUES (
    %(id)s, %(name)s, %(brand)s, %(segment)s,
    %(price_min)s, %(price_max)s,
    %(fuel_types)s, %(mileage_data)s, %(engine_cc)s, %(seating)s,
    %(transmission)s, %(features)s, %(service_interval_km)s,
    %(popular_in)s, %(best_for)s
)
ON CONFLICT (id) DO NOTHING;
"""

INSERT_ISSUE = """
INSERT INTO common_issues (issue_key, causes, most_likely, severity, cost_range)
VALUES (%(issue_key)s, %(causes)s, %(most_likely)s, %(severity)s, %(cost_range)s)
ON CONFLICT (issue_key) DO NOTHING;
"""

INSERT_SERVICE = """
INSERT INTO service_intervals (item_key, interval_km, cost)
VALUES (%(item_key)s, %(interval_km)s, %(cost)s)
ON CONFLICT (item_key) DO NOTHING;
"""


# ─────────────────────────────────────────────
# Main Migration
# ─────────────────────────────────────────────

def migrate():
    database_url = os.getenv("DATABASE_URL")
    if not database_url:
        print("❌  DATABASE_URL not set. Add it to your .env file:")
        print("    DATABASE_URL=postgresql://user:password@localhost:5432/autobot_db")
        print()
        print("  Free cloud PostgreSQL options:")
        print("  • Supabase → https://supabase.com     (500 MB free)")
        print("  • Neon     → https://neon.tech         (512 MB free)")
        print("  • Render   → https://render.com        (90-day free tier)")
        sys.exit(1)

    # Load seed JSON source
    json_path = Path(__file__).parent.parent / "data" / "seed.json"
    if not json_path.exists():
        print(f"❌  Seed data JSON not found at {json_path}")
        sys.exit(1)

    with open(json_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    print("🔌  Connecting to PostgreSQL...")
    conn = psycopg2.connect(database_url, cursor_factory=psycopg2.extras.RealDictCursor)
    cur = conn.cursor()

    # ── Create tables & indexes ────────────────────────────────
    print("🏗️   Creating tables and indexes (IF NOT EXISTS)...")
    cur.execute(CREATE_CARS_TABLE)
    cur.execute(CREATE_ISSUES_TABLE)
    cur.execute(CREATE_SERVICE_TABLE)
    cur.execute(CREATE_USERS_TABLE)
    cur.execute(CREATE_CONVERSATIONS_TABLE)
    cur.execute(CREATE_MESSAGES_TABLE)
    conn.commit()
    print("    ✓ cars, common_issues, service_intervals, users, conversations, chat_messages tables ready")

    # ── Insert cars ────────────────────────────────────────────
    cars = data.get("cars", [])
    cars_inserted = 0
    print(f"\n🚗  Migrating {len(cars)} cars...")
    for car in cars:
        import json as json_lib
        cur.execute(INSERT_CAR, {
            "id":                   car["id"],
            "name":                 car["name"],
            "brand":                car["brand"],
            "segment":              car.get("segment"),
            "price_min":            car["price_lakh"]["min"],
            "price_max":            car["price_lakh"]["max"],
            "fuel_types":           car.get("fuel_type", []),
            "mileage_data":         json_lib.dumps(car.get("mileage_kmpl", {})),
            "engine_cc":            car.get("engine_cc"),
            "seating":              car.get("seating", 5),
            "transmission":         car.get("transmission", []),
            "features":             car.get("features", []),
            "service_interval_km":  car.get("service_interval_km", 10000),
            "popular_in":           car.get("popular_in", []),
            "best_for":             car.get("best_for", []),
        })
        cars_inserted += 1
        print(f"    ✓ [{car['id']:02d}] {car['name']}")
    conn.commit()

    # Synchronize sequence generator for cars.id
    cur.execute("SELECT setval(pg_get_serial_sequence('cars', 'id'), COALESCE(MAX(id), 1)) FROM cars;")
    conn.commit()

    # ── Insert common issues ───────────────────────────────────
    issues = data.get("common_issues", {})
    issues_inserted = 0
    print(f"\n🔧  Migrating {len(issues)} common issues...")
    for key, val in issues.items():
        cur.execute(INSERT_ISSUE, {
            "issue_key":   key,
            "causes":      val.get("causes", []),
            "most_likely": val.get("most_likely", ""),
            "severity":    val.get("severity", "Medium"),
            "cost_range":  val.get("cost_range", ""),
        })
        issues_inserted += 1
        print(f"    ✓ {key}")
    conn.commit()

    # ── Insert service intervals ───────────────────────────────
    intervals = data.get("service_intervals", {})
    svc_inserted = 0
    print(f"\n📅  Migrating {len(intervals)} service intervals...")
    for key, val in intervals.items():
        cur.execute(INSERT_SERVICE, {
            "item_key":    key,
            "interval_km": val.get("interval_km"),
            "cost":        val.get("cost", ""),
        })
        svc_inserted += 1
        print(f"    ✓ {key} — every {val['interval_km']:,} km")
    conn.commit()

    # ── Verify row counts ──────────────────────────────────────
    cur.execute("SELECT COUNT(*) AS n FROM cars")
    car_count = cur.fetchone()["n"]
    cur.execute("SELECT COUNT(*) AS n FROM common_issues")
    issue_count = cur.fetchone()["n"]
    cur.execute("SELECT COUNT(*) AS n FROM service_intervals")
    svc_count = cur.fetchone()["n"]

    cur.close()
    conn.close()

    print("\n" + "─" * 50)
    print("✅  Migration complete!")
    print(f"   • cars:              {car_count} rows")
    print(f"   • common_issues:     {issue_count} rows")
    print(f"   • service_intervals: {svc_count} rows")
    print("─" * 50)
    print("\n▶  You can now start AutoBot: python main.py")


if __name__ == "__main__":
    migrate()
