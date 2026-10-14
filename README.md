# AutoBot — AI Automobile Assistant 🚗⚡

[![Python 3.10+](https://img.shields.io/badge/Python-3.10%2B-blue.svg)](https://www.python.org/)
[![Pydantic AI](https://img.shields.io/badge/Agent-Pydantic%20AI-e92063.svg)](https://ai.pydantic.dev/)
[![Model](https://img.shields.io/badge/LLM-Gemini%203.6%20Flash-4285F4.svg)](https://aistudio.google.com/)
[![FastAPI](https://img.shields.io/badge/API-FastAPI-009688.svg)](https://fastapi.tiangolo.com/)
[![Gradio UI](https://img.shields.io/badge/UI-Gradio-ff7c00.svg)](https://www.gradio.app/)
[![Database](https://img.shields.io/badge/Database-PostgreSQL%2016-336791.svg)](https://www.postgresql.org/)
[![Docker](https://img.shields.io/badge/Docker-Ready-2496ED.svg)](https://www.docker.com/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

**AutoBot** is an enterprise-grade, full-stack automotive AI assistant engineered specifically for the **Indian automobile market**. Powered by **Pydantic AI**, **Google Gemini 3.6 Flash**, **FastAPI**, **Gradio**, and **PostgreSQL**, AutoBot seamlessly delivers vehicle recommendations, safety-first diagnostic triage, maintenance schedules, and deterministic loan EMI calculations within a single unified conversation.

---

## 🌟 Key Capabilities

- **🧠 LLM-Led Intent Classification**: Dynamically routes user requests into single or combined intents (`buying`, `diagnostics`, `service`, `finance`, `general`) without brittle regex or keyword routers.
- **🏷️ Modern Car Recommendation Card View**: Formats vehicle recommendations into interactive HTML cards featuring category badges, ex-showroom price pills, responsive 4-column specifications grids, feature tags, and buyer suitability callouts.
- **⚡ Prefix KV-Caching**: Sized to ~4,500 prompt tokens (surpassing Gemini's 4,096-token KV-cache threshold). Delivers ~90% cost reduction and ultra-low Time-To-First-Token (TTFT) on multi-turn dialogues.
- **🛡️ Safe Tool Boundary**: The LLM calls parameterised Python tools for database lookups and EMI arithmetic. Never executes raw SQL or accesses database credentials directly.
- **⚡ Real-Time Token Streaming**: Event-driven token emission via `run_stream_events` feeds both the Gradio UI and FastAPI Server-Sent Events (SSE) stream simultaneously.
- **🚨 Diagnostic Safety Policy**: Enforces strict automotive safety guidelines. Flags severe drivability hazards (brake failure, steering loss, overheating, smoke) with immediate "do not drive" warnings.
- **🗄️ PostgreSQL Persistence & History**: Multi-user authentication (salted PBKDF2-HMAC-SHA256 password hashing), session persistence, and instant conversation reload.
- **⚡ 5-Minute In-Memory TTL Cache**: Thread-safe in-memory caching (`TTLCache`) eliminates redundant database queries for frequent vehicle catalogue searches (~0ms latency).
- **🔍 RapidFuzz Fuzzy Fallback Engine**: Uses token-set ratio matching to normalize slang, typos, and natural-language symptoms (e.g. *"car shaking at high speed"*) against catalogued records.
- **🌐 Dual Server Support**: Run the complete Gradio Web UI (`:7860`), standalone FastAPI server (`:8000`), or both via Docker Compose.

---

## 🏗️ Architecture & Workflow

```text
┌──────────────────────────────────────┐       ┌──────────────────────────────────────┐
│        Gradio Web UI (:7860)         │       │     FastAPI REST/SSE API (:8000)     │
│   • Modern Glassmorphic SPA Layout   │       │   • POST /chat (SSE Token Stream)    │
│   • Interactive Car Cards & History  │       │   • POST /chat/sync (Full JSON)      │
│   • Built-in Auth & Session Manager  │       │   • POST /auth/* & GET /sessions/*   │
└──────────────────┬───────────────────┘       └──────────────────┬───────────────────┘
                   │                                              │
                   └──────────────────────┬───────────────────────┘
                                          ▼
                             ┌────────────────────────┐
                             │   Pydantic AI Agent    │
                             │   (Gemini 3.6 Flash)   │
                             │  KV Prefix Cached (>4k)│
                             └────────────┬───────────┘
                                          │ LLM-Selected Tools
                                          ▼
┌────────────────────────────────────────────────────────────────────────────────────────┐
│ Safe Tool Execution Boundary (Parameterised SQL & Deterministic Math)                  │
│  • record_intent_classification     • search_catalog_by_segment   • get_vehicle        │
│  • search_catalog                   • search_catalog_by_fuel      • search_known_issue │
│  • search_catalog_by_budget         • calculate_loan_emi          • get_service_ivs    │
└─────────────────────────────────────────┬──────────────────────────────────────────────┘
                                          ▼
                             ┌────────────────────────┐
                             │  PostgreSQL + TTLCache │
                             │   (Pool: min 2, max 10)│
                             └────────────────────────┘
```

---

## 📁 Project Structure

```text
Auto-Bot/
├── main.py                     # Primary Application (Gradio UI on :7860)
├── api_server.py               # Standalone FastAPI Server (REST & SSE on :8000)
├── run_all.py                  # Multi-service launcher (runs UI + API concurrently)
├── Dockerfile                  # Multi-stage production Python container
├── docker-compose.yml          # Full-stack orchestrator (PostgreSQL 16 + AutoBot App)
├── requirements.txt            # Production Python dependencies
├── .env.example                # Environment variable configuration template
│
├── api/                        # ── FastAPI REST & SSE Layer ─────────────────────────
│   ├── app.py                  # App factory, lifespan context manager, CORS, middleware
│   ├── schemas.py              # Pydantic v2 request and response models
│   ├── dependencies.py         # Async thread-pool wrappers for synchronous DB queries
│   ├── logger.py               # Centralized structured logging configuration
│   └── routers/
│       ├── health.py           # GET /health (Liveness & readiness probe)
│       ├── chat.py             # POST /chat (SSE Stream) & POST /chat/sync
│       ├── auth.py             # POST /auth/signup & POST /auth/login
│       └── sessions.py         # GET /sessions/{session_id}/history
│
├── agents/
│   └── automotive_agent.py     # Pydantic AI Agent, cached prompt, tool registrations
│
├── tools/
│   └── car_tools.py            # Catalogue retrieval helpers & EMI loan calculations
│
├── db/
│   ├── connection.py           # ThreadedConnectionPool + in-memory TTLCache
│   ├── queries.py              # Parameterised SQL queries & conversation persistence
│   ├── fuzzy_queries.py        # RapidFuzz search engine, alias maps, and synonym logic
│   ├── auth.py                 # PBKDF2 password hashing & authentication
│   └── migrate.py              # DDL migrations & seed catalog importer
│
├── models/
│   └── schemas.py              # QueryPlan and FuzzyCarFilter domain models
│
├── ui/
│   └── app.py                  # Gradio UI components, Car Card formatter, and CSS
│
├── data/
│   └── seed.json               # Curated Indian automotive seed database
│
├── scripts/
│   ├── docker-entrypoint.sh    # Docker entrypoint (auto-migration & service execution)
│   └── smoke_test.sh           # End-to-end API & UI smoke test suite
│
└── tests/                      # ── Automated Test Suite (pytest) ───────────────────
    ├── conftest.py             # Test environment path & dependency configuration
    ├── test_api.py             # FastAPI endpoint integration tests
    ├── test_cards.py           # Car Recommendation Card View formatting tests
    ├── test_emi.py             # Loan EMI mathematical precision tests
    └── test_fuzzy.py           # RapidFuzz synonym & diagnostic matching tests
```

---

## 🚀 Quick Start Guide

### 1. Prerequisites
- **Python 3.10+**
- **Docker Desktop** — [Download Docker](https://www.docker.com/products/docker-desktop/)
- **Google Gemini API Key** — [Get a free key from Google AI Studio](https://aistudio.google.com/app/apikey)

---

### 2. Option A: Full Stack with Docker Compose (Recommended)

Run everything (PostgreSQL, Gradio UI, and FastAPI API) with a single command:

```bash
# 1. Clone the repository
git clone https://github.com/your-username/Auto-Bot.git
cd Auto-Bot

# 2. Configure environment variables
cp .env.example .env
# Open .env and add your GEMINI_API_KEY
```

```bash
# 3. Build and launch all services in background
docker compose up -d --build
```

That's it! AutoBot will:
1. Start PostgreSQL 16 on port `5433` with healthchecks.
2. Automatically run database migrations and seed catalogue data.
3. Start the **Gradio Web UI** on **`http://localhost:7860`**.
4. Start the **FastAPI REST/SSE Server** on **`http://localhost:8000`** (Swagger docs at **`http://localhost:8000/docs`**).

#### Daily Docker Management:
```bash
docker compose logs -f app      # Stream live application logs
docker compose ps               # Check container health status
docker compose down             # Stop services (database data is preserved)
docker compose down -v          # Stop AND wipe database volume (fresh start)
```

---

### 3. Option B: Local Python Environment

#### 1. Setup Virtual Environment:
```bash
python3 -m venv .venv
source .venv/bin/activate    # On Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

#### 2. Start PostgreSQL:
You can start just the database using Docker:
```bash
docker compose up -d db
```

#### 3. Run Database Migrations:
```bash
python db/migrate.py
```

#### 4. Launch Services:
- **Run both UI and API together**:
  ```bash
  python run_all.py
  ```
- **Run Gradio UI only** (`http://localhost:7860`):
  ```bash
  python main.py
  ```
- **Run FastAPI server only** (`http://localhost:8000`):
  ```bash
  python api_server.py
  ```

---

## 🎨 Car Recommendation Card View

AutoBot presents recommended vehicles using structured, interactive automotive cards:

```text
┌────────────────────────────────────────────────────────────────────────┐
│ 🚗 Tata Nexon EV   [ Compact SUV EV ]               [ 💰 ₹14.49L - ₹19.49L ] │
├────────────────────────────────────────────────────────────────────────┤
│  ⛽ FUEL & ENGINE              📊 RANGE / MILEAGE                       │
│  Electric (EV) · 40.5 kWh       465 km per charge                      │
│                                                                        │
│  ⚙️ TRANSMISSION                🛡️ SAFETY RATING                        │
│  Automatic                      5-Star BNCAP, 6 Airbags                │
├────────────────────────────────────────────────────────────────────────┤
│  🌟 KEY FEATURES                                                       │
│  [ Connected Car Tech ]  [ Level 2 ADAS ]  [ Sunroof ]  [ V2L Charger ] │
├────────────────────────────────────────────────────────────────────────┤
│  💡 Best For: Daily urban commuting and eco-conscious family roadtrips │
└────────────────────────────────────────────────────────────────────────┘
```

The card engine converts vehicle markdown blocks automatically into styled `.car-card` DOM structures, ensuring consistent visual appeal across the Gradio UI and REST API clients.

---

## 📡 FastAPI REST & SSE API Reference

| Method | Path | Summary | Description |
|:---|:---|:---|:---|
| `GET` | `/` | Root Status | Overall service health, database check, agent readiness probe |
| `GET` | `/health` | Health Check | Kubernetes / Docker liveness and readiness probe |
| `POST` | `/chat` | SSE Chat Stream | Real-time Server-Sent Events stream emitting tokens and metadata |
| `POST` | `/chat/sync` | Synchronous Chat | Synchronous query returning complete markdown, intent, and timing |
| `POST` | `/auth/signup` | Register User | Creates user account with PBKDF2-hashed password |
| `POST` | `/auth/login` | Login User | Authenticates user credentials and returns session user record |
| `GET` | `/sessions/{id}/history` | Chat History | Fetches ordered conversation turn history for an authenticated user |

### cURL Examples

#### 1. Synchronous Query
```bash
curl -X POST http://localhost:8000/chat/sync \
  -H "Content-Type: application/json" \
  -d '{
    "message": "Recommend top petrol SUVs under 15 Lakhs in India with price and mileage"
  }'
```

#### 2. Server-Sent Events (SSE) Streaming
```bash
curl -N -X POST http://localhost:8000/chat \
  -H "Content-Type: application/json" \
  -d '{
    "message": "Calculate EMI for an 8 Lakh car loan at 9.5% for 5 years with 1.5 Lakh down payment"
  }'
```

#### 3. User Authentication & History
```bash
# Register
curl -X POST http://localhost:8000/auth/signup \
  -H "Content-Type: application/json" \
  -d '{"username":"goutham","email":"goutham@autobot.dev","password":"SecretPassword123"}'

# Login
curl -X POST http://localhost:8000/auth/login \
  -H "Content-Type: application/json" \
  -d '{"username_or_email":"goutham","password":"SecretPassword123"}'
```

---

## 🧪 Testing Suite

AutoBot includes a complete automated test suite powered by `pytest`.

Run all automated unit and integration tests:
```bash
python3 -m pytest -v
```

### Test Coverage Breakdown:
- **`tests/test_api.py`**: Validates `/health`, `/`, user signup/login lifecycle, and 404 session handling.
- **`tests/test_cards.py`**: Tests vehicle recommendation markdown conversion to modern HTML car cards.
- **`tests/test_emi.py`**: Tests mathematical precision of car loan EMI, zero-interest edge cases, and currency formatting.
- **`tests/test_fuzzy.py`**: Tests RapidFuzz alias mapping for fuel types, car segments, brands, and symptom search.

### Automated Smoke Tests:
Run the live shell test script against running instances:
```bash
chmod +x scripts/smoke_test.sh
./scripts/smoke_test.sh
```

---

## ⚙️ Environment Variables

All settings are configured via `.env` or system environment variables:

| Variable | Required | Default | Description |
|:---|:---:|:---:|:---|
| `GEMINI_API_KEY` | **Yes** | — | Google AI Studio Gemini API Key |
| `DATABASE_URL` | **Yes** | — | PostgreSQL connection URI (`postgresql://user:pass@host:port/db`) |
| `GEMINI_MODEL` | No | `gemini-3.6-flash` | Gemini model variant |
| `API_PORT` | No | `8000` | Port for FastAPI REST/SSE server |
| `API_HOST` | No | `0.0.0.0` | Host binding for FastAPI server |
| `CORS_ORIGINS` | No | `*` | Allowed CORS origins (comma-separated for production) |
| `LOG_LEVEL` | No | `INFO` | Application log level (`DEBUG`, `INFO`, `WARNING`, `ERROR`) |

---

## 🗃️ Database Schema

The database consists of 6 core tables configured with foreign keys and indexes:

1. **`cars`**: Vehicle inventory catalog (name, brand, segment, price ranges, fuel types, mileage JSONB, features).
2. **`common_issues`**: Diagnostic records (symptoms, probable causes, severity rating, repair cost ranges).
3. **`service_intervals`**: Preventive maintenance checklist (item key, recommended km interval, estimated cost).
4. **`users`**: User account credentials (username, email, salted PBKDF2 password hashes).
5. **`conversations`**: Chat session headers linked to user IDs (`session_id` UUID, title, timestamp).
6. **`chat_messages`**: Chat turn history (`session_id`, `role`, `content`, detected `intent`, `created_at`).

---

## 📄 License

This project is licensed under the [MIT License](LICENSE).
