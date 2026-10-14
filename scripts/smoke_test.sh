#!/usr/bin/env bash
# smoke_test.sh — AutoBot FastAPI smoke tests
#
# Usage:
#   chmod +x smoke_test.sh
#   ./smoke_test.sh
#
# Requires: curl, python3 (for JSON pretty-print)
# Run api_server.py first: python api_server.py

set -euo pipefail

BASE_URL="${API_BASE_URL:-http://localhost:8000}"
PASS=0
FAIL=0

GREEN='\033[0;32m'
RED='\033[0;31m'
CYAN='\033[0;36m'
NC='\033[0m'

check() {
    local name="$1"
    local status="$2"
    if [ "$status" -eq 0 ]; then
        echo -e "${GREEN}✅ PASS${NC}  $name"
        PASS=$((PASS + 1))
    else
        echo -e "${RED}❌ FAIL${NC}  $name"
        FAIL=$((FAIL + 1))
    fi
}

echo ""
echo -e "${CYAN}━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━${NC}"
echo -e "${CYAN}  AutoBot FastAPI Smoke Tests  →  $BASE_URL${NC}"
echo -e "${CYAN}━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━${NC}"
echo ""

# ── 1. Health check ───────────────────────────────────────────────────────────
echo "1. GET /health"
HEALTH=$(curl -s -o /dev/null -w "%{http_code}" "$BASE_URL/health")
check "Health returns 200 or 503" "$([ "$HEALTH" -eq 200 ] || [ "$HEALTH" -eq 503 ]; echo $?)"

HEALTH_BODY=$(curl -s "$BASE_URL/health")
echo "   Response: $HEALTH_BODY"
echo ""

# ── 2. Sync chat — buying intent ──────────────────────────────────────────────
echo "2. POST /chat/sync — buying intent"
SYNC_STATUS=$(curl -s -o /tmp/autobot_sync.json -w "%{http_code}" \
    -X POST "$BASE_URL/chat/sync" \
    -H "Content-Type: application/json" \
    -d '{"message": "Show me petrol SUVs under 15 lakhs"}')

check "Sync chat returns 200 or 429" "$([ "$SYNC_STATUS" -eq 200 ] || [ "$SYNC_STATUS" -eq 429 ]; echo $?)"
if [ "$SYNC_STATUS" -eq 200 ]; then
    echo "   Intent:  $(python3 -c "import json,sys; d=json.load(open('/tmp/autobot_sync.json')); print(d.get('intent','?'))")"
    echo "   Elapsed: $(python3 -c "import json,sys; d=json.load(open('/tmp/autobot_sync.json')); print(d.get('elapsed_seconds','?'))")s"
fi
echo ""

# ── 3. Sync chat — EMI / finance intent ──────────────────────────────────────
echo "3. POST /chat/sync — finance intent (EMI tool)"
EMI_STATUS=$(curl -s -o /tmp/autobot_emi.json -w "%{http_code}" \
    -X POST "$BASE_URL/chat/sync" \
    -H "Content-Type: application/json" \
    -d '{"message": "Calculate EMI for 10 lakh loan at 9% for 5 years"}')

check "EMI sync chat returns 200 or 429" "$([ "$EMI_STATUS" -eq 200 ] || [ "$EMI_STATUS" -eq 429 ]; echo $?)"
echo ""

# ── 4. Auth: signup ───────────────────────────────────────────────────────────
echo "4. POST /auth/signup"
SIGNUP_STATUS=$(curl -s -o /tmp/autobot_signup.json -w "%{http_code}" \
    -X POST "$BASE_URL/auth/signup" \
    -H "Content-Type: application/json" \
    -d '{"username":"smoketest_user","email":"smoketest@autobot.dev","password":"test1234"}')

check "Signup returns 200 or 400 (conflict if user exists)" \
    "$([ "$SIGNUP_STATUS" -eq 200 ] || [ "$SIGNUP_STATUS" -eq 400 ]; echo $?)"
echo "   Response: $(cat /tmp/autobot_signup.json)"
echo ""

# ── 5. Auth: login ────────────────────────────────────────────────────────────
echo "5. POST /auth/login"
LOGIN_STATUS=$(curl -s -o /tmp/autobot_login.json -w "%{http_code}" \
    -X POST "$BASE_URL/auth/login" \
    -H "Content-Type: application/json" \
    -d '{"username_or_email":"smoketest_user","password":"test1234"}')

check "Login returns 200 or 401" \
    "$([ "$LOGIN_STATUS" -eq 200 ] || [ "$LOGIN_STATUS" -eq 401 ]; echo $?)"
echo ""

# ── 6. SSE streaming (first 10 seconds sample) ────────────────────────────────
echo "6. POST /chat — SSE streaming (10s sample)"
SSE_OUTPUT=$(curl -s -N --max-time 10 \
    -X POST "$BASE_URL/chat" \
    -H "Content-Type: application/json" \
    -d '{"message": "What are the best EVs in India under 20 lakhs?"}' 2>/dev/null || true)

check "SSE stream starts without error" \
    "$(echo "$SSE_OUTPUT" | grep -q 'data:' && echo 0 || echo 1)"
echo ""

# ── 7. Gradio unaffected ─────────────────────────────────────────────────────
echo "7. Gradio UI still running on :7860"
GRADIO_STATUS=$(curl -s -o /dev/null -w "%{http_code}" --max-time 3 "http://localhost:7860" 2>/dev/null || echo "000")
check "Gradio returns 200" "$([ "$GRADIO_STATUS" -eq 200 ]; echo $?)"
echo ""

# ── Summary ───────────────────────────────────────────────────────────────────
echo -e "${CYAN}━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━${NC}"
echo -e "  Results: ${GREEN}${PASS} passed${NC}  ${RED}${FAIL} failed${NC}"
echo -e "${CYAN}━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━${NC}"
echo ""

[ "$FAIL" -eq 0 ]
