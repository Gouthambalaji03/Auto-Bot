"""
tests/test_api.py — Integration tests for AutoBot FastAPI endpoints
"""

import uuid
import pytest
from fastapi.testclient import TestClient
from api.app import app


@pytest.fixture(scope="module")
def client():
    with TestClient(app) as c:
        yield c


def test_health_endpoint(client):
    """GET /health should return 200 with status ok."""
    response = client.get("/health")
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "ok"
    assert data["db"] == "connected"
    assert "agent" in data
    assert "model" in data


def test_root_status_endpoint(client):
    """GET / should return 200 with status ok."""
    response = client.get("/")
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "ok"
    assert data["services"]["database"]["status"] == "connected"
    assert "docs" in data


def test_auth_workflow(client):
    """Test user signup, duplicate rejection, and login."""
    unique_id = uuid.uuid4().hex[:8]
    username = f"usr_{unique_id}"
    email = f"{username}@example.com"
    password = "SecurePassword123"

    # 1. Signup
    signup_resp = client.post(
        "/auth/signup",
        json={"username": username, "email": email, "password": password},
    )
    assert signup_resp.status_code == 200
    signup_data = signup_resp.json()
    assert signup_data["success"] is True
    assert signup_data["user"]["username"] == username

    # 2. Duplicate signup should be rejected with 400
    dup_resp = client.post(
        "/auth/signup",
        json={"username": username, "email": email, "password": password},
    )
    assert dup_resp.status_code == 400

    # 3. Login
    login_resp = client.post(
        "/auth/login",
        json={"username_or_email": username, "password": password},
    )
    assert login_resp.status_code == 200
    login_data = login_resp.json()
    assert login_data["success"] is True
    assert login_data["user"]["username"] == username

    # 4. Invalid login should be rejected with 401
    bad_login = client.post(
        "/auth/login",
        json={"username_or_email": username, "password": "WrongPassword"},
    )
    assert bad_login.status_code == 401


def test_session_history_not_found(client):
    """GET /sessions/{id}/history should return 404 for unknown session."""
    response = client.get("/sessions/unknown-session-12345/history?user_id=99999")
    assert response.status_code == 404
