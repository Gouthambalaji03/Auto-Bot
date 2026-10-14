"""
db/auth.py — User Authentication & Password Hashing

Provides secure password hashing (PBKDF2-HMAC-SHA256 with random salt),
user registration (sign up), and credential authentication (login).
"""

import hashlib
import hmac
import secrets
from typing import Optional, Tuple

from db.connection import get_conn


# ─────────────────────────────────────────────
# Password Hashing Utilities
# ─────────────────────────────────────────────

def hash_password(password: str) -> str:
    """
    Hashes a plain-text password using PBKDF2-HMAC-SHA256 with 100,000 iterations.
    Format returned: 'salt_hex$hash_hex'
    """
    salt = secrets.token_bytes(16)
    pw_hash = hashlib.pbkdf2_hmac("sha256", password.encode("utf-8"), salt, 100_000)
    return f"{salt.hex()}${pw_hash.hex()}"


def verify_password(password: str, stored_hash: str) -> bool:
    """
    Verifies a plain-text password against a stored 'salt_hex$hash_hex' string.
    Uses constant-time hmac.compare_digest to prevent timing attacks.
    """
    try:
        salt_hex, hash_hex = stored_hash.split("$", 1)
        salt = bytes.fromhex(salt_hex)
        expected_hash = bytes.fromhex(hash_hex)
        computed_hash = hashlib.pbkdf2_hmac("sha256", password.encode("utf-8"), salt, 100_000)
        return hmac.compare_digest(computed_hash, expected_hash)
    except Exception:
        return False


# ─────────────────────────────────────────────
# User Account Operations (Sign Up & Login)
# ─────────────────────────────────────────────

def db_create_user(username: Optional[str], email: Optional[str], password: Optional[str]) -> Tuple[bool, str, Optional[dict]]:
    """
    Registers a new user in the PostgreSQL users table.

    Returns:
        (success, message, user_dict_or_none)
    """
    username = (username or "").strip()
    email = (email or "").strip().lower()
    password = password or ""

    if not username or len(username) < 3:
        return False, "Username must be at least 3 characters long.", None

    if not email or "@" not in email or "." not in email:
        return False, "Please enter a valid email address.", None

    if not password or len(password) < 6:
        return False, "Password must be at least 6 characters long.", None

    pw_hash = hash_password(password)

    try:
        with get_conn() as conn:
            with conn.cursor() as cur:
                # Check for existing username or email
                cur.execute("SELECT username, email FROM users WHERE username ILIKE %s OR email ILIKE %s", (username, email))
                existing = cur.fetchone()
                if existing:
                    if existing["username"].lower() == username.lower():
                        return False, "Username is already taken. Please choose another.", None
                    if existing["email"].lower() == email:
                        return False, "An account with this email already exists.", None

                cur.execute(
                    """
                    INSERT INTO users (username, email, password_hash)
                    VALUES (%s, %s, %s)
                    RETURNING id, username, email, created_at
                    """,
                    (username, email, pw_hash)
                )
                row = cur.fetchone()
                user = dict(row)
                return True, f"Welcome to AutoBot, {user['username']}! Account created successfully.", user
    except Exception as e:
        return False, f"Failed to create account: {str(e)}", None


def db_authenticate_user(username_or_email: Optional[str], password: Optional[str]) -> Tuple[bool, str, Optional[dict]]:
    """
    Authenticates a user by username or email and password.

    Returns:
        (success, message, user_dict_or_none)
    """
    identifier = (username_or_email or "").strip().lower()
    password = password or ""
    if not identifier or not password:
        return False, "Please enter both username/email and password.", None

    try:
        with get_conn() as conn:
            with conn.cursor() as cur:
                cur.execute(
                    "SELECT id, username, email, password_hash, created_at FROM users WHERE LOWER(username) = %s OR LOWER(email) = %s",
                    (identifier, identifier)
                )
                row = cur.fetchone()
                if not row:
                    return False, "Invalid username/email or password.", None

                user = dict(row)
                if not verify_password(password, user["password_hash"]):
                    return False, "Invalid username/email or password.", None

                user_dict = {
                    "id": user["id"],
                    "username": user["username"],
                    "email": user["email"],
                    "created_at": str(user["created_at"]),
                }
                return True, f"Welcome back, {user['username']}!", user_dict
    except Exception as e:
        return False, f"Authentication error: {str(e)}", None
