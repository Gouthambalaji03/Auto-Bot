"""
api/routers/auth.py — POST /auth/signup  |  POST /auth/login

Exposes the existing db/auth.py user registration and authentication logic
over HTTP. No JWT tokens — login returns the user record ({id, username, email})
which the caller stores and passes as user_id in subsequent chat requests.

Nothing in db/auth.py is modified.
"""

from __future__ import annotations

import logging
import time

from fastapi import APIRouter, HTTPException

from api.dependencies import async_create_user, async_authenticate_user
from api.schemas import (
    SignupRequest,
    SignupResponse,
    LoginRequest,
    LoginResponse,
)

logger = logging.getLogger(__name__)
router = APIRouter(prefix="/auth", tags=["Auth"])


@router.post(
    "/signup",
    response_model=SignupResponse,
    summary="Register a new AutoBot user account",
    description=(
        "Creates a new user with a PBKDF2-hashed password. "
        "Returns the new user record on success or a descriptive error message on failure."
    ),
)
async def signup(body: SignupRequest) -> SignupResponse:
    start = time.perf_counter()
    logger.info("Signup attempt — username=%r email=%r", body.username, body.email)

    success, message, user = await async_create_user(
        body.username, body.email, body.password
    )
    elapsed_ms = (time.perf_counter() - start) * 1000

    if not success:
        logger.warning(
            "Signup FAILED — username=%r email=%r reason=%r  (%.0fms)",
            body.username, body.email, message, elapsed_ms,
        )
        raise HTTPException(status_code=400, detail=message)

    logger.info(
        "Signup SUCCESS — user_id=%s username=%r email=%r  (%.0fms)",
        user.get("id") if user else "?",
        body.username,
        body.email,
        elapsed_ms,
    )
    return SignupResponse(success=True, message=message, user=user)


@router.post(
    "/login",
    response_model=LoginResponse,
    summary="Authenticate an existing AutoBot user",
    description=(
        "Validates username/email + password. "
        "Returns the user record on success. Store the returned user.id and "
        "pass it as user_id in chat requests to enable conversation persistence."
    ),
)
async def login(body: LoginRequest) -> LoginResponse:
    start = time.perf_counter()
    logger.info("Login attempt — identifier=%r", body.username_or_email)

    success, message, user = await async_authenticate_user(
        body.username_or_email, body.password
    )
    elapsed_ms = (time.perf_counter() - start) * 1000

    if not success:
        logger.warning(
            "Login FAILED — identifier=%r reason=%r  (%.0fms)",
            body.username_or_email, message, elapsed_ms,
        )
        raise HTTPException(status_code=401, detail=message)

    logger.info(
        "Login SUCCESS — user_id=%s username=%r  (%.0fms)",
        user.get("id") if user else "?",
        user.get("username") if user else "?",
        elapsed_ms,
    )
    return LoginResponse(success=True, message=message, user=user)
