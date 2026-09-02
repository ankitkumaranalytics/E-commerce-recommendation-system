"""Authentication helpers for the marketplace API.

The module intentionally uses only standard-library cryptography so the
development SQLite setup remains easy to run. Tokens are signed, short-lived
JWT-compatible payloads and passwords use PBKDF2-HMAC with a per-user salt.
"""

from __future__ import annotations

import base64
import hashlib
import hmac
import json
import os
import secrets
import time
from typing import Any, Dict

from fastapi import Depends, HTTPException, status
from fastapi.security import HTTPAuthorizationCredentials, HTTPBearer
from dotenv import load_dotenv

from src.database.models import User
from src.database.connection import get_db
from sqlalchemy.orm import Session


load_dotenv()
ALGORITHM = "HS256"
PASSWORD_ITERATIONS = 310_000
bearer_scheme = HTTPBearer(auto_error=False)


def _secret_key() -> bytes:
    secret = os.getenv("SECRET_KEY", "")
    if len(secret) < 32:
        raise RuntimeError("SECRET_KEY must contain at least 32 characters")
    return secret.encode("utf-8")


def hash_password(password: str) -> str:
    """Hash a password using a random salt and PBKDF2-HMAC-SHA256."""
    salt = secrets.token_bytes(16)
    digest = hashlib.pbkdf2_hmac(
        "sha256", password.encode("utf-8"), salt, PASSWORD_ITERATIONS
    )
    return "pbkdf2_sha256${}${}${}".format(
        PASSWORD_ITERATIONS,
        _encode(salt),
        _encode(digest),
    )


def verify_password(password: str, encoded: str) -> bool:
    """Constant-time verification of a stored password hash."""
    try:
        algorithm, iterations, salt, expected = encoded.split("$", 3)
        if algorithm != "pbkdf2_sha256":
            return False
        digest = hashlib.pbkdf2_hmac(
            "sha256",
            password.encode("utf-8"),
            _decode(salt),
            int(iterations),
        )
        return hmac.compare_digest(_encode(digest), expected)
    except (TypeError, ValueError):
        return False


def create_access_token(user: User, expires_seconds: int | None = None) -> str:
    """Create a signed access token containing only non-sensitive claims."""
    lifetime = expires_seconds or int(os.getenv("ACCESS_TOKEN_EXPIRE_MINUTES", "30")) * 60
    payload = {
        "sub": user.user_id,
        "role": user.role or "CUSTOMER",
        "iat": int(time.time()),
        "exp": int(time.time()) + lifetime,
    }
    encoded_header = _encode_json({"alg": ALGORITHM, "typ": "JWT"})
    encoded_payload = _encode_json(payload)
    signing_input = f"{encoded_header}.{encoded_payload}".encode("ascii")
    signature = hmac.new(_secret_key(), signing_input, hashlib.sha256).digest()
    return f"{encoded_header}.{encoded_payload}.{_encode(signature)}"


def decode_access_token(token: str) -> Dict[str, Any]:
    """Validate a token signature and expiration, returning its claims."""
    try:
        header, payload, encoded_signature = token.split(".")
        signing_input = f"{header}.{payload}".encode("ascii")
        expected = hmac.new(_secret_key(), signing_input, hashlib.sha256).digest()
        if not hmac.compare_digest(_decode(encoded_signature), expected):
            raise ValueError("invalid signature")
        claims = json.loads(_decode(payload))
        if int(claims["exp"]) <= int(time.time()):
            raise ValueError("expired token")
        if not claims.get("sub") or not claims.get("role"):
            raise ValueError("missing claims")
        return claims
    except (KeyError, TypeError, ValueError, json.JSONDecodeError) as exc:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid or expired access token",
            headers={"WWW-Authenticate": "Bearer"},
        ) from exc


def get_current_user(
    credentials: HTTPAuthorizationCredentials | None = Depends(bearer_scheme),
    db_session: Session = Depends(get_db),
) -> User:
    """Resolve the authenticated user from a bearer token."""
    if credentials is None:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Authentication required",
            headers={"WWW-Authenticate": "Bearer"},
        )
    claims = decode_access_token(credentials.credentials)
    user = db_session.query(User).filter(User.user_id == claims["sub"]).first()
    if user is None or not user.is_active:
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="User is inactive")
    return user


def require_role(*roles: str):
    """Build a dependency that enforces one of the supplied roles."""
    allowed = {role.upper() for role in roles}

    def dependency(user: User = Depends(get_current_user)) -> User:
        if (user.role or "CUSTOMER").upper() not in allowed:
            raise HTTPException(status_code=status.HTTP_403_FORBIDDEN, detail="Insufficient permissions")
        return user

    return dependency


def _encode(value: bytes) -> str:
    return base64.urlsafe_b64encode(value).rstrip(b"=").decode("ascii")


def _decode(value: str) -> bytes:
    return base64.urlsafe_b64decode(value + "=" * (-len(value) % 4))


def _encode_json(value: Dict[str, Any]) -> str:
    return _encode(json.dumps(value, separators=(",", ":"), sort_keys=True).encode("utf-8"))
