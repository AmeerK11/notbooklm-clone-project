from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Any

from fastapi import Depends, HTTPException, Request, status
from itsdangerous import BadSignature, SignatureExpired, URLSafeTimedSerializer
from sqlalchemy.orm import Session
from starlette.middleware.sessions import SessionMiddleware

from data import crud
from data.db import get_db

AUTH_MODE_DEV = "dev"
AUTH_MODE_HF = "hf_oauth"
AUTH_BRIDGE_SALT = "streamlit-auth-bridge"
DEFAULT_DEV_SESSION_SECRET = "dev-only-session-secret-change-me"


@dataclass(frozen=True)
class CurrentUser:
    id: int
    email: str
    display_name: str | None = None
    avatar_url: str | None = None


class AuthBridgeTokenError(ValueError):
    pass


def get_auth_mode() -> str:
    configured = os.getenv("AUTH_MODE", AUTH_MODE_DEV).strip().lower()
    return configured if configured in {AUTH_MODE_DEV, AUTH_MODE_HF} else AUTH_MODE_DEV


def configure_session_middleware(app) -> None:
    """Attach Starlette session middleware once during app setup."""
    secret = os.getenv("APP_SESSION_SECRET", DEFAULT_DEV_SESSION_SECRET).strip()
    auth_mode = get_auth_mode()
    if auth_mode == AUTH_MODE_HF and (not secret or secret == DEFAULT_DEV_SESSION_SECRET):
        raise RuntimeError("APP_SESSION_SECRET must be set to a non-default value in hf_oauth mode.")
    same_site = os.getenv("SESSION_COOKIE_SAMESITE", "lax").strip().lower()
    if same_site not in {"lax", "strict", "none"}:
        same_site = "lax"
    secure_default = "1" if auth_mode == AUTH_MODE_HF else "0"
    https_only = os.getenv("SESSION_COOKIE_SECURE", secure_default).strip().lower() in {
        "1",
        "true",
        "yes",
        "on",
    }
    app.add_middleware(
        SessionMiddleware,
        secret_key=secret,
        same_site=same_site,
        https_only=https_only,
        max_age=60 * 60 * 24 * 7,  # 7 days
    )


def _bridge_serializer() -> URLSafeTimedSerializer:
    secret = os.getenv("APP_SESSION_SECRET", DEFAULT_DEV_SESSION_SECRET)
    return URLSafeTimedSerializer(secret_key=secret, salt=AUTH_BRIDGE_SALT)


def _session_user_to_current_user(session_user: dict[str, Any]) -> CurrentUser | None:
    try:
        return CurrentUser(
            id=int(session_user["id"]),
            email=str(session_user["email"]),
            display_name=(str(session_user["display_name"]) if session_user.get("display_name") else None),
            avatar_url=(str(session_user["avatar_url"]) if session_user.get("avatar_url") else None),
        )
    except (KeyError, TypeError, ValueError):
        return None


def get_session_user(request: Request) -> CurrentUser | None:
    raw_user = request.session.get("user")
    if not isinstance(raw_user, dict):
        return None
    return _session_user_to_current_user(raw_user)


def set_session_user(request: Request, user: CurrentUser) -> None:
    request.session["user"] = {
        "id": user.id,
        "email": user.email,
        "display_name": user.display_name,
        "avatar_url": user.avatar_url,
    }


def clear_session_user(request: Request) -> None:
    request.session.pop("user", None)


def generate_auth_bridge_token(user: CurrentUser) -> str:
    payload = {
        "id": user.id,
        "email": user.email,
        "display_name": user.display_name,
        "avatar_url": user.avatar_url,
    }
    return _bridge_serializer().dumps(payload)


def decode_auth_bridge_token(token: str) -> CurrentUser:
    ttl_seconds = int(os.getenv("AUTH_BRIDGE_TOKEN_TTL_SECONDS", "300"))
    try:
        payload = _bridge_serializer().loads(token, max_age=ttl_seconds)
    except SignatureExpired as exc:
        raise AuthBridgeTokenError("Bridge token expired. Please sign in again.") from exc
    except BadSignature as exc:
        raise AuthBridgeTokenError("Invalid bridge token.") from exc

    if not isinstance(payload, dict):
        raise AuthBridgeTokenError("Invalid bridge token payload.")

    user = _session_user_to_current_user(payload)
    if user is None:
        raise AuthBridgeTokenError("Invalid bridge token payload.")
    return user


def _ensure_dev_user(request: Request, db: Session) -> CurrentUser:
    dev_user_id = int(os.getenv("AUTH_DEV_USER_ID", "1"))
    dev_email = os.getenv("AUTH_DEV_EMAIL", "dev@example.com")
    dev_name = os.getenv("AUTH_DEV_DISPLAY_NAME", "Dev User")

    user = crud.get_or_create_user(
        db=db,
        user_id=dev_user_id,
        email=dev_email,
        display_name=dev_name,
    )
    current = CurrentUser(
        id=user.id,
        email=user.email,
        display_name=user.display_name,
        avatar_url=user.avatar_url,
    )
    set_session_user(request, current)
    return current


def require_current_user(request: Request, db: Session = Depends(get_db)) -> CurrentUser:
    """Resolve current user from session; auto-provision in dev mode."""
    session_user = get_session_user(request)
    if session_user:
        return session_user

    if get_auth_mode() == AUTH_MODE_DEV:
        return _ensure_dev_user(request, db)

    raise HTTPException(
        status_code=status.HTTP_401_UNAUTHORIZED,
        detail="Authentication required.",
    )
