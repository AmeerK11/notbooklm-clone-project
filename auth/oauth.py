from __future__ import annotations

import os
import secrets
from dataclasses import dataclass
from typing import Any
from urllib.parse import urlencode

import httpx


class HFOAuthError(RuntimeError):
    pass


@dataclass(frozen=True)
class HFOAuthSettings:
    client_id: str
    client_secret: str
    authorize_url: str
    token_url: str
    userinfo_url: str
    scope: str


def get_hf_oauth_settings() -> HFOAuthSettings:
    client_id = os.getenv("HF_OAUTH_CLIENT_ID", "").strip()
    client_secret = os.getenv("HF_OAUTH_CLIENT_SECRET", "").strip()
    if not client_id or not client_secret:
        raise HFOAuthError("HF OAuth client configuration is missing.")

    return HFOAuthSettings(
        client_id=client_id,
        client_secret=client_secret,
        authorize_url=os.getenv("HF_OAUTH_AUTHORIZE_URL", "https://huggingface.co/oauth/authorize").strip(),
        token_url=os.getenv("HF_OAUTH_TOKEN_URL", "https://huggingface.co/oauth/token").strip(),
        userinfo_url=os.getenv("HF_OAUTH_USERINFO_URL", "https://huggingface.co/oauth/userinfo").strip(),
        scope=os.getenv("HF_OAUTH_SCOPE", "openid profile email").strip(),
    )


def generate_oauth_state() -> str:
    return secrets.token_urlsafe(32)


def build_hf_authorize_url(redirect_uri: str, state: str) -> str:
    settings = get_hf_oauth_settings()
    query = urlencode(
        {
            "client_id": settings.client_id,
            "redirect_uri": redirect_uri,
            "response_type": "code",
            "scope": settings.scope,
            "state": state,
        }
    )
    return f"{settings.authorize_url}?{query}"


async def exchange_code_for_hf_user(code: str, redirect_uri: str) -> dict[str, Any]:
    settings = get_hf_oauth_settings()

    timeout = httpx.Timeout(20.0, connect=10.0)
    async with httpx.AsyncClient(timeout=timeout) as client:
        token_response = await client.post(
            settings.token_url,
            data={
                "grant_type": "authorization_code",
                "code": code,
                "redirect_uri": redirect_uri,
                "client_id": settings.client_id,
                "client_secret": settings.client_secret,
            },
            headers={"Accept": "application/json"},
        )
        if token_response.status_code >= 400:
            raise HFOAuthError(
                f"HF token exchange failed: {token_response.status_code} {token_response.text}"
            )
        token_payload = token_response.json()
        access_token = token_payload.get("access_token")
        if not access_token:
            raise HFOAuthError("HF token response missing access_token.")

        userinfo_response = await client.get(
            settings.userinfo_url,
            headers={"Authorization": f"Bearer {access_token}", "Accept": "application/json"},
        )
        if userinfo_response.status_code >= 400:
            raise HFOAuthError(
                f"HF userinfo failed: {userinfo_response.status_code} {userinfo_response.text}"
            )
        userinfo = userinfo_response.json()

    email = str(
        userinfo.get("email")
        or userinfo.get("preferred_username")
        or userinfo.get("sub")
        or ""
    ).strip()
    if not email:
        raise HFOAuthError("Unable to resolve user identity from HF OAuth response.")

    display_name = userinfo.get("name") or userinfo.get("preferred_username") or None
    avatar_url = userinfo.get("picture") or None
    return {
        "email": email,
        "display_name": str(display_name) if display_name else None,
        "avatar_url": str(avatar_url) if avatar_url else None,
        "provider_sub": str(userinfo.get("sub") or ""),
    }
