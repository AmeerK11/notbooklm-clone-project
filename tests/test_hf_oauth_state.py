from __future__ import annotations

import pathlib
import sys

import pytest
from fastapi.testclient import TestClient
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import app as app_module
from app import app
from auth.oauth import generate_oauth_state, is_valid_oauth_state
from data.db import Base, get_db


@pytest.fixture()
def db_engine(tmp_path):
    db_file = tmp_path / "test_hf_oauth_state.db"
    engine = create_engine(
        f"sqlite:///{db_file}",
        connect_args={"check_same_thread": False},
    )
    import data.models  # noqa: F401

    Base.metadata.create_all(bind=engine)
    yield engine
    Base.metadata.drop_all(bind=engine)
    engine.dispose()


@pytest.fixture()
def db_session(db_engine):
    Session = sessionmaker(autocommit=False, autoflush=False, bind=db_engine)
    session = Session()
    yield session
    session.close()


@pytest.fixture()
def client(db_session):
    def _override_get_db():
        yield db_session

    app.dependency_overrides[get_db] = _override_get_db
    with TestClient(app, raise_server_exceptions=True) as c:
        yield c
    app.dependency_overrides.clear()


def test_oauth_state_is_signed_and_valid(monkeypatch):
    monkeypatch.setenv("APP_SESSION_SECRET", "oauth-state-test-secret")
    state = generate_oauth_state()
    assert state
    assert is_valid_oauth_state(state) is True
    assert is_valid_oauth_state("plain-random-state") is False


def test_callback_accepts_signed_state_without_session(client, monkeypatch):
    monkeypatch.setenv("AUTH_MODE", "hf_oauth")
    monkeypatch.setenv("APP_SESSION_SECRET", "oauth-state-test-secret")
    monkeypatch.setenv("HF_OAUTH_CLIENT_ID", "test-client-id")
    monkeypatch.setenv("HF_OAUTH_CLIENT_SECRET", "test-client-secret")
    monkeypatch.setenv("HF_OAUTH_REDIRECT_URI", "http://testserver/auth/callback")
    monkeypatch.setenv("AUTH_SUCCESS_REDIRECT_URL", "http://testserver/")

    async def _fake_exchange(*, code: str, redirect_uri: str):
        return {
            "email": "oauth-user@example.com",
            "display_name": "OAuth User",
            "avatar_url": None,
            "provider_sub": "sub-123",
        }

    monkeypatch.setattr(app_module, "exchange_code_for_hf_user", _fake_exchange)

    response = client.get(
        "/auth/callback",
        params={"state": generate_oauth_state(), "code": "test-code"},
        follow_redirects=False,
    )
    assert response.status_code == 302
    location = response.headers.get("location", "")
    assert "auth_bridge=" in location


def test_callback_rejects_invalid_state_without_session(client, monkeypatch):
    monkeypatch.setenv("AUTH_MODE", "hf_oauth")
    monkeypatch.setenv("APP_SESSION_SECRET", "oauth-state-test-secret")
    monkeypatch.setenv("HF_OAUTH_CLIENT_ID", "test-client-id")
    monkeypatch.setenv("HF_OAUTH_CLIENT_SECRET", "test-client-secret")

    response = client.get(
        "/auth/callback",
        params={"state": "invalid-state", "code": "test-code"},
        follow_redirects=False,
    )
    assert response.status_code == 400
    assert response.json()["detail"] == "Invalid OAuth state."
