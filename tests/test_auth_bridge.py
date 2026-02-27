"""
Tests for Hugging Face OAuth bridge handoff used by the Streamlit client.
"""
from __future__ import annotations

import pathlib
import sys

import pytest
from fastapi.testclient import TestClient
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from app import app
from auth.session import CurrentUser, generate_auth_bridge_token
from data.db import Base, get_db


@pytest.fixture()
def db_engine(tmp_path):
    db_file = tmp_path / "test_auth_bridge.db"
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


def test_auth_bridge_exchange_hf_mode_success(client, monkeypatch):
    monkeypatch.setenv("AUTH_MODE", "hf_oauth")
    monkeypatch.setenv("APP_SESSION_SECRET", "auth-bridge-test-secret")

    token = generate_auth_bridge_token(
        CurrentUser(id=999, email="hf-user@example.com", display_name="HF User")
    )
    resp = client.post("/auth/bridge/exchange", json={"token": token})
    assert resp.status_code == 200
    payload = resp.json()
    assert payload["authenticated"] is True
    assert payload["mode"] == "hf_oauth"
    assert payload["user"]["email"] == "hf-user@example.com"

    status_resp = client.get("/auth/status")
    assert status_resp.status_code == 200
    status_payload = status_resp.json()
    assert status_payload["authenticated"] is True
    assert status_payload["user"]["email"] == "hf-user@example.com"


def test_auth_bridge_exchange_disabled_in_dev_mode(client, monkeypatch):
    monkeypatch.setenv("AUTH_MODE", "dev")
    monkeypatch.setenv("APP_SESSION_SECRET", "auth-bridge-test-secret")

    token = generate_auth_bridge_token(
        CurrentUser(id=777, email="dev-user@example.com", display_name="Dev User")
    )
    resp = client.post("/auth/bridge/exchange", json={"token": token})
    assert resp.status_code == 400
    assert "only available in hf_oauth mode" in resp.json()["detail"]
