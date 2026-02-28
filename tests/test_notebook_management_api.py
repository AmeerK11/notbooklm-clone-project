"""
Integration tests for notebook rename/delete management endpoints.
"""
from __future__ import annotations

import os
import pathlib
import sys
from unittest.mock import patch

import pytest
from fastapi.testclient import TestClient
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import app as app_module
from app import app
from data.db import Base, get_db


@pytest.fixture()
def db_engine(tmp_path):
    db_file = tmp_path / "test_notebook_mgmt.db"
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
def client(db_session, monkeypatch, tmp_path):
    monkeypatch.setenv("AUTH_MODE", "dev")
    monkeypatch.setenv("APP_SESSION_SECRET", "notebook-mgmt-test-secret")
    monkeypatch.setenv("STORAGE_BASE_DIR", str(tmp_path / "storage"))
    monkeypatch.setattr("app.UPLOADS_ROOT", tmp_path / "uploads")

    def _override_get_db():
        yield db_session

    app.dependency_overrides[get_db] = _override_get_db
    with TestClient(app, raise_server_exceptions=True) as c:
        yield c
    app.dependency_overrides.clear()


def test_rename_notebook_success(client):
    create_resp = client.post("/notebooks", json={"title": "Original title"})
    assert create_resp.status_code == 200
    notebook_id = create_resp.json()["id"]

    rename_resp = client.patch(f"/notebooks/{notebook_id}", json={"title": "Renamed title"})
    assert rename_resp.status_code == 200
    payload = rename_resp.json()
    assert payload["id"] == notebook_id
    assert payload["title"] == "Renamed title"

    list_resp = client.get("/notebooks")
    assert list_resp.status_code == 200
    titles = [n["title"] for n in list_resp.json()]
    assert "Renamed title" in titles


def test_rename_notebook_unknown_returns_404(client):
    rename_resp = client.patch("/notebooks/999999", json={"title": "No notebook"})
    assert rename_resp.status_code == 404


def test_delete_notebook_success(client):
    create_resp = client.post("/notebooks", json={"title": "Delete me"})
    assert create_resp.status_code == 200
    notebook_id = create_resp.json()["id"]

    delete_resp = client.delete(f"/notebooks/{notebook_id}")
    assert delete_resp.status_code == 200
    assert delete_resp.json()["status"] == "deleted"

    list_resp = client.get("/notebooks")
    assert list_resp.status_code == 200
    ids = [n["id"] for n in list_resp.json()]
    assert notebook_id not in ids


def test_delete_notebook_other_user_returns_404(client):
    create_resp = client.post("/notebooks", json={"title": "User1 notebook"})
    assert create_resp.status_code == 200
    notebook_id = create_resp.json()["id"]

    logout_resp = client.post("/auth/logout")
    assert logout_resp.status_code == 200
    login_user2 = client.post(
        "/auth/dev-login",
        json={"email": "user2@example.com", "display_name": "User Two"},
    )
    assert login_user2.status_code == 200

    delete_resp = client.delete(f"/notebooks/{notebook_id}")
    assert delete_resp.status_code == 404


def test_create_url_source_rejects_localhost_target(client):
    create_resp = client.post("/notebooks", json={"title": "URL validation"})
    assert create_resp.status_code == 200
    notebook_id = create_resp.json()["id"]

    source_resp = client.post(
        f"/notebooks/{notebook_id}/sources",
        json={
            "type": "url",
            "title": "Bad URL",
            "url": "http://127.0.0.1:8000/health",
            "status": "pending",
        },
    )
    assert source_resp.status_code == 400
    assert "restricted IP" in source_resp.json()["detail"]


def test_create_url_source_requires_url_field(client):
    create_resp = client.post("/notebooks", json={"title": "Missing URL"})
    assert create_resp.status_code == 200
    notebook_id = create_resp.json()["id"]

    source_resp = client.post(
        f"/notebooks/{notebook_id}/sources",
        json={
            "type": "url",
            "title": "No URL",
            "status": "pending",
        },
    )
    assert source_resp.status_code == 400
    assert "URL is required" in source_resp.json()["detail"]


def test_create_url_source_accepts_public_url(client):
    create_resp = client.post("/notebooks", json={"title": "Good URL"})
    assert create_resp.status_code == 200
    notebook_id = create_resp.json()["id"]

    with patch("src.ingestion.extractors.socket.getaddrinfo") as mock_getaddrinfo, patch(
        "app.ingest_source", return_value=3
    ) as mock_ingest:
        mock_getaddrinfo.return_value = [
            (
                2,
                1,
                6,
                "",
                ("93.184.216.34", 0),
            )
        ]
        source_resp = client.post(
            f"/notebooks/{notebook_id}/sources",
            json={
                "type": "url",
                "title": "Example URL",
                "url": "https://example.com/article",
                "status": "pending",
            },
        )

    assert source_resp.status_code == 200
    payload = source_resp.json()
    assert payload["type"] == "url"
    assert payload["url"] == "https://example.com/article"
    assert payload["status"] == "ready"
    assert payload["ingested_at"] is not None
    mock_ingest.assert_called_once()


def test_upload_source_sanitizes_filename(client):
    create_resp = client.post("/notebooks", json={"title": "Uploads"})
    assert create_resp.status_code == 200
    notebook_id = create_resp.json()["id"]

    with patch("app.ingest_source", return_value=1):
        upload_resp = client.post(
            f"/notebooks/{notebook_id}/sources/upload",
            data={"status": "pending"},
            files={"file": ("../../../../evil.txt", b"hello world", "text/plain")},
        )

    assert upload_resp.status_code == 200
    payload = upload_resp.json()
    assert payload["original_name"] == "evil.txt"
    assert payload["storage_path"] is not None
    assert ".." not in payload["storage_path"]
    assert f"notebook_{notebook_id}" in payload["storage_path"]
    assert pathlib.Path(payload["storage_path"]).exists()


def test_delete_notebook_removes_notebook_storage_and_uploads(client):
    create_resp = client.post("/notebooks", json={"title": "Delete storage"})
    assert create_resp.status_code == 200
    notebook_id = create_resp.json()["id"]

    storage_root = pathlib.Path(os.environ["STORAGE_BASE_DIR"]) / "users" / "1" / "notebooks" / str(notebook_id)
    storage_root.mkdir(parents=True, exist_ok=True)
    (storage_root / "marker.txt").write_text("x", encoding="utf-8")

    upload_root = pathlib.Path(app_module.UPLOADS_ROOT) / f"notebook_{notebook_id}"
    upload_root.mkdir(parents=True, exist_ok=True)
    (upload_root / "upload.txt").write_text("x", encoding="utf-8")

    delete_resp = client.delete(f"/notebooks/{notebook_id}")
    assert delete_resp.status_code == 200

    assert not storage_root.exists()
    assert not upload_root.exists()
