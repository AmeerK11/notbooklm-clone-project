"""
Integration tests for citation persistence in chat threads.
"""
from __future__ import annotations

import pathlib
import sys
from unittest.mock import patch

import pytest
from fastapi.testclient import TestClient
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from app import app
from data.db import Base, get_db


@pytest.fixture()
def db_engine(tmp_path):
    db_file = tmp_path / "test_chat_citations.db"
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
def client(db_session, monkeypatch):
    monkeypatch.setenv("AUTH_MODE", "dev")
    monkeypatch.setenv("APP_SESSION_SECRET", "chat-citations-test-secret")

    def _override_get_db():
        yield db_session

    app.dependency_overrides[get_db] = _override_get_db
    with TestClient(app, raise_server_exceptions=True) as c:
        yield c
    app.dependency_overrides.clear()


def test_thread_messages_include_persisted_citations(client):
    create_notebook = client.post("/notebooks", json={"title": "Citation Notebook"})
    assert create_notebook.status_code == 200
    notebook_id = int(create_notebook.json()["id"])

    create_source = client.post(
        f"/notebooks/{notebook_id}/sources",
        json={
            "type": "text",
            "title": "Lecture Notes",
            "status": "ready",
        },
    )
    assert create_source.status_code == 200
    source_id = int(create_source.json()["id"])

    create_thread = client.post(
        f"/notebooks/{notebook_id}/threads",
        json={"title": "Q&A"},
    )
    assert create_thread.status_code == 200
    thread_id = int(create_thread.json()["id"])

    retrieval_rows = [
        {
            "chunk_id": "chunk-1",
            "score": 0.12,
            "document": "Neural networks learn from examples.",
            "metadata": {
                "source_id": str(source_id),
                "source_title": "Lecture Notes",
                "chunk_index": 0,
            },
        }
    ]

    with patch("app.query_notebook_chunks", return_value=retrieval_rows), patch(
        "app.generate_chat_completion", return_value="They learn from examples in the data."
    ):
        chat_resp = client.post(
            f"/threads/{thread_id}/chat",
            params={"notebook_id": notebook_id},
            json={"question": "How do neural networks learn?", "top_k": 5},
        )

    assert chat_resp.status_code == 200
    chat_payload = chat_resp.json()
    assert len(chat_payload["citations"]) == 1
    assert int(chat_payload["citations"][0]["source_id"]) == source_id

    messages_resp = client.get(
        f"/threads/{thread_id}/messages",
        params={"notebook_id": notebook_id},
    )
    assert messages_resp.status_code == 200
    messages = messages_resp.json()
    assistant_message = next((m for m in messages if m["role"] == "assistant"), None)
    assert assistant_message is not None
    assert len(assistant_message["citations"]) == 1
    assert int(assistant_message["citations"][0]["source_id"]) == source_id
    assert assistant_message["citations"][0]["source_title"] == "Lecture Notes"
