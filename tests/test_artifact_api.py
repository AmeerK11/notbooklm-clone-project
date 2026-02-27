"""
Integration tests for artifact API endpoints.

Uses FastAPI TestClient with an in-memory SQLite database so no external
services are required.
"""
from __future__ import annotations

import json
import pathlib
import sys
from unittest.mock import MagicMock, patch

import pytest
from fastapi.testclient import TestClient
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from data.db import Base, get_db
from data import crud
from app import app


# ── Database fixtures ─────────────────────────────────────────────────────────


@pytest.fixture()
def db_engine(tmp_path):
    """Fresh SQLite database for each test."""
    db_file = tmp_path / "test.db"
    engine = create_engine(
        f"sqlite:///{db_file}",
        connect_args={"check_same_thread": False},
    )
    # Import all models so Base knows about them
    import data.models  # noqa: F401
    Base.metadata.create_all(bind=engine)
    yield engine
    Base.metadata.drop_all(bind=engine)
    engine.dispose()


@pytest.fixture()
def db_session(db_engine):
    """Single session shared across one test."""
    Session = sessionmaker(autocommit=False, autoflush=False, bind=db_engine)
    session = Session()
    yield session
    session.close()


@pytest.fixture()
def client(db_session):
    """TestClient with database overridden to the test session."""

    def _override_get_db():
        yield db_session

    app.dependency_overrides[get_db] = _override_get_db
    with TestClient(app, raise_server_exceptions=True) as c:
        yield c
    app.dependency_overrides.clear()


@pytest.fixture()
def notebook(db_session):
    """A test user + notebook ready in the test database."""
    crud.get_or_create_user(db=db_session, user_id=1)
    nb = crud.create_notebook(db=db_session, owner_user_id=1, title="Test Notebook")
    return nb


# ── Quiz endpoint tests ───────────────────────────────────────────────────────


class TestQuizEndpoint:
    def test_generate_quiz_success(self, client, notebook):
        """POST quiz returns 200 with status=ready and content populated."""
        mock_result = {
            "questions": [
                {
                    "id": 1,
                    "question": "What is ML?",
                    "options": ["A) a", "B) b", "C) c", "D) d"],
                    "correct_answer": "A",
                    "explanation": "Because.",
                    "difficulty": "easy",
                    "topic": "ML",
                }
            ],
            "metadata": {"num_questions": 1},
        }

        with patch("app.QuizGenerator") as MockGen:
            mock_gen = MagicMock()
            mock_gen.generate_quiz.return_value = mock_result
            mock_gen.to_markdown.return_value = "# Quiz\n\n## Questions\n\n1. What is ML?"
            mock_gen.save_quiz.return_value = "/tmp/quiz.md"
            MockGen.return_value = mock_gen

            resp = client.post(
                f"/notebooks/{notebook.id}/artifacts/quiz",
                json={"num_questions": 1, "difficulty": "easy"},
            )

        assert resp.status_code == 200
        data = resp.json()
        assert data["type"] == "quiz"
        assert data["status"] == "ready"
        assert data["content"] is not None
        assert data["content"].startswith("# Quiz")
        assert data["file_path"] == "/tmp/quiz.md"

    def test_generate_quiz_no_content_fails(self, client, notebook):
        """POST quiz marks artifact as failed when generator reports no content."""
        with patch("app.QuizGenerator") as MockGen:
            mock_gen = MagicMock()
            mock_gen.generate_quiz.return_value = {
                "error": "No content found in notebook.",
                "questions": [],
                "metadata": {},
            }
            MockGen.return_value = mock_gen

            resp = client.post(
                f"/notebooks/{notebook.id}/artifacts/quiz",
                json={},
            )

        assert resp.status_code == 200
        assert resp.json()["status"] == "failed"
        assert resp.json()["error_message"] is not None

    def test_generate_quiz_unknown_notebook_404(self, client):
        """POST quiz for a non-existent notebook returns 404."""
        resp = client.post("/notebooks/9999/artifacts/quiz", json={})
        assert resp.status_code == 404

    def test_generate_quiz_stores_metadata(self, client, notebook):
        """Artifact metadata reflects the request parameters."""
        mock_result = {"questions": [], "metadata": {}}
        with patch("app.QuizGenerator") as MockGen:
            mock_gen = MagicMock()
            mock_gen.generate_quiz.return_value = {
                "error": "No content.",
                "questions": [],
                "metadata": {},
            }
            MockGen.return_value = mock_gen

            resp = client.post(
                f"/notebooks/{notebook.id}/artifacts/quiz",
                json={"num_questions": 7, "difficulty": "hard", "title": "Hard Quiz"},
            )

        data = resp.json()
        assert data["title"] == "Hard Quiz"
        assert data["metadata"]["num_questions"] == 7
        assert data["metadata"]["difficulty"] == "hard"


# ── Podcast endpoint tests ────────────────────────────────────────────────────


class TestPodcastEndpoint:
    def test_generate_podcast_returns_pending(self, client, notebook):
        """POST podcast returns 200 with status=pending immediately."""
        with patch("app._run_podcast_background"):
            resp = client.post(
                f"/notebooks/{notebook.id}/artifacts/podcast",
                json={"duration": "5min"},
            )

        assert resp.status_code == 200
        data = resp.json()
        assert data["type"] == "podcast"
        assert data["status"] == "pending"
        assert data["metadata"]["duration"] == "5min"

    def test_generate_podcast_unknown_notebook_404(self, client):
        """POST podcast for a non-existent notebook returns 404."""
        resp = client.post("/notebooks/9999/artifacts/podcast", json={})
        assert resp.status_code == 404

    def test_generate_podcast_stores_topic(self, client, notebook):
        """topic_focus is persisted in artifact metadata."""
        with patch("app._run_podcast_background"):
            resp = client.post(
                f"/notebooks/{notebook.id}/artifacts/podcast",
                json={"duration": "10min", "topic_focus": "neural nets"},
            )

        data = resp.json()
        assert data["metadata"]["topic_focus"] == "neural nets"


# ── List artifacts tests ──────────────────────────────────────────────────────


class TestListArtifacts:
    def test_list_empty(self, client, notebook):
        """GET /artifacts returns empty list when none exist."""
        resp = client.get(f"/notebooks/{notebook.id}/artifacts")
        assert resp.status_code == 200
        assert resp.json() == []

    def test_list_returns_all(self, client, notebook, db_session):
        """GET /artifacts returns all artifacts for the notebook."""
        crud.create_artifact(db=db_session, notebook_id=notebook.id, artifact_type="quiz")
        crud.create_artifact(db=db_session, notebook_id=notebook.id, artifact_type="podcast")

        resp = client.get(f"/notebooks/{notebook.id}/artifacts")
        assert resp.status_code == 200
        assert len(resp.json()) == 2

    def test_list_filter_by_type(self, client, notebook, db_session):
        """GET /artifacts?artifact_type=quiz returns only quizzes."""
        crud.create_artifact(db=db_session, notebook_id=notebook.id, artifact_type="quiz")
        crud.create_artifact(db=db_session, notebook_id=notebook.id, artifact_type="podcast")

        resp = client.get(f"/notebooks/{notebook.id}/artifacts?artifact_type=quiz")
        data = resp.json()
        assert len(data) == 1
        assert data[0]["type"] == "quiz"

    def test_list_unknown_notebook_404(self, client):
        """GET /artifacts for unknown notebook returns 404."""
        resp = client.get("/notebooks/9999/artifacts")
        assert resp.status_code == 404


# ── Get single artifact tests ─────────────────────────────────────────────────


class TestGetArtifact:
    def test_get_existing_artifact(self, client, notebook, db_session):
        """GET /artifacts/{id} returns the correct artifact."""
        artifact = crud.create_artifact(
            db=db_session,
            notebook_id=notebook.id,
            artifact_type="quiz",
            title="My Quiz",
        )

        resp = client.get(f"/notebooks/{notebook.id}/artifacts/{artifact.id}")
        assert resp.status_code == 200
        data = resp.json()
        assert data["id"] == artifact.id
        assert data["title"] == "My Quiz"
        assert data["type"] == "quiz"

    def test_get_nonexistent_artifact_404(self, client, notebook):
        """GET /artifacts/9999 returns 404."""
        resp = client.get(f"/notebooks/{notebook.id}/artifacts/9999")
        assert resp.status_code == 404

    def test_get_artifact_wrong_notebook_404(self, client, db_session):
        """GET /artifacts/{id} returns 404 if artifact belongs to a different notebook."""
        crud.get_or_create_user(db=db_session, user_id=1)
        nb1 = crud.create_notebook(db=db_session, owner_user_id=1, title="NB1")
        nb2 = crud.create_notebook(db=db_session, owner_user_id=1, title="NB2")
        artifact = crud.create_artifact(db=db_session, notebook_id=nb1.id, artifact_type="quiz")

        # Try to access artifact via the wrong notebook
        resp = client.get(f"/notebooks/{nb2.id}/artifacts/{artifact.id}")
        assert resp.status_code == 404


# ── Audio download tests ──────────────────────────────────────────────────────


class TestPodcastAudio:
    def test_audio_pending_returns_409(self, client, notebook, db_session):
        """GET /audio for a pending podcast returns 409."""
        artifact = crud.create_artifact(
            db=db_session,
            notebook_id=notebook.id,
            artifact_type="podcast",
        )

        resp = client.get(f"/notebooks/{notebook.id}/artifacts/{artifact.id}/audio")
        assert resp.status_code == 409

    def test_audio_quiz_artifact_returns_400(self, client, notebook, db_session):
        """GET /audio for a quiz artifact returns 400 (wrong type)."""
        artifact = crud.create_artifact(
            db=db_session,
            notebook_id=notebook.id,
            artifact_type="quiz",
        )
        crud.update_artifact(db_session, artifact.id, status="ready")

        resp = client.get(f"/notebooks/{notebook.id}/artifacts/{artifact.id}/audio")
        assert resp.status_code == 400

    def test_audio_ready_no_file_returns_404(self, client, notebook, db_session, tmp_path):
        """GET /audio returns 404 when audio file is missing from disk."""
        artifact = crud.create_artifact(
            db=db_session,
            notebook_id=notebook.id,
            artifact_type="podcast",
        )
        crud.update_artifact(
            db_session,
            artifact.id,
            status="ready",
            file_path=str(tmp_path / "nonexistent.mp3"),
        )

        resp = client.get(f"/notebooks/{notebook.id}/artifacts/{artifact.id}/audio")
        assert resp.status_code == 404

    def test_audio_ready_with_file_returns_200(self, client, notebook, db_session, tmp_path):
        """GET /audio returns 200 and file content when audio exists."""
        audio_file = tmp_path / "podcast.mp3"
        audio_file.write_bytes(b"fake mp3 content")

        artifact = crud.create_artifact(
            db=db_session,
            notebook_id=notebook.id,
            artifact_type="podcast",
        )
        crud.update_artifact(
            db_session,
            artifact.id,
            status="ready",
            file_path=str(audio_file),
        )

        resp = client.get(f"/notebooks/{notebook.id}/artifacts/{artifact.id}/audio")
        assert resp.status_code == 200
        assert resp.headers["content-type"].startswith("audio/mpeg")
        assert resp.content == b"fake mp3 content"


class TestPodcastTranscript:
    def test_transcript_ready_from_content_returns_200(self, client, notebook, db_session):
        """GET /transcript returns markdown content when transcript file path is absent."""
        artifact = crud.create_artifact(
            db=db_session,
            notebook_id=notebook.id,
            artifact_type="podcast",
        )
        crud.update_artifact(
            db_session,
            artifact.id,
            status="ready",
            content="# Podcast Transcript\n\nHello world",
        )

        resp = client.get(f"/notebooks/{notebook.id}/artifacts/{artifact.id}/transcript")
        assert resp.status_code == 200
        assert resp.headers["content-type"].startswith("text/markdown")
        assert b"Podcast Transcript" in resp.content

    def test_transcript_ready_from_file_returns_200(self, client, notebook, db_session, tmp_path):
        """GET /transcript streams file when transcript_path exists in metadata."""
        transcript_file = tmp_path / "transcript.md"
        transcript_file.write_text("# Transcript\n\nBody", encoding="utf-8")

        artifact = crud.create_artifact(
            db=db_session,
            notebook_id=notebook.id,
            artifact_type="podcast",
            metadata={"transcript_path": str(transcript_file)},
        )
        crud.update_artifact(
            db_session,
            artifact.id,
            status="ready",
            content="# fallback",
        )

        resp = client.get(f"/notebooks/{notebook.id}/artifacts/{artifact.id}/transcript")
        assert resp.status_code == 200
        assert resp.headers["content-type"].startswith("text/markdown")
        assert resp.content == b"# Transcript\n\nBody"


# ── Report endpoint tests ─────────────────────────────────────────────────────


class TestReportEndpoint:
    def test_generate_report_success(self, client, notebook):
        """POST report returns 200 with status=ready and markdown content."""
        with patch("app.ReportGenerator") as MockGen:
            mock_gen = MagicMock()
            mock_gen.generate_report.return_value = {
                "markdown": "# Notebook Report\n\n## Executive Summary\nSummary text.",
                "metadata": {"notebook_id": str(notebook.id)},
            }
            mock_gen.save_report.return_value = "/tmp/report.md"
            MockGen.return_value = mock_gen

            resp = client.post(
                f"/notebooks/{notebook.id}/artifacts/report",
                json={"title": "My Report", "topic_focus": "overview"},
            )

        assert resp.status_code == 200
        data = resp.json()
        assert data["type"] == "report"
        assert data["status"] == "ready"
        assert data["content"].startswith("# Notebook Report")
        assert data["file_path"] == "/tmp/report.md"

    def test_generate_report_error_marks_failed(self, client, notebook):
        """POST report marks artifact failed when generator returns error."""
        with patch("app.ReportGenerator") as MockGen:
            mock_gen = MagicMock()
            mock_gen.generate_report.return_value = {
                "error": "No content found in notebook.",
                "markdown": "",
                "metadata": {},
            }
            MockGen.return_value = mock_gen

            resp = client.post(f"/notebooks/{notebook.id}/artifacts/report", json={})

        assert resp.status_code == 200
        data = resp.json()
        assert data["type"] == "report"
        assert data["status"] == "failed"
        assert data["error_message"] is not None


# ── Generic download tests ────────────────────────────────────────────────────


class TestArtifactDownload:
    def test_download_pending_returns_409(self, client, notebook, db_session):
        artifact = crud.create_artifact(
            db=db_session,
            notebook_id=notebook.id,
            artifact_type="report",
        )
        resp = client.get(f"/notebooks/{notebook.id}/artifacts/{artifact.id}/download")
        assert resp.status_code == 409

    def test_download_ready_markdown_returns_200(self, client, notebook, db_session, tmp_path):
        report_file = tmp_path / "report.md"
        report_file.write_text("# Report\n\nBody", encoding="utf-8")

        artifact = crud.create_artifact(
            db=db_session,
            notebook_id=notebook.id,
            artifact_type="report",
        )
        crud.update_artifact(
            db_session,
            artifact.id,
            status="ready",
            file_path=str(report_file),
        )

        resp = client.get(f"/notebooks/{notebook.id}/artifacts/{artifact.id}/download")
        assert resp.status_code == 200
        assert resp.headers["content-type"].startswith("text/markdown")
        assert resp.content == b"# Report\n\nBody"
