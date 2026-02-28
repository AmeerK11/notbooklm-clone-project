"""
Integration tests for artifact API endpoints.

Uses FastAPI TestClient with an in-memory SQLite database so no external
services are required.
"""
from __future__ import annotations

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
import app as app_module
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
            mock_gen.format_quiz_markdown.return_value = "# Quiz\n\n## Questions\n\n## Answer Key\n"
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
        assert "## Answer Key" in data["content"]
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


# ── Report endpoint tests ─────────────────────────────────────────────────────


class TestReportEndpoint:
    def test_generate_report_success(self, client, notebook):
        mock_result = {"content": "# Report\n\nGenerated report content.", "detail_level": "medium"}
        with patch("app.ReportGenerator") as MockGen:
            mock_gen = MagicMock()
            mock_gen.generate_report.return_value = mock_result
            mock_gen.save_report.return_value = "/tmp/report.md"
            MockGen.return_value = mock_gen

            resp = client.post(
                f"/notebooks/{notebook.id}/artifacts/report",
                json={"detail_level": "medium"},
            )

        assert resp.status_code == 200
        data = resp.json()
        assert data["type"] == "report"
        assert data["status"] == "ready"
        assert data["content"].startswith("# Report")
        assert data["file_path"] == "/tmp/report.md"

    def test_generate_report_no_content_fails(self, client, notebook):
        with patch("app.ReportGenerator") as MockGen:
            mock_gen = MagicMock()
            mock_gen.generate_report.return_value = {"error": "No content found in notebook."}
            MockGen.return_value = mock_gen

            resp = client.post(
                f"/notebooks/{notebook.id}/artifacts/report",
                json={"detail_level": "short"},
            )

        assert resp.status_code == 200
        assert resp.json()["status"] == "failed"
        assert "No content" in str(resp.json()["error_message"])

    def test_generate_report_invalid_detail_level_400(self, client, notebook):
        resp = client.post(
            f"/notebooks/{notebook.id}/artifacts/report",
            json={"detail_level": "super-long"},
        )
        assert resp.status_code == 400

    def test_generate_report_unknown_notebook_404(self, client):
        resp = client.post("/notebooks/9999/artifacts/report", json={"detail_level": "medium"})
        assert resp.status_code == 404


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

    def test_background_podcast_failure_persists_transcript(self, notebook, db_session):
        """If audio fails but transcript exists, artifact should be failed with transcript content."""
        artifact = crud.create_artifact(
            db=db_session,
            notebook_id=notebook.id,
            artifact_type="podcast",
            metadata={"duration": "5min"},
        )

        result_payload = {
            "error": "Transcript generated but audio synthesis failed for all segments.",
            "transcript": [{"speaker": "Alex", "text": "Intro text"}],
            "audio_path": None,
            "metadata": {"tts_provider": "elevenlabs"},
        }

        with patch("app.PodcastGenerator") as MockGen, patch("app.SessionLocal", return_value=db_session):
            mock_gen = MagicMock()
            mock_gen.generate_podcast.return_value = result_payload
            mock_gen.format_transcript_markdown.return_value = "# Podcast Transcript\n\n**Alex:** Intro text"
            mock_gen.save_transcript.return_value = "/tmp/transcript.md"
            MockGen.return_value = mock_gen

            app_module._run_podcast_background(
                artifact_id=artifact.id,
                user_id=1,
                notebook_id=notebook.id,
                duration="5min",
                topic_focus=None,
            )

        updated = crud.get_artifact(db_session, artifact.id)
        assert updated is not None
        assert updated.status == "failed"
        assert updated.content is not None
        assert "Podcast Transcript" in updated.content
        assert updated.error_message is not None
        assert "audio synthesis failed" in updated.error_message.lower()


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
