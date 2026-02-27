"""
Unit tests for quiz, report, and podcast artifact generators.

All external dependencies (LLM, ChromaDB, TTS) are mocked so these tests
run without network access or API keys.
"""
from __future__ import annotations

import json
import os
import pathlib
import sys
from unittest.mock import MagicMock, patch

import pytest

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.artifacts.quiz_generator import QuizGenerator
from src.artifacts.podcast_generator import PodcastGenerator
from src.artifacts.report_generator import ReportGenerator

# ── Shared fixtures ───────────────────────────────────────────────────────────

MOCK_QUIZ_LLM_RESPONSE = {
    "questions": [
        {
            "id": 1,
            "question": "What is machine learning?",
            "options": [
                "A) A type of computer hardware",
                "B) A method for training models on data",
                "C) A programming language",
                "D) A database technology",
            ],
            "correct_answer": "B",
            "explanation": "Machine learning trains models on data to make predictions.",
            "difficulty": "easy",
            "topic": "Machine Learning Basics",
        }
    ]
}

MOCK_PODCAST_LLM_RESPONSE = {
    "segments": [
        {"speaker": "Alex", "text": "Welcome to our podcast about machine learning!"},
        {"speaker": "Jordan", "text": "Thanks, Alex! Machine learning is fascinating."},
        {"speaker": "Alex", "text": "What is the core idea behind it?"},
        {"speaker": "Jordan", "text": "The core idea is training models on data."},
    ]
}

MOCK_REPORT_MARKDOWN = "# Notebook Report\n\n## Executive Summary\nMachine learning overview."

MOCK_CHROMA_RESULTS = [
    ("chunk-1", 0.85, {"document": "Machine learning is a subset of AI.", "metadata": {}}),
    ("chunk-2", 0.80, {"document": "Models are trained on labelled datasets.", "metadata": {}}),
]


def _make_openai_chat_response(content_dict: dict) -> MagicMock:
    """Build a mock that mimics openai.chat.completions.create() return value."""
    mock_response = MagicMock()
    mock_response.choices[0].message.content = json.dumps(content_dict)
    return mock_response


def _chroma_dir(tmp_path: pathlib.Path, user: str = "1", nb: str = "1") -> pathlib.Path:
    """Create and return the expected chroma directory under tmp_path."""
    d = tmp_path / "data" / "users" / user / "notebooks" / nb / "chroma"
    d.mkdir(parents=True, exist_ok=True)
    return d


# ── QuizGenerator tests ───────────────────────────────────────────────────────


class TestQuizGenerator:
    def test_generate_quiz_returns_questions(self, tmp_path):
        """Returns correct questions dict when context and LLM are available."""
        _chroma_dir(tmp_path)

        mock_store = MagicMock()
        mock_store.query.return_value = MOCK_CHROMA_RESULTS

        env = {"STORAGE_BASE_DIR": str(tmp_path / "data")}
        with patch.dict(os.environ, env):
            with patch("src.artifacts.quiz_generator.ChromaAdapter", return_value=mock_store):
                with patch(
                    "src.artifacts.quiz_generator.generate_chat_completion",
                    return_value=json.dumps(MOCK_QUIZ_LLM_RESPONSE),
                ):
                    gen = QuizGenerator()
                    result = gen.generate_quiz(
                        user_id="1",
                        notebook_id="1",
                        num_questions=1,
                        difficulty="easy",
                    )

        assert "questions" in result
        assert len(result["questions"]) == 1
        assert result["questions"][0]["correct_answer"] == "B"
        assert result["metadata"]["difficulty"] == "easy"
        assert result["metadata"]["num_questions"] == 1

    def test_generate_quiz_no_chroma_dir_returns_error(self, tmp_path):
        """Returns error dict when the chroma directory does not exist."""
        env = {"STORAGE_BASE_DIR": str(tmp_path / "nonexistent")}
        with patch.dict(os.environ, env):
            gen = QuizGenerator()
            result = gen.generate_quiz(user_id="1", notebook_id="1")

        assert "error" in result
        assert result["questions"] == []

    def test_generate_quiz_empty_vectorstore_returns_error(self, tmp_path):
        """Returns error dict when vectorstore returns no chunks."""
        _chroma_dir(tmp_path)

        mock_store = MagicMock()
        mock_store.query.return_value = []

        env = {"STORAGE_BASE_DIR": str(tmp_path / "data")}
        with patch.dict(os.environ, env):
            with patch("src.artifacts.quiz_generator.ChromaAdapter", return_value=mock_store):
                gen = QuizGenerator()
                result = gen.generate_quiz(user_id="1", notebook_id="1")

        assert "error" in result

    def test_generate_quiz_defaults_applied(self, tmp_path):
        """Default num_questions and difficulty are read from env vars."""
        _chroma_dir(tmp_path)

        mock_store = MagicMock()
        mock_store.query.return_value = MOCK_CHROMA_RESULTS

        env = {
            "STORAGE_BASE_DIR": str(tmp_path / "data"),
            "DEFAULT_QUIZ_QUESTIONS": "3",
            "DEFAULT_QUIZ_DIFFICULTY": "hard",
        }
        with patch.dict(os.environ, env):
            with patch("src.artifacts.quiz_generator.ChromaAdapter", return_value=mock_store):
                with patch(
                    "src.artifacts.quiz_generator.generate_chat_completion",
                    return_value=json.dumps({"questions": [MOCK_QUIZ_LLM_RESPONSE["questions"][0]] * 3}),
                ):
                    gen = QuizGenerator()
                    result = gen.generate_quiz(user_id="1", notebook_id="1")

        assert result["metadata"]["num_questions"] == 3
        assert result["metadata"]["difficulty"] == "hard"

    def test_to_markdown_and_save_quiz_creates_markdown_file(self, tmp_path):
        """Quiz markdown includes answer key and is saved as .md."""
        quiz_data = {
            "questions": MOCK_QUIZ_LLM_RESPONSE["questions"],
            "metadata": {"num_questions": 1, "difficulty": "easy"},
        }

        gen = QuizGenerator()
        markdown = gen.to_markdown(quiz_data, title="ML Quiz")
        saved_path = gen.save_quiz(markdown, "1", "1")

        p = pathlib.Path(saved_path)
        assert p.exists()
        assert p.suffix == ".md"
        saved_text = p.read_text(encoding="utf-8")
        assert "## Questions" in saved_text
        assert "## Answer Key" in saved_text


# ── PodcastGenerator tests ────────────────────────────────────────────────────


class TestPodcastGenerator:
    def _make_generator(self, tmp_path: pathlib.Path, extra_env: dict | None = None):
        """Convenience: build a PodcastGenerator with EdgeTTS mocked out."""
        env = {
            "STORAGE_BASE_DIR": str(tmp_path / "data"),
            "OPENAI_API_KEY": "test-key",
            "TTS_PROVIDER": "edge",
            **(extra_env or {}),
        }
        with patch.dict(os.environ, env):
            with patch("src.artifacts.tts_adapter.EdgeTTS"):
                return PodcastGenerator(), env

    def test_generate_podcast_returns_transcript(self, tmp_path):
        """Returns transcript list and audio_path when all mocks succeed."""
        _chroma_dir(tmp_path)

        mock_store = MagicMock()
        mock_store.query.return_value = MOCK_CHROMA_RESULTS

        fake_audio = str(tmp_path / "podcast.mp3")

        env = {
            "STORAGE_BASE_DIR": str(tmp_path / "data"),
            "OPENAI_API_KEY": "test-key",
            "TTS_PROVIDER": "edge",
        }
        with patch.dict(os.environ, env):
            with patch("src.artifacts.tts_adapter.EdgeTTS"):
                with patch(
                    "src.artifacts.podcast_generator.ChromaAdapter", return_value=mock_store
                ):
                    with patch(
                        "src.artifacts.podcast_generator.generate_chat_completion",
                        return_value=json.dumps(MOCK_PODCAST_LLM_RESPONSE),
                    ):
                        gen = PodcastGenerator()

                        with patch.object(gen, "_synthesize_segments", return_value=[fake_audio]):
                            with patch.object(gen, "_combine_audio", return_value=fake_audio):
                                result = gen.generate_podcast(
                                    user_id="1",
                                    notebook_id="1",
                                    duration_target="5min",
                                )

        assert "transcript" in result
        assert len(result["transcript"]) == 4
        assert result["audio_path"] == fake_audio
        assert result["metadata"]["duration_target"] == "5min"

    def test_generate_podcast_no_chroma_dir_returns_error(self, tmp_path):
        """Returns error dict when chroma directory does not exist."""
        env = {
            "STORAGE_BASE_DIR": str(tmp_path / "nonexistent"),
            "OPENAI_API_KEY": "test-key",
            "TTS_PROVIDER": "edge",
        }
        with patch.dict(os.environ, env):
            with patch("src.artifacts.tts_adapter.EdgeTTS"):
                gen = PodcastGenerator()
                result = gen.generate_podcast(user_id="1", notebook_id="1")

        assert "error" in result
        assert result["transcript"] == []


class TestReportGenerator:
    def test_generate_report_returns_markdown(self, tmp_path):
        """Report generator returns markdown when context and LLM are available."""
        _chroma_dir(tmp_path)

        mock_store = MagicMock()
        mock_store.query.return_value = MOCK_CHROMA_RESULTS

        env = {"STORAGE_BASE_DIR": str(tmp_path / "data"), "OPENAI_API_KEY": "test-key"}
        with patch.dict(os.environ, env):
            with patch("src.artifacts.report_generator.ChromaAdapter", return_value=mock_store):
                with patch(
                    "src.artifacts.report_generator.generate_chat_completion",
                    return_value=MOCK_REPORT_MARKDOWN,
                ):
                    gen = ReportGenerator()
                    result = gen.generate_report(user_id="1", notebook_id="1", title="My Report")

        assert "error" not in result
        assert result["markdown"].startswith("# Notebook Report")
        assert result["metadata"]["notebook_id"] == "1"

    def test_save_report_creates_markdown_file(self, tmp_path):
        """save_report writes a .md file under the reports artifact folder."""
        env = {"OPENAI_API_KEY": "test-key"}
        with patch.dict(os.environ, env):
            gen = ReportGenerator()
            saved_path = gen.save_report(MOCK_REPORT_MARKDOWN, "1", "1")

        p = pathlib.Path(saved_path)
        assert p.exists()
        assert p.suffix == ".md"
        assert p.read_text(encoding="utf-8").startswith("# Notebook Report")


class TestPodcastGeneratorMore:
    def test_generate_podcast_empty_vectorstore_returns_error(self, tmp_path):
        """Returns error dict when vectorstore has no chunks."""
        _chroma_dir(tmp_path)

        mock_store = MagicMock()
        mock_store.query.return_value = []

        env = {
            "STORAGE_BASE_DIR": str(tmp_path / "data"),
            "OPENAI_API_KEY": "test-key",
            "TTS_PROVIDER": "edge",
        }
        with patch.dict(os.environ, env):
            with patch("src.artifacts.tts_adapter.EdgeTTS"):
                with patch(
                    "src.artifacts.podcast_generator.ChromaAdapter", return_value=mock_store
                ):
                    gen = PodcastGenerator()
                    result = gen.generate_podcast(user_id="1", notebook_id="1")

        assert "error" in result

    def test_save_transcript_creates_markdown_file(self, tmp_path):
        """save_transcript writes a markdown transcript file at the expected path."""
        podcast_data = {
            "transcript": MOCK_PODCAST_LLM_RESPONSE["segments"],
            "audio_path": str(tmp_path / "podcast.mp3"),
            "metadata": {
                "duration_target": "5min",
                "tts_provider": "edge",
                "generated_at": "2026-01-01T00:00:00",
            },
        }

        env = {"OPENAI_API_KEY": "test-key", "TTS_PROVIDER": "edge"}
        with patch.dict(os.environ, env):
            with patch("src.artifacts.tts_adapter.EdgeTTS"):
                gen = PodcastGenerator()
                markdown = gen.to_markdown(podcast_data, title="Podcast Transcript")
                saved_path = gen.save_transcript(markdown, "1", "1")

        p = pathlib.Path(saved_path)
        assert p.exists()
        assert p.suffix == ".md"
        saved_text = p.read_text(encoding="utf-8")
        assert "# Podcast Transcript" in saved_text
        assert "## Transcript" in saved_text

    def test_generate_podcast_topic_focus(self, tmp_path):
        """topic_focus is passed through to metadata."""
        _chroma_dir(tmp_path)

        mock_store = MagicMock()
        mock_store.query.return_value = MOCK_CHROMA_RESULTS

        env = {
            "STORAGE_BASE_DIR": str(tmp_path / "data"),
            "OPENAI_API_KEY": "test-key",
            "TTS_PROVIDER": "edge",
        }
        with patch.dict(os.environ, env):
            with patch("src.artifacts.tts_adapter.EdgeTTS"):
                with patch(
                    "src.artifacts.podcast_generator.ChromaAdapter", return_value=mock_store
                ):
                    with patch(
                        "src.artifacts.podcast_generator.generate_chat_completion",
                        return_value=json.dumps(MOCK_PODCAST_LLM_RESPONSE),
                    ):
                        gen = PodcastGenerator()
                        with patch.object(gen, "_synthesize_segments", return_value=[]):
                            with patch.object(gen, "_combine_audio", return_value=""):
                                result = gen.generate_podcast(
                                    user_id="1",
                                    notebook_id="1",
                                    topic_focus="neural networks",
                                )

        assert result["metadata"]["topic_focus"] == "neural networks"
