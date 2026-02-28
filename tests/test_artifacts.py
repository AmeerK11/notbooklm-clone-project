"""
Unit tests for quiz and podcast artifact generators.

All external dependencies (OpenAI, ChromaDB, TTS) are mocked so these tests
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

        mock_llm_resp = _make_openai_chat_response(MOCK_QUIZ_LLM_RESPONSE)

        env = {"STORAGE_BASE_DIR": str(tmp_path / "data"), "OPENAI_API_KEY": "test-key"}
        with patch.dict(os.environ, env):
            with patch("src.artifacts.quiz_generator.ChromaAdapter", return_value=mock_store):
                with patch("src.artifacts.quiz_generator.OpenAI") as mock_openai_cls:
                    mock_client = MagicMock()
                    mock_client.chat.completions.create.return_value = mock_llm_resp
                    mock_openai_cls.return_value = mock_client

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
        env = {"STORAGE_BASE_DIR": str(tmp_path / "nonexistent"), "OPENAI_API_KEY": "test-key"}
        with patch.dict(os.environ, env):
            with patch("src.artifacts.quiz_generator.OpenAI"):
                gen = QuizGenerator()
                result = gen.generate_quiz(user_id="1", notebook_id="1")

        assert "error" in result
        assert result["questions"] == []

    def test_generate_quiz_empty_vectorstore_returns_error(self, tmp_path):
        """Returns error dict when vectorstore returns no chunks."""
        _chroma_dir(tmp_path)

        mock_store = MagicMock()
        mock_store.query.return_value = []

        env = {"STORAGE_BASE_DIR": str(tmp_path / "data"), "OPENAI_API_KEY": "test-key"}
        with patch.dict(os.environ, env):
            with patch("src.artifacts.quiz_generator.ChromaAdapter", return_value=mock_store):
                with patch("src.artifacts.quiz_generator.OpenAI"):
                    gen = QuizGenerator()
                    result = gen.generate_quiz(user_id="1", notebook_id="1")

        assert "error" in result

    def test_generate_quiz_defaults_applied(self, tmp_path):
        """Default num_questions and difficulty are read from env vars."""
        _chroma_dir(tmp_path)

        mock_store = MagicMock()
        mock_store.query.return_value = MOCK_CHROMA_RESULTS

        mock_llm_resp = _make_openai_chat_response(
            {"questions": [MOCK_QUIZ_LLM_RESPONSE["questions"][0]] * 3}
        )

        env = {
            "STORAGE_BASE_DIR": str(tmp_path / "data"),
            "OPENAI_API_KEY": "test-key",
            "DEFAULT_QUIZ_QUESTIONS": "3",
            "DEFAULT_QUIZ_DIFFICULTY": "hard",
        }
        with patch.dict(os.environ, env):
            with patch("src.artifacts.quiz_generator.ChromaAdapter", return_value=mock_store):
                with patch("src.artifacts.quiz_generator.OpenAI") as mock_openai_cls:
                    mock_client = MagicMock()
                    mock_client.chat.completions.create.return_value = mock_llm_resp
                    mock_openai_cls.return_value = mock_client

                    gen = QuizGenerator()
                    result = gen.generate_quiz(user_id="1", notebook_id="1")

        assert result["metadata"]["num_questions"] == 3
        assert result["metadata"]["difficulty"] == "hard"

    def test_save_quiz_creates_markdown_file(self, tmp_path):
        """save_quiz writes a markdown file with questions and answer key."""
        quiz_data = {
            "questions": MOCK_QUIZ_LLM_RESPONSE["questions"],
            "metadata": {"num_questions": 1, "difficulty": "easy"},
        }

        with patch("src.artifacts.quiz_generator.OpenAI"):
            gen = QuizGenerator()
            markdown = gen.format_quiz_markdown(quiz_data, title="Quiz")
            saved_path = gen.save_quiz(markdown, "1", "1")

        p = pathlib.Path(saved_path)
        assert p.exists()
        assert p.suffix == ".md"
        saved = p.read_text(encoding="utf-8")
        assert "## Questions" in saved
        assert "## Answer Key" in saved
        assert "1. **B**" in saved

    def test_generate_quiz_normalizes_multiline_options(self, tmp_path):
        """Multiline option strings are normalized into labeled bullet options."""
        _chroma_dir(tmp_path)

        mock_store = MagicMock()
        mock_store.query.return_value = MOCK_CHROMA_RESULTS
        raw_payload = {
            "questions": [
                {
                    "id": 1,
                    "question": "What is the goal?",
                    "options": "A) One\nB) Two\nC) Three\nD) Four",
                    "correct_answer": "B) Two",
                    "explanation": "Two is correct.",
                    "topic": "Goals",
                }
            ]
        }
        mock_llm_resp = _make_openai_chat_response(raw_payload)

        env = {"STORAGE_BASE_DIR": str(tmp_path / "data"), "OPENAI_API_KEY": "test-key"}
        with patch.dict(os.environ, env):
            with patch("src.artifacts.quiz_generator.ChromaAdapter", return_value=mock_store):
                with patch("src.artifacts.quiz_generator.OpenAI") as mock_openai_cls:
                    mock_client = MagicMock()
                    mock_client.chat.completions.create.return_value = mock_llm_resp
                    mock_openai_cls.return_value = mock_client

                    gen = QuizGenerator()
                    result = gen.generate_quiz(user_id="1", notebook_id="1", num_questions=1)
                    markdown = gen.format_quiz_markdown(result, title="Quiz")

        assert "error" not in result
        assert result["questions"][0]["options"] == ["A) One", "B) Two", "C) Three", "D) Four"]
        assert "- A) One" in markdown
        assert "- D) Four" in markdown


# ── PodcastGenerator tests ────────────────────────────────────────────────────


class TestPodcastGenerator:
    def _make_generator(self, tmp_path: pathlib.Path, extra_env: dict | None = None):
        """Convenience: build a PodcastGenerator with EdgeTTS mocked out."""
        env = {
            "STORAGE_BASE_DIR": str(tmp_path / "data"),
            "OPENAI_API_KEY": "test-key",
            "TRANSCRIPT_LLM_PROVIDER": "openai",
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

        mock_llm_resp = _make_openai_chat_response(MOCK_PODCAST_LLM_RESPONSE)

        fake_audio = str(tmp_path / "podcast.mp3")
        pathlib.Path(fake_audio).write_bytes(b"fake-audio")

        env = {
            "STORAGE_BASE_DIR": str(tmp_path / "data"),
            "OPENAI_API_KEY": "test-key",
            "TRANSCRIPT_LLM_PROVIDER": "openai",
            "TTS_PROVIDER": "edge",
        }
        with patch.dict(os.environ, env):
            with patch("src.artifacts.tts_adapter.EdgeTTS"):
                with patch(
                    "src.artifacts.podcast_generator.ChromaAdapter", return_value=mock_store
                ):
                    with patch("src.artifacts.podcast_generator.OpenAI") as mock_openai_cls:
                        mock_client = MagicMock()
                        mock_client.chat.completions.create.return_value = mock_llm_resp
                        mock_openai_cls.return_value = mock_client

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
            "TRANSCRIPT_LLM_PROVIDER": "openai",
            "TTS_PROVIDER": "edge",
        }
        with patch.dict(os.environ, env):
            with patch("src.artifacts.tts_adapter.EdgeTTS"):
                with patch("src.artifacts.podcast_generator.OpenAI"):
                    gen = PodcastGenerator()
                    result = gen.generate_podcast(user_id="1", notebook_id="1")

        assert "error" in result
        assert result["transcript"] == []

    def test_generate_podcast_empty_vectorstore_returns_error(self, tmp_path):
        """Returns error dict when vectorstore has no chunks."""
        _chroma_dir(tmp_path)

        mock_store = MagicMock()
        mock_store.query.return_value = []

        env = {
            "STORAGE_BASE_DIR": str(tmp_path / "data"),
            "OPENAI_API_KEY": "test-key",
            "TRANSCRIPT_LLM_PROVIDER": "openai",
            "TTS_PROVIDER": "edge",
        }
        with patch.dict(os.environ, env):
            with patch("src.artifacts.tts_adapter.EdgeTTS"):
                with patch(
                    "src.artifacts.podcast_generator.ChromaAdapter", return_value=mock_store
                ):
                    with patch("src.artifacts.podcast_generator.OpenAI"):
                        gen = PodcastGenerator()
                        result = gen.generate_podcast(user_id="1", notebook_id="1")

        assert "error" in result

    def test_save_transcript_creates_markdown_file(self, tmp_path):
        """save_transcript writes markdown transcript at the expected path."""
        podcast_data = {
            "transcript": MOCK_PODCAST_LLM_RESPONSE["segments"],
            "audio_path": str(tmp_path / "podcast.mp3"),
            "metadata": {"duration_target": "5min"},
        }

        env = {
            "OPENAI_API_KEY": "test-key",
            "TRANSCRIPT_LLM_PROVIDER": "openai",
            "TTS_PROVIDER": "edge",
        }
        with patch.dict(os.environ, env):
            with patch("src.artifacts.tts_adapter.EdgeTTS"):
                with patch("src.artifacts.podcast_generator.OpenAI"):
                    gen = PodcastGenerator()
                    saved_path = gen.save_transcript(podcast_data, "1", "1")

        p = pathlib.Path(saved_path)
        assert p.exists()
        assert p.suffix == ".md"
        saved = p.read_text(encoding="utf-8")
        assert "# Podcast Transcript" in saved
        assert "## Conversation" in saved
        assert "**Alex:**" in saved

    def test_generate_podcast_topic_focus(self, tmp_path):
        """topic_focus is passed through to metadata."""
        _chroma_dir(tmp_path)

        mock_store = MagicMock()
        mock_store.query.return_value = MOCK_CHROMA_RESULTS

        mock_llm_resp = _make_openai_chat_response(MOCK_PODCAST_LLM_RESPONSE)

        env = {
            "STORAGE_BASE_DIR": str(tmp_path / "data"),
            "OPENAI_API_KEY": "test-key",
            "TRANSCRIPT_LLM_PROVIDER": "openai",
            "TTS_PROVIDER": "edge",
        }
        with patch.dict(os.environ, env):
            with patch("src.artifacts.tts_adapter.EdgeTTS"):
                with patch(
                    "src.artifacts.podcast_generator.ChromaAdapter", return_value=mock_store
                ):
                    with patch("src.artifacts.podcast_generator.OpenAI") as mock_openai_cls:
                        mock_client = MagicMock()
                        mock_client.chat.completions.create.return_value = mock_llm_resp
                        mock_openai_cls.return_value = mock_client

                        gen = PodcastGenerator()
                        with patch.object(gen, "_synthesize_segments", return_value=[]):
                            with patch.object(gen, "_combine_audio", return_value=""):
                                result = gen.generate_podcast(
                                    user_id="1",
                                    notebook_id="1",
                                    topic_focus="neural networks",
                                )

        assert result["metadata"]["topic_focus"] == "neural networks"

    def test_generate_podcast_when_tts_fails_returns_error_with_transcript(self, tmp_path):
        """If TTS produces no audio segments, generator returns an explicit error."""
        _chroma_dir(tmp_path)

        mock_store = MagicMock()
        mock_store.query.return_value = MOCK_CHROMA_RESULTS
        mock_llm_resp = _make_openai_chat_response(MOCK_PODCAST_LLM_RESPONSE)

        env = {
            "STORAGE_BASE_DIR": str(tmp_path / "data"),
            "OPENAI_API_KEY": "test-key",
            "TRANSCRIPT_LLM_PROVIDER": "openai",
            "TTS_PROVIDER": "edge",
        }
        with patch.dict(os.environ, env):
            with patch("src.artifacts.tts_adapter.EdgeTTS"):
                with patch(
                    "src.artifacts.podcast_generator.ChromaAdapter", return_value=mock_store
                ):
                    with patch("src.artifacts.podcast_generator.OpenAI") as mock_openai_cls:
                        mock_client = MagicMock()
                        mock_client.chat.completions.create.return_value = mock_llm_resp
                        mock_openai_cls.return_value = mock_client

                        gen = PodcastGenerator()
                        with patch.object(gen, "_synthesize_segments", return_value=[]):
                            result = gen.generate_podcast(user_id="1", notebook_id="1")

        assert "error" in result
        assert "audio synthesis failed" in str(result["error"]).lower()
        assert isinstance(result.get("transcript"), list)
        assert len(result["transcript"]) > 0
