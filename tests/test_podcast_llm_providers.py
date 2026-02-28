"""
Provider-specific tests for podcast transcript generation.
"""
from __future__ import annotations

import os
import pathlib
import sys
from unittest.mock import MagicMock, patch

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.artifacts.podcast_generator import PodcastGenerator


def _prepare_common_mocks(mock_store_cls):
    mock_store = MagicMock()
    mock_store.query.return_value = [
        ("chunk-1", 0.1, {"document": "Context block for podcast generation.", "metadata": {}})
    ]
    mock_store_cls.return_value = mock_store


def test_podcast_generator_ollama_provider_without_openai_key(tmp_path):
    env = {
        "STORAGE_BASE_DIR": str(tmp_path / "data"),
        "TRANSCRIPT_LLM_PROVIDER": "ollama",
        "TRANSCRIPT_LLM_MODEL": "qwen2.5:3b",
        "OLLAMA_BASE_URL": "http://127.0.0.1:11434",
        "TTS_PROVIDER": "edge",
        "OPENAI_API_KEY": "",
    }
    with patch.dict(os.environ, env, clear=False):
        with patch("src.artifacts.tts_adapter.EdgeTTS"):
            with patch("src.artifacts.podcast_generator.Path.exists", return_value=True):
                with patch("src.artifacts.podcast_generator.ChromaAdapter") as mock_store_cls:
                    _prepare_common_mocks(mock_store_cls)

                    mock_resp = MagicMock()
                    mock_resp.raise_for_status.return_value = None
                    mock_resp.json.return_value = {
                        "response": '{"segments":[{"speaker":"Alex","text":"Hello from Ollama."}]}'
                    }

                    with patch("src.artifacts.podcast_generator.requests.post", return_value=mock_resp):
                        generator = PodcastGenerator(llm_provider="ollama")
                        with patch.object(generator, "_synthesize_segments", return_value=[]):
                            with patch.object(generator, "_combine_audio", return_value=""):
                                result = generator.generate_podcast("1", "1")

    assert "error" not in result
    assert result["metadata"]["tts_provider"] == "edge"
    assert len(result["transcript"]) == 1
    assert result["transcript"][0]["speaker"] == "Alex"


def test_podcast_generator_groq_provider_without_openai_key(tmp_path):
    env = {
        "STORAGE_BASE_DIR": str(tmp_path / "data"),
        "TRANSCRIPT_LLM_PROVIDER": "groq",
        "TRANSCRIPT_LLM_MODEL": "llama-3.1-8b-instant",
        "GROQ_API_KEY": "gsk-test",
        "TTS_PROVIDER": "edge",
        "OPENAI_API_KEY": "",
    }
    with patch.dict(os.environ, env, clear=False):
        with patch("src.artifacts.tts_adapter.EdgeTTS"):
            with patch("src.artifacts.podcast_generator.Path.exists", return_value=True):
                with patch("src.artifacts.podcast_generator.ChromaAdapter") as mock_store_cls:
                    _prepare_common_mocks(mock_store_cls)

                    with patch("groq.Groq") as mock_groq_cls:
                        mock_groq = MagicMock()
                        mock_groq.chat.completions.create.return_value = MagicMock(
                            choices=[
                                MagicMock(
                                    message=MagicMock(
                                        content='{"segments":[{"speaker":"Jordan","text":"Hello from Groq."}]}'
                                    )
                                )
                            ]
                        )
                        mock_groq_cls.return_value = mock_groq

                        generator = PodcastGenerator(llm_provider="groq")
                        with patch.object(generator, "_synthesize_segments", return_value=[]):
                            with patch.object(generator, "_combine_audio", return_value=""):
                                result = generator.generate_podcast("1", "1")

    assert "error" not in result
    assert len(result["transcript"]) == 1
    assert result["transcript"][0]["speaker"] == "Jordan"
