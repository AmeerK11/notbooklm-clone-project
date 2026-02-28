"""
Provider-specific tests for report generation.
"""
from __future__ import annotations

import os
import pathlib
import sys
from unittest.mock import MagicMock, patch

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.artifacts.report_generator import ReportGenerator


def _prepare_store(mock_store_cls):
    mock_store = MagicMock()
    mock_store.query.return_value = [
        ("chunk-1", 0.1, {"document": "Context block for report generation.", "metadata": {}})
    ]
    mock_store_cls.return_value = mock_store


def test_report_generator_ollama_provider_without_openai_key(tmp_path):
    env = {
        "STORAGE_BASE_DIR": str(tmp_path / "data"),
        "REPORT_LLM_PROVIDER": "ollama",
        "REPORT_LLM_MODEL": "qwen2.5:3b",
        "OLLAMA_BASE_URL": "http://127.0.0.1:11434",
        "OPENAI_API_KEY": "",
    }
    with patch.dict(os.environ, env, clear=False):
        with patch("src.artifacts.report_generator.Path.exists", return_value=True):
            with patch("src.artifacts.report_generator.ChromaAdapter") as mock_store_cls:
                _prepare_store(mock_store_cls)

                mock_resp = MagicMock()
                mock_resp.raise_for_status.return_value = None
                mock_resp.json.return_value = {"response": "# Report\n\nGenerated from Ollama."}

                with patch("src.artifacts.report_generator.requests.post", return_value=mock_resp):
                    generator = ReportGenerator(llm_provider="ollama")
                    result = generator.generate_report("1", "1")

    assert "error" not in result
    assert "content" in result
    assert "Generated from Ollama." in result["content"]
    assert result["llm_provider"] == "ollama"


def test_report_generator_groq_provider_without_openai_key(tmp_path):
    env = {
        "STORAGE_BASE_DIR": str(tmp_path / "data"),
        "REPORT_LLM_PROVIDER": "groq",
        "REPORT_LLM_MODEL": "llama-3.1-8b-instant",
        "GROQ_API_KEY": "gsk-test",
        "OPENAI_API_KEY": "",
    }
    with patch.dict(os.environ, env, clear=False):
        with patch("src.artifacts.report_generator.Path.exists", return_value=True):
            with patch("src.artifacts.report_generator.ChromaAdapter") as mock_store_cls:
                _prepare_store(mock_store_cls)

                with patch("groq.Groq") as mock_groq_cls:
                    mock_groq = MagicMock()
                    mock_groq.chat.completions.create.return_value = MagicMock(
                        choices=[MagicMock(message=MagicMock(content="# Report\n\nGenerated from Groq."))]
                    )
                    mock_groq_cls.return_value = mock_groq

                    generator = ReportGenerator(llm_provider="groq")
                    result = generator.generate_report("1", "1")

    assert "error" not in result
    assert "content" in result
    assert "Generated from Groq." in result["content"]
    assert result["llm_provider"] == "groq"
