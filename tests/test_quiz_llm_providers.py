"""
Provider-specific tests for quiz generation.
"""
from __future__ import annotations

import os
import pathlib
import sys
from unittest.mock import MagicMock, patch

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.artifacts.quiz_generator import QuizGenerator


def _prepare_common_mocks(mock_store_cls):
    mock_store = MagicMock()
    mock_store.query.return_value = [
        ("chunk-1", 0.1, {"document": "Context block for quiz generation.", "metadata": {}})
    ]
    mock_store_cls.return_value = mock_store


def test_quiz_generator_ollama_provider_without_openai_key(tmp_path):
    env = {
        "STORAGE_BASE_DIR": str(tmp_path / "data"),
        "QUIZ_LLM_PROVIDER": "ollama",
        "QUIZ_LLM_MODEL": "qwen2.5:3b",
        "OLLAMA_BASE_URL": "http://127.0.0.1:11434",
        "OPENAI_API_KEY": "",
    }
    with patch.dict(os.environ, env, clear=False):
        with patch("src.artifacts.quiz_generator.Path.exists", return_value=True):
            with patch("src.artifacts.quiz_generator.ChromaAdapter") as mock_store_cls:
                _prepare_common_mocks(mock_store_cls)

                mock_resp = MagicMock()
                mock_resp.raise_for_status.return_value = None
                mock_resp.json.return_value = {
                    "response": (
                        '{"questions":[{"id":1,"question":"What is context?",'
                        '"options":["A) A","B) B","C) C","D) D"],'
                        '"correct_answer":"B","explanation":"Because.","topic":"Basics"}]}'
                    )
                }

                with patch("src.artifacts.quiz_generator.requests.post", return_value=mock_resp):
                    generator = QuizGenerator(llm_provider="ollama")
                    result = generator.generate_quiz("1", "1", num_questions=1, difficulty="easy")

    assert "error" not in result
    assert result["metadata"]["llm_provider"] == "ollama"
    assert result["metadata"]["llm_model"] == "qwen2.5:3b"
    assert len(result["questions"]) == 1
    assert result["questions"][0]["correct_answer"] == "B"


def test_quiz_generator_groq_provider_without_openai_key(tmp_path):
    env = {
        "STORAGE_BASE_DIR": str(tmp_path / "data"),
        "QUIZ_LLM_PROVIDER": "groq",
        "QUIZ_LLM_MODEL": "llama-3.1-8b-instant",
        "GROQ_API_KEY": "gsk-test",
        "OPENAI_API_KEY": "",
    }
    with patch.dict(os.environ, env, clear=False):
        with patch("src.artifacts.quiz_generator.Path.exists", return_value=True):
            with patch("src.artifacts.quiz_generator.ChromaAdapter") as mock_store_cls:
                _prepare_common_mocks(mock_store_cls)

                with patch("groq.Groq") as mock_groq_cls:
                    mock_groq = MagicMock()
                    mock_groq.chat.completions.create.return_value = MagicMock(
                        choices=[
                            MagicMock(
                                message=MagicMock(
                                    content=(
                                        '{"questions":[{"id":1,"question":"What is Groq?",'
                                        '"options":["A) A","B) B","C) C","D) D"],'
                                        '"correct_answer":"A","explanation":"Because.","topic":"LLM"}]}'
                                    )
                                )
                            )
                        ]
                    )
                    mock_groq_cls.return_value = mock_groq

                    generator = QuizGenerator(llm_provider="groq")
                    result = generator.generate_quiz("1", "1", num_questions=1, difficulty="easy")

    assert "error" not in result
    assert result["metadata"]["llm_provider"] == "groq"
    assert result["metadata"]["llm_model"] == "llama-3.1-8b-instant"
    assert len(result["questions"]) == 1
    assert result["questions"][0]["correct_answer"] == "A"
