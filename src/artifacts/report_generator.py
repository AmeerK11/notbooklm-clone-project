"""
Report generator using RAG context from ingested notebook content.
"""
from __future__ import annotations

import os
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

from dotenv import load_dotenv
from openai import OpenAI
import requests

from src.ingestion.vectorstore import ChromaAdapter

load_dotenv()

SUPPORTED_REPORT_LLM_PROVIDERS = {"openai", "groq", "ollama"}
DEFAULT_REPORT_MODELS = {
    "openai": "gpt-4o-mini",
    "groq": "llama-3.1-8b-instant",
    "ollama": "qwen2.5:3b",
}
REPORT_SYSTEM_PROMPT = (
    "You write high quality reports grounded only in provided source context. "
    "Do not invent facts."
)


class ReportGenerator:
    def __init__(
        self,
        api_key: Optional[str] = None,
        model: Optional[str] = None,
        llm_provider: Optional[str] = None,
    ):
        provider_default = (
            llm_provider
            or os.getenv("REPORT_LLM_PROVIDER", "").strip()
            or os.getenv("QUIZ_LLM_PROVIDER", "").strip()
            or os.getenv("TRANSCRIPT_LLM_PROVIDER", "").strip()
            or "openai"
        )
        self.llm_provider = provider_default.strip().lower()
        if self.llm_provider not in SUPPORTED_REPORT_LLM_PROVIDERS:
            raise ValueError(
                f"Unsupported REPORT_LLM_PROVIDER='{self.llm_provider}'. "
                f"Choose from: {sorted(SUPPORTED_REPORT_LLM_PROVIDERS)}"
            )

        self.model = self._resolve_model_name(model)
        self._openai_client: OpenAI | None = None
        self._groq_client = None
        self._ollama_base_url = os.getenv("OLLAMA_BASE_URL", "http://127.0.0.1:11434").rstrip("/")

        if self.llm_provider == "openai":
            self.api_key = api_key or os.getenv("OPENAI_API_KEY")
            self._openai_client = OpenAI(api_key=self.api_key)
        elif self.llm_provider == "groq":
            from groq import Groq

            groq_api_key = os.getenv("GROQ_API_KEY")
            if not groq_api_key:
                raise ValueError("GROQ_API_KEY is required when REPORT_LLM_PROVIDER=groq")
            self._groq_client = Groq(api_key=groq_api_key)
        else:
            self.api_key = None

    def _resolve_model_name(self, explicit_model: Optional[str]) -> str:
        if explicit_model and explicit_model.strip():
            return explicit_model.strip()
        configured = os.getenv("REPORT_LLM_MODEL", "").strip()
        if configured:
            return configured
        legacy = os.getenv("LLM_MODEL", "").strip()
        if legacy:
            return legacy
        return DEFAULT_REPORT_MODELS.get(self.llm_provider, "gpt-4o-mini")

    def generate_report(
        self,
        user_id: str,
        notebook_id: str,
        detail_level: str = "medium",
        topic_focus: str | None = None,
    ) -> dict[str, str]:
        context = self._get_report_context(user_id, notebook_id, topic_focus)
        if not context:
            return {"error": "No content found in notebook. Please ingest documents first."}

        report_markdown = self._generate_markdown(context=context, detail_level=detail_level, topic_focus=topic_focus)
        if not report_markdown.strip():
            return {"error": "Failed to generate report content."}

        return {
            "content": report_markdown,
            "detail_level": detail_level,
            "llm_provider": self.llm_provider,
            "llm_model": self.model,
        }

    def _get_report_context(self, user_id: str, notebook_id: str, topic_focus: str | None) -> str:
        storage_base = os.getenv("STORAGE_BASE_DIR", "data")
        chroma_dir = Path(storage_base) / "users" / user_id / "notebooks" / notebook_id / "chroma"
        if not chroma_dir.exists():
            return ""

        store = ChromaAdapter(persist_directory=str(chroma_dir))
        if topic_focus:
            queries = [topic_focus]
        else:
            queries = [
                "main ideas and summary",
                "key evidence and facts",
                "conclusions and action items",
            ]

        chunks: list[str] = []
        for query in queries:
            try:
                results = store.query(user_id, notebook_id, query, top_k=6)
            except Exception:
                continue
            for _, _, chunk_data in results:
                text = str(chunk_data.get("document", "")).strip()
                if text:
                    chunks.append(text)

        if not chunks:
            return ""

        unique_chunks = list(dict.fromkeys(chunks))
        return "\n\n".join(unique_chunks[:14])

    def _generate_markdown(self, context: str, detail_level: str, topic_focus: str | None) -> str:
        target_words = {
            "short": 400,
            "medium": 800,
            "long": 1400,
        }.get(detail_level, 800)

        focus_line = f"Focus area: {topic_focus}" if topic_focus else "Focus area: broad summary of notebook content"
        prompt = f"""
Write a polished Markdown report from the notebook context below.

{focus_line}
Target length: around {target_words} words.

Notebook context:
{context}

Requirements:
- Use Markdown headings and concise sections.
- Include: Executive Summary, Key Insights, Evidence/Examples, Risks or Open Questions, Next Steps.
- Stay faithful to provided context. Do not fabricate unsupported claims.
- Keep tone professional and clear.
- Return Markdown only (no code fences).
"""

        try:
            return self._generate_report_content(prompt)
        except Exception:
            return ""

    def _generate_report_content(self, prompt: str) -> str:
        if self.llm_provider == "openai":
            assert self._openai_client is not None
            response = self._openai_client.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": REPORT_SYSTEM_PROMPT},
                    {"role": "user", "content": prompt},
                ],
                temperature=0.4,
            )
            return str(response.choices[0].message.content or "").strip()

        if self.llm_provider == "groq":
            assert self._groq_client is not None
            response = self._groq_client.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": REPORT_SYSTEM_PROMPT},
                    {"role": "user", "content": prompt},
                ],
                temperature=0.4,
            )
            return str(response.choices[0].message.content or "").strip()

        payload = {
            "model": self.model,
            "system": REPORT_SYSTEM_PROMPT,
            "prompt": prompt,
            "stream": False,
            "options": {"temperature": 0.4},
        }
        response = requests.post(
            f"{self._ollama_base_url}/api/generate",
            json=payload,
            timeout=120,
        )
        response.raise_for_status()
        body = response.json()
        return str(body.get("response", "")).strip()

    def save_report(self, markdown_text: str, user_id: str, notebook_id: str) -> str:
        storage_base = os.getenv("STORAGE_BASE_DIR", "data")
        report_dir = Path(storage_base) / "users" / user_id / "notebooks" / notebook_id / "artifacts" / "reports"
        report_dir.mkdir(parents=True, exist_ok=True)

        timestamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
        path = report_dir / f"report_{timestamp}.md"
        path.write_text(markdown_text, encoding="utf-8")
        return str(path)
