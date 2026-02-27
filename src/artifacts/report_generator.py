"""
Report generator using RAG context from ingested notebook content.
"""
from __future__ import annotations

import os
from datetime import datetime
from pathlib import Path
from typing import Optional

from dotenv import load_dotenv
from openai import OpenAI

from src.ingestion.vectorstore import ChromaAdapter

load_dotenv()


class ReportGenerator:
    def __init__(self, api_key: Optional[str] = None, model: Optional[str] = None):
        self.api_key = api_key or os.getenv("OPENAI_API_KEY")
        self.model = model or os.getenv("LLM_MODEL", "gpt-4o-mini")
        self.client = OpenAI(api_key=self.api_key)

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
            response = self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {
                        "role": "system",
                        "content": (
                            "You write high quality reports grounded only in provided source context. "
                            "Do not invent facts."
                        ),
                    },
                    {"role": "user", "content": prompt},
                ],
                temperature=0.4,
            )
            content = response.choices[0].message.content or ""
            return str(content).strip()
        except Exception:
            return ""

    def save_report(self, markdown_text: str, user_id: str, notebook_id: str) -> str:
        report_dir = Path(f"data/users/{user_id}/notebooks/{notebook_id}/artifacts/reports")
        report_dir.mkdir(parents=True, exist_ok=True)

        timestamp = datetime.utcnow().strftime("%Y%m%d_%H%M%S")
        path = report_dir / f"report_{timestamp}.md"
        path.write_text(markdown_text, encoding="utf-8")
        return str(path)
