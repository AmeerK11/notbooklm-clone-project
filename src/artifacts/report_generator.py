"""
Report generator using RAG context from ingested notebook content.
"""
from __future__ import annotations

import os
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

from dotenv import load_dotenv

from src.ingestion.vectorstore import ChromaAdapter
from utils.llm_client import generate_chat_completion

load_dotenv()


class ReportGenerator:
    def __init__(self, api_key: Optional[str] = None, model: Optional[str] = None):
        self.api_key = api_key or os.getenv("OPENAI_API_KEY")
        self.model = model or os.getenv("LLM_MODEL", "gpt-4o-mini")

    def generate_report(
        self,
        user_id: str,
        notebook_id: str,
        title: Optional[str] = None,
        topic_focus: Optional[str] = None,
    ) -> Dict[str, Any]:
        context_blocks = self._get_report_context(user_id=user_id, notebook_id=notebook_id, topic_focus=topic_focus)
        if not context_blocks:
            return {
                "error": "No content found in notebook. Please ingest documents first.",
                "markdown": "",
                "metadata": {},
            }

        prompt = self._build_report_prompt(context_blocks=context_blocks, title=title, topic_focus=topic_focus)
        try:
            markdown = generate_chat_completion(
                system_prompt=(
                    "You are an expert technical writer. "
                    "Write clear, concise markdown reports grounded in the provided source chunks."
                ),
                user_prompt=prompt,
            )
        except Exception as exc:
            return {
                "error": f"Failed to generate report: {exc}",
                "markdown": "",
                "metadata": {},
            }

        if not markdown.strip():
            return {
                "error": "Report generation returned empty content.",
                "markdown": "",
                "metadata": {},
            }

        return {
            "markdown": markdown.strip(),
            "metadata": {
                "notebook_id": notebook_id,
                "topic_focus": topic_focus,
                "title": title,
                "model": self.model,
                "generated_at": datetime.utcnow().isoformat(),
                "context_chunks_used": len(context_blocks),
            },
        }

    def save_report(self, markdown: str, user_id: str, notebook_id: str) -> str:
        report_dir = Path(f"data/users/{user_id}/notebooks/{notebook_id}/artifacts/reports")
        report_dir.mkdir(parents=True, exist_ok=True)
        timestamp = datetime.utcnow().strftime("%Y%m%d_%H%M%S")
        file_path = report_dir / f"report_{timestamp}.md"
        file_path.write_text(markdown, encoding="utf-8")
        return str(file_path)

    def _get_report_context(
        self,
        user_id: str,
        notebook_id: str,
        topic_focus: Optional[str] = None,
    ) -> List[str]:
        data_base = os.getenv("STORAGE_BASE_DIR", "data")
        chroma_dir = Path(data_base) / "users" / user_id / "notebooks" / notebook_id / "chroma"
        if not chroma_dir.exists():
            return []

        store = ChromaAdapter(persist_directory=str(chroma_dir))
        queries = [topic_focus] if topic_focus else [
            "document overview and key ideas",
            "important definitions and facts",
            "notable findings and conclusions",
            "examples and practical details",
        ]

        context_blocks: List[str] = []
        seen_docs: set[str] = set()
        for q in queries:
            if not q:
                continue
            try:
                results = store.query(user_id=user_id, notebook_id=notebook_id, query_text=q, top_k=4)
            except Exception:
                continue
            for _, _, chunk_data in results:
                doc = chunk_data.get("document", "").strip()
                if not doc or doc in seen_docs:
                    continue
                metadata = chunk_data.get("metadata", {})
                source_id = metadata.get("source_id", "?")
                source_title = metadata.get("source_title", "Unknown")
                chunk_index = metadata.get("chunk_index", "?")
                context_blocks.append(
                    f"[source_id={source_id}; source_title={source_title}; chunk_index={chunk_index}]\n{doc}"
                )
                seen_docs.add(doc)

        return context_blocks[:16]

    def _build_report_prompt(
        self,
        context_blocks: List[str],
        title: Optional[str],
        topic_focus: Optional[str],
    ) -> str:
        report_title = title or "Notebook Report"
        joined_context = "\n\n".join(context_blocks)
        return f"""
Create a markdown report from the context below.

Report title: {report_title}
Topic focus: {topic_focus or "General"}

Context chunks:
{joined_context}

Requirements:
- Output valid markdown only.
- Include sections: Executive Summary, Key Insights, Detailed Notes, Open Questions, Sources.
- In Sources, list every source/chunk tag you relied on.
- Do not invent facts not present in the context.
- Keep it concise and useful for study/review.
"""
