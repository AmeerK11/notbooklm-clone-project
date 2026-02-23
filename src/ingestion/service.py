from __future__ import annotations

import os
from pathlib import Path
from typing import Any

from data.models import Source

from .chunker import chunk_text
from .embeddings import EmbeddingAdapter
from .extractors import (
    extract_text_from_pdf,
    extract_text_from_pptx,
    extract_text_from_txt,
    extract_text_from_url,
)
from .storage import LocalStorageAdapter
from .vectorstore import ChromaAdapter


def _coerce_user_notebook_ids(owner_user_id: int, notebook_id: int) -> tuple[str, str]:
    return str(owner_user_id), str(notebook_id)


def _extract_source_text(source: Source) -> str:
    if source.type == "url" and source.url:
        return extract_text_from_url(source.url).get("text", "")

    if not source.storage_path:
        return ""

    file_path = Path(source.storage_path)
    if not file_path.exists() or not file_path.is_file():
        return ""

    suffix = file_path.suffix.lower()
    if suffix == ".pdf":
        return extract_text_from_pdf(file_path, use_ocr=False).get("text", "")
    if suffix == ".pptx":
        return extract_text_from_pptx(file_path).get("text", "")
    if suffix in {".txt", ".md"}:
        return extract_text_from_txt(file_path).get("text", "")

    try:
        return file_path.read_text(encoding="utf-8")
    except UnicodeDecodeError:
        return file_path.read_text(encoding="latin-1")


def _build_store(owner_user_id: int, notebook_id: int) -> tuple[ChromaAdapter, str, str]:
    user_id, notebook_key = _coerce_user_notebook_ids(owner_user_id, notebook_id)
    adapter = LocalStorageAdapter(base_dir=os.getenv("STORAGE_BASE_DIR", "data"))
    notebook_path = adapter.ensure_notebook(user_id, notebook_key)
    chroma_dir = str((notebook_path / "chroma").resolve())
    return ChromaAdapter(persist_directory=chroma_dir), user_id, notebook_key


def ingest_source(source: Source, owner_user_id: int) -> int:
    text = _extract_source_text(source)
    if not text.strip():
        return 0

    chunk_model = os.getenv("CHUNK_MODEL", "sentence-transformers/all-MiniLM-L6-v2")
    chunks = chunk_text(text, model_name=chunk_model)
    if not chunks:
        return 0

    source_id = str(source.id)
    for idx, c in enumerate(chunks):
        c["source_id"] = source_id
        c["source_title"] = source.title or source.original_name or ""
        c["chunk_index"] = idx
        c["page"] = None

    provider = os.getenv("EMBEDDING_PROVIDER", "local")
    model_name = os.getenv("EMBEDDING_MODEL", "all-MiniLM-L6-v2")
    embedder = EmbeddingAdapter(model_name=model_name, provider=provider)
    embeddings = embedder.embed_texts([c["text"] for c in chunks], batch_size=32)

    store, user_id, notebook_key = _build_store(owner_user_id, source.notebook_id)
    store.delete_source(user_id=user_id, notebook_id=notebook_key, source_id=source_id)
    store.upsert_chunks(user_id=user_id, notebook_id=notebook_key, chunks=chunks, embeddings=embeddings)
    return len(chunks)


def query_notebook_chunks(
    owner_user_id: int,
    notebook_id: int,
    query_text: str,
    top_k: int = 5,
    source_id: int | None = None,
) -> list[dict[str, Any]]:
    store, user_id, notebook_key = _build_store(owner_user_id, notebook_id)
    rows = store.query(
        user_id=user_id,
        notebook_id=notebook_key,
        query_text=query_text,
        top_k=top_k,
        source_id=(str(source_id) if source_id is not None else None),
        query_embedding=EmbeddingAdapter(
            model_name=os.getenv("EMBEDDING_MODEL", "all-MiniLM-L6-v2"),
            provider=os.getenv("EMBEDDING_PROVIDER", "local"),
        ).embed_texts([query_text], batch_size=1)[0],
    )
    return [
        {
            "chunk_id": chunk_id,
            "score": score,
            "document": data.get("document", ""),
            "metadata": data.get("metadata", {}),
        }
        for chunk_id, score, data in rows
    ]
