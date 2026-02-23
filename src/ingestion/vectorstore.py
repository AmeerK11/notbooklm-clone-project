from __future__ import annotations
from typing import List, Dict, Any, Optional
import chromadb


class ChromaAdapter:
    """Wrapper around Chroma client for per-user-notebook collections.

    Uses a collection named `<user_id>_<notebook_id>` for isolation.
    """

    def __init__(self, persist_directory: Optional[str] = None):
        # Use new Chroma API (1.5+): EphemeralClient or PersistentClient
        if persist_directory:
            self.client = chromadb.PersistentClient(path=persist_directory)
        else:
            self.client = chromadb.EphemeralClient()

    def _collection_name(self, user_id: str, notebook_id: str) -> str:
        return f"{user_id}_{notebook_id}"

    def get_or_create_collection(self, user_id: str, notebook_id: str):
        name = self._collection_name(user_id, notebook_id)
        try:
            return self.client.get_collection(name)
        except Exception:
            return self.client.create_collection(name)

    def upsert_chunks(self, user_id: str, notebook_id: str, chunks: List[Dict[str, Any]], embeddings: List[List[float]]):
        col = self.get_or_create_collection(user_id, notebook_id)
        ids = [c["chunk_id"] for c in chunks]
        documents = [c.get("text", "") for c in chunks]
        metadatas = [
            {
                "user_id": user_id,
                "notebook_id": notebook_id,
                "source_id": c.get("source_id"),
                "page": c.get("page"),
                "char_start": c.get("char_start"),
                "char_end": c.get("char_end"),
                "text_preview": c.get("text_preview"),
            }
            for c in chunks
        ]
        col.upsert(ids=ids, embeddings=embeddings, metadatas=metadatas, documents=documents)
