from __future__ import annotations
from typing import List, Dict, Any, Optional, Tuple
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

    def upsert_chunks(
        self,
        user_id: str,
        notebook_id: str,
        chunks: List[Dict[str, Any]],
        embeddings: Optional[List[List[float]]] = None,
    ):
        col = self.get_or_create_collection(user_id, notebook_id)
        ids = [c["chunk_id"] for c in chunks]
        documents = [c.get("text", "") for c in chunks]
        metadatas = [
            {
                "user_id": user_id,
                "notebook_id": notebook_id,
                "source_id": str(c.get("source_id", "")),
                "source_title": str(c.get("source_title", "")),
                "chunk_index": int(c.get("chunk_index", -1)),
                "page": int(c.get("page", -1)) if c.get("page") is not None else -1,
                "char_start": int(c.get("char_start", -1)) if c.get("char_start") is not None else -1,
                "char_end": int(c.get("char_end", -1)) if c.get("char_end") is not None else -1,
                "text_preview": str(c.get("text_preview", "")),
            }
            for c in chunks
        ]
        if embeddings is None:
            col.upsert(ids=ids, metadatas=metadatas, documents=documents)
        else:
            col.upsert(ids=ids, embeddings=embeddings, metadatas=metadatas, documents=documents)

    def delete_source(self, user_id: str, notebook_id: str, source_id: str) -> None:
        col = self.get_or_create_collection(user_id, notebook_id)
        col.delete(where={"source_id": str(source_id)})

    def query(
        self,
        user_id: str,
        notebook_id: str,
        query_text: str,
        top_k: int = 5,
        source_id: Optional[str] = None,
        query_embedding: Optional[List[float]] = None,
    ) -> List[Tuple[str, float, Dict[str, Any]]]:
        col = self.get_or_create_collection(user_id, notebook_id)
        where = {"source_id": str(source_id)} if source_id is not None else None
        if query_embedding is None:
            res = col.query(query_texts=[query_text], n_results=top_k, where=where)
        else:
            res = col.query(query_embeddings=[query_embedding], n_results=top_k, where=where)

        ids = res.get("ids", [[]])[0] if res.get("ids") else []
        documents = res.get("documents", [[]])[0] if res.get("documents") else []
        metadatas = res.get("metadatas", [[]])[0] if res.get("metadatas") else []
        distances = res.get("distances", [[]])[0] if res.get("distances") else []

        rows: List[Tuple[str, float, Dict[str, Any]]] = []
        for idx, chunk_id in enumerate(ids):
            doc = documents[idx] if idx < len(documents) else ""
            metadata = metadatas[idx] if idx < len(metadatas) and isinstance(metadatas[idx], dict) else {}
            distance = float(distances[idx]) if idx < len(distances) else 0.0
            rows.append(
                (
                    str(chunk_id),
                    distance,
                    {"document": doc, "metadata": metadata},
                )
            )
        return rows
