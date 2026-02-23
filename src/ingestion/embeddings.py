from __future__ import annotations
from typing import List
from sentence_transformers import SentenceTransformer
import os


class EmbeddingAdapter:
    """Flexible embedding adapter supporting multiple providers: local, OpenAI, HuggingFace."""

    def __init__(
        self,
        model_name: str = "all-MiniLM-L6-v2",
        provider: str = "local",  # "local" | "openai" | "huggingface"
    ):
        """
        Args:
            model_name: Model identifier. Defaults to sentence-transformers all-MiniLM-L6-v2.
            provider: One of "local", "openai", "huggingface".
                     - "local": Uses sentence-transformers (offline, no API key).
                     - "openai": Uses OpenAI's embedding API (requires OPENAI_API_KEY).
                     - "huggingface": Uses HF's inference API (requires HF_API_TOKEN).
        """
        self.provider = provider.lower()
        self.model_name = model_name
        self._embedding_model = None

        if self.provider == "local":
            self._init_local(model_name)
        elif self.provider == "openai":
            self._init_openai(model_name)
        elif self.provider == "huggingface":
            self._init_huggingface(model_name)
        else:
            raise ValueError(f"Unknown provider: {self.provider}")

    def _init_local(self, model_name: str):
        """Initialize local sentence-transformers model."""
        prefixed = (
            model_name if model_name.startswith("sentence-transformers/") else f"sentence-transformers/{model_name}"
        )
        self._embedding_model = SentenceTransformer(prefixed)

    def _init_openai(self, model_name: str):
        """Initialize OpenAI embedding client."""
        try:
            from openai import OpenAI
        except ImportError:
            raise ImportError("OpenAI provider requires 'openai' package. Install with: pip install openai")

        api_key = os.getenv("OPENAI_API_KEY")
        if not api_key:
            raise ValueError("OPENAI_API_KEY environment variable not set")

        self._openai_client = OpenAI(api_key=api_key)
        self._openai_model = model_name or "text-embedding-3-small"

    def _init_huggingface(self, model_name: str):
        """Initialize HuggingFace inference API client."""
        try:
            from huggingface_hub import InferenceClient
        except ImportError:
            raise ImportError("HuggingFace provider requires 'huggingface_hub' package. Install with: pip install huggingface-hub")

        hf_token = os.getenv("HF_API_TOKEN")
        if not hf_token:
            raise ValueError("HF_API_TOKEN environment variable not set")

        self._hf_client = InferenceClient(token=hf_token)
        self._hf_model = model_name or "sentence-transformers/all-MiniLM-L6-v2"

    def embed_texts(self, texts: List[str], batch_size: int = 64) -> List[List[float]]:
        """Return list of embedding vectors for `texts`.
        
        Automatically handles batching based on provider.
        """
        if self.provider == "local":
            return self._embed_local(texts, batch_size)
        elif self.provider == "openai":
            return self._embed_openai(texts, batch_size)
        elif self.provider == "huggingface":
            return self._embed_huggingface(texts, batch_size)

    def _embed_local(self, texts: List[str], batch_size: int) -> List[List[float]]:
        """Local embedding using sentence-transformers."""
        embs = self._embedding_model.encode(texts, batch_size=batch_size, show_progress_bar=False)
        # Ensure lists (convert numpy arrays)
        return [e.tolist() if hasattr(e, "tolist") else list(e) for e in embs]

    def _embed_openai(self, texts: List[str], batch_size: int) -> List[List[float]]:
        """OpenAI embedding API (with batching)."""
        embeddings = []
        for i in range(0, len(texts), batch_size):
            batch = texts[i : i + batch_size]
            try:
                response = self._openai_client.embeddings.create(model=self._openai_model, input=batch)
                embeddings.extend([item.embedding for item in response.data])
            except Exception as e:
                raise RuntimeError(f"OpenAI embedding failed: {e}")
        return embeddings

    def _embed_huggingface(self, texts: List[str], batch_size: int) -> List[List[float]]:
        """HuggingFace embedding API (with batching)."""
        embeddings = []
        for i in range(0, len(texts), batch_size):
            batch = texts[i : i + batch_size]
            try:
                response = self._hf_client.feature_extraction(inputs=batch, model=self._hf_model)
                embeddings.extend(response)
            except Exception as e:
                raise RuntimeError(f"HuggingFace embedding failed: {e}")
        return embeddings
