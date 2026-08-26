# app/llm/embeddings.py
from __future__ import annotations
from sentence_transformers import SentenceTransformer

Vector = list[float]


class EmbeddingsClient:
    """Concrete implementation of the embeddings interface expected by Embedder."""

    def __init__(self, model_name: str = "all-MiniLM-L6-v2"):
        self._model = SentenceTransformer(model_name)

    def embed_batch(self, texts: list[str]) -> list[Vector]:
        if not texts:
            return []
        vectors = self._model.encode(texts, batch_size=32, show_progress_bar=False)
        return vectors.tolist()

    def embed(self, text: str) -> Vector:
        return self._model.encode([text], show_progress_bar=False)[0].tolist()