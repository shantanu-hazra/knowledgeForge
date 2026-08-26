# embedder.py
from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from rag.chunker import Chunk

Vector = list[float]


@dataclass
class EmbeddedChunk:
    chunk: Chunk
    vector: Vector

    @property
    def text(self) -> str:
        return self.chunk.text

    @property
    def metadata(self) -> dict[str, Any]:
        return self.chunk.metadata


class LLMEmbeddingsClient:
    """Expected interface from app/llm/embeddings.py (for reference)."""
    def embed_batch(self, texts: list[str]) -> list[Vector]: ...
    def embed(self, text: str) -> Vector: ...


class Embedder:
    def __init__(self, model_client: LLMEmbeddingsClient):
        self.model_client = model_client

    def embed_chunks(self, chunks: list[Chunk]) -> list[EmbeddedChunk]:
        if not chunks:
            return []

        texts = [c.text for c in chunks]
        vectors = self.model_client.embed_batch(texts)

        if len(vectors) != len(chunks):
            raise ValueError(
                f"embed_batch returned {len(vectors)} vectors for {len(chunks)} chunks"
            )

        return [
            EmbeddedChunk(chunk=c, vector=v)
            for c, v in zip(chunks, vectors)
        ]

    def embed_query(self, query: str) -> Vector:
        return self.model_client.embed(query)