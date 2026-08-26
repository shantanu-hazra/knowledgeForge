from dataclasses import dataclass
from typing import Any, Optional
import logging

logger = logging.getLogger(__name__)


@dataclass
class RetrieverConfig:
    top_k: int = 5
    score_threshold: float = 0.0


class Retriever:
    """Wraps an embedder + vector store to turn a query into ranked, filtered chunks."""

    def __init__(
        self,
        embedder: Any,
        vector_store: "VectorStore",
        top_k: int = 5,
        score_threshold: float = 0.0,
        fetch_multiplier: int = 3,
    ):
        self.embedder = embedder
        self.vector_store = vector_store
        self.top_k = top_k
        self.score_threshold = score_threshold
        self.fetch_multiplier = fetch_multiplier

    def retrieve(
        self,
        query: str,
        top_k: Optional[int] = None,
        score_threshold: Optional[float] = None,
    ) -> list["RetrievedChunk"]:
        """Embed the query, search the vector store, filter by score, return ranked chunks."""
        if not query or not query.strip():
            raise ValueError("query must be a non-empty string")

        effective_top_k = top_k if top_k is not None else self.top_k
        effective_threshold = score_threshold if score_threshold is not None else self.score_threshold

        try:
            query_vector = self.embedder.embed_query(query)
        except Exception as e:
            logger.error(f"Failed to embed query: {e}")
            raise

        try:
            # Over-fetch since threshold filtering happens after search;
            # otherwise a strict threshold can leave you with fewer than top_k
            # results even though better matches exist further down.
            fetch_k = effective_top_k * self.fetch_multiplier
            raw_results = self.vector_store.search(query_vector, top_k=fetch_k)
        except Exception as e:
            logger.error(f"Vector store search failed: {e}")
            raise

        filtered = [r for r in raw_results if r.score >= effective_threshold]
        ranked = sorted(filtered, key=lambda r: r.score, reverse=True)[:effective_top_k]

        logger.info(
            f"Retrieved {len(ranked)}/{len(raw_results)} chunks for query "
            f"(top_k={effective_top_k}, threshold={effective_threshold})"
        )
        return ranked
    
from database.vector_store import VectorStoreConfig, get_vector_store
from rag.embedder import Embedder
from llm.embeddings import EmbeddingsClient

_retriever_instance: Optional[Retriever] = None

def get_retriever() -> Retriever:
    """Builds (once) and returns the shared Retriever instance."""
    global _retriever_instance
    if _retriever_instance is not None:
        return _retriever_instance

    config = RetrieverConfig(top_k=5, score_threshold=0.3)

    vector_config = VectorStoreConfig(
        vector_size=384,
        backend="qdrant",
        collection_name="forge_documents",
        persist_directory="./qdrant_data",
    )

    _retriever_instance = Retriever(
        embedder=Embedder(EmbeddingsClient(model_name="all-MiniLM-L6-v2")),
        vector_store=get_vector_store(vector_config),
        top_k=config.top_k,
        score_threshold=config.score_threshold,
    )
    return _retriever_instance


def run_retrieval(query: str) -> list["RetrievedChunk"]:
    return get_retriever().retrieve(query)