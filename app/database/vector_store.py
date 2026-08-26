from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any, Optional
import logging
import uuid

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Data model
# ---------------------------------------------------------------------------

@dataclass
class RetrievedChunk:
    id: str
    document: str
    metadata: dict[str, Any] = field(default_factory=dict)
    score: float = 0.0


# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

@dataclass
class VectorStoreConfig:
    backend: str  # "chroma" | "qdrant" | "pinecone"
    collection_name: str
    persist_directory: Optional[str] = None      # chroma
    url: Optional[str] = None                    # qdrant / pinecone
    api_key: Optional[str] = None                # qdrant cloud / pinecone
    vector_size: Optional[int] = None             # required by qdrant/pinecone at collection creation
    distance_metric: str = "cosine"                # "cosine" | "euclidean" | "dot"


# ---------------------------------------------------------------------------
# Abstract interface
# ---------------------------------------------------------------------------

class VectorStore(ABC):
    """Abstract interface every vector store backend must implement."""

    @abstractmethod
    def add(
        self,
        ids: list[str],
        vectors: list[list[float]],
        documents: list[str],
        metadatas: list[dict[str, Any]],
    ) -> None:
        ...

    @abstractmethod
    def search(self, query_vector: list[float], top_k: int = 5) -> list[RetrievedChunk]:
        ...

    @abstractmethod
    def update(
        self,
        id: str,
        vector: Optional[list[float]] = None,
        document: Optional[str] = None,
        metadata: Optional[dict[str, Any]] = None,
    ) -> None:
        ...

    @abstractmethod
    def delete(self, ids: list[str]) -> None:
        ...

# ---------------------------------------------------------------------------
# Qdrant implementation
# ---------------------------------------------------------------------------

class QdrantVectorStore(VectorStore):
    def __init__(
        self,
        collection_name: str,
        url: Optional[str] = None,
        api_key: Optional[str] = None,
        path: Optional[str] = None,
        vector_size: int = 384,
        distance_metric: str = "cosine",
    ):
        from qdrant_client import QdrantClient
        from qdrant_client.http import models as qmodels

        self._qmodels = qmodels
        self.collection_name = collection_name
        if url:
            self.client = QdrantClient(url=url, api_key=api_key)
        elif path:
            self.client = QdrantClient(path=path)   # <-- physical, on-disk, embedded
        else:
            self.client = QdrantClient(":memory:")

        distance_map = {
            "cosine": qmodels.Distance.COSINE,
            "euclidean": qmodels.Distance.EUCLID,
            "dot": qmodels.Distance.DOT,
        }
        if not self.client.collection_exists(collection_name):
            self.client.create_collection(
                collection_name=collection_name,
                vectors_config=qmodels.VectorParams(
                    size=vector_size,
                    distance=distance_map.get(distance_metric, qmodels.Distance.COSINE),
                ),
            )

    def to_qdrant_id(self, original_id: str) -> str:
        """Deterministically map an arbitrary string id to a valid UUID."""
        return str(uuid.uuid5(uuid.NAMESPACE_DNS, original_id))

    def add(
        self,
        ids: list[str],
        vectors: list[list[float]],
        documents: list[str],
        metadatas: list[dict[str, Any]],
    ) -> None:
        if not (len(ids) == len(vectors) == len(documents) == len(metadatas)):
            raise ValueError("ids, vectors, documents, metadatas must be the same length")

        points = [
            self._qmodels.PointStruct(
                id=self.to_qdrant_id(ids[i]),
                vector=vectors[i],
                payload={"document": documents[i], "original_id": ids[i], **metadatas[i]},
            )
            for i in range(len(ids))
        ]
        self.client.upsert(collection_name=self.collection_name, points=points)

    def search(self, query_vector: list[float], top_k: int = 5) -> list[RetrievedChunk]:
        # NOTE: QdrantClient.search() was removed in newer qdrant-client versions.
        # query_points() is the replacement — note the `query=` kwarg (not
        # `query_vector=`) and that results live under `response.points`.
        response = self.client.query_points(
            collection_name=self.collection_name,
            query=query_vector,
            limit=top_k,
        )
        chunks: list[RetrievedChunk] = []
        for r in response.points:
            payload = dict(r.payload or {})
            document = payload.pop("document", "")
            original_id = payload.pop("original_id", str(r.id))
            chunks.append(
                RetrievedChunk(
                    id=original_id,          # return the human-readable id, not the UUID
                    document=document,
                    metadata=payload,
                    score=r.score,
                )
            )
        return chunks

    def update(
        self,
        id: str,
        vector: Optional[list[float]] = None,
        document: Optional[str] = None,
        metadata: Optional[dict[str, Any]] = None,
    ) -> None:
        qid = self.to_qdrant_id(id)
        if vector is not None:
            self.client.update_vectors(
                collection_name=self.collection_name,
                points=[self._qmodels.PointVectors(id=qid, vector=vector)],
            )
        payload_update: dict[str, Any] = {}
        if document is not None:
            payload_update["document"] = document
        if metadata is not None:
            payload_update.update(metadata)
        if payload_update:
            self.client.set_payload(
                collection_name=self.collection_name,
                payload=payload_update,
                points=[qid],
            )

    def delete(self, ids: list[str]) -> None:
        self.client.delete(
            collection_name=self.collection_name,
            points_selector=self._qmodels.PointIdsList(points=[self.to_qdrant_id(i) for i in ids]),
        )


# ---------------------------------------------------------------------------
# Factory (singleton per collection)
# ---------------------------------------------------------------------------

_vector_stores: dict[str, "VectorStore"] = {}

def get_vector_store(config: VectorStoreConfig) -> VectorStore:
    """Returns a cached VectorStore instance per collection_name.

    Embedded/on-disk backends (like Qdrant with a `path=`) hold an exclusive
    file lock for the life of the process, so the underlying client (and the
    collection-existence check in __init__) must only ever run once per
    process, not once per request.
    """
    cache_key = f"{config.backend}:{config.collection_name}"
    if cache_key in _vector_stores:
        return _vector_stores[cache_key]

    if config.backend != "qdrant":
        raise NotImplementedError(f"backend '{config.backend}' not yet implemented")

    if not config.vector_size:
        raise ValueError("qdrant backend requires config.vector_size")

    store = QdrantVectorStore(
        collection_name=config.collection_name,
        url=config.url,
        api_key=config.api_key,
        path=config.persist_directory,      # <-- local on-disk path
        vector_size=config.vector_size,
        distance_metric=config.distance_metric,
    )
    _vector_stores[cache_key] = store
    return store

    # elif config.backend == "pinecone":
    #     return PineconeVectorStore(...)



# ---------------------------------------------------------------------------
# Example usage
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    config = VectorStoreConfig(
        backend="qdrant",
        collection_name="knowledgeforge_docs",
        persist_directory="./qdrant_data",
        vector_size=1536,
    )
    store = get_vector_store(config)

    store.add(
        ids=["doc1_chunk0"],
        vectors=[[0.01] * 1536],
        documents=["Example chunk text."],
        metadatas=[{"source": "example.pdf", "page": 1}],
    )

    results = store.search(query_vector=[0.01] * 1536, top_k=3)
    for r in results:
        print(r.id, r.score, r.document[:50])