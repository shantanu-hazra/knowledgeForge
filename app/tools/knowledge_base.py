"""
tools/knowledge_base.py

Generic agent-tool wrapper around the existing RAG retriever.
Owns no retrieval logic — calls only the existing, unmodified pipeline:

    knowledge_base_search
            │
            ▼
       get_retriever()
            │
            ▼
     retriever.retrieve()
            │
            ▼
          Qdrant
            │
            ▼
      RetrievedChunk[]

No embedding, no Qdrant client, no ingestion code lives here. This
module's only job is to call the retriever and reshape RetrievedChunk[]
into a plain dict, matching the same (validated kwargs) -> dict|list
contract every other tool.fn already follows in ToolDispatcher.call().
"""

from typing import Any

from rag.retriever import run_retrieval  # existing, unmodified pipeline


def knowledge_base_search(query: str) -> dict[str, Any]:
    """
    Matches the calling convention ToolDispatcher uses for every tool:
        result = tool.fn(**validated.model_dump())
    Since this returns a dict, ToolDispatcher.call() will json.dumps()
    it directly (same path as `weather`, `search`, etc.) — no special
    casing needed in the dispatcher.
    """
    chunks = run_retrieval(query)
    print(f"[knowledge_base_search] query={query!r} -> {len(chunks)} chunks")
    return {"results": [_serialize_chunk(chunk) for chunk in chunks]}


def _serialize_chunk(chunk: Any) -> dict[str, Any]:
    """
    Handles both dict-shaped and object-shaped chunks. Your retriever
    returns dicts with a "document" key for text content, "score", and
    "metadata" — adjust further only if this doesn't match exactly.
    """
    if isinstance(chunk, dict):
        get = chunk.get
    else:
        get = lambda key, default=None: getattr(chunk, key, default)

    return {
        "content": get("document") or get("text") or get("content"),
        "source": get("source") or (get("metadata") or {}).get("source"),
        "score": get("score"),
        "metadata": get("metadata") or {},
    }