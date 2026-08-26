"""
agents/retrieval_agent.py

Executes tasks the planner routed to "retrieval_agent" — semantic
search over the private knowledge base via the existing RAG pipeline.

Builds one Retriever (embedder + vector store) at import time and
reuses it across every task — embedding model / vector store
connections are expensive to spin up per-task.
"""

from typing import Any

from database.vector_store import VectorStoreConfig, get_vector_store
from rag.retriever import run_retrieval, Retriever
from agent.agents._utils import coerce_to_text

DEFAULT_TOP_K = 5
DEFAULT_SCORE_THRESHOLD = 0.3

_vector_config = VectorStoreConfig(
    vector_size=384,
    backend="qdrant",
    collection_name="forge_documents",
    persist_directory="./qdrant_data",  # local folder where it'll persist to disk
)




def run(task_input: dict[str, Any], results: dict[str, Any]) -> dict[str, Any]:
    """
    task_input expected shape: {"query": str, "top_k": int (optional)}.
    Falls back to task_input["task"] if "query" wasn't supplied, so a
    plan that only gave a free-text instruction still works. Routed
    through coerce_to_text() since the planning LLM's JSON isn't
    type-checked per field — "query" can come back as a list, number,
    etc. instead of a plain string.
    """
    raw_query = task_input.get("query") or task_input.get("task")
    if not raw_query:
        raise ValueError("retrieval_agent requires task_input['query'] or ['task']")
    query = coerce_to_text(raw_query, field_name="query", agent_name="retrieval_agent")

    top_k = task_input.get("top_k", DEFAULT_TOP_K)

    hits = run_retrieval(query)

    return {
        "query": query,
        "chunks": [
            {
                "text": hit.document,
                "source": hit.metadata.get("source", "unknown"),
                "score": hit.score,
                "citation": hit.metadata.get("citation"),
            }
            for hit in hits
        ],
    }