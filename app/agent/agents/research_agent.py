"""
agents/research_agent.py

Executes tasks the planner routed to "research_agent" — external
research when the answer needs information outside the private
knowledge base (current events, public sources, anything the RAG
index doesn't cover).

Wraps tools/web_search.web_search(query, max_results) -> {"query":
str, "results": [{"title", "url", "snippet"}, ...], "error": str |
None}. Note this is a single dict, not a list — a prior version of
this file assumed web_search() returned list[Result] with attribute
access (hit.snippet etc.), which actually iterated the dict's keys
("query"/"results"/"error" as bare strings) and crashed on the first
.snippet access. The real payload is results["results"], and each
entry there is a plain dict, not an object — use dict access
throughout.
"""

from typing import Any

from tools.web_search import web_search as search
from agent.agents._utils import coerce_to_text

DEFAULT_MAX_RESULTS = 5


def run(task_input: dict[str, Any], results: dict[str, Any]) -> dict[str, Any]:
    """
    task_input expected shape: {"query": str, "max_results": int (optional)}.
    Falls back to task_input["task"] if "query" wasn't supplied. Routed
    through coerce_to_text() since the planning LLM's JSON isn't
    type-checked per field — "query" can come back as a list, number,
    etc. instead of a plain string.
    """
    raw_query = task_input.get("query") or task_input.get("task")
    if not raw_query:
        raise ValueError("research_agent requires task_input['query'] or ['task']")
    query = coerce_to_text(raw_query, field_name="query", agent_name="research_agent")

    max_results = task_input.get("max_results", DEFAULT_MAX_RESULTS)

    search_response = search(query, max_results=max_results)

    # web_search() fails closed rather than raising — a DDGS-side error
    # (rate limit, network issue, etc.) comes back as
    # {"results": [], "error": "<message>"} rather than an exception.
    # Surface that as a real error here too, instead of silently
    # returning zero findings and letting downstream agents treat "no
    # results" as if the search legitimately found nothing.
    if search_response.get("error"):
        raise RuntimeError(
            f"research_agent's web search failed for query {query!r}: "
            f"{search_response['error']}"
        )

    hits = search_response.get("results", [])

    return {
        "query": query,
        "findings": [
            {
                "text": hit.get("snippet", ""),
                "source": hit.get("title", ""),
                "url": hit.get("url", ""),
            }
            for hit in hits
        ],
    }