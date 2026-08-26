"""
agents/writer_agent.py

Executes tasks the planner routed to "writer_agent" — generates the
final user-facing answer from whatever evidence and analysis are
available in `results` (retrieval_agent / research_agent / analysis_agent
outputs, depending on what this task depends_on in the plan). Must stay
grounded in the provided evidence rather than inventing facts.
"""

from typing import Any

from llm.client import LLM
from llm.schemas import Message
from agent.agents._utils import collect_evidence, format_evidence_block, coerce_to_text

_llm = LLM()

SYSTEM_PROMPT = (
    "You are the writer agent in a multi-agent workflow. Your job is to "
    "produce the final answer for the user, grounded strictly in the "
    "evidence and analysis provided below — never invent facts, sources, "
    "or figures that aren't present in them. Write in clear, direct "
    "prose suited to answering the user's original question. If the "
    "evidence is incomplete, acknowledge the gap rather than guessing."
)


def run(task_input: dict[str, Any], results: dict[str, Any]) -> dict[str, Any]:
    """
    task_input expected shape: {"query": str} — the original user
    question this answer is for (falls back to ["task"]). `results`
    holds outputs from whatever this task depends_on, typically
    analysis_agent and/or the raw retrieval_agent/research_agent
    evidence.
    """
    user_query = task_input.get("query") or task_input.get("task")
    if not user_query:
        raise ValueError("writer_agent requires task_input['query'] or ['task']")
    user_query = coerce_to_text(user_query, field_name="query", agent_name="writer_agent")

    analyses = [
        output["analysis"]
        for output in results.values()
        if isinstance(output, dict) and "analysis" in output
    ]
    evidence = collect_evidence(results)
    evidence_block = format_evidence_block(evidence)

    context_parts = [f"Original user question:\n{user_query}"]
    if analyses:
        context_parts.append("Analysis from upstream agents:\n" + "\n---\n".join(analyses))
    context_parts.append(f"Supporting evidence:\n{evidence_block}")

    messages = [
        Message(role="system", content=SYSTEM_PROMPT),
        Message(role="user", content="\n\n".join(context_parts)),
    ]

    response = _llm.chat(messages)

    return {
        "answer": response.content,
        "evidence_used": evidence,
    }