"""
agents/reviewer_agent.py

Executes tasks the planner routed to "reviewer_agent" — checks the
writer_agent's answer (found in `results`, since this task typically
depends_on writer_agent) against the available evidence for
unsupported claims, missing information, contradictions, and whether
the user's original question was actually answered.
"""

from typing import Any
import json

from llm.client import LLM
from llm.schemas import Message
from agent.agents._utils import collect_evidence, format_evidence_block, coerce_to_text

_llm = LLM()

SYSTEM_PROMPT = (
    "You are the reviewer agent in a multi-agent workflow. You are given "
    "a draft answer and the evidence it was supposed to be grounded in. "
    "Check the answer for: claims not supported by the evidence, "
    "contradictions with the evidence, missing information the evidence "
    "contains but the answer omits, and whether it actually addresses "
    "the user's original question. "
    "Respond with ONLY a JSON object, no other text, of the shape: "
    '{"approved": true|false, "issues": ["..."], "notes": "..."}. '
    "\"issues\" should be empty if there are none. Set \"approved\" to "
    "false if there are any unsupported claims, contradictions, or the "
    "question is not adequately answered."
)


def run(task_input: dict[str, Any], results: dict[str, Any]) -> dict[str, Any]:
    """
    task_input expected shape: {"query": str} — the original user
    question, used to judge completeness (falls back to ["task"]).
    `results` must include the writer_agent output this task
    depends_on; also folds in any retrieval_agent/research_agent
    evidence this task depends on directly, if present.
    """
    user_query = task_input.get("query") or task_input.get("task") or ""
    if user_query:
        user_query = coerce_to_text(user_query, field_name="query", agent_name="reviewer_agent")

    draft_answer = None
    for output in results.values():
        if isinstance(output, dict) and "answer" in output:
            draft_answer = output["answer"]
            break
    if draft_answer is None:
        raise ValueError(
            "reviewer_agent found no writer_agent output in `results` — "
            "does this task depend_on the writer_agent task?"
        )

    evidence = collect_evidence(results)
    evidence_block = format_evidence_block(evidence)

    messages = [
        Message(role="system", content=SYSTEM_PROMPT),
        Message(
            role="user",
            content=(
                f"Original user question:\n{user_query}\n\n"
                f"Draft answer to review:\n{draft_answer}\n\n"
                f"Evidence it should be grounded in:\n{evidence_block}"
            ),
        ),
    ]

    response = _llm.chat(messages)

    try:
        verdict = json.loads(response.content)
    except (json.JSONDecodeError, TypeError):
        # Model didn't return clean JSON — fail closed rather than
        # silently approving an unreviewed answer.
        verdict = {
            "approved": False,
            "issues": ["reviewer_agent could not parse a structured verdict"],
            "notes": response.content,
        }

    verdict.setdefault("approved", False)
    verdict.setdefault("issues", [])
    verdict.setdefault("notes", "")

    return {
        "approved": verdict["approved"],
        "issues": verdict["issues"],
        "notes": verdict["notes"],
        "reviewed_answer": draft_answer,
    }