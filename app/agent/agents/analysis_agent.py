"""
agents/analysis_agent.py

Executes tasks the planner routed to "analysis_agent" — reasoning,
comparison, and synthesis over evidence collected by upstream agents
(retrieval_agent, research_agent, or both). Depends on those tasks in
the plan, so their outputs arrive via `results`.
"""

from typing import Any

from llm.client import LLM
from llm.schemas import Message
from agent.agents._utils import collect_evidence, format_evidence_block, coerce_to_text

_llm = LLM()

SYSTEM_PROMPT = (
    "You are an analysis agent in a multi-agent workflow. You are given "
    "evidence gathered by other agents and an analytical task. Reason "
    "carefully over the evidence provided — compare, identify patterns, "
    "and draw conclusions. Do not introduce facts that are not present "
    "in the evidence; if the evidence is insufficient to complete the "
    "task, say so explicitly rather than filling gaps with assumptions."
)


def run(task_input: dict[str, Any], results: dict[str, Any]) -> dict[str, Any]:
    """
    task_input expected shape: {"instruction": str} (or ["task"] as a
    fallback). `results` holds the outputs of every task this one
    depends_on — typically retrieval_agent and/or research_agent.
    """
    instruction = task_input.get("instruction") or task_input.get("task")
    if not instruction:
        raise ValueError("analysis_agent requires task_input['instruction'] or ['task']")
    instruction = coerce_to_text(instruction, field_name="instruction", agent_name="analysis_agent")

    evidence = collect_evidence(results)
    evidence_block = format_evidence_block(evidence)

    messages = [
        Message(role="system", content=SYSTEM_PROMPT),
        Message(
            role="user",
            content=(
                f"Analytical task:\n{instruction}\n\n"
                f"Evidence gathered so far:\n{evidence_block}"
            ),
        ),
    ]

    response = _llm.chat(messages)

    return {
        "analysis": response.content,
        "evidence_used": evidence,
    }