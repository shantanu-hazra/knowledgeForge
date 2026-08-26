"""
agents/_utils.py

Small shared helpers so analysis_agent, writer_agent, and
reviewer_agent don't each reinvent "pull the evidence out of upstream
results". Not registered in AGENT_REGISTRY — this is plumbing, not an
agent.
"""

from typing import Any


def coerce_to_text(value: Any, field_name: str, agent_name: str) -> str:
    """
    Normalizes a task_input field that's supposed to be a plain string
    (e.g. "query", "instruction") but may not be one.

    task_input comes from the planner's LLM-generated JSON — nothing
    validates that individual fields match what a given agent expects,
    only that task_input as a whole is a dict. It's common for the
    planning LLM to wrap a single value in a list (e.g. {"query":
    ["some question"]}) or otherwise return the wrong shape. Rather
    than let that surface as an opaque AttributeError deep in a
    downstream call (e.g. `'list' object has no attribute 'strip'`),
    every agent that expects a string field should route it through
    this first.

    Raises ValueError with a clear message if `value` can't reasonably
    be turned into non-empty text, instead of silently guessing.
    """
    if isinstance(value, str):
        text = value.strip()
    elif isinstance(value, (list, tuple)):
        # Most common LLM slip: wrapping a single string in a list.
        # Join multiple items rather than silently dropping any.
        parts = [str(v).strip() for v in value if v is not None and str(v).strip()]
        text = " ".join(parts)
    elif value is None:
        text = ""
    else:
        text = str(value).strip()

    if not text:
        raise ValueError(
            f"{agent_name} received an empty or unusable '{field_name}' "
            f"in task_input: {value!r}"
        )
    return text


def collect_evidence(results: dict[str, Any]) -> list[dict[str, Any]]:
    """
    Flattens whatever retrieval_agent / research_agent (or any other
    upstream task) put into `results` into one list of evidence items,
    each normalized to at least {"source", "text"} so downstream
    prompts can iterate over a single consistent shape regardless of
    which agents ran.
    """
    evidence: list[dict[str, Any]] = []

    for task_id, output in results.items():
        if not isinstance(output, dict):
            evidence.append({"source": task_id, "text": str(output)})
            continue

        # retrieval_agent shape: {"query": ..., "chunks": [{"text", "source", ...}]}
        if "chunks" in output:
            for chunk in output["chunks"]:
                evidence.append(
                    {
                        "source": chunk.get("source", task_id),
                        "text": chunk.get("text", ""),
                        "citation": chunk.get("citation"),
                        "score": chunk.get("score"),
                    }
                )
        # research_agent shape: {"query": ..., "findings": [{"text", "source", ...}]}
        elif "findings" in output:
            for finding in output["findings"]:
                evidence.append(
                    {
                        "source": finding.get("source", task_id),
                        "text": finding.get("text", ""),
                        "url": finding.get("url"),
                    }
                )
        else:
            evidence.append({"source": task_id, "text": str(output)})

    return evidence


def format_evidence_block(evidence: list[dict[str, Any]]) -> str:
    """Renders evidence items as a numbered block for a prompt."""
    if not evidence:
        return "(no evidence was gathered by upstream agents)"

    lines = []
    for i, item in enumerate(evidence, start=1):
        source = item.get("source", "unknown")
        text = item.get("text", "")
        lines.append(f"[{i}] ({source}) {text}")
    return "\n".join(lines)