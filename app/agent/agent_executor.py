"""
agent_executor.py

Execution step: given a PlannerDecision's task list, run each task
against the matching entry in AGENT_REGISTRY, respecting `depends_on`
ordering, and hand back a plain dict of results keyed by task_id.

This module has no planning logic — it only executes what the planner
already decided. It has no per-agent business logic either: each
agent module just needs to expose a `run(task_input, results) -> Any`
callable; wiring a new one in is a one-line addition to AGENT_RUNNERS.

NOTE: this is the supervisor's execution step. Earlier versions of
this module mirrored OpenAI's tool-call/tool-result message protocol
(matching call.id -> tool_call_id) so results could be fed back into
another chat() round. That's gone — the supervisor invokes agents
directly off PlannerDecision.tasks and gets results back as a plain
dict, with no provider-specific message shape involved. Whatever the
supervisor does with those results next (summarize them, feed them
into the next planning prompt, etc.) is its own concern, not this
module's.
"""

from typing import Any
import asyncio

from agent.agent_registry import AGENT_REGISTRY
from agent.planner import AgentTask

# Plug your actual agent implementations in here. Each must expose a
# `run(task_input: dict, results: dict[str, Any]) -> Any` callable —
# `results` gives an agent access to the outputs of the tasks it
# depends_on, keyed by task_id.
from agent.agents import (
    retrieval_agent,
    research_agent,
    analysis_agent,
    writer_agent,
    reviewer_agent,
)

AGENT_RUNNERS = {
    "retrieval_agent": retrieval_agent.run,
    "research_agent": research_agent.run,
    "analysis_agent": analysis_agent.run,
    "writer_agent": writer_agent.run,
    "reviewer_agent": reviewer_agent.run,
}

assert AGENT_RUNNERS.keys() == AGENT_REGISTRY.keys(), (
    "AGENT_RUNNERS must stay in lockstep with agent.registry.AGENT_REGISTRY — "
    "every registered agent needs a runner, and vice versa."
)


async def run_plan(tasks: list[AgentTask]) -> dict[str, Any]:
    """
    Runs `tasks` to completion, respecting depends_on, and returns the
    raw results dict keyed by task_id. This is what the supervisor
    calls directly with PlannerDecision.tasks — there's no
    tool_call/Message translation step anymore.

    Tasks whose dependencies are already satisfied run concurrently;
    tasks with no dependency relationship to one another (e.g.
    retrieval_agent and research_agent both required by analysis_agent)
    are dispatched in parallel automatically.
    """
    results: dict[str, Any] = {}
    remaining = {t.task_id: t for t in tasks}

    while remaining:
        ready = [
            t for t in remaining.values()
            if all(dep in results for dep in t.depends_on)
        ]
        if not ready:
            raise ValueError(
                f"Unresolvable task dependencies among: {list(remaining)}"
            )

        outputs = await asyncio.gather(*(_run_one(t, results) for t in ready))
        for task, output in zip(ready, outputs):
            results[task.task_id] = output
            del remaining[task.task_id]

    return results


async def _run_one(task: AgentTask, results: dict[str, Any]) -> Any:
    runner = AGENT_RUNNERS[task.agent_name]  # agent_name already validated by AgentTask
    return await asyncio.to_thread(runner, task.task_input, results)