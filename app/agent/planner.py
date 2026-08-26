"""
planner.py

Decision-only step: given a user query, conversation memory, and the
hardcoded agent registry, ask the LLM which agent(s) it needs and what
task each one should perform. Returns a PlannerDecision.

This module NEVER executes an agent — that's the supervisor's job
(see agent_executor.run_plan). It has no per-agent business logic and
no keyword/heuristic branching — agent selection and task
decomposition are delegated entirely to the LLM, which is asked to
return a plan as JSON matching the PlannerDecision schema below.
Adding a new agent requires zero changes here: register it in
agent.registry.AGENT_SCHEMAS (including its required_input_keys) and
the planner picks it up automatically.

NOTE: this intentionally does NOT use OpenAI-native function-calling
(no `tools=` param, no `response.tool_calls`). The LLM is prompted to
emit a JSON object shaped like PlannerDecision and we validate it
directly with pydantic — this keeps the planner provider-agnostic and
lets the supervisor invoke agents directly rather than round-tripping
through a tool-call protocol.
"""

from typing import Any, Optional
import json

from pydantic import BaseModel, Field, ValidationError, model_validator

from llm.client import LLM
from llm.schemas import Message
from agent.prompt_builder import build_prompt, DEFAULT_SYSTEM_PROMPT
from agent.agent_registry import AGENT_SCHEMAS, AGENT_REGISTRY
from database.session import Session

EVIDENCE_AGENTS = {"retrieval_agent", "research_agent", "analysis_agent"}
SYNTHESIS_AGENTS = {"writer_agent"}

class AgentTask(BaseModel):
    task_id: str = Field(
        ..., description="Unique id for this task within the plan, used by depends_on."
    )
    agent_name: str = Field(
        ..., description="Name of the selected agent, must match an entry in AGENT_REGISTRY."
    )
    task_input: dict[str, Any] = Field(
        default_factory=dict,
        description="Arguments/instructions for the agent, matching that agent's expected input.",
    )
    depends_on: list[str] = Field(
        default_factory=list,
        description="task_ids of other tasks in this plan that must complete before this one runs.",
    )

    @model_validator(mode="after")
    def _agent_must_be_registered(self) -> "AgentTask":
        if self.agent_name not in AGENT_REGISTRY:
            raise ValueError(
                f"Planner selected unknown agent '{self.agent_name}'; "
                f"expected one of {list(AGENT_REGISTRY)}"
            )
        return self

    @model_validator(mode="after")
    def _task_input_has_required_keys(self) -> "AgentTask":
        """
        Cross-checks task_input against the agent's declared
        required_input_keys (see AGENT_SCHEMAS). This is what turns a
        malformed plan into a ValidationError the caller can catch and
        retry/replan on, instead of a bare ValueError raised deep
        inside that agent's own run() at execution time — by then
        it's a worker-thread crash, not a planning-time failure.

        Every agent in AGENT_REGISTRY also accepts a generic "task"
        key as a fallback for its declared primary key (verified
        against each agent's own run() — e.g. writer_agent does
        `task_input.get("query") or task_input.get("task")`), so this
        check is satisfied by either the agent's specific key(s) or
        "task", not only by an exact match on required_input_keys.

        Deliberately does NOT skip this check for tasks with
        depends_on: none of the current agents source their primary
        text field from `results` — writer_agent and reviewer_agent
        both read task_input directly regardless of what they
        depend_on, so a depends_on carve-out here would just let the
        original bug back in for the (very common) case of a
        writer_agent task that depends on retrieval/analysis.
        """
        schema = AGENT_REGISTRY.get(self.agent_name, {})
        required = schema.get("required_input_keys", [])
        if not required:
            return self
        accepted_keys = set(required) | {"task"}
        if not any(k in self.task_input for k in accepted_keys):
            raise ValueError(
                f"Task '{self.task_id}' for agent '{self.agent_name}' is missing "
                f"a required task_input key — needs one of {sorted(accepted_keys)}; "
                f"got keys {list(self.task_input)}"
            )
        return self


class PlannerDecision(BaseModel):
    tasks_required: bool = Field(
        ..., description="Whether any agent task is needed to answer this turn."
    )
    tasks: Optional[list[AgentTask]] = Field(
        default=None,
        description="Ordered/dependent set of agent tasks that make up the plan.",
    )
    reasoning: Optional[str] = Field(
        default=None,
        description="Optional free-text rationale, never used for control flow.",
    )

    @model_validator(mode="after")
    def _synthesis_must_depend_on_evidence(self) -> "PlannerDecision":
        if not self.tasks:
            return self
        ids_by_agent: dict[str, list[str]] = {}
        for t in self.tasks:
            ids_by_agent.setdefault(t.agent_name, []).append(t.task_id)

        evidence_ids = {
            tid for agent in EVIDENCE_AGENTS for tid in ids_by_agent.get(agent, [])
        }
        if not evidence_ids:
            return self

        for t in self.tasks:
            if t.agent_name in SYNTHESIS_AGENTS and not (set(t.depends_on) & evidence_ids):
                raise ValueError(
                    f"Task '{t.task_id}' ({t.agent_name}) must depends_on at least "
                    f"one of the evidence-gathering tasks in this plan "
                    f"{sorted(evidence_ids)}, otherwise it will synthesize an "
                    f"answer with no retrieved context."
                )
        return self


def _build_planning_system_prompt() -> str:
    """
    Renders AGENT_SCHEMAS into a system prompt that asks the LLM to
    return a plan as raw JSON matching PlannerDecision. This is the
    replacement for passing AGENT_SCHEMAS as an OpenAI `tools` param —
    the schema is described in the prompt itself instead.

    Each agent's required_input_keys are spelled out explicitly here —
    capabilities/description alone don't tell the LLM what task_input
    shape a given agent expects, which previously let it invent
    arbitrary keys (e.g. omitting "query" for writer_agent) that only
    failed once the agent actually ran.
    """
    def _agent_line(a: dict[str, Any]) -> str:
        required = a.get("required_input_keys", [])
        if required:
            req_str = (
                f'; task_input MUST include key(s): {", ".join(required)} '
                f'(a generic "task" key is also accepted as a fallback)'
            )
        else:
            req_str = "; task_input has no strictly required keys"
        return (
            f"- {a['name']}: {a['description']} "
            f"(capabilities: {', '.join(a['capabilities'])}{req_str})"
        )

    agents_block = "\n".join(_agent_line(a) for a in AGENT_SCHEMAS)
    return (
        "You are the supervisor's planning step in a multi-agent workflow. "
        "Given the conversation so far, decide whether any of the following "
        "agents are needed to answer the user, and if so, what task(s) to "
        "give them:\n\n"
        f"{agents_block}\n\n"
        "Respond with ONLY a single JSON object, no other text, no markdown "
        "fences, matching this shape:\n"
        '{"tasks_required": true|false, '
        '"tasks": [{"task_id": "t1", "agent_name": "<one of the names above>", '
        '"task_input": {...arguments for that agent, including every key '
        'marked as required above...}, '
        '"depends_on": ["<task_id>", ...]}], '
        '"reasoning": "<optional short rationale>"}\n\n'
        'If no agent is needed, respond with {"tasks_required": false, "tasks": null, '
        '"reasoning": "..."}. '
        "task_id values must be unique within the plan. depends_on must only "
        "reference task_ids defined elsewhere in the same plan — use it to "
        "sequence tasks whose input depends on another task's output (e.g. "
        "analysis_agent depending on retrieval_agent), and omit it (or leave "
        "it empty) for tasks that can run independently. Every task's "
        "task_input must include that agent's required key(s) listed above, "
        "REGARDLESS of depends_on — depends_on only threads upstream evidence "
        "into an agent's `results`, it never substitutes for that agent's own "
        "required task_input key(s). For example a writer_agent task must "
        "always include a \"query\" key with the user's original question, "
        "even when it also depends_on a retrieval_agent or analysis_agent "
        "task for supporting evidence. reviewer_agent tasks should always "
        "depends_on the writer_agent task they're meant to review, since "
        "that's how reviewer_agent finds the draft answer to check. "
        "Likewise, any writer_agent task that is meant to use evidence "
        "gathered by a retrieval_agent, research_agent, or analysis_agent "
        "task in THIS SAME plan MUST include that task's task_id in its own "
        "depends_on — omitting it means writer_agent will run with NO "
        "access to that evidence and will incorrectly report the "
        "information as missing, even though it was successfully retrieved. "
        "Fields inside task_input that expect a single piece of text "
        "(e.g. \"query\", \"instruction\") must be a plain string — never "
        "a list, even if there's only one item — e.g. "
        '{"query": "Where does the user work?"}, not '
        '{"query": ["Where does the user work?"]}.'
    )


class Planner:
    """
    Thin wrapper around the LLM that turns a conversation into a
    structured PlannerDecision. Holds no agent-specific logic —
    AGENT_SCHEMAS is the hardcoded registry defined in agent.registry
    and is used directly here rather than being threaded through as a
    parameter. The supervisor (agent_executor.run_plan) is responsible
    for actually invoking the agents named in the resulting tasks.
    """

    def __init__(self, llm: Optional[LLM] = None):
        self.llm = llm or LLM()
        self.available_agents = AGENT_SCHEMAS

    def plan(
        self,
        user_query: str,
        conversation_memory: Session,
    ) -> PlannerDecision:
        """
        Turn-start convenience wrapper: builds the initial prompt from
        memory, then delegates to decide(). Use this for round 1 of a
        turn, when there's no in-progress agent-call transcript yet.
        """
        prompt: list[Message] = build_prompt(
            session=conversation_memory,
            user_message=user_query,
            system_prompt=DEFAULT_SYSTEM_PROMPT,
        )
        return self.decide(prompt)

    def decide(self, prompt: list[Message]) -> PlannerDecision:
        """
        Core decision step, reusable across every round of the
        multi-agent loop — not just turn-start. Callers (e.g. the
        supervisor in agent.py) pass in the plain conversation so far —
        they don't need to know this step requires JSON-plan
        instructions; decide() prepends those itself so that concern
        stays inside the planner rather than leaking into every caller.

        Strips any existing system message(s) from `prompt` first. The
        caller's conversation may carry a system prompt written for a
        different purpose (e.g. instructions for unrelated tools this
        planning call never gets access to) — mixing that into the
        planning call risks confusing the JSON-plan instructions below,
        so this call gets a clean, single system message of its own.
        """
        conversation_only = [m for m in prompt if getattr(m, "role", None) != "system"]
        planning_prompt = [
            Message(role="system", content=_build_planning_system_prompt()),
            *conversation_only,
        ]
        response = self.llm.chat(planning_prompt, response_format={"type": "json_object"})
        return self._to_decision(response)

    @staticmethod
    def _to_decision(response) -> PlannerDecision:
        """
        Parses the LLM's JSON response into a PlannerDecision. This is
        the only place that touches the raw LLM response — everything
        downstream works with the structured decision instead.

        Provider-agnostic by design: this only assumes `response`
        exposes a `.content` string, not any tool-calling shape.
        """
        content = getattr(response, "content", None)
        if not content:
            raise ValueError("Planner received an empty response from the LLM")

        cleaned = content.strip()
        if cleaned.startswith("```"):
            # tolerate a stray markdown fence even though the prompt asks for none
            cleaned = cleaned.strip("`")
            if cleaned.lower().startswith("json"):
                cleaned = cleaned[4:]
            cleaned = cleaned.strip()

        try:
            raw = json.loads(cleaned)
        except json.JSONDecodeError as e:
            raise ValueError(
                f"Planner could not parse a JSON plan from the LLM response: {content!r}"
            ) from e

        try:
            return PlannerDecision.model_validate(raw)
        except ValidationError as e:
            raise ValueError(
                f"Planner's JSON plan did not match the expected schema: {raw!r}"
            ) from e