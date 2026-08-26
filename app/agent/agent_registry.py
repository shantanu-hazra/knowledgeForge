"""
Agent registry for the multi-agent workflow.

The planner imports AGENT_SCHEMAS and AGENT_REGISTRY directly from
this module and uses them to decide:
- which agents are required
- what task each agent should perform
- what dependencies exist between tasks

The registry itself does not execute agents.

`required_input_keys` records which task_input key(s) each agent's
run() treats as its primary text field (verified against each
agent's actual `.get(...)` calls, not guessed). Every agent in this
registry also accepts a generic "task" key as a fallback for its
primary key — that convention is uniform across all five agents, so
it's handled once in the planner rather than repeated here. An empty
list means the agent has no required key it will raise on
(reviewer_agent's "query" is used if present but defaults to "" and
never raises).
"""
from typing import Any

AGENT_SCHEMAS: list[dict[str, Any]] = [
    {
        "name": "retrieval_agent",
        "description": (
            "Retrieves relevant information from the application's "
            "private knowledge base using the existing RAG pipeline. "
            "Use this agent when the answer requires information from "
            "uploaded documents or internal knowledge."
        ),
        "capabilities": [
            "semantic document retrieval",
            "private knowledge base search",
            "document-grounded evidence retrieval",
            "citation metadata retrieval",
        ],
        # retrieval_agent.run(): task_input.get("query") or task_input.get("task")
        "required_input_keys": ["query"],
    },
    {
        "name": "research_agent",
        "description": (
            "Performs external research using available research tools. "
            "Use this agent when the task requires information that is "
            "not available in the private knowledge base or requires "
            "external/current sources."
        ),
        "capabilities": [
            "external research",
            "web search",
            "source discovery",
            "fact collection",
        ],
        # research_agent.run(): task_input.get("query") or task_input.get("task")
        "required_input_keys": ["query"],
    },
    {
        "name": "analysis_agent",
        "description": (
            "Analyzes and synthesizes information collected by other "
            "agents. Use this agent for comparison, reasoning, identifying "
            "patterns, extracting conclusions, or combining multiple "
            "sources of evidence."
        ),
        "capabilities": [
            "reasoning",
            "comparison",
            "synthesis",
            "evidence analysis",
            "conclusion generation",
        ],
        # analysis_agent.run(): task_input.get("instruction") or task_input.get("task")
        # NOTE: primary key is "instruction", not "query" — distinct from the
        # other agents.
        "required_input_keys": ["instruction"],
    },
    {
        "name": "writer_agent",
        "description": (
            "Generates the final response using the information and "
            "analysis available in shared state. It should not invent "
            "facts and should remain grounded in the provided evidence."
        ),
        "capabilities": [
            "answer generation",
            "structured writing",
            "summarization",
            "citation-aware response generation",
        ],
        # writer_agent.run(): task_input.get("query") or task_input.get("task").
        # Always required even when this task depends_on another — writer_agent
        # never pulls the user's question from `results`, only evidence/analysis.
        "required_input_keys": ["query"],
    },
    {
        "name": "reviewer_agent",
        "description": (
            "Reviews the generated answer against the available evidence. "
            "Checks for unsupported claims, missing information, contradictions, "
            "and whether the user's question has been adequately answered."
        ),
        "capabilities": [
            "answer validation",
            "evidence verification",
            "unsupported-claim detection",
            "completeness checking",
            "quality review",
        ],
        # reviewer_agent.run(): task_input.get("query") or task_input.get("task") or "" —
        # genuinely optional, never raises if absent. Strongly recommended
        # (used to judge completeness) but not enforced here.
        # reviewer_agent's task MUST depends_on the writer_agent task, but
        # that's a depends_on/results-shape requirement, not a task_input
        # key — not something required_input_keys can express.
        "required_input_keys": [],
    },
]

AGENT_REGISTRY = {
    agent["name"]: agent
    for agent in AGENT_SCHEMAS
}