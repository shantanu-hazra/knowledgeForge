from dataclasses import dataclass, field
from typing import Any


@dataclass
class AgentState:
    user_query: str

    plan: list[dict[str, Any]] = field(default_factory=list)

    retrieved_documents: list[dict[str, Any]] = field(default_factory=list)

    research_results: list[dict[str, Any]] = field(default_factory=list)

    analysis: str = ""

    draft: str = ""

    review: str = ""

    final_answer: str = ""