"""
prompt_builder.py

Assembles the final LLM-ready message list for a turn:

    messages = [
        system,
        summary,
        *recent_messages,
        user_message,
    ]

Pulls summary + recent turns from Session (no new persistence logic),
adds a fixed system prompt and the incoming user message, and returns
something ready to hand straight to your LLM client.
"""

from llm.schemas import Message
from database.session import Session

DEFAULT_SYSTEM_PROMPT = (
    "You are a helpful assistant. For any arithmetic, you must never "
    "compute results yourself, even simple ones. Break every expression "
    "down into individual binary operations (one operator at a time, "
    "respecting order of operations) and call the calculator tool for "
    "each step. Do not combine multiple operations into a single tool "
    "call, and do not state a numeric result unless it came from a tool call."
    "The weather tool requires exact "
    "latitude and longitude — it does not accept city names. If the "
    "user asks about weather/temperature for a place, first use the "
    "search tool to find that place's coordinates, then call the "
    "weather tool with those coordinates. Never guess or estimate "
    "coordinates yourself — always get them from a search result."
    "For any question that might be answered by the user's own documents "
    "or internal organizational data (e.g. salary, CTC, offer letter, "
    "policies), use the knowledge_base_search tool before responding — "
    "never state you lack access to personal data without first checking "
    "the knowledge base."
)


def build_prompt(
    session: Session,
    user_message: str,
    system_prompt: str = DEFAULT_SYSTEM_PROMPT,
    recent_limit: int = 10,
) -> list[Message]:
    """
    Build the full message list for one LLM call:
    [system] + [summary, if any] + [recent messages] + [new user message]

    Does NOT persist the user_message — call session.add_user_message()
    separately (before or after building the prompt, your call) so this
    function stays a pure read + assemble step.
    """
    messages: list[Message] = []

    # 1. system
    messages.append(Message(role="system", content=system_prompt))

    # 2. summary (only if one exists yet)
    summary = session.summary()
    if summary:
        messages.append(
            Message(role="system", content=f"Conversation summary so far: {summary}")
        )

    # 3. recent messages, oldest -> newest
    for record in session.recent_messages(limit=recent_limit):
        messages.append(record.to_llm_message())

    # 4. the new user turn
    messages.append(Message(role="user", content=user_message))

    return messages