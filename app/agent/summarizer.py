"""
summarizer.py

    Old Messages -> LLM -> Summary

Condenses a batch of old messages into an updated running summary.

Preserve: user preferences, goals, important decisions
Remove:   greetings, small talk, repetition
"""

from database.models import MessageRecord
from llm.client import LLM  # your LLM wrapper

llm = LLM()

SUMMARIZER_SYSTEM_PROMPT = """\
You summarize conversation history for an AI agent's long-term memory.

Given an existing summary (if any) and a batch of new messages, produce
an updated summary.

Preserve:
- user preferences (stated likes, dislikes, constraints, style choices)
- goals (what the user is trying to accomplish, short- or long-term)
- important decisions (choices made, options ruled out, and why)

Remove:
- greetings and small talk
- repetition — if something is restated, keep it once
- filler that doesn't affect future turns

Keep it concise: prefer short factual statements over narrative prose.
Write in third person about "the user". Do not invent details that
aren't in the messages.
"""


def summarize(
    old_messages: list[MessageRecord],
    prior_summary: str = "",
) -> str:
    """
    Fold `old_messages` into `prior_summary` and return the new summary.
    Pure function: no DB reads/writes — caller (memory_handler.save /
    Session.set_summary) is responsible for persisting the result.
    """
    if not old_messages:
        return prior_summary

    transcript = _format_transcript(old_messages)

    prompt = (
        f"Existing summary:\n{prior_summary or '(none yet)'}\n\n"
        f"New messages to fold in:\n{transcript}\n\n"
        "Updated summary:"
    )

    return llm.complete(prompt, system=SUMMARIZER_SYSTEM_PROMPT).strip()


def _format_transcript(messages: list[MessageRecord]) -> str:
    return "\n".join(f"{m.role}: {m.content}" for m in messages)