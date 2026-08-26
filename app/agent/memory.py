"""
memory_handler.py

Orchestrates the load -> summarize-if-needed -> save loop:

    history = load(session_id, user_id)
    history = summarize_if_needed(history)
    save(history)

No new persistence logic lives here — it composes Session (session.py)
with an LLM summarization call and writes back through
Session.set_summary(). The SessionHistory object carries state between
the three functions so summarize_if_needed doesn't need to re-query
the DB to know what it just changed.
"""

from dataclasses import dataclass, field

from database.crud import load_messages, DB_PATH
from database.models import MessageRecord
from database.session import Session

from agent.summarizer import summarize

# --- tunables ---------------------------------------------------------
SUMMARIZE_AFTER_MESSAGES = 5   # trigger once raw history exceeds this
KEEP_RECENT_MESSAGES = 10       # always keep this many verbatim after summarizing


@dataclass
class SessionHistory:
    session: Session
    total_message_count: int
    summary: str
    recent: list[MessageRecord] = field(default_factory=list)


def load(session_id: int, user_id: str, db_path: str = DB_PATH) -> SessionHistory:
    """
    Resume a conversation by id and load enough state to decide
    whether a summarization pass is needed this turn.
    """
    session = Session(user_id=user_id, conversation_id=session_id, db_path=db_path)

    # limit=None -> full history, oldest first. We need the count to
    # decide whether to summarize, and the rows themselves are handed
    # straight to the summarizer if summarize_if_needed fires.
    all_rows = load_messages(session_id, limit=None, db_path=db_path)

    return SessionHistory(
        session=session,
        total_message_count=len(all_rows),
        summary=session.summary(),
        recent=[MessageRecord(**r) for r in all_rows],
    )


def summarize_if_needed(history: SessionHistory) -> SessionHistory:
    print(f"[summarize_if_needed] total_message_count={history.total_message_count}, summary='{history.summary}'")
    if history.total_message_count <= SUMMARIZE_AFTER_MESSAGES:
        return history

    to_condense = history.recent[:-KEEP_RECENT_MESSAGES]
    still_recent = history.recent[-KEEP_RECENT_MESSAGES:]

    if not to_condense:
        return history

    new_summary = summarize(old_messages=to_condense, prior_summary=history.summary)

    history.session.set_summary(new_summary)
    history.summary = new_summary
    history.recent = still_recent
    history.total_message_count = len(still_recent)

    print(f"[summarize_if_needed] conversation summary updated to: {new_summary}")

    return history


def _summarize_messages(prior_summary: str, messages: list[MessageRecord]) -> str:
    """
    Calls the LLM to fold `messages` into `prior_summary`.
    Swap the body for your actual LLM client call.
    """
    from llm.client import complete  # your LLM wrapper

    transcript = "\n".join(f"{m.role}: {m.content}" for m in messages)
    prompt = (
        "Update the running summary of this conversation to incorporate "
        "the new turns below. Keep it concise but preserve facts, "
        "decisions, and open threads a later turn might need.\n\n"
        f"Existing summary:\n{prior_summary or '(none yet)'}\n\n"
        f"New turns to fold in:\n{transcript}\n\n"
        "Updated summary:"
    )
    return complete(prompt)


def save(history: SessionHistory) -> None:
    """
    Persist the summary state. Individual messages are already written
    as they happen (via session.add_user_message / add_assistant_message
    inside the agent loop) — this is the explicit, idempotent commit
    point for whatever summarize_if_needed produced.
    """
    history.session.set_summary(history.summary)