"""
Pydantic models for the SQLite conversation store — typed views over
the raw dict rows that db.py returns, plus a converter to the LLM's
Message schema so the DB layer and the LLM layer share one shape.
"""

from datetime import datetime

from pydantic import BaseModel


class MessageRecord(BaseModel):
    """A single stored turn."""
    role: str          # "system" | "user" | "assistant" | "tool"
    content: str
    created_at: datetime

    def to_llm_message(self):
        """Convert to the LLM client's Message schema."""
        from llm.schemas import Message  # local import avoids a hard dependency
        return Message(role=self.role, content=self.content)


class ConversationRecord(BaseModel):
    """A conversation thread's metadata."""
    id: int
    user_id: str | None = None   # not returned by load_messages/get_summary, set where available
    summary: str = ""
    created_at: datetime
    updated_at: datetime
