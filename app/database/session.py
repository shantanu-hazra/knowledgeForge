"""
Session: ties a user_id + conversation_id together so the agent loop
doesn't have to thread conversation_id through every call by hand.
Thin wrapper over db.py — no new persistence logic, just convenience.
"""

from database.crud import (
    create_conversation,
    save_message,
    load_messages,
    update_summary,
    get_summary,
    conversation_exists,
    DB_PATH,
)
from database.models import MessageRecord


class Session:
    def __init__(self, user_id, conversation_id=None, db_path=DB_PATH):
        self.user_id = user_id
        self.db_path = db_path
        if conversation_id is not None and conversation_exists(conversation_id, db_path):
            self.conversation_id = conversation_id
        else:
            self.conversation_id = create_conversation(user_id, db_path)

    def add_user_message(self, content: str) -> None:
        save_message(self.conversation_id, "user", content, self.db_path)

    def add_assistant_message(self, content: str) -> None:
        save_message(self.conversation_id, "assistant", content, self.db_path)

    def add_message(self, role: str, content: str) -> None:
        save_message(self.conversation_id, role, content, self.db_path)

    def recent_messages(self, limit: int = 10) -> list[MessageRecord]:
        rows = load_messages(self.conversation_id, limit=limit, db_path=self.db_path)
        return [MessageRecord(**r) for r in rows]

    def summary(self) -> str:
        return get_summary(self.conversation_id, self.db_path)

    def set_summary(self, summary: str) -> None:
        update_summary(self.conversation_id, summary, self.db_path)

    def build_context(self, limit: int = 10):
        """
        Returns the LLM-ready message list for this session:
        [system summary if any] + [recent messages], converted to
        the LLM client's Message schema.
        """
        from llm.schemas import Message

        context = []
        summary = self.summary()
        if summary:
            context.append(Message(role="system", content=f"Conversation summary so far: {summary}"))

        for record in self.recent_messages(limit=limit):
            context.append(record.to_llm_message())

        return context