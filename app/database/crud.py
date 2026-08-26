"""
SQLite-backed conversation store for the agent.

Schema:
    conversations(id, user_id, summary, created_at, updated_at)
        - one row per conversation thread; a user can have many.
    messages(id, conversation_id, role, content, created_at)
        - one row per turn, linked to a conversation.

The `summary` column on `conversations` is what keeps context focused:
once a conversation gets long, summarize the older messages and store
the summary via update_summary(), then load_messages() with a `limit`
to pull only the recent turns. Feed [summary] + [recent messages] to
the LLM instead of the full history.
"""

import sqlite3
from contextlib import contextmanager
from datetime import datetime, timezone

DB_PATH = "agent.db"


@contextmanager
def _connect(db_path: str = DB_PATH):
    conn = sqlite3.connect(db_path)
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA foreign_keys = ON")
    try:
        yield conn
        conn.commit()
    finally:
        conn.close()


def init_db(db_path: str = DB_PATH) -> None:
    """Create tables if they don't exist. Call once at startup."""
    with _connect(db_path) as conn:
        conn.execute("""
            CREATE TABLE IF NOT EXISTS conversations (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                user_id TEXT NOT NULL,
                summary TEXT NOT NULL DEFAULT '',
                created_at TEXT NOT NULL,
                updated_at TEXT NOT NULL
            )
        """)
        conn.execute("""
            CREATE TABLE IF NOT EXISTS messages (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                conversation_id INTEGER NOT NULL,
                role TEXT NOT NULL,
                content TEXT NOT NULL,
                created_at TEXT NOT NULL,
                FOREIGN KEY (conversation_id) REFERENCES conversations(id) ON DELETE CASCADE
            )
        """)
        conn.execute("""
            CREATE INDEX IF NOT EXISTS idx_messages_conversation
            ON messages(conversation_id, created_at)
        """)
        conn.execute("""
            CREATE INDEX IF NOT EXISTS idx_conversations_user
            ON conversations(user_id)
        """)


def create_conversation(user_id: str, db_path: str = DB_PATH) -> int:
    """Start a new conversation thread for a user. Returns conversation_id."""
    now = datetime.now(timezone.utc).isoformat()
    with _connect(db_path) as conn:
        cur = conn.execute(
            "INSERT INTO conversations (user_id, summary, created_at, updated_at) "
            "VALUES (?, '', ?, ?)",
            (user_id, now, now),
        )
        return cur.lastrowid


def save_message(
    conversation_id: int,
    role: str,
    content: str,
    db_path: str = DB_PATH,
) -> int:
    """
    Append a message to a conversation. role is typically
    "system" / "user" / "assistant" / "tool". Returns the new message id.
    """
    now = datetime.now(timezone.utc).isoformat()
    with _connect(db_path) as conn:
        cur = conn.execute(
            "INSERT INTO messages (conversation_id, role, content, created_at) "
            "VALUES (?, ?, ?, ?)",
            (conversation_id, role, content, now),
        )
        conn.execute(
            "UPDATE conversations SET updated_at = ? WHERE id = ?",
            (now, conversation_id),
        )
        return cur.lastrowid

def conversation_exists(conversation_id: int, db_path: str = DB_PATH) -> bool:
    """Check whether a conversation id actually exists before trusting a
    client-supplied value — prevents FOREIGN KEY errors from stale/invalid ids."""
    with _connect(db_path) as conn:
        row = conn.execute(
            "SELECT 1 FROM conversations WHERE id = ?",
            (conversation_id,),
        ).fetchone()
        return row is not None

def load_messages(
    conversation_id: int,
    limit: int | None = None,
    db_path: str = DB_PATH,
) -> list[dict]:
    """
    Return messages for a conversation, oldest first.
    Pass `limit` to fetch only the most recent N turns (recommended
    once a conversation is long — combine with the stored summary
    instead of loading the entire history every call).
    """
    with _connect(db_path) as conn:
        if limit is None:
            rows = conn.execute(
                "SELECT role, content, created_at FROM messages "
                "WHERE conversation_id = ? ORDER BY id ASC",
                (conversation_id,),
            ).fetchall()
        else:
            rows = conn.execute(
                "SELECT role, content, created_at FROM messages "
                "WHERE conversation_id = ? ORDER BY id DESC LIMIT ?",
                (conversation_id, limit),
            ).fetchall()
            rows = list(reversed(rows))  # back to chronological order

        return [dict(r) for r in rows]


def update_summary(
    conversation_id: int,
    summary: str,
    db_path: str = DB_PATH,
) -> None:
    """
    Overwrite the running summary for a conversation. Call this after
    condensing older turns (e.g. via an LLM summarization pass) so
    future loads can send [summary] + [recent messages] instead of
    the full history.
    """
    now = datetime.now(timezone.utc).isoformat()
    with _connect(db_path) as conn:
        conn.execute(
            "UPDATE conversations SET summary = ?, updated_at = ? WHERE id = ?",
            (summary, now, conversation_id),
        )


def get_summary(conversation_id: int, db_path: str = DB_PATH) -> str:
    """Fetch the current stored summary for a conversation."""
    with _connect(db_path) as conn:
        row = conn.execute(
            "SELECT summary FROM conversations WHERE id = ?",
            (conversation_id,),
        ).fetchone()
        return row["summary"] if row else ""


def list_conversations(user_id: str, db_path: str = DB_PATH) -> list[dict]:
    """List all conversations for a user, most recently updated first."""
    with _connect(db_path) as conn:
        rows = conn.execute(
            "SELECT id, summary, created_at, updated_at FROM conversations "
            "WHERE user_id = ? ORDER BY updated_at DESC",
            (user_id,),
        ).fetchall()
        return [dict(r) for r in rows]


if __name__ == "__main__":
    init_db("demo.db")
    convo_id = create_conversation("user_123", "demo.db")

    save_message(convo_id, "user", "What's the weather in Berlin?", "demo.db")
    save_message(convo_id, "assistant", "It's 18°C and cloudy.", "demo.db")
    save_message(convo_id, "user", "Search for good pizza places nearby.", "demo.db")

    print("Recent messages:")
    for m in load_messages(convo_id, limit=2, db_path="demo.db"):
        print(m)

    update_summary(convo_id, "User asked about Berlin weather and nearby pizza.", "demo.db")
    print("\nStored summary:", get_summary(convo_id, "demo.db"))

    print("\nConversations for user:")
    print(list_conversations("user_123", "demo.db"))