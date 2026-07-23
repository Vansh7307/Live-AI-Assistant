from __future__ import annotations

import os
from typing import Any

from sqlalchemy import create_engine, text
from sqlalchemy.engine import Engine

_ENGINE: Engine | None = None


def _get_engine() -> Engine:
    """Module-level singleton engine so we don't open a new SQLite
    connection pool on every single chat request."""
    global _ENGINE
    if _ENGINE is None:
        root = os.path.dirname(__file__)
        db_path = os.getenv("MEMORY_DB_PATH") or os.path.join(root, "memory.sqlite3")
        _ENGINE = create_engine(
            f"sqlite:///{db_path}",
            connect_args={"check_same_thread": False},
        )
        _init_db(_ENGINE)
    return _ENGINE


def _init_db(engine: Engine) -> None:
    with engine.begin() as conn:
        conn.execute(
            text(
                """
                CREATE TABLE IF NOT EXISTS messages (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    session_id TEXT NOT NULL DEFAULT 'default',
                    role TEXT NOT NULL,
                    content TEXT NOT NULL,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                );
                """
            )
        )
        # Migrate older DBs created before session_id existed.
        existing_cols = {
            row[1] for row in conn.execute(text("PRAGMA table_info(messages)")).fetchall()
        }
        if "session_id" not in existing_cols:
            conn.execute(
                text("ALTER TABLE messages ADD COLUMN session_id TEXT NOT NULL DEFAULT 'default'")
            )
        conn.execute(
            text("CREATE INDEX IF NOT EXISTS idx_messages_session_id ON messages(session_id, id)")
        )


class SQLiteMemory:
    """Per-session conversation memory backed by SQLite.

    Every conversation is scoped to a `session_id` so concurrent users never
    see each other's chat history. Uses a shared engine (see `_get_engine`)
    instead of opening a fresh connection pool per call.
    """

    def __init__(self):
        self.engine = _get_engine()

    def get_recent_messages(self, session_id: str, limit: int = 8) -> list[dict[str, Any]]:
        with self.engine.begin() as conn:
            rows = conn.execute(
                text(
                    """
                    SELECT role, content, created_at
                    FROM messages
                    WHERE session_id = :session_id
                    ORDER BY id DESC
                    LIMIT :limit
                    """
                ),
                {"session_id": session_id, "limit": limit},
            ).fetchall()
        rows.reverse()
        return [{"role": r[0], "content": r[1], "created_at": str(r[2])} for r in rows]

    def append_user_message(self, session_id: str, content: str):
        self._append(session_id, "user", content)

    def append_assistant_message(self, session_id: str, content: str):
        self._append(session_id, "assistant", content)

    def _append(self, session_id: str, role: str, content: str):
        with self.engine.begin() as conn:
            conn.execute(
                text(
                    "INSERT INTO messages (session_id, role, content) VALUES (:session_id, :role, :content)"
                ),
                {"session_id": session_id, "role": role, "content": content},
            )
