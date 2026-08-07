"""Persistent conversation memory.

Backed by either PostgreSQL (recommended for production multi-instance) or
SQLite (default, good for single-instance / local dev), selected via the
``DATABASE_URL`` environment variable. Provides per-session history, a
running conversation summary, and a retention cleanup routine.
"""

from __future__ import annotations

import os
import time
from typing import Any

from sqlalchemy import create_engine, text
from sqlalchemy.engine import Engine

_ENGINE: Engine | None = None


def _get_engine() -> Engine:
    """Module-level singleton engine so we don't open a new connection pool
    on every single chat request."""
    global _ENGINE
    if _ENGINE is None:
        db_url = os.getenv("DATABASE_URL")
        if db_url:
            # Render/Heroku-style postgres:// URLs need the psycopg driver.
            if db_url.startswith("postgres://"):
                db_url = db_url.replace("postgres://", "postgresql+psycopg://", 1)
            _ENGINE = create_engine(db_url, pool_pre_ping=True)
        else:
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
                CREATE TABLE IF NOT EXISTS sessions (
                    session_id TEXT PRIMARY KEY,
                    summary TEXT NOT NULL DEFAULT '',
                    message_count INTEGER NOT NULL DEFAULT 0,
                    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                );
                """
            )
        )
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


class Memory:
    """Per-session conversation memory.

    Every conversation is scoped to a ``session_id`` so concurrent users never
    see each other's chat history. Uses a shared engine (see ``_get_engine``)
    instead of opening a fresh connection pool per call.
    """

    def __init__(self):
        self.engine = _get_engine()

    # --- message history -------------------------------------------------

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
            conn.execute(
                text(
                    """
                    INSERT INTO sessions (session_id, message_count)
                    VALUES (:session_id, 1)
                    ON CONFLICT(session_id) DO UPDATE SET
                        message_count = sessions.message_count + 1,
                        updated_at = CURRENT_TIMESTAMP
                    """
                ),
                {"session_id": session_id},
            )

    # --- summary ---------------------------------------------------------

    def get_summary(self, session_id: str) -> str:
        with self.engine.begin() as conn:
            row = conn.execute(
                text("SELECT summary FROM sessions WHERE session_id = :session_id"),
                {"session_id": session_id},
            ).fetchone()
        return (row[0] if row else "") or ""

    def set_summary(self, session_id: str, summary: str):
        with self.engine.begin() as conn:
            conn.execute(
                text(
                    """
                    INSERT INTO sessions (session_id, summary)
                    VALUES (:session_id, :summary)
                    ON CONFLICT(session_id) DO UPDATE SET summary = :summary
                    """
                ),
                {"session_id": session_id, "summary": summary},
            )

    def get_message_count(self, session_id: str) -> int:
        with self.engine.begin() as conn:
            row = conn.execute(
                text("SELECT message_count FROM sessions WHERE session_id = :session_id"),
                {"session_id": session_id},
            ).fetchone()
        return row[0] if row else 0

    def list_sessions(self, limit: int = 50) -> list[dict[str, Any]]:
        """Return most-recent sessions with their summaries (for the
        frontend's session/history sidebar)."""
        with self.engine.begin() as conn:
            rows = conn.execute(
                text(
                    """
                    SELECT session_id, summary, message_count, updated_at
                    FROM sessions
                    ORDER BY updated_at DESC
                    LIMIT :limit
                    """
                ),
                {"limit": limit},
            ).fetchall()
        return [
            {
                "session_id": r[0],
                "summary": r[1] or "",
                "message_count": r[2],
                "updated_at": str(r[3]),
            }
            for r in rows
        ]

    # --- retention -------------------------------------------------------

    def cleanup_old_sessions(self, max_age_days: int = 30) -> int:
        """Delete sessions (and their messages) older than ``max_age_days``.
        Returns number of sessions removed. Call from a scheduled job."""
        cutoff_ts = int(time.time()) - max_age_days * 86400
        is_postgres = str(self.engine.url).startswith("postgresql")
        with self.engine.begin() as conn:
            if is_postgres:
                expired = conn.execute(
                    text(
                        "SELECT session_id FROM sessions WHERE updated_at < to_timestamp(:cutoff)"
                    ),
                    {"cutoff": cutoff_ts},
                ).fetchall()
            else:
                expired = conn.execute(
                    text(
                        "SELECT session_id FROM sessions WHERE updated_at < datetime(:cutoff, 'unixepoch')"
                    ),
                    {"cutoff": cutoff_ts},
                ).fetchall()
            ids = [r[0] for r in expired]
            if not ids:
                return 0
            if is_postgres:
                conn.execute(
                    text("DELETE FROM messages WHERE session_id = ANY(:ids)"),
                    {"ids": ids},
                )
                conn.execute(
                    text("DELETE FROM sessions WHERE session_id = ANY(:ids)"),
                    {"ids": ids},
                )
            else:
                # SQLite has no ANY() - build an explicit IN clause.
                placeholders = ",".join(f":id_{i}" for i in range(len(ids)))
                params = {f"id_{i}": sid for i, sid in enumerate(ids)}
                conn.execute(
                    text(f"DELETE FROM messages WHERE session_id IN ({placeholders})"),
                    params,
                )
                conn.execute(
                    text(f"DELETE FROM sessions WHERE session_id IN ({placeholders})"),
                    params,
                )
        return len(ids)


# Backwards-compatible alias so existing imports/tests keep working.
SQLiteMemory = Memory
