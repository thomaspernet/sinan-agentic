"""SQLite Session Store - Persistent conversation storage.

Provides SQLite-backed persistence for chat sessions:
- Store and retrieve messages by session
- Session metadata (created_at, title, archived)
- Archive old sessions
- OpenAI-compatible history format
- Capability snapshots (one JSON blob per capability per session)

Schema migrations are forward-only: each new release adds tables/columns
without altering existing rows. Older databases pick up new tables on the
next ``__init__``.

Usage:
    from sinan_agentic_core.session.sqlite_store import SQLiteSessionStore

    store = SQLiteSessionStore("data/conversations.db")

    store.get_or_create_session("session-123")
    store.add_message("session-123", "user", "Hello!")
    history = store.get_conversation_history("session-123")
"""

import json
import logging
import sqlite3
from contextlib import contextmanager
from pathlib import Path
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)


class SQLiteSessionStore:
    """SQLite-backed storage for chat sessions and messages.

    Handles:
    - Creating sessions on demand
    - Storing messages with role/content
    - Retrieving history in OpenAI-compatible format
    - Archiving and deleting sessions
    """

    def __init__(self, db_path: str = "data/conversations.db"):
        """
        Args:
            db_path: Path to the SQLite database file.
                     Directory will be created if it doesn't exist.
        """
        self.db_path = Path(db_path)
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self._init_schema()

    @contextmanager
    def _connect(self):
        """Context manager for database connections."""
        conn = sqlite3.connect(str(self.db_path))
        conn.row_factory = sqlite3.Row
        try:
            yield conn
        finally:
            conn.close()

    def _init_schema(self):
        """Create tables if they don't exist."""
        with self._connect() as conn:
            cursor = conn.cursor()

            cursor.execute("""
                CREATE TABLE IF NOT EXISTS sessions (
                    id TEXT PRIMARY KEY,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    archived_at TIMESTAMP DEFAULT NULL,
                    is_archived INTEGER DEFAULT 0,
                    title TEXT DEFAULT NULL,
                    metadata TEXT DEFAULT '{}'
                )
            """)

            cursor.execute("""
                CREATE TABLE IF NOT EXISTS messages (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    session_id TEXT NOT NULL,
                    role TEXT NOT NULL,
                    content TEXT NOT NULL,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    metadata TEXT DEFAULT '{}',
                    FOREIGN KEY (session_id) REFERENCES sessions(id) ON DELETE CASCADE
                )
            """)

            cursor.execute("""
                CREATE INDEX IF NOT EXISTS idx_messages_session
                ON messages(session_id)
            """)

            cursor.execute("""
                CREATE INDEX IF NOT EXISTS idx_sessions_active
                ON sessions(is_archived, updated_at)
            """)

            cursor.execute("""
                CREATE TABLE IF NOT EXISTS capability_snapshots (
                    session_id TEXT NOT NULL,
                    capability_key TEXT NOT NULL,
                    snapshot TEXT NOT NULL,
                    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    PRIMARY KEY (session_id, capability_key),
                    FOREIGN KEY (session_id) REFERENCES sessions(id) ON DELETE CASCADE
                )
            """)

            conn.commit()
            logger.debug(f"Database initialized at {self.db_path}")

    # -----------------------------------------------------------------
    # Session management
    # -----------------------------------------------------------------

    def get_or_create_session(self, session_id: str) -> Dict[str, Any]:
        """Get existing session or create a new one.

        Returns:
            Session metadata dict
        """
        with self._connect() as conn:
            cursor = conn.cursor()
            cursor.execute(
                "SELECT * FROM sessions WHERE id = ? AND is_archived = 0",
                (session_id,),
            )
            row = cursor.fetchone()
            if row:
                return dict(row)

            cursor.execute("INSERT INTO sessions (id) VALUES (?)", (session_id,))
            conn.commit()

            cursor.execute("SELECT * FROM sessions WHERE id = ?", (session_id,))
            return dict(cursor.fetchone())

    def archive_session(self, session_id: str) -> bool:
        """Archive a session (keeps data, marks as archived).

        Returns:
            True if session was archived
        """
        with self._connect() as conn:
            cursor = conn.cursor()
            cursor.execute(
                """
                UPDATE sessions
                SET is_archived = 1, archived_at = CURRENT_TIMESTAMP
                WHERE id = ? AND is_archived = 0
                """,
                (session_id,),
            )
            conn.commit()
            return cursor.rowcount > 0

    def clear_session(self, session_id: str) -> bool:
        """Delete a session and all its messages permanently.

        Returns:
            True if session was deleted
        """
        with self._connect() as conn:
            cursor = conn.cursor()
            cursor.execute("DELETE FROM messages WHERE session_id = ?", (session_id,))
            cursor.execute(
                "DELETE FROM capability_snapshots WHERE session_id = ?",
                (session_id,),
            )
            cursor.execute("DELETE FROM sessions WHERE id = ?", (session_id,))
            conn.commit()
            return cursor.rowcount > 0

    def get_active_sessions(self) -> List[Dict[str, Any]]:
        """Get all active (non-archived) sessions with message counts."""
        with self._connect() as conn:
            cursor = conn.cursor()
            cursor.execute("""
                SELECT s.*, COUNT(m.id) as message_count
                FROM sessions s
                LEFT JOIN messages m ON s.id = m.session_id
                WHERE s.is_archived = 0
                GROUP BY s.id
                ORDER BY s.updated_at DESC
            """)
            return [dict(row) for row in cursor.fetchall()]

    def get_archived_sessions(self) -> List[Dict[str, Any]]:
        """Get all archived sessions with message counts."""
        with self._connect() as conn:
            cursor = conn.cursor()
            cursor.execute("""
                SELECT s.*, COUNT(m.id) as message_count
                FROM sessions s
                LEFT JOIN messages m ON s.id = m.session_id
                WHERE s.is_archived = 1
                GROUP BY s.id
                ORDER BY s.archived_at DESC
            """)
            return [dict(row) for row in cursor.fetchall()]

    def get_message_count(self, session_id: str) -> int:
        """Get the number of messages in a session."""
        with self._connect() as conn:
            cursor = conn.cursor()
            cursor.execute(
                "SELECT COUNT(*) as count FROM messages WHERE session_id = ?",
                (session_id,),
            )
            return cursor.fetchone()["count"]

    # -----------------------------------------------------------------
    # Message operations
    # -----------------------------------------------------------------

    def add_message(
        self,
        session_id: str,
        role: str,
        content: str,
        metadata: Optional[Dict] = None,
    ) -> int:
        """Add a message to a session.

        Args:
            session_id: Session to add to (auto-creates if needed)
            role: Message role (user, assistant, system)
            content: Message content
            metadata: Optional metadata dict

        Returns:
            The message ID
        """
        self.get_or_create_session(session_id)

        with self._connect() as conn:
            cursor = conn.cursor()

            cursor.execute(
                """
                INSERT INTO messages (session_id, role, content, metadata)
                VALUES (?, ?, ?, ?)
                """,
                (session_id, role, content, json.dumps(metadata or {})),
            )
            message_id = cursor.lastrowid

            # Update session timestamp
            cursor.execute(
                "UPDATE sessions SET updated_at = CURRENT_TIMESTAMP WHERE id = ?",
                (session_id,),
            )

            # Set title from first user message
            if role == "user":
                title = content[:100] + "..." if len(content) > 100 else content
                cursor.execute(
                    "UPDATE sessions SET title = ? WHERE id = ? AND title IS NULL",
                    (title, session_id),
                )

            conn.commit()
            return message_id

    def get_messages(self, session_id: str, limit: int = 100) -> List[Dict[str, Any]]:
        """Get messages for a session with full metadata.

        Returns:
            List of message dicts with role, content, created_at, metadata
        """
        with self._connect() as conn:
            cursor = conn.cursor()
            cursor.execute(
                """
                SELECT role, content, created_at, metadata
                FROM messages
                WHERE session_id = ?
                ORDER BY created_at ASC
                LIMIT ?
                """,
                (session_id, limit),
            )

            messages = []
            for row in cursor.fetchall():
                messages.append({
                    "role": row["role"],
                    "content": row["content"],
                    "created_at": row["created_at"],
                    "metadata": json.loads(row["metadata"]) if row["metadata"] else {},
                })
            return messages

    def get_conversation_history(self, session_id: str) -> List[Dict[str, str]]:
        """Get conversation history in OpenAI-compatible format.

        Returns:
            List of {"role": str, "content": str} dicts
        """
        messages = self.get_messages(session_id)
        return [{"role": m["role"], "content": m["content"]} for m in messages]

    # -----------------------------------------------------------------
    # Capability snapshots
    # -----------------------------------------------------------------

    def save_capability_snapshot(
        self,
        session_id: str,
        capability_key: str,
        snapshot: dict[str, Any],
    ) -> None:
        """Persist a capability's snapshot for a session.

        Each ``(session_id, capability_key)`` pair stores at most one row;
        repeated calls overwrite the previous blob. ``capability_key`` is a
        stable identifier the caller chooses (typically the capability's
        class name) — it must match the key used at rehydrate time.

        Args:
            session_id: Session the snapshot belongs to. The session row is
                created on demand so callers do not need to seed it first.
            capability_key: Stable identifier for the capability instance.
            snapshot: JSON-serializable dict produced by
                ``Capability.to_snapshot()``.
        """
        self.get_or_create_session(session_id)
        payload = json.dumps(snapshot)
        with self._connect() as conn:
            cursor = conn.cursor()
            cursor.execute(
                """
                INSERT INTO capability_snapshots (session_id, capability_key, snapshot)
                VALUES (?, ?, ?)
                ON CONFLICT(session_id, capability_key) DO UPDATE SET
                    snapshot = excluded.snapshot,
                    updated_at = CURRENT_TIMESTAMP
                """,
                (session_id, capability_key, payload),
            )
            conn.commit()

    def load_capability_snapshot(
        self,
        session_id: str,
        capability_key: str,
    ) -> Optional[dict[str, Any]]:
        """Return the stored snapshot dict, or ``None`` when absent."""
        with self._connect() as conn:
            cursor = conn.cursor()
            cursor.execute(
                """
                SELECT snapshot FROM capability_snapshots
                WHERE session_id = ? AND capability_key = ?
                """,
                (session_id, capability_key),
            )
            row = cursor.fetchone()
            if row is None:
                return None
            data = json.loads(row["snapshot"])
            return data if isinstance(data, dict) else None

    def load_all_capability_snapshots(self, session_id: str) -> dict[str, dict[str, Any]]:
        """Return every snapshot for a session keyed by capability key."""
        with self._connect() as conn:
            cursor = conn.cursor()
            cursor.execute(
                "SELECT capability_key, snapshot FROM capability_snapshots WHERE session_id = ?",
                (session_id,),
            )
            result: dict[str, dict[str, Any]] = {}
            for row in cursor.fetchall():
                data = json.loads(row["snapshot"])
                if isinstance(data, dict):
                    result[row["capability_key"]] = data
            return result

    def delete_capability_snapshots(self, session_id: str) -> None:
        """Drop every capability snapshot stored under this session."""
        with self._connect() as conn:
            cursor = conn.cursor()
            cursor.execute(
                "DELETE FROM capability_snapshots WHERE session_id = ?",
                (session_id,),
            )
            conn.commit()
