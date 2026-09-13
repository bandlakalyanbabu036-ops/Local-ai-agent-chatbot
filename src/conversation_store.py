import sqlite3
from datetime import datetime, timezone
from pathlib import Path
from contextlib import contextmanager
from typing import Iterator


class ConversationStoreError(RuntimeError):
    """Raised when conversation history cannot be read or written."""


class ConversationStore:
    """Persist chat messages for a conversation in a local SQLite database."""

    def __init__(self, database_path: str | Path):
        self.database_path = Path(database_path)
        self.database_path.parent.mkdir(parents=True, exist_ok=True)
        self._initialize_database()

    @contextmanager
    def _connect(self) -> Iterator[sqlite3.Connection]:
        connection = sqlite3.connect(self.database_path)
        connection.row_factory = sqlite3.Row
        try:
            yield connection
            connection.commit()
        except Exception:
            connection.rollback()
            raise
        finally:
            connection.close()

    def _initialize_database(self) -> None:
        try:
            with self._connect() as connection:
                connection.execute(
                    """
                    CREATE TABLE IF NOT EXISTS messages (
                        id INTEGER PRIMARY KEY AUTOINCREMENT,
                        conversation_id TEXT NOT NULL,
                        role TEXT NOT NULL CHECK (role IN ('user', 'assistant')),
                        content TEXT NOT NULL,
                        timestamp TEXT NOT NULL
                    )
                    """
                )
                connection.execute(
                    """
                    CREATE INDEX IF NOT EXISTS idx_messages_conversation
                    ON messages (conversation_id, id)
                    """
                )
        except sqlite3.Error as error:
            raise ConversationStoreError(
                f"Could not initialize conversation database: {error}"
            ) from error

    def load_messages(self, conversation_id: str) -> list[dict[str, str]]:
        try:
            with self._connect() as connection:
                rows = connection.execute(
                    """
                    SELECT role, content
                    FROM messages
                    WHERE conversation_id = ?
                    ORDER BY id
                    """,
                    (conversation_id,),
                ).fetchall()
            return [
                {"role": row["role"], "content": row["content"]}
                for row in rows
            ]
        except sqlite3.Error as error:
            raise ConversationStoreError(
                f"Could not load conversation history: {error}"
            ) from error

    def save_message(
        self,
        conversation_id: str,
        role: str,
        content: str,
    ) -> None:
        if role not in {"user", "assistant"}:
            raise ValueError(f"Unsupported conversation message role: {role}")
        try:
            with self._connect() as connection:
                connection.execute(
                    """
                    INSERT INTO messages (
                        conversation_id, role, content, timestamp
                    ) VALUES (?, ?, ?, ?)
                    """,
                    (
                        conversation_id,
                        role,
                        content,
                        datetime.now(timezone.utc).isoformat(),
                    ),
                )
        except sqlite3.Error as error:
            raise ConversationStoreError(
                f"Could not save conversation message: {error}"
            ) from error

    def clear_conversation(self, conversation_id: str) -> None:
        try:
            with self._connect() as connection:
                connection.execute(
                    "DELETE FROM messages WHERE conversation_id = ?",
                    (conversation_id,),
                )
        except sqlite3.Error as error:
            raise ConversationStoreError(
                f"Could not clear conversation history: {error}"
            ) from error