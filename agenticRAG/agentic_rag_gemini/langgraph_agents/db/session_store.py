"""PostgreSQL session store — replaces Firebase Firestore for LangGraph agents.

Coexists with memory/session_store.py (Firebase) for the old code path.
"""

from __future__ import annotations

import json
import uuid

from langgraph_agents.db.postgres import PostgresClient


class SessionStore:
    """Async session persistence backed by the conversations table."""

    def __init__(self, pg: PostgresClient = None):
        self.pg = pg or PostgresClient()

    async def save_session(
        self,
        user_id: str,
        session_id: str,
        messages: list[dict],
        summary: str = None,
    ):
        """Insert or update a conversation row."""
        await self.pg.connect()

        # Upsert: check for existing row by user_id + session_id
        existing = await self.pg.fetchrow(
            "SELECT id FROM conversations WHERE user_id = $1 AND session_id = $2",
            user_id,
            session_id,
        )

        if existing:
            await self.pg.execute(
                "UPDATE conversations SET messages = $1, summary = COALESCE($2, summary) WHERE id = $3",
                json.dumps(messages, ensure_ascii=False),
                summary,
                str(existing["id"]),
            )
        else:
            await self.pg.execute(
                """INSERT INTO conversations (id, user_id, session_id, messages, summary)
                   VALUES ($1, $2, $3, $4, $5)""",
                str(uuid.uuid4()),
                user_id,
                session_id,
                json.dumps(messages, ensure_ascii=False),
                summary,
            )

    async def load_session(self, user_id: str, session_id: str) -> dict | None:
        """Load a single session by user and session ID."""
        row = await self.pg.fetchrow(
            "SELECT * FROM conversations WHERE user_id = $1 AND session_id = $2",
            user_id,
            session_id,
        )
        if not row:
            return None
        return {
            "id": str(row["id"]),
            "user_id": row["user_id"],
            "session_id": row["session_id"],
            "messages": row["messages"] or [],
            "summary": row["summary"],
            "created_at": row["created_at"].isoformat(),
        }

    async def list_sessions(self, user_id: str, limit: int = 10) -> list[dict]:
        """List recent sessions for a user."""
        rows = await self.pg.fetch(
            "SELECT id, session_id, summary, created_at FROM conversations "
            "WHERE user_id = $1 ORDER BY created_at DESC LIMIT $2",
            user_id,
            limit,
        )
        return [
            {
                "id": str(row["id"]),
                "session_id": row["session_id"],
                "summary": row["summary"],
                "created_at": row["created_at"].isoformat(),
            }
            for row in rows
        ]
