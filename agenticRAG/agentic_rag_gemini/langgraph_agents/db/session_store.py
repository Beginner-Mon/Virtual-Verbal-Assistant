"""PostgreSQL session store — replaces Firebase Firestore for LangGraph agents.

Coexists with memory/session_store.py (Firebase) for the old code path.
"""

from __future__ import annotations

import json
import uuid
from datetime import datetime, timezone
from typing import Optional

import redis.asyncio as aioredis

from langgraph_agents.db.postgres import PostgresClient
from langgraph_agents.shared import get_pg_client


_REDIS_URL = "redis://localhost:6379/0"
_STM_MAX = 3


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

        existing = await self.pg.fetchrow(
            "SELECT id FROM conversations WHERE user_id = $1 AND session_id = $2",
            user_id,
            session_id,
        )

        if existing:
            await self.pg.execute(
                "UPDATE conversations SET messages = $1, summary = COALESCE($2, summary), updated_at = now() WHERE id = $3",
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


# ── Standalone async helpers (Phase 5) ──────────────────────────────


async def list_user_sessions(user_id: str, limit: int = 50) -> list[dict]:
    pg = get_pg_client()
    await pg.connect()
    rows = await pg.fetch(
        """
        SELECT session_id::text AS session_id,
               created_at, updated_at,
               jsonb_array_length(COALESCE(messages, '[]'::jsonb)) AS message_count,
               COALESCE(
                 SUBSTRING(
                   (SELECT m->>'content' FROM jsonb_array_elements(messages) AS m
                    WHERE m->>'role' = 'user' LIMIT 1),
                   1, 80
                 ),
                 '(empty)'
               ) AS first_user_message_preview
        FROM conversations
        WHERE user_id = $1::uuid
        ORDER BY updated_at DESC
        LIMIT $2
        """,
        user_id, limit,
    )
    return [
        {
            "session_id": r["session_id"],
            "created_at": r["created_at"].isoformat(),
            "updated_at": r["updated_at"].isoformat(),
            "first_user_message_preview": r["first_user_message_preview"],
            "message_count": r["message_count"],
        }
        for r in rows
    ]


async def load_session_messages(user_id: str, session_id: str) -> Optional[dict]:
    pg = get_pg_client()
    await pg.connect()
    row = await pg.fetchrow(
        """SELECT session_id::text, messages, updated_at
           FROM conversations
           WHERE user_id = $1::uuid AND session_id = $2::uuid""",
        user_id, session_id,
    )
    return dict(row) if row else None


async def populate_stm_from_messages(session_id: str, messages: list[dict]) -> None:
    """Pick last 3 Q&A pairs from full message log → write to Redis STM."""
    pairs = []
    pending_user = None
    for m in messages:
        if m["role"] == "user":
            pending_user = m["content"]
        elif m["role"] == "assistant" and pending_user:
            pairs.append({
                "q": pending_user,
                "a": m["content"],
                "ts": m.get("timestamp", ""),
            })
            pending_user = None

    stm = pairs[-_STM_MAX:]

    r = aioredis.from_url(_REDIS_URL)
    try:
        await r.setex(f"stm:{session_id}", 7200, json.dumps(stm))
    finally:
        close_fn = getattr(r, "aclose", None) or r.close
        await close_fn()


async def write_session_turn(
    user_id: str,
    session_id: str,
    user_query: str,
    assistant_answer: str,
    intent: str,
    tokens: int,
) -> None:
    """Append 1 user message + 1 assistant message to conversations.

    INSERT if session_id new, UPDATE (append to messages JSONB) if exists.
    Also update Redis STM (FIFO 3 Q&A pairs).
    """
    pg = get_pg_client()
    await pg.connect()
    ts = datetime.now(timezone.utc).isoformat()

    new_turn = [
        {"role": "user", "content": user_query, "timestamp": ts},
        {"role": "assistant", "content": assistant_answer, "timestamp": ts,
         "metadata": {"intent": intent, "tokens": tokens}},
    ]

    await pg.execute(
        """
        INSERT INTO conversations (id, user_id, session_id, messages, created_at, updated_at)
        VALUES (gen_random_uuid(), $1::uuid, $2::uuid, $3::jsonb, now(), now())
        ON CONFLICT (session_id) DO UPDATE
        SET messages = conversations.messages || $3::jsonb,
            updated_at = now()
        """,
        user_id, session_id, json.dumps(new_turn),
    )

    await _append_stm(session_id, user_query, assistant_answer, ts)


async def _append_stm(session_id: str, q: str, a: str, ts: str) -> None:
    r = aioredis.from_url(_REDIS_URL)
    try:
        raw = await r.get(f"stm:{session_id}")
        stm = json.loads(raw) if raw else []
        stm.append({"q": q, "a": a, "ts": ts})
        stm = stm[-_STM_MAX:]
        await r.setex(f"stm:{session_id}", 7200, json.dumps(stm))
    except Exception:
        pass
    finally:
        close_fn = getattr(r, "aclose", None) or r.close
        await close_fn()
