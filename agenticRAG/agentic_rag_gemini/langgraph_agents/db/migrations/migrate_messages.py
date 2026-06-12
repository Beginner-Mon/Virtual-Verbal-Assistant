"""One-time migration: conversations.messages JSONB → messages table.

Usage:
    python -m langgraph_agents.db.migrations.migrate_messages
"""
import asyncio
import json
import os
from pathlib import Path

import asyncpg
from dotenv import load_dotenv

load_dotenv(Path(__file__).resolve().parents[3] / ".env")

_DSN = os.getenv(
    "DATABASE_URL",
    "postgresql://postgres:postgres@localhost:5432/vva"
)


async def main():
    conn = await asyncpg.connect(_DSN)
    rows = await conn.fetch(
        "SELECT session_id, messages FROM conversations WHERE _migrated = false"
    )
    print(f"Migrating {len(rows)} sessions…")
    migrated = 0
    for row in rows:
        msgs = row["messages"]
        if isinstance(msgs, str):
            try:
                msgs = json.loads(msgs)
            except json.JSONDecodeError:
                msgs = []
        if not msgs:
            await conn.execute(
                "UPDATE conversations SET _migrated = true WHERE session_id = $1",
                row["session_id"],
            )
            continue

        records = [
            (
                row["session_id"],
                m["role"],
                m.get("content", ""),
                None,
                None,
                None,
                m.get("timestamp"),
            )
            for m in msgs
            if m.get("role") in ("user", "assistant") and m.get("content")
        ]
        if records:
            await conn.executemany(
                """INSERT INTO messages (session_id, role, content, intent, tokens, grader_result, created_at)
                   VALUES ($1::uuid, $2, $3, $4, $5, $6, COALESCE($7::timestamptz, now()))
                   ON CONFLICT DO NOTHING""",
                records,
            )
        await conn.execute(
            "UPDATE conversations SET _migrated = true WHERE session_id = $1",
            row["session_id"],
        )
        migrated += 1

    await conn.close()
    print(f"Done. {migrated}/{len(rows)} sessions migrated.")


if __name__ == "__main__":
    asyncio.run(main())
