"""One-time migration: conversations.messages JSONB → messages table.

Usage:
    python -m langgraph_agents.db.migrations.migrate_messages
"""
import asyncio
import json

import asyncpg

# Resolve the database the same way the application does, rather than a third
# way. This script used to load the repo-root `.env` (which the app does not
# read) and take the DSN from `DATABASE_URL` (which nothing else sets),
# defaulting to `postgresql://postgres:postgres@localhost:5432/vva` — a
# different port, a different user, and a different database from both the local
# container (:5433, user vva) and the managed one. Running it therefore migrated
# something that was not the application's data, or failed to connect at all.
from langgraph_agents.db.postgres import get_default_dsn

_DSN = get_default_dsn()


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
