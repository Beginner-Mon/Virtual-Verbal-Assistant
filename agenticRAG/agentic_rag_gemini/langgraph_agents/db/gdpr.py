"""GDPR deletion cascade — M.8 (D12, D17).

CRITICAL: summaries have NO FK to messages (D19). Deleting a message does NOT
automatically cascade to summaries. We must manually mark affected summary chunks
as 'dirty' and exclude them from context + search until re-summarized.

Key rules (M.8):
  1. Mark-dirty = APPLICATION LOGIC (not DB cascade)
  2. HARD delete (not soft)
  3. Empty-chunk after delete → DELETE chunk + embedding
  4. Dirty window: chunk excluded from context assembly + memory_search
  5. Re-summarize = background task (fixes dirty → active)
  6. Re-summarize + regen embedding in SAME transaction (prevents stale vectors)
"""

from __future__ import annotations

from langgraph_agents.shared import get_pg_client
from langgraph_agents.shared.logging import get_logger

logger = get_logger("langgraph.gdpr")


async def delete_message(message_id: str, session_id: str) -> dict:
    """Delete a single message and mark affected summary chunks as dirty.

    Step 1: Get seq_id of message to delete
    Step 2: Mark summaries covering this seq_id as 'dirty'
    Step 3: Hard-delete the message
    Step 4: Remove empty summary chunks

    Returns: {deleted: true, dirty_chunks: int, empty_chunks_deleted: int}
    """
    pg = get_pg_client()
    await pg.connect()

    async with pg._pool.acquire() as conn:
        async with conn.transaction():
            # Step 1: Get seq_id
            row = await conn.fetchrow(
                "SELECT seq_id FROM messages WHERE id = $1 AND session_id = $2",
                message_id, session_id,
            )
            if not row:
                return {"deleted": False, "reason": "not_found"}

            seq = row["seq_id"]

            # Step 2: Mark affected summaries as dirty (M.8 #1)
            result = await conn.execute(
                """
                UPDATE summaries
                SET status = 'dirty'
                WHERE session_id = $1
                  AND covers_from_seq <= $2
                  AND $2 <= covers_up_to_seq
                  AND status = 'active'
                """,
                session_id, seq,
            )
            # asyncpg execute returns "UPDATE N" string; parse it
            dirty_count = int(result.split()[-1]) if result else 0

            # Step 3: Hard-delete the message (M.8 #2)
            await conn.execute(
                "DELETE FROM messages WHERE id = $1 AND session_id = $2",
                message_id, session_id,
            )

            # Step 4: Delete empty summary chunks (M.8 #3)
            # A chunk is empty if no messages exist in its range after deletion
            empty_result = await conn.execute(
                """
                DELETE FROM summaries
                WHERE session_id = $1
                  AND status = 'dirty'
                  AND NOT EXISTS (
                      SELECT 1 FROM messages
                      WHERE messages.session_id = $1
                        AND messages.seq_id BETWEEN summaries.covers_from_seq
                                                AND summaries.covers_up_to_seq
                  )
                """,
                session_id,
            )
            empty_deleted = int(empty_result.split()[-1]) if empty_result else 0

            logger.info("gdpr_delete_message", extra={
                "message_id": message_id,
                "session_id": session_id,
                "seq": seq,
                "dirty_chunks": dirty_count,
                "empty_chunks_deleted": empty_deleted,
            })

            return {
                "deleted": True,
                "dirty_chunks": dirty_count,
                "empty_chunks_deleted": empty_deleted,
            }


async def delete_session(session_id: str) -> dict:
    """Hard-delete an entire session and all associated data.

    FK ON DELETE CASCADE handles:
      - messages (session_id FK → CASCADE)
      - summaries (session_id FK → CASCADE)
    conversations row is deleted explicitly.
    """
    pg = get_pg_client()
    await pg.connect()

    await pg.execute(
        "DELETE FROM conversations WHERE session_id = $1",
        session_id,
    )

    logger.info("gdpr_delete_session", extra={"session_id": session_id})
    return {"deleted": True, "session_id": session_id}


async def delete_user(user_id: str) -> dict:
    """Hard-delete a user and ALL associated data (GDPR erasure).

    FK ON DELETE CASCADE handles:
      - conversations (user_id FK → CASCADE)
        - messages (session_id FK → CASCADE)
        - summaries (session_id FK → CASCADE)
      - user_memory (user_id FK → CASCADE)
    users row is deleted explicitly.

    Per D19: messages + summaries have NO user_id — cascade through conversations
    works correctly (conversations.user_id → conversations.session_id → messages/summaries).
    """
    pg = get_pg_client()
    await pg.connect()

    # Count before deletion for audit
    conv_count = await pg.fetchval(
        "SELECT COUNT(*) FROM conversations WHERE user_id = $1", user_id
    )

    await pg.execute("DELETE FROM users WHERE id = $1", user_id)

    logger.info("gdpr_delete_user", extra={
        "user_id": user_id,
        "conversations_deleted": conv_count,
    })
    return {
        "deleted": True,
        "user_id": user_id,
        "conversations_deleted": conv_count,
    }


async def get_dirty_chunks(session_id: str) -> list[dict]:
    """List summary chunks with 'dirty' status for a session.

    These chunks are EXCLUDED from context assembly + memory_search (D17).
    """
    pg = get_pg_client()
    await pg.connect()

    rows = await pg.fetch(
        """
        SELECT id, covers_from_seq, covers_up_to_seq, created_at
        FROM summaries
        WHERE session_id = $1 AND status = 'dirty'
        ORDER BY covers_from_seq
        """,
        session_id,
    )
    return [
        {
            "id": str(r["id"]),
            "covers_from_seq": r["covers_from_seq"],
            "covers_up_to_seq": r["covers_up_to_seq"],
            "created_at": r["created_at"].isoformat(),
        }
        for r in rows
    ]


async def re_summarize_chunk(
    session_id: str,
    chunk_id: str,
    new_summary_text: str,
    new_embedding: list[float],
) -> bool:
    """Replace a dirty chunk with new summary + embedding in ONE transaction.

    M.8 #6: regen embedding in SAME transaction as summary_text update
    to prevent stale vectors pointing to deleted content.
    """
    pg = get_pg_client()
    await pg.connect()

    async with pg._pool.acquire() as conn:
        async with conn.transaction():
            result = await conn.execute(
                """
                UPDATE summaries
                SET summary_text = $1,
                    embedding = $2,
                    status = 'active'
                WHERE id = $3
                  AND session_id = $4
                  AND status = 'dirty'
                """,
                new_summary_text, new_embedding, chunk_id, session_id,
            )
            updated = int(result.split()[-1]) if result else 0

    logger.info("gdpr_re_summarize", extra={
        "session_id": session_id,
        "chunk_id": chunk_id,
        "updated": updated > 0,
    })
    return updated > 0
