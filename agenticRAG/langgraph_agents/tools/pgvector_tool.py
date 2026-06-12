"""Retrieval tools — M.2 + M.6 (kb_search + memory_search + resume_last_session).

M.4 schema targets:
  - kb_search → kb_embeddings table (public KB, FK documents)
  - memory_search → summaries table (2-step tenant scope, D19)

Decisions encoded:
  D11:  Memory tách Knowledge — 2 index/tool riêng
  D19:  NO user_id on summaries — tenant via 2-step (conversations→summaries)
  D23:  Empty {found:[]} ≠ error {error}
  D28:  Top-k caps: kb=5, memory=3
  M.6:  memory_search: topic→vector, temporal→SQL filter (tách 2 trục)
"""

from __future__ import annotations

from typing import Optional

from langchain_core.runnables import RunnableConfig
from langchain_core.tools import tool

from langgraph_agents.shared import get_embedding_service, get_pg_client
from langgraph_agents.shared.logging import get_logger

logger = get_logger("langgraph.tools")


def _to_uuid(value: str) -> str:
    """Coerce user-supplied string into a deterministic UUID."""
    import uuid
    try:
        return str(uuid.UUID(value))
    except (ValueError, TypeError):
        return str(uuid.uuid5(uuid.NAMESPACE_DNS, str(value)))


# ═══════════════════════════════════════════════════════════════════════════
# kb_search — public knowledge base (kb_embeddings table)
# ═══════════════════════════════════════════════════════════════════════════

@tool
async def kb_search(query: str, top_k: int = 5) -> list[dict]:
    """Search the internal PT/wellness knowledge base.

    Use for: exercises, stretches, anatomy, physiotherapy techniques,
    health/wellness facts. Returns chunks with content + similarity score.

    Args:
        query: Search query in Vietnamese or English.
        top_k: Max results (default 5, capped at 5 per D28).

    Returns:
        List of {content, similarity, source_type, document_title} dicts.
        Empty list if no results found (NOT an error — D23).
    """
    top_k = min(top_k, 5)  # D28 cap

    try:
        embed_svc = get_embedding_service()
        pg = get_pg_client()
        await pg.connect()

        # Generate embedding with query: prefix for search
        query_vec = await embed_svc.aembed_query(query)

        rows = await pg.fetch(
            """
            SELECT ke.content, ke.chunk_index,
                   1 - (ke.embedding <=> $1) AS similarity,
                   d.source_type, d.title
            FROM kb_embeddings ke
            JOIN documents d ON ke.document_id = d.id
            ORDER BY ke.embedding <=> $1
            LIMIT $2
            """,
            query_vec,
            top_k,
        )

        results = [
            {
                "content": r["content"],
                "similarity": round(float(r["similarity"]), 4),
                "source_type": r["source_type"],
                "document_title": r["title"] or "",
                "chunk_index": r["chunk_index"],
            }
            for r in rows
        ]

        logger.info("kb_search_done", extra={
            "query": query[:80], "results": len(results),
        })
        return results

    except Exception as exc:
        # D23: error → raise so retriever can retry
        logger.error("kb_search_error", extra={"query": query[:80], "error": str(exc)})
        raise


# ═══════════════════════════════════════════════════════════════════════════
# memory_search — user's past session summaries (summaries table, 2-step)
# ═══════════════════════════════════════════════════════════════════════════

@tool
async def memory_search(
    query: str,
    since_days: Optional[int] = None,
    top_k: int = 3,
    config: RunnableConfig = None,
) -> dict:
    """Search user's past conversation summaries for relevant context.

    Use for: "lần trước", "tuần trước", "tôi đã hỏi", "nhắc lại",
    "tiếp tục bài tập". Topic search uses vector similarity;
    temporal search uses SQL date filter (M.6: tách 2 trục).

    Args:
        query: What to search for in past conversations.
        since_days: Optional — only search within N days (e.g. 7 for "tuần trước").
        top_k: Max results (default 3, capped at 3 per D28).

    Returns:
        {found: true, results: [{summary_text, session_id, similarity, created_at}]}
        OR {found: false} if no results (NOT error — D23).
    """
    top_k = min(top_k, 3)  # D28 cap

    # Scope injected from runtime config — LLM does NOT see these (M.6/D19)
    user_id = _to_uuid(config["configurable"]["user_id"])
    current_session_id = config["configurable"]["session_id"]

    try:
        embed_svc = get_embedding_service()
        pg = get_pg_client()
        await pg.connect()

        # Step 1: Get user's session_ids (tenant scope — M.6)
        session_rows = await pg.fetch(
            "SELECT session_id FROM conversations WHERE user_id = $1",
            user_id,
        )
        user_session_ids = [r["session_id"] for r in session_rows]

        if not user_session_ids:
            return {"found": False}

        # Step 2: Vector search scoped to user's summaries (D19)
        query_vec = await embed_svc.aembed_query(query)

        # Build temporal filter (M.6: tách topic/temporal — 2 trục)
        if since_days is not None and since_days > 0:
            rows = await pg.fetch(
                """
                SELECT s.summary_text, s.session_id, s.created_at,
                       1 - (s.embedding <=> $1) AS similarity
                FROM summaries s
                WHERE s.session_id = ANY($2)
                  AND s.session_id != $3
                  AND s.status = 'active'
                  AND s.created_at >= now() - ($4 || ' days')::interval
                ORDER BY s.embedding <=> $1
                LIMIT $5
                """,
                query_vec, user_session_ids, current_session_id,
                str(since_days), top_k,
            )
        else:
            rows = await pg.fetch(
                """
                SELECT s.summary_text, s.session_id, s.created_at,
                       1 - (s.embedding <=> $1) AS similarity
                FROM summaries s
                WHERE s.session_id = ANY($2)
                  AND s.session_id != $3
                  AND s.status = 'active'
                ORDER BY s.embedding <=> $1
                LIMIT $4
                """,
                query_vec, user_session_ids, current_session_id, top_k,
            )

        results = [
            {
                "summary_text": r["summary_text"],
                "session_id": str(r["session_id"]),
                "similarity": round(float(r["similarity"]), 4),
                "created_at": r["created_at"].isoformat(),
            }
            for r in rows
        ]

        logger.info("memory_search_done", extra={
            "query": query[:80],
            "user_sessions": len(user_session_ids),
            "results": len(results),
            "since_days": since_days,
        })
        return {"found": True, "results": results} if results else {"found": False}

    except Exception as exc:
        logger.error("memory_search_error", extra={"query": query[:80], "error": str(exc)})
        raise


# ═══════════════════════════════════════════════════════════════════════════
# resume_last_session — tool riêng cho "tiếp tục session vừa rồi" (M.6)
# ═══════════════════════════════════════════════════════════════════════════

@tool
async def resume_last_session(
    since_days: Optional[int] = None,
    config: RunnableConfig = None,
) -> dict:
    """Resume the most recent past session for this user.

    Use for: "tiếp tục", "làm tiếp bài hôm trước", "quay lại bài tập".
    Returns summaries + recent messages from the most recent session
    (excluding current session). No vector search needed — recency-based.

    Args:
        since_days: Optional time window (e.g. 1 for "hôm qua", 7 for "tuần trước").

    Returns:
        {found: true, session_id, summary_chunks, recent_messages}
        OR {found: false}
    """
    # Scope injected from runtime config — LLM does NOT see these
    user_id = _to_uuid(config["configurable"]["user_id"])
    current_session_id = config["configurable"]["session_id"]

    try:
        pg = get_pg_client()
        await pg.connect()

        # Step 1: Find most recent session
        if since_days is not None and since_days > 0:
            row = await pg.fetchrow(
                """
                SELECT session_id, title, updated_at
                FROM conversations
                WHERE user_id = $1
                  AND session_id != $2
                  AND updated_at >= now() - ($3 || ' days')::interval
                ORDER BY updated_at DESC
                LIMIT 1
                """,
                user_id, current_session_id, str(since_days),
            )
        else:
            row = await pg.fetchrow(
                """
                SELECT session_id, title, updated_at
                FROM conversations
                WHERE user_id = $1
                  AND session_id != $2
                ORDER BY updated_at DESC
                LIMIT 1
                """,
                user_id, current_session_id,
            )

        if not row:
            return {"found": False}

        target_session = str(row["session_id"])

        # Step 2: Load summaries (frozen chunks)
        summary_rows = await pg.fetch(
            """
            SELECT summary_text, covers_from_seq, covers_up_to_seq
            FROM summaries
            WHERE session_id = $1 AND status = 'active'
            ORDER BY covers_from_seq
            """,
            target_session,
        )
        summary_chunks = [r["summary_text"] for r in summary_rows]

        # Step 3: Load recent messages after last summary
        last_seq = summary_rows[-1]["covers_up_to_seq"] if summary_rows else 0
        msg_rows = await pg.fetch(
            """
            SELECT role, content, seq_id
            FROM messages
            WHERE session_id = $1 AND seq_id > $2
            ORDER BY seq_id
            LIMIT 10
            """,
            target_session,
            last_seq,
        )
        recent_messages = [
            {"role": r["role"], "content": r["content"]} for r in msg_rows
        ]

        logger.info("resume_last_session_done", extra={
            "target_session": target_session,
            "summary_chunks": len(summary_chunks),
            "recent_messages": len(recent_messages),
        })

        return {
            "found": True,
            "session_id": target_session,
            "summary_chunks": summary_chunks,
            "recent_messages": recent_messages,
        }

    except Exception as exc:
        logger.error("resume_last_session_error", extra={"error": str(exc)})
        raise


# ── Exports ───────────────────────────────────────────────────────────────
# Legacy name pgvector_search → new name kb_search (backwards compat)
pgvector_search = kb_search
