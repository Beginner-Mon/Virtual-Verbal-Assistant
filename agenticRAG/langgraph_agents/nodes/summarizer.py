"""Background summarizer — M.5 (chunk-based, frozen, CAS idempotent).

Decisions encoded:
  D13:  Single threshold = 10k tokens; recent-raw window = chunk_size
  M.5:  Summary chunk-based, trigger by TOKEN cumulative, runs BACKGROUND,
        frozen on write (never re-compressed), CAS idempotent
  M.7:  Summary chunks = STATIC prefix → prompt cache works (append-only)

Trigger: after write_session_turn, check cumulative tokens since last summary.
If >= 10k → fire background task. Never blocks the chat response.

Edge A (M.5): summarize fail → retry 1× → still fail → log WARNING, leave.
Hard cap: memory node uses existing token budget to prevent unbounded growth.
"""

from __future__ import annotations

import asyncio
import time
from datetime import datetime, timezone

from langgraph_agents.shared import get_pg_client, get_embedding_service
from langgraph_agents.shared.logging import get_logger
from langgraph_agents.llm import get_chat_model

logger = get_logger("langgraph.summarizer")

# Strong references to fire-and-forget summarizer tasks (same pattern as TTS)
_pending_summarizer_tasks: set = set()

_SUMMARY_THRESHOLD = 10_000   # D13: single threshold
_MAX_RETRY = 1                # Edge A: retry once


def _token_estimate(text: str) -> int:
    """Rough token count: ~4 chars per token for Vietnamese/English."""
    return len(text) // 4


async def maybe_summarize(session_id: str) -> None:
    """Check if summarization is needed and fire background task.

    Called after write_session_turn. Never raises — all errors logged.
    """
    try:
        pg = get_pg_client()
        await pg.connect()

        # Get last summary chunk's covers_up_to_seq (0 if none)
        last_chunk = await pg.fetchval(
            """SELECT COALESCE(MAX(covers_up_to_seq), 0)
               FROM summaries
               WHERE session_id = $1 AND status = 'active'""",
            session_id,
        )

        # Count uncached token budget since last summary
        rows = await pg.fetch(
            """SELECT token_count, content FROM messages
               WHERE session_id = $1 AND seq_id > $2
               ORDER BY seq_id""",
            session_id, last_chunk or 0,
        )

        total_tokens = 0
        for r in rows:
            total_tokens += r["token_count"] or _token_estimate(r["content"] or "")

        if total_tokens < _SUMMARY_THRESHOLD:
            return

        # Fire background task (same pattern as _pending_tts_tasks)
        task = asyncio.create_task(
            _run_summarize(session_id, last_chunk or 0)
        )
        _pending_summarizer_tasks.add(task)
        task.add_done_callback(_pending_summarizer_tasks.discard)

    except Exception as exc:
        logger.warning("summarizer_trigger_failed", extra={
            "session_id": session_id, "error": str(exc),
        })


async def _summarize_messages(rows: list, session_id: str) -> str | None:
    """LLM-summarize a list of message rows → summary_text or None if fail.

    rows = records with {role, content, token_count}.
    Uses the same prompt + retry logic as _run_summarize.
    Extracted so rebuild_dirty_chunk can reuse it (M.8 #5).
    """
    lines = []
    for r in rows:
        label = "User" if r["role"] == "user" else "Assistant"
        lines.append(f"{label}: {r['content']}")

    conversation = "\n".join(lines)

    llm = get_chat_model("planner")
    # "The same language as the conversation", not a named language. This used to
    # force Vietnamese, so an English session was summarised into Vietnamese and
    # then fed back to the model as context on the next turn — pushing it toward
    # a language the user had not written in.
    prompt = (
        "Summarise the following conversation in 2-4 sentences, written in the "
        "SAME language as the conversation itself. Keep what matters clinically: "
        "symptoms, exercises recommended, contraindications, and the user's "
        "progress.\n\n"
        f"{conversation}"
    )

    summary_text = ""
    for attempt in range(_MAX_RETRY + 1):
        try:
            ai_msg = await llm.ainvoke(prompt)
            summary_text = ai_msg.content.strip() if ai_msg.content else ""
            if summary_text:
                break
        except Exception as exc:
            if attempt < _MAX_RETRY:
                logger.warning("summarizer_llm_retry", extra={
                    "session_id": session_id, "attempt": attempt + 1,
                    "error": str(exc),
                })
                await asyncio.sleep(0.5)
            else:
                logger.error("summarizer_llm_failed", extra={
                    "session_id": session_id, "errors": str(exc),
                })
                return None

    if not summary_text:
        logger.warning("summarizer_empty_response", extra={
            "session_id": session_id,
        })
        return None

    return summary_text


async def _run_summarize(session_id: str, from_seq: int) -> None:
    """Summarize messages in (from_seq, new_covers_up_to_seq]."""
    t0 = time.perf_counter()

    try:
        pg = get_pg_client()
        await pg.connect()

        # Load messages not yet covered
        rows = await pg.fetch(
            """SELECT seq_id, role, content, token_count
               FROM messages
               WHERE session_id = $1 AND seq_id > $2
               ORDER BY seq_id""",
            session_id, from_seq,
        )

        if not rows:
            return

        total_tokens = 0
        for r in rows:
            total_tokens += r["token_count"] or _token_estimate(r["content"] or "")

        covers_up_to_seq = rows[-1]["seq_id"]

        # ── LLM summarize (cheap model tier) ───────────────────────────
        summary_text = await _summarize_messages(rows, session_id)
        if not summary_text:
            return

        # ── Embed + INSERT (same transaction) ───────────────────────────
        embed_svc = get_embedding_service()
        embedding = await embed_svc.aembed_passage(summary_text)

        await pg.execute(
            """INSERT INTO summaries
               (session_id, summary_text, covers_from_seq, covers_up_to_seq,
                embedding, status)
               VALUES ($1, $2, $3, $4, $5, 'active')
               ON CONFLICT ON CONSTRAINT uq_chunk DO NOTHING""",
            session_id, summary_text, from_seq + 1, covers_up_to_seq, embedding,
        )

        elapsed_ms = round((time.perf_counter() - t0) * 1000)
        logger.info("summarizer_done", extra={
            "session_id": session_id,
            "covers_from_seq": from_seq + 1,
            "covers_up_to_seq": covers_up_to_seq,
            "token_count": total_tokens,
            "summary_length": len(summary_text),
            "elapsed_ms": elapsed_ms,
        })

    except Exception as exc:
        logger.error("summarizer_fatal", extra={
            "session_id": session_id, "error": str(exc),
        })


async def rebuild_dirty_chunk(session_id: str, chunk_id: str) -> bool:
    """Re-summarize 1 dirty chunk from messages remaining in its range (M.8 #5,#6).

    Returns True if chunk is back to active; False if skipped (empty/LLM fail).
    """
    pg = get_pg_client()
    await pg.connect()

    # 1. Get range of the dirty chunk
    chunk = await pg.fetchrow(
        "SELECT covers_from_seq, covers_up_to_seq FROM summaries "
        "WHERE id = $1 AND session_id = $2 AND status = 'dirty'",
        chunk_id, session_id,
    )
    if not chunk:
        return False   # already handled / no longer dirty (idempotent)

    # 2. Load messages REMAINING in range (deleted ones are gone)
    rows = await pg.fetch(
        "SELECT role, content, token_count FROM messages "
        "WHERE session_id = $1 AND seq_id BETWEEN $2 AND $3 ORDER BY seq_id",
        session_id, chunk["covers_from_seq"], chunk["covers_up_to_seq"],
    )
    if not rows:
        return False   # empty-chunk: delete_message already removed it (M.8 #3)

    # 3. Re-summarize + embed + UPDATE (call gdpr with correct 4-arg signature)
    summary_text = await _summarize_messages(rows, session_id)
    if not summary_text:
        return False   # LLM fail → leave dirty, retry next time

    embedding = await get_embedding_service().aembed_passage(summary_text)

    from langgraph_agents.db.gdpr import re_summarize_chunk
    return await re_summarize_chunk(session_id, chunk_id, summary_text, embedding)
