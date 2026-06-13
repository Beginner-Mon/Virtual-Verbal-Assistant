"""Acceptance tests — cụm A re-work (A1 clarify, A2 GDPR, A3 user_memory).

R2 required tests from FIX-R1-GDPR-RESUMMARIZE.md.
Conventions:
  - HAS_PG guard + _make_config pattern from test_pr1_memory_fix.py
  - AsyncMock patterns from test_pr2_summarizer.py
  - Markers: unit / integration
"""

from __future__ import annotations

import json
import uuid
import pytest
from unittest.mock import AsyncMock, MagicMock, patch

from langchain_core.runnables import RunnableConfig

# ── PG availability guard (same pattern as test_pr1_memory_fix.py) ─────────

HAS_PG = False
try:
    import asyncpg
    async def _ping():
        conn = await asyncpg.connect("postgresql://vva:vva_dev@localhost:5433/vva")
        await conn.close()
        return True
    import asyncio
    HAS_PG = asyncio.run(_ping())
except Exception:
    pass


def _make_config(user_id="test-user-a", session_id=None):
    return RunnableConfig(configurable={
        "user_id": user_id,
        "session_id": session_id or str(uuid.uuid4()),
        "query": "test",
        "persona_id": "eca_default",
        "request_id": "req-test",
    })


# ═══════════════════════════════════════════════════════════════════════════
# A2 — GDPR (integration, real PG)
# ═══════════════════════════════════════════════════════════════════════════

@pytest.mark.integration
@pytest.mark.skipif(not HAS_PG, reason="PostgreSQL not available on port 5433")
@pytest.mark.asyncio
async def test_a2_dirty_window_excludes_from_load_and_search():
    """Dirty chunk excluded from _load_summary_chunks and memory_search (M.8 #4)."""
    from langgraph_agents.shared import get_pg_client, get_embedding_service
    from langgraph_agents.nodes.memory import _load_summary_chunks
    from langgraph_agents.tools.pgvector_tool import memory_search

    pg = get_pg_client()
    await pg.connect()
    embed = get_embedding_service()

    user_id = str(uuid.uuid4())
    session_id = str(uuid.uuid4())

    await pg.execute("INSERT INTO users (id) VALUES ($1) ON CONFLICT DO NOTHING", user_id)
    await pg.execute(
        "INSERT INTO conversations (session_id, user_id) VALUES ($1, $2) ON CONFLICT DO NOTHING",
        session_id, user_id,
    )

    try:
        # Insert messages seq 1-10
        for seq in range(1, 11):
            await pg.execute(
                """INSERT INTO messages (session_id, seq_id, role, content, token_count)
                   VALUES ($1, $2, 'user', $3, 10)""",
                session_id, seq, f"message {seq}",
            )

        # Insert active summary chunk covering seq 1-10
        vec = await embed.aembed_passage("Người dùng hỏi về bài tập vai và được khuyên bài tập xoay vai.")
        await pg.execute(
            """INSERT INTO summaries
               (session_id, summary_text, covers_from_seq, covers_up_to_seq, embedding, status)
               VALUES ($1, $2, 1, 10, $3, 'active')
               ON CONFLICT ON CONSTRAINT uq_chunk DO NOTHING""",
            session_id,
            "Người dùng hỏi về bài tập vai và được khuyên bài tập xoay vai.",
            vec,
        )

        # Verify chunk is visible before mark-dirty
        chunks_before = await _load_summary_chunks(session_id)
        assert len(chunks_before) == 1, "Chunk should be active before deletion"

        # Mark chunk dirty (simulate delete_message side-effect)
        await pg.execute(
            "UPDATE summaries SET status = 'dirty' WHERE session_id = $1 AND status = 'active'",
            session_id,
        )

        # Dirty chunk must NOT appear in _load_summary_chunks
        chunks_after = await _load_summary_chunks(session_id)
        assert chunks_after == [], "_load_summary_chunks must exclude dirty chunks"

        # Dirty chunk must NOT appear in memory_search results
        search_session = str(uuid.uuid4())
        config = _make_config(user_id=user_id, session_id=search_session)
        result = await memory_search.ainvoke({"query": "bài tập vai"}, config)
        # Either not found or found results must all be active
        if result.get("found"):
            for r in result["results"]:
                assert r["session_id"] != session_id, (
                    "memory_search must not return results from dirty-chunk session"
                )

    finally:
        await pg.execute("DELETE FROM summaries WHERE session_id = $1", session_id)
        await pg.execute("DELETE FROM messages WHERE session_id = $1", session_id)
        await pg.execute("DELETE FROM conversations WHERE session_id = $1", session_id)
        await pg.execute("DELETE FROM users WHERE id = $1", user_id)


@pytest.mark.integration
@pytest.mark.skipif(not HAS_PG, reason="PostgreSQL not available on port 5433")
@pytest.mark.asyncio
async def test_a2_rebuild_dirty_chunk_round_trip():
    """rebuild_dirty_chunk → chunk active, summary_text updated, embedding changed (M.8 #5,#6)."""
    from langgraph_agents.shared import get_pg_client, get_embedding_service
    from langgraph_agents.nodes.summarizer import rebuild_dirty_chunk

    pg = get_pg_client()
    await pg.connect()
    embed = get_embedding_service()

    user_id = str(uuid.uuid4())
    session_id = str(uuid.uuid4())

    await pg.execute("INSERT INTO users (id) VALUES ($1) ON CONFLICT DO NOTHING", user_id)
    await pg.execute(
        "INSERT INTO conversations (session_id, user_id) VALUES ($1, $2) ON CONFLICT DO NOTHING",
        session_id, user_id,
    )

    try:
        # Insert messages seq 1-5 (seq 3 will be "deleted")
        contents = {
            1: "Tôi bị đau lưng",
            2: "Bạn nên tập cầu mông nhẹ",
            # 3 is the deleted message (we simply don't insert it)
            4: "Cảm ơn bác sĩ",
            5: "Chào mừng bạn, chúc tập luyện tốt",
        }
        for seq, content in contents.items():
            await pg.execute(
                """INSERT INTO messages (session_id, seq_id, role, content, token_count)
                   VALUES ($1, $2, 'user', $3, 10)""",
                session_id, seq, content,
            )

        # Insert a dirty summary chunk covering seq 1-5
        old_summary = "Người dùng bị đau lưng và hỏi về bài tập vai (TIN CŨ CẦN XÓA)."
        old_vec = await embed.aembed_passage(old_summary)
        chunk_id = await pg.fetchval(
            """INSERT INTO summaries
               (session_id, summary_text, covers_from_seq, covers_up_to_seq, embedding, status)
               VALUES ($1, $2, 1, 5, $3, 'dirty')
               RETURNING id""",
            session_id, old_summary, old_vec,
        )
        chunk_id = str(chunk_id)

        old_vec_list = old_vec if isinstance(old_vec, list) else list(old_vec)

        # Mock LLM to return a predictable new summary (avoids live LLM call)
        new_summary_text = "Người dùng bị đau lưng, được khuyên tập cầu mông nhẹ."
        mock_llm = MagicMock()
        mock_ai = MagicMock()
        mock_ai.content = new_summary_text
        mock_llm.ainvoke = AsyncMock(return_value=mock_ai)

        with patch("langgraph_agents.nodes.summarizer.get_chat_model", return_value=mock_llm):
            result = await rebuild_dirty_chunk(session_id, chunk_id)

        assert result is True, "rebuild_dirty_chunk must return True on success"

        # Verify chunk is now active
        row = await pg.fetchrow(
            "SELECT status, summary_text, embedding FROM summaries WHERE id = $1::uuid",
            chunk_id,
        )
        assert row is not None
        assert row["status"] == "active", f"Chunk status must be 'active', got {row['status']}"
        assert row["summary_text"] == new_summary_text, (
            f"summary_text must be updated, got: {row['summary_text']}"
        )
        # Embedding must differ from old (new text → new vector)
        new_vec = list(row["embedding"])
        assert new_vec != old_vec_list, "embedding must change after re-summarize"

    finally:
        await pg.execute("DELETE FROM summaries WHERE session_id = $1", session_id)
        await pg.execute("DELETE FROM messages WHERE session_id = $1", session_id)
        await pg.execute("DELETE FROM conversations WHERE session_id = $1", session_id)
        await pg.execute("DELETE FROM users WHERE id = $1", user_id)


@pytest.mark.integration
@pytest.mark.skipif(not HAS_PG, reason="PostgreSQL not available on port 5433")
@pytest.mark.asyncio
async def test_a2_empty_chunk_deleted_after_all_messages_removed():
    """Deleting all messages in chunk range → chunk row removed (M.8 #3)."""
    from langgraph_agents.shared import get_pg_client, get_embedding_service
    from langgraph_agents.db.gdpr import delete_message as gdpr_delete_message

    pg = get_pg_client()
    await pg.connect()
    embed = get_embedding_service()

    user_id = str(uuid.uuid4())
    session_id = str(uuid.uuid4())

    await pg.execute("INSERT INTO users (id) VALUES ($1) ON CONFLICT DO NOTHING", user_id)
    await pg.execute(
        "INSERT INTO conversations (session_id, user_id) VALUES ($1, $2) ON CONFLICT DO NOTHING",
        session_id, user_id,
    )

    try:
        # Insert exactly 1 message
        msg_id = await pg.fetchval(
            """INSERT INTO messages (session_id, seq_id, role, content, token_count)
               VALUES ($1, 1, 'user', 'single message', 5) RETURNING id""",
            session_id,
        )
        msg_id = str(msg_id)

        # Insert active summary covering only that message
        vec = await embed.aembed_passage("Người dùng gửi 1 tin nhắn duy nhất.")
        await pg.execute(
            """INSERT INTO summaries
               (session_id, summary_text, covers_from_seq, covers_up_to_seq, embedding, status)
               VALUES ($1, $2, 1, 1, $3, 'active')
               ON CONFLICT ON CONSTRAINT uq_chunk DO NOTHING""",
            session_id, "Người dùng gửi 1 tin nhắn duy nhất.", vec,
        )

        # Delete the only message → chunk should be removed (empty chunk)
        result = await gdpr_delete_message(msg_id, session_id)
        assert result["deleted"] is True

        # Chunk must no longer exist
        remaining = await pg.fetchval(
            "SELECT COUNT(*) FROM summaries WHERE session_id = $1",
            session_id,
        )
        assert remaining == 0, (
            f"Empty chunk must be deleted by gdpr_delete_message, found {remaining} rows"
        )

    finally:
        await pg.execute("DELETE FROM summaries WHERE session_id = $1", session_id)
        await pg.execute("DELETE FROM messages WHERE session_id = $1", session_id)
        await pg.execute("DELETE FROM conversations WHERE session_id = $1", session_id)
        await pg.execute("DELETE FROM users WHERE id = $1", user_id)


@pytest.mark.integration
@pytest.mark.skipif(not HAS_PG, reason="PostgreSQL not available on port 5433")
@pytest.mark.asyncio
async def test_a2_ownership_other_user_cannot_delete_message():
    """User B cannot delete User A's message — returns not_found (M.8 ownership)."""
    from langgraph_agents.shared import get_pg_client
    from langgraph_agents.db.gdpr import delete_message as gdpr_delete_message

    pg = get_pg_client()
    await pg.connect()

    user_a = str(uuid.uuid4())
    user_b = str(uuid.uuid4())
    session_a = str(uuid.uuid4())

    await pg.execute("INSERT INTO users (id) VALUES ($1) ON CONFLICT DO NOTHING", user_a)
    await pg.execute("INSERT INTO users (id) VALUES ($1) ON CONFLICT DO NOTHING", user_b)
    await pg.execute(
        "INSERT INTO conversations (session_id, user_id) VALUES ($1, $2) ON CONFLICT DO NOTHING",
        session_a, user_a,
    )

    try:
        msg_id = await pg.fetchval(
            """INSERT INTO messages (session_id, seq_id, role, content, token_count)
               VALUES ($1, 1, 'user', 'secret message', 5) RETURNING id""",
            session_a,
        )
        msg_id = str(msg_id)

        # User B tries to delete via a different session_id they don't own
        session_b_fake = str(uuid.uuid4())
        result = await gdpr_delete_message(msg_id, session_b_fake)

        # Must return not_found (wrong session_id → no row match)
        assert result["deleted"] is False, (
            "Deleting another user's message via wrong session must return deleted=False"
        )

        # Original message must still exist
        still_there = await pg.fetchval(
            "SELECT COUNT(*) FROM messages WHERE id = $1::uuid",
            msg_id,
        )
        assert still_there == 1, "Message must still exist after failed cross-user delete"

    finally:
        await pg.execute("DELETE FROM messages WHERE session_id = $1", session_a)
        await pg.execute("DELETE FROM conversations WHERE session_id = $1", session_a)
        await pg.execute("DELETE FROM users WHERE id = ANY($1)", [user_a, user_b])


@pytest.mark.integration
@pytest.mark.skipif(not HAS_PG, reason="PostgreSQL not available on port 5433")
@pytest.mark.asyncio
async def test_a2_delete_user_cascades_all_data():
    """DELETE /users/{id} → 0 rows in conversations/messages/summaries/user_memory (M.8)."""
    from langgraph_agents.shared import get_pg_client, get_embedding_service
    from langgraph_agents.db.gdpr import delete_user

    pg = get_pg_client()
    await pg.connect()
    embed = get_embedding_service()

    user_id = str(uuid.uuid4())
    session_id = str(uuid.uuid4())

    await pg.execute("INSERT INTO users (id) VALUES ($1) ON CONFLICT DO NOTHING", user_id)
    await pg.execute(
        "INSERT INTO conversations (session_id, user_id) VALUES ($1, $2) ON CONFLICT DO NOTHING",
        session_id, user_id,
    )
    await pg.execute(
        """INSERT INTO messages (session_id, seq_id, role, content, token_count)
           VALUES ($1, 1, 'user', 'hello', 5)""",
        session_id,
    )
    vec = await embed.aembed_passage("Hello summary.")
    await pg.execute(
        """INSERT INTO summaries
           (session_id, summary_text, covers_from_seq, covers_up_to_seq, embedding, status)
           VALUES ($1, $2, 1, 1, $3, 'active')
           ON CONFLICT ON CONSTRAINT uq_chunk DO NOTHING""",
        session_id, "Hello summary.", vec,
    )
    await pg.execute(
        "INSERT INTO users (id) VALUES ($1) ON CONFLICT DO NOTHING", user_id
    )
    await pg.execute(
        """INSERT INTO user_memory (user_id, fact_text)
           VALUES ($1::uuid, 'bị đau lưng')""",
        user_id,
    )

    result = await delete_user(user_id)
    assert result["deleted"] is True

    # All rows must be gone (cascade)
    conv_count = await pg.fetchval(
        "SELECT COUNT(*) FROM conversations WHERE user_id = $1::uuid", user_id,
    )
    msg_count = await pg.fetchval(
        "SELECT COUNT(*) FROM messages WHERE session_id = $1", session_id,
    )
    sum_count = await pg.fetchval(
        "SELECT COUNT(*) FROM summaries WHERE session_id = $1", session_id,
    )
    mem_count = await pg.fetchval(
        "SELECT COUNT(*) FROM user_memory WHERE user_id = $1::uuid", user_id,
    )

    assert conv_count == 0, f"conversations must be 0 after delete_user, got {conv_count}"
    assert msg_count == 0, f"messages must be 0 after delete_user, got {msg_count}"
    assert sum_count == 0, f"summaries must be 0 after delete_user, got {sum_count}"
    assert mem_count == 0, f"user_memory must be 0 after delete_user, got {mem_count}"


@pytest.mark.unit
@pytest.mark.asyncio
async def test_a2_regression_r1_rebuild_dirty_chunk_no_raise():
    """R1 regression: rebuild_dirty_chunk(session_id, chunk_id) does NOT raise TypeError.

    Mocks pg + LLM + embed so no real DB needed.
    Verifies the correct 4-arg gdpr.re_summarize_chunk is called (not the old 2-arg bug).
    """
    from langgraph_agents.nodes.summarizer import rebuild_dirty_chunk

    mock_pg = AsyncMock()
    mock_pg.connect = AsyncMock()

    # fetchrow returns chunk range
    mock_pg.fetchrow = AsyncMock(return_value={
        "covers_from_seq": 1,
        "covers_up_to_seq": 10,
    })
    # fetch returns 2 remaining messages
    mock_pg.fetch = AsyncMock(return_value=[
        {"role": "user", "content": "bị đau lưng", "token_count": 10},
        {"role": "assistant", "content": "tập cầu mông nhẹ", "token_count": 10},
    ])

    mock_embed = MagicMock()
    mock_embed.aembed_passage = AsyncMock(return_value=[0.1] * 384)

    mock_llm = MagicMock()
    mock_ai = MagicMock()
    mock_ai.content = "Người dùng bị đau lưng, được khuyên tập cầu mông."
    mock_llm.ainvoke = AsyncMock(return_value=mock_ai)

    mock_re_summarize = AsyncMock(return_value=True)

    with patch("langgraph_agents.nodes.summarizer.get_pg_client", return_value=mock_pg):
        with patch("langgraph_agents.nodes.summarizer.get_embedding_service", return_value=mock_embed):
            with patch("langgraph_agents.nodes.summarizer.get_chat_model", return_value=mock_llm):
                with patch("langgraph_agents.db.gdpr.re_summarize_chunk", mock_re_summarize):
                    # Must not raise — this was the R1 bug (TypeError on wrong args)
                    result = await rebuild_dirty_chunk("test-session", "test-chunk-id")

    assert result is True, "rebuild_dirty_chunk must return True when gdpr.re_summarize_chunk returns True"

    # Verify gdpr.re_summarize_chunk was called with correct 4-arg signature
    mock_re_summarize.assert_called_once()
    call_args = mock_re_summarize.call_args[0]
    assert call_args[0] == "test-session", "First arg must be session_id"
    assert call_args[1] == "test-chunk-id", "Second arg must be chunk_id"
    assert isinstance(call_args[2], str) and len(call_args[2]) > 0, "Third arg must be summary_text string"
    assert isinstance(call_args[3], list), "Fourth arg must be embedding list"


# ═══════════════════════════════════════════════════════════════════════════
# A1 — clarify động (unit)
# ═══════════════════════════════════════════════════════════════════════════

@pytest.mark.unit
@pytest.mark.asyncio
async def test_a1_resume_within_24h_returns_ambiguous():
    """resume_last_session: 2 sessions with gap < 24h → ambiguous=true + ≤3 candidates."""
    from langgraph_agents.tools.pgvector_tool import resume_last_session

    from datetime import datetime, timezone, timedelta
    now = datetime.now(timezone.utc)

    session_a = str(uuid.uuid4())
    session_b = str(uuid.uuid4())

    mock_pg = AsyncMock()
    mock_pg.connect = AsyncMock()

    # top2 sessions: updated_at gap = 1 hour (< 24h → ambiguous)
    mock_top2 = [
        MagicMock(**{
            "__getitem__": lambda self, k: {
                "session_id": session_a,
                "title": "Session A",
                "updated_at": now - timedelta(hours=1),
            }[k]
        }),
        MagicMock(**{
            "__getitem__": lambda self, k: {
                "session_id": session_b,
                "title": "Session B",
                "updated_at": now - timedelta(hours=2),
            }[k]
        }),
    ]
    # Patch __len__ so len(top2) >= 2
    mock_top2_list = [
        {"session_id": session_a, "title": "Session A", "updated_at": now - timedelta(hours=1)},
        {"session_id": session_b, "title": "Session B", "updated_at": now - timedelta(hours=2)},
    ]

    # fetchrow for preview messages
    async def fetchrow_side(query, *args):
        return {"content": "Xin chào, tôi bị đau vai."}

    mock_pg.fetch = AsyncMock(side_effect=[
        mock_top2_list,       # top2 sessions query
        [],                   # summary_rows
        [],                   # recent messages
    ])
    mock_pg.fetchrow = AsyncMock(side_effect=fetchrow_side)

    config = _make_config(user_id="user-a", session_id=str(uuid.uuid4()))

    with patch("langgraph_agents.tools.pgvector_tool.get_pg_client", return_value=mock_pg):
        result = await resume_last_session.ainvoke({}, config)

    assert result["found"] is True
    assert result.get("ambiguous") is True, (
        f"Expected ambiguous=True for sessions with 1h gap, got: {result}"
    )
    candidates = result.get("candidates", [])
    assert 1 <= len(candidates) <= 3, f"Expected 1-3 candidates, got {len(candidates)}"


@pytest.mark.unit
@pytest.mark.asyncio
async def test_a1_resume_beyond_24h_no_ambiguous():
    """resume_last_session: 2 sessions with gap > 3 days → no ambiguous key."""
    from langgraph_agents.tools.pgvector_tool import resume_last_session

    from datetime import datetime, timezone, timedelta
    now = datetime.now(timezone.utc)

    session_a = str(uuid.uuid4())
    session_b = str(uuid.uuid4())

    mock_top2_list = [
        {"session_id": session_a, "title": "Session A", "updated_at": now - timedelta(days=1)},
        {"session_id": session_b, "title": "Session B", "updated_at": now - timedelta(days=4)},
    ]

    mock_pg = AsyncMock()
    mock_pg.connect = AsyncMock()
    mock_pg.fetch = AsyncMock(side_effect=[
        mock_top2_list,  # top2 query
        [],              # summary_rows
        [],              # recent messages
    ])
    mock_pg.fetchrow = AsyncMock(return_value=None)

    config = _make_config(user_id="user-a", session_id=str(uuid.uuid4()))

    with patch("langgraph_agents.tools.pgvector_tool.get_pg_client", return_value=mock_pg):
        result = await resume_last_session.ainvoke({}, config)

    assert result["found"] is True
    assert "ambiguous" not in result, (
        f"Expected no ambiguous key for 3-day gap, got: {result}"
    )


@pytest.mark.unit
@pytest.mark.asyncio
async def test_a1_memory_search_ambiguous_when_top2_different_sessions_small_gap():
    """memory_search: top-2 from different sessions, sim gap < 0.05 → ambiguous=true."""
    from langgraph_agents.tools.pgvector_tool import memory_search
    from datetime import datetime, timezone

    session_x = str(uuid.uuid4())
    session_y = str(uuid.uuid4())
    user_id = str(uuid.uuid4())

    now = datetime.now(timezone.utc)

    mock_pg = AsyncMock()
    mock_pg.connect = AsyncMock()
    mock_pg.fetch = AsyncMock(side_effect=[
        # session_rows: user has 2 sessions
        [{"session_id": session_x}, {"session_id": session_y}],
        # search rows: two results from different sessions, similarity gap < 0.05
        [
            {"summary_text": "Bài tập cho đau vai session X",
             "session_id": session_x, "created_at": now, "similarity": 0.92},
            {"summary_text": "Bài tập cho đau vai session Y",
             "session_id": session_y, "created_at": now, "similarity": 0.91},
        ],
    ])

    mock_embed = MagicMock()
    mock_embed.aembed_query = AsyncMock(return_value=[0.1] * 384)

    config = _make_config(user_id=user_id, session_id=str(uuid.uuid4()))

    with patch("langgraph_agents.tools.pgvector_tool.get_pg_client", return_value=mock_pg):
        with patch("langgraph_agents.tools.pgvector_tool.get_embedding_service", return_value=mock_embed):
            result = await memory_search.ainvoke({"query": "đau vai"}, config)

    assert result.get("found") is True
    assert result.get("ambiguous") is True, (
        f"Expected ambiguous=True when top-2 from different sessions with gap 0.01, got: {result}"
    )
    assert len(result.get("candidates", [])) >= 2


@pytest.mark.unit
@pytest.mark.asyncio
async def test_a1_memory_search_no_ambiguous_when_same_session():
    """memory_search: top-2 from same session → no ambiguous (not a cross-session conflict)."""
    from langgraph_agents.tools.pgvector_tool import memory_search
    from datetime import datetime, timezone

    session_x = str(uuid.uuid4())
    user_id = str(uuid.uuid4())
    now = datetime.now(timezone.utc)

    mock_pg = AsyncMock()
    mock_pg.connect = AsyncMock()
    mock_pg.fetch = AsyncMock(side_effect=[
        [{"session_id": session_x}],
        [
            {"summary_text": "Chunk 1 session X", "session_id": session_x,
             "created_at": now, "similarity": 0.93},
            {"summary_text": "Chunk 2 session X", "session_id": session_x,
             "created_at": now, "similarity": 0.92},
        ],
    ])
    mock_embed = MagicMock()
    mock_embed.aembed_query = AsyncMock(return_value=[0.1] * 384)

    config = _make_config(user_id=user_id, session_id=str(uuid.uuid4()))

    with patch("langgraph_agents.tools.pgvector_tool.get_pg_client", return_value=mock_pg):
        with patch("langgraph_agents.tools.pgvector_tool.get_embedding_service", return_value=mock_embed):
            result = await memory_search.ainvoke({"query": "đau vai"}, config)

    assert result.get("found") is True
    assert "ambiguous" not in result, (
        f"Expected no ambiguous when top-2 are from same session, got: {result}"
    )


@pytest.mark.unit
def test_a1_derive_mode_clarify_when_tool_ambiguous():
    """synthesizer._derive_mode → 'clarify' when ToolMessage contains ambiguous: true."""
    from langgraph_agents.nodes.synthesizer import _derive_mode
    from langchain_core.messages import ToolMessage

    ambiguous_result = json.dumps({
        "found": True,
        "ambiguous": True,
        "results": [{"summary_text": "...", "session_id": "s1", "similarity": 0.9}],
        "candidates": [{"summary_text": "...", "session_id": "s1"}],
    })

    state = {
        "needs_clarification": False,
        "required_outputs": [],
        "messages": [
            ToolMessage(content=ambiguous_result, tool_call_id="call-1", name="memory_search"),
        ],
    }

    mode = _derive_mode(state)
    assert mode == "clarify", (
        f"Expected 'clarify' when ToolMessage contains ambiguous=true, got '{mode}'"
    )


@pytest.mark.unit
def test_a1_derive_mode_not_clarify_when_no_ambiguous():
    """synthesizer._derive_mode → not 'clarify' when tool result has no ambiguous flag."""
    from langgraph_agents.nodes.synthesizer import _derive_mode
    from langchain_core.messages import ToolMessage

    normal_result = json.dumps({
        "found": True,
        "results": [{"summary_text": "Bài tập squat", "session_id": "s1", "similarity": 0.9}],
    })

    state = {
        "needs_clarification": False,
        "required_outputs": ["exercise_protocol"],
        "messages": [
            ToolMessage(content=normal_result, tool_call_id="call-1", name="memory_search"),
        ],
    }

    mode = _derive_mode(state)
    assert mode in ("synthesize", "refuse", "chat"), (
        f"Expected synthesize/refuse/chat without ambiguous, got '{mode}'"
    )
    assert mode != "clarify"


# ═══════════════════════════════════════════════════════════════════════════
# A3 — user_memory (integration)
# ═══════════════════════════════════════════════════════════════════════════

@pytest.mark.integration
@pytest.mark.skipif(not HAS_PG, reason="PostgreSQL not available on port 5433")
@pytest.mark.asyncio
async def test_a3_user_memory_fact_injected_into_system_message():
    """POST user_memory fact → memory node injects [USER FACTS] into SystemMessage."""
    from langgraph_agents.shared import get_pg_client
    from langgraph_agents.nodes.memory import _load_user_facts, _assemble_system_message

    pg = get_pg_client()
    await pg.connect()

    user_id = str(uuid.uuid4())
    await pg.execute("INSERT INTO users (id) VALUES ($1) ON CONFLICT DO NOTHING", user_id)

    try:
        # Insert a fact directly (simulating POST /users/{id}/memory)
        await pg.execute(
            "INSERT INTO user_memory (user_id, fact_text) VALUES ($1::uuid, $2)",
            user_id, "Tôi bị đau lưng mãn tính",
        )

        # _load_user_facts must return the fact
        facts = await _load_user_facts(user_id)
        assert len(facts) >= 1
        assert any("đau lưng" in f for f in facts), (
            f"Expected fact about đau lưng in facts, got: {facts}"
        )

        # _assemble_system_message must include [USER FACTS] block
        sys_msg = _assemble_system_message(facts, [])
        assert sys_msg is not None, "SystemMessage must not be None when user_facts exist"
        assert "[USER FACTS]" in sys_msg.content, (
            f"SystemMessage must contain [USER FACTS] block, got: {sys_msg.content[:200]}"
        )
        assert "đau lưng" in sys_msg.content

    finally:
        await pg.execute("DELETE FROM user_memory WHERE user_id = $1::uuid", user_id)
        await pg.execute("DELETE FROM users WHERE id = $1", user_id)


@pytest.mark.integration
@pytest.mark.skipif(not HAS_PG, reason="PostgreSQL not available on port 5433")
@pytest.mark.asyncio
async def test_a3_user_memory_ownership_other_user_gets_404():
    """User B deleting User A's fact returns DELETE 0 (not found / not owned)."""
    from langgraph_agents.shared import get_pg_client

    pg = get_pg_client()
    await pg.connect()

    user_a = str(uuid.uuid4())
    user_b = str(uuid.uuid4())

    await pg.execute("INSERT INTO users (id) VALUES ($1) ON CONFLICT DO NOTHING", user_a)
    await pg.execute("INSERT INTO users (id) VALUES ($1) ON CONFLICT DO NOTHING", user_b)

    try:
        # User A inserts a fact
        fact_id = await pg.fetchval(
            "INSERT INTO user_memory (user_id, fact_text) VALUES ($1::uuid, $2) RETURNING id",
            user_a, "Bí mật của user A",
        )

        # User B tries to delete User A's fact
        result = await pg.execute(
            "DELETE FROM user_memory WHERE id = $1::uuid AND user_id = $2::uuid",
            fact_id, user_b,
        )
        assert result == "DELETE 0", (
            f"User B must not be able to delete User A's fact, got result: {result}"
        )

        # Fact must still exist
        still_there = await pg.fetchval(
            "SELECT COUNT(*) FROM user_memory WHERE id = $1::uuid",
            fact_id,
        )
        assert still_there == 1, "Fact must still exist after cross-user delete attempt"

    finally:
        await pg.execute("DELETE FROM user_memory WHERE user_id = ANY($1)", [user_a, user_b])
        await pg.execute("DELETE FROM users WHERE id = ANY($1)", [user_a, user_b])


@pytest.mark.unit
def test_a3_user_memory_create_validates_empty_fact_text():
    """UserMemoryCreate schema: fact_text='' → Pydantic ValidationError (min_length=1)."""
    from pydantic import ValidationError
    from langgraph_agents.api.schemas import UserMemoryCreate

    with pytest.raises(ValidationError):
        UserMemoryCreate(fact_text="")


@pytest.mark.unit
def test_a3_user_memory_create_validates_too_long_fact_text():
    """UserMemoryCreate schema: fact_text >500 chars → Pydantic ValidationError (max_length=500)."""
    from pydantic import ValidationError
    from langgraph_agents.api.schemas import UserMemoryCreate

    with pytest.raises(ValidationError):
        UserMemoryCreate(fact_text="x" * 501)


@pytest.mark.unit
def test_a3_user_memory_create_accepts_valid_fact_text():
    """UserMemoryCreate schema: valid fact_text → no error."""
    from langgraph_agents.api.schemas import UserMemoryCreate

    item = UserMemoryCreate(fact_text="Tôi bị đau lưng mãn tính", category="health")
    assert item.fact_text == "Tôi bị đau lưng mãn tính"
    assert item.category == "health"
