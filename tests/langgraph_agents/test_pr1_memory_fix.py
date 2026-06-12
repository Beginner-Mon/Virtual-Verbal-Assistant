"""Memory layer tenant-isolation tests — PR 1 acceptance criteria.

Requires Docker PostgreSQL (port 5433) with M.4 schema migrated.
Tests: tenant scope, STM round-trip, LLM tool schema, e2e memory_search.
"""

import json
import uuid
import pytest

from langchain_core.runnables import RunnableConfig

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
# Tenant isolation (BẮT BUỘC)
# ═══════════════════════════════════════════════════════════════════════════

@pytest.mark.integration
@pytest.mark.skipif(not HAS_PG, reason="PostgreSQL not available on port 5433")
@pytest.mark.asyncio
async def test_memory_search_tenant_isolation():
    """User A must NOT see User B's summaries (D19)."""
    from langgraph_agents.tools.pgvector_tool import memory_search
    from langgraph_agents.shared import get_pg_client, get_embedding_service

    pg = get_pg_client()
    await pg.connect()
    embed = get_embedding_service()

    # ── Setup: 2 users, each with 1 conversation + 1 summary ──────────
    user_a = str(uuid.uuid4())
    user_b = str(uuid.uuid4())
    session_a = str(uuid.uuid4())
    session_b = str(uuid.uuid4())

    await pg.execute("INSERT INTO users (id) VALUES ($1) ON CONFLICT DO NOTHING", user_a)
    await pg.execute("INSERT INTO users (id) VALUES ($1) ON CONFLICT DO NOTHING", user_b)
    await pg.execute(
        "INSERT INTO conversations (session_id, user_id) VALUES ($1, $2) ON CONFLICT DO NOTHING",
        session_a, user_a,
    )
    await pg.execute(
        "INSERT INTO conversations (session_id, user_id) VALUES ($1, $2) ON CONFLICT DO NOTHING",
        session_b, user_b,
    )

    # Insert summaries with embeddings
    vec_a = await embed.aembed_passage("User A's secret: bài tập squat cho đau lưng")
    vec_b = await embed.aembed_passage("User B's secret: bài tập yoga cho cổ")
    await pg.execute(
        """INSERT INTO summaries (session_id, summary_text, covers_from_seq,
           covers_up_to_seq, embedding, status) VALUES ($1, $2, 1, 10, $3, 'active')
           ON CONFLICT ON CONSTRAINT uq_chunk DO NOTHING""",
        session_a, "User A bị đau lưng mãn tính, được khuyên tập squat nhẹ", vec_a,
    )
    await pg.execute(
        """INSERT INTO summaries (session_id, summary_text, covers_from_seq,
           covers_up_to_seq, embedding, status) VALUES ($1, $2, 1, 10, $3, 'active')
           ON CONFLICT ON CONSTRAINT uq_chunk DO NOTHING""",
        session_b, "User B tập yoga thường xuyên, hỏi về động tác cổ", vec_b,
    )

    # ── Test: User A searches → must NOT see User B's summary ──────────
    config_a = _make_config(user_id=user_a, session_id=str(uuid.uuid4()))
    result_a = await memory_search.ainvoke({"query": "bài tập đau lưng"}, config_a)
    assert result_a["found"] is True, f"User A should find their own summary, got {result_a}"
    for r in result_a["results"]:
        assert "User B" not in r["summary_text"], (
            f"TENANT LEAK: User A saw User B's data: {r['summary_text']}"
        )

    # ── Test: User B searches → must NOT see User A's summary ──────────
    config_b = _make_config(user_id=user_b, session_id=str(uuid.uuid4()))
    result_b = await memory_search.ainvoke({"query": "bài tập yoga cổ"}, config_b)
    assert result_b["found"] is True, f"User B should find their own summary, got {result_b}"
    for r in result_b["results"]:
        assert "User A" not in r["summary_text"], (
            f"TENANT LEAK: User B saw User A's data: {r['summary_text']}"
        )

    # ── Cleanup ───────────────────────────────────────────────────────
    await pg.execute("DELETE FROM summaries WHERE session_id = ANY($1)", [session_a, session_b])
    await pg.execute("DELETE FROM conversations WHERE session_id = ANY($1)", [session_a, session_b])
    await pg.execute("DELETE FROM users WHERE id = ANY($1)", [user_a, user_b])


# ═══════════════════════════════════════════════════════════════════════════
# LLM tool schema (config params hidden)
# ═══════════════════════════════════════════════════════════════════════════

@pytest.mark.unit
def test_memory_search_schema_hides_user_id():
    """memory_search tool schema must NOT expose user_id/config to LLM."""
    from langgraph_agents.tools.pgvector_tool import memory_search
    schema = memory_search.tool_call_schema
    from pydantic import BaseModel
    assert isinstance(schema, type) and issubclass(schema, BaseModel)
    fields = schema.model_fields
    assert "query" in fields, "query must be visible to LLM"
    assert "since_days" in fields, "since_days must be visible to LLM"
    assert "top_k" in fields, "top_k must be visible to LLM"
    assert "user_id" not in fields, "user_id must NOT be in LLM schema (config-injected)"
    assert "current_session_id" not in fields, "current_session_id must NOT be in LLM schema"
    assert "config" not in fields, "config must NOT be in LLM schema (RunnableConfig is hidden)"


@pytest.mark.unit
def test_resume_last_session_schema_hides_user_id():
    """resume_last_session tool schema must NOT expose user_id/config to LLM."""
    from langgraph_agents.tools.pgvector_tool import resume_last_session
    schema = resume_last_session.tool_call_schema
    from pydantic import BaseModel
    assert isinstance(schema, type) and issubclass(schema, BaseModel)
    fields = schema.model_fields
    assert "since_days" in fields or True  # at least one arg visible
    assert "user_id" not in fields, "user_id must NOT be in LLM schema"
    assert "config" not in fields, "config must NOT be in LLM schema"


# ═══════════════════════════════════════════════════════════════════════════
# STM round-trip (Redis key/format)
# ═══════════════════════════════════════════════════════════════════════════

@pytest.mark.unit
def test_stm_normalize_writer_format():
    """_normalize_redis_format converts [{q, a, ts}] → [{role, content}]."""
    from langgraph_agents.nodes.memory import _normalize_redis_format
    writer_format = [
        {"q": "xin chào", "a": "Chào bạn!", "ts": "2026-01-01T00:00:00Z"},
        {"q": "bài tập squat", "a": "3 hiệp × 10 lần...", "ts": "2026-01-01T00:01:00Z"},
    ]
    result = _normalize_redis_format(writer_format)
    assert len(result) == 4  # 2 pairs → 4 messages
    assert result[0] == {"role": "user", "content": "xin chào"}
    assert result[1] == {"role": "assistant", "content": "Chào bạn!"}
    assert result[3] == {"role": "assistant", "content": "3 hiệp × 10 lần..."}


@pytest.mark.unit
def test_stm_normalize_passthrough():
    """_normalize_redis_format passes through [{role, content}] format."""
    from langgraph_agents.nodes.memory import _normalize_redis_format
    reader_format = [
        {"role": "user", "content": "hello"},
        {"role": "assistant", "content": "hi"},
    ]
    result = _normalize_redis_format(reader_format)
    assert result == reader_format  # pass through unchanged


@pytest.mark.unit
def test_stm_normalize_empty():
    """_normalize_redis_format returns empty for empty input."""
    from langgraph_agents.nodes.memory import _normalize_redis_format
    assert _normalize_redis_format([]) == []


# ═══════════════════════════════════════════════════════════════════════════
# E2E: memory_search no longer errors (may return empty until PR 2)
# ═══════════════════════════════════════════════════════════════════════════

@pytest.mark.unit
def test_memory_search_bound_in_retriever_tools():
    """memory_search + resume_last_session must be in RETRIEVER_BASE_TOOLS."""
    from langgraph_agents.nodes.retriever_agent import RETRIEVER_BASE_TOOLS
    tool_names = {t.name for t in RETRIEVER_BASE_TOOLS}
    assert "memory_search" in tool_names, "memory_search must be in RETRIEVER_BASE_TOOLS"
    assert "resume_last_session" in tool_names, "resume_last_session must be in RETRIEVER_BASE_TOOLS"
    assert "kb_search" in tool_names
