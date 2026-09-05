"""Regression tests for the 3 "silent memory" bugs found live on 18/06/2026.

These guard the conversational-memory path that 245 existing tests missed:
  1. PostgresClient.executemany missing       → write_session_turn crashed → no persistence
  2. write_session_turn passed an ISO *string* for created_at::timestamptz under
     executemany() binary binding              → type error → no persistence
  3. synthesizer dropped conversation history  → could not answer follow-ups

Tests assert the MECHANISM (rows persisted / history reaches the LLM), NOT the
LLM's wording — so they stay deterministic and don't need a live model.
"""

import uuid
import pytest
from unittest.mock import MagicMock, patch


# ── PG availability guard (mirrors test_pr1_memory_fix.py) ──────────────────
HAS_PG = False
try:
    import asyncio
    import asyncpg

    async def _ping():
        conn = await asyncpg.connect("postgresql://vva:vva_dev@localhost:5433/vva")
        await conn.close()
        return True

    HAS_PG = asyncio.run(_ping())
except Exception:
    pass


# ═══════════════════════════════════════════════════════════════════════════
# Bug #1 + #2 — write_session_turn must persist user + assistant rows
# ═══════════════════════════════════════════════════════════════════════════

@pytest.mark.integration
@pytest.mark.skipif(not HAS_PG, reason="PostgreSQL not available on port 5433")
@pytest.mark.asyncio
async def test_write_session_turn_persists_two_messages():
    """write_session_turn must INSERT one user + one assistant row.

    Regresses: PostgresClient.executemany missing (#1) and the str-timestamp
    type error under executemany (#2) — both made persistence silently fail.
    """
    from langgraph_agents.db.session_store import write_session_turn, _to_uuid
    from langgraph_agents.shared import get_pg_client

    user_id = "regr-" + str(uuid.uuid4())
    session_id = str(uuid.uuid4())
    pg = get_pg_client()
    await pg.connect()

    try:
        await write_session_turn(
            user_id=user_id,
            session_id=session_id,
            user_query="hello",
            assistant_answer="hi there",
            total_tokens=5,
        )

        rows = await pg.fetch(
            "SELECT role, content, token_count FROM messages "
            "WHERE session_id = $1::uuid ORDER BY seq_id",
            session_id,
        )
        assert len(rows) == 2, f"expected 2 rows, got {len(rows)}"
        assert rows[0]["role"] == "user" and rows[0]["content"] == "hello"
        assert rows[1]["role"] == "assistant" and rows[1]["content"] == "hi there"
        assert rows[1]["token_count"] == 5
    finally:
        await pg.execute("DELETE FROM conversations WHERE session_id = $1::uuid", session_id)
        await pg.execute("DELETE FROM users WHERE id = $1::uuid", _to_uuid(user_id))


# ═══════════════════════════════════════════════════════════════════════════
# Bug #3 — synthesizer must include prior conversation in its LLM call
# ═══════════════════════════════════════════════════════════════════════════

@pytest.mark.unit
@pytest.mark.asyncio
async def test_synthesizer_includes_conversation_history():
    """The synthesizer must pass prior user/assistant turns to the LLM.

    Regresses: synthesizer built msgs = [System, Human(query)] only, ignoring the
    history memory_node loaded into state['messages'] → no follow-up recall.
    Also asserts tool noise (ToolMessage / tool-call AIMessage) is NOT forwarded.
    """
    from langchain_core.messages import (
        SystemMessage, HumanMessage, AIMessage, ToolMessage,
    )
    from langgraph_agents.nodes import synthesizer as syn

    captured = {}

    class FakeAI:
        content = "ok"
        usage_metadata = None

    async def fake_ainvoke(msgs):
        captured["msgs"] = msgs
        return FakeAI()

    mock_llm = MagicMock()
    mock_llm.ainvoke = fake_ainvoke

    tool_call_ai = AIMessage(content="", tool_calls=[
        {"name": "kb_search", "args": {}, "id": "t1"}
    ])

    state = {
        "messages": [
            SystemMessage(content="[USER FACTS]\n- none"),   # memory system → excluded
            HumanMessage(content="My name is Nguyen"),        # prior turn → included
            AIMessage(content="Nice to meet you, Nguyen"),    # prior turn → included
            tool_call_ai,                                     # tool-call AI → excluded
            ToolMessage(content="[]", name="kb_search", tool_call_id="t1"),  # tool result → excluded
        ],
        "resolved_query": "what is my name",
        "required_outputs": [],
        "needs_clarification": False,
        "total_tokens": 0,
    }
    config = {"configurable": {
        "request_id": "r", "persona_id": "anne", "query": "what is my name",
    }}

    # No stream writer outside a running graph → synthesizer uses ainvoke(msgs).
    with patch.object(syn, "get_chat_model", return_value=mock_llm):
        await syn.synthesizer_node(state, config)

    msgs = captured.get("msgs")
    assert msgs is not None, "synthesizer never called the LLM"
    contents = [getattr(m, "content", "") for m in msgs]

    # Prior conversation must reach the model
    assert "My name is Nguyen" in contents, "prior USER turn must be in the LLM messages"
    assert "Nice to meet you, Nguyen" in contents, "prior ASSISTANT turn must be in the LLM messages"
    # The current query must be the last thing the USER says. It is no longer
    # the last message overall: a short voice card is appended after it, so the
    # closest text to the generation point describes the character instead of a
    # tag contract. See _persona_loader.build_voice_card.
    humans = [m for m in msgs if isinstance(m, HumanMessage)]
    assert humans[-1].content == "what is my name"
    assert isinstance(msgs[-1], SystemMessage)
    assert "Who is speaking" in msgs[-1].content
    # Tool noise must NOT be forwarded as conversation
    assert "[]" not in contents, "ToolMessage content must not be forwarded as history"
