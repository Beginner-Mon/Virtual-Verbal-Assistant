"""Memory unit tests — M.5 context assembly, token budget, mock DB/Redis.
"""

import json
import pytest
from unittest.mock import AsyncMock, MagicMock, patch

from langgraph_agents.state import AgentState


@pytest.mark.unit
class TestContextAssembly:
    """Verify _assemble_system_message and _recent_raw_to_messages."""

    def test_assemble_with_user_facts_and_summaries(self):
        from langgraph_agents.nodes.memory import _assemble_system_message
        msg = _assemble_system_message(
            user_facts=["65 tuoi", "dau man tinh"],
            summary_chunks=["Session 1: da hoi ve squat", "Session 2: da hoi ve L4-L5"],
        )
        assert msg is not None
        content = msg.content
        assert "[USER FACTS]" in content
        assert "- 65 tuoi" in content
        assert "- dau man tinh" in content
        assert "[SESSION HISTORY]" in content
        assert "squat" in content

    def test_assemble_facts_only(self):
        from langgraph_agents.nodes.memory import _assemble_system_message
        msg = _assemble_system_message(
            user_facts=["da tap yoga 2 nam"],
            summary_chunks=[],
        )
        assert msg is not None
        assert "[USER FACTS]" in msg.content
        assert "[SESSION HISTORY]" not in msg.content

    def test_assemble_summaries_only(self):
        from langgraph_agents.nodes.memory import _assemble_system_message
        msg = _assemble_system_message(
            user_facts=[],
            summary_chunks=["Session 1: test"],
        )
        assert msg is not None
        assert "[SESSION HISTORY]" in msg.content
        assert "[USER FACTS]" not in msg.content

    def test_assemble_empty_returns_none(self):
        from langgraph_agents.nodes.memory import _assemble_system_message
        msg = _assemble_system_message([], [])
        assert msg is None

    def test_recent_raw_to_messages(self):
        from langgraph_agents.nodes.memory import _recent_raw_to_messages
        from langchain_core.messages import HumanMessage, AIMessage
        raw = [
            {"role": "user", "content": "xin chao"},
            {"role": "assistant", "content": "Chao ban!"},
        ]
        msgs = _recent_raw_to_messages(raw)
        assert len(msgs) == 2
        assert isinstance(msgs[0], HumanMessage)
        assert isinstance(msgs[1], AIMessage)
        assert msgs[0].content == "xin chao"

    def test_recent_raw_skips_empty_content(self):
        from langgraph_agents.nodes.memory import _recent_raw_to_messages
        raw = [{"role": "user", "content": ""}]
        msgs = _recent_raw_to_messages(raw)
        assert msgs[0].content == ""

    def test_recent_raw_unknown_role_skipped(self):
        from langgraph_agents.nodes.memory import _recent_raw_to_messages
        raw = [{"role": "system", "content": "beep"}]
        msgs = _recent_raw_to_messages(raw)
        # system role is not user or assistant → not converted
        assert len(msgs) == 0


@pytest.mark.unit
class TestTokenBudget:
    """Verify _token_estimate and _select_recent_raw."""

    def test_token_estimate_empty(self):
        from langgraph_agents.nodes.memory import _token_estimate
        assert _token_estimate("") == 0

    def test_token_estimate_vietnamese(self):
        from langgraph_agents.nodes.memory import _token_estimate
        # ~40 chars → ~10 tokens
        assert _token_estimate("Xin chao ban, toi can bai tap cho dau lung") == 10

    def test_select_recent_raw_within_budget(self):
        from langgraph_agents.nodes.memory import _select_recent_raw
        raw = [{"content": "short"}, {"content": "also short"}]
        selected = _select_recent_raw(raw, budget=100)
        assert len(selected) == 2

    def test_select_recent_raw_exceeds_budget(self):
        from langgraph_agents.nodes.memory import _select_recent_raw
        raw = [{"content": "x" * 1000, "role": "user"}, {"content": "hi", "role": "assistant"}]
        selected = _select_recent_raw(raw, budget=100)  # "hi" = 2/4=0 tokens, 1000 chars=250 tokens
        # Only "hi" fits (0 tokens) and then 1000x exceeds 100 budget
        assert len(selected) == 1
        assert "hi" == selected[0]["content"]


@pytest.mark.unit
class TestMemoryLoaderMocks:
    """Test async loader functions with mocked PostgreSQL."""

    @pytest.mark.asyncio
    async def test_load_user_facts_returns_facts(self):
        from langgraph_agents.nodes.memory import _load_user_facts
        mock_pg = AsyncMock()
        mock_pg.fetch.return_value = [
            {"fact_text": "65 tuổi"},
            {"fact_text": "đau lưng mãn tính"},
        ]

        with patch("langgraph_agents.nodes.memory.get_pg_client", return_value=mock_pg):
            facts = await _load_user_facts("test-user-id")
            assert len(facts) == 2
            assert "65 tuổi" in facts

    @pytest.mark.asyncio
    async def test_load_user_facts_handles_db_error(self):
        from langgraph_agents.nodes.memory import _load_user_facts
        mock_pg = AsyncMock()
        mock_pg.fetch.side_effect = RuntimeError("DB connection failed")

        with patch("langgraph_agents.nodes.memory.get_pg_client", return_value=mock_pg):
            facts = await _load_user_facts("test-user-id")
            assert facts == []  # graceful degradation

    @pytest.mark.asyncio
    async def test_load_summary_chunks_returns_active(self):
        from langgraph_agents.nodes.memory import _load_summary_chunks
        mock_pg = AsyncMock()
        mock_pg.fetch.return_value = [
            {"summary_text": "User asked about squat exercises"},
            {"summary_text": "User asked about L4-L5 pain"},
        ]

        with patch("langgraph_agents.nodes.memory.get_pg_client", return_value=mock_pg):
            chunks = await _load_summary_chunks("test-session")
            assert len(chunks) == 2
            assert "squat" in chunks[0]

    @pytest.mark.asyncio
    async def test_load_summary_chunks_handles_db_error(self):
        from langgraph_agents.nodes.memory import _load_summary_chunks
        mock_pg = AsyncMock()
        mock_pg.fetch.side_effect = RuntimeError("DB down")

        with patch("langgraph_agents.nodes.memory.get_pg_client", return_value=mock_pg):
            chunks = await _load_summary_chunks("test-session")
            assert chunks == []

    @pytest.mark.asyncio
    async def test_load_recent_raw_redis_hit(self):
        from langgraph_agents.nodes.memory import _load_recent_raw_redis
        mock_redis = AsyncMock()
        mock_redis.get.return_value = json.dumps([
            {"role": "user", "content": "xin chào"},
            {"role": "assistant", "content": "Chào bạn!"},
        ])

        with patch("redis.asyncio.from_url", return_value=mock_redis):
            result = await _load_recent_raw_redis("test-session")
            assert result is not None
            assert len(result) == 2
            assert result[0]["role"] == "user"

    @pytest.mark.asyncio
    async def test_load_recent_raw_redis_miss(self):
        from langgraph_agents.nodes.memory import _load_recent_raw_redis
        mock_redis = AsyncMock()
        mock_redis.get.return_value = None

        with patch("redis.asyncio.from_url", return_value=mock_redis):
            result = await _load_recent_raw_redis("test-session")
            assert result is None

    @pytest.mark.asyncio
    async def test_load_recent_raw_redis_connection_error(self):
        from langgraph_agents.nodes.memory import _load_recent_raw_redis
        with patch("redis.asyncio.from_url", side_effect=ConnectionError("Redis down")):
            result = await _load_recent_raw_redis("test-session")
            assert result is None

    @pytest.mark.asyncio
    async def test_load_recent_raw_db_fallback(self):
        from langgraph_agents.nodes.memory import _load_recent_raw_db
        mock_pg = AsyncMock()
        mock_pg.fetch.return_value = [
            {"role": "assistant", "content": "Chào bạn!"},
            {"role": "user", "content": "xin chào"},
        ]

        with patch("langgraph_agents.nodes.memory.get_pg_client", return_value=mock_pg):
            result = await _load_recent_raw_db("test-session")
            # reversed: chronological order
            assert len(result) == 2
            assert result[0]["role"] == "user"
            assert result[0]["content"] == "xin chào"

    @pytest.mark.asyncio
    async def test_load_recent_raw_db_error_returns_empty(self):
        from langgraph_agents.nodes.memory import _load_recent_raw_db
        mock_pg = AsyncMock()
        mock_pg.fetch.side_effect = RuntimeError("DB down")

        with patch("langgraph_agents.nodes.memory.get_pg_client", return_value=mock_pg):
            result = await _load_recent_raw_db("test-session")
            assert result == []

    @pytest.mark.asyncio
    async def test_load_recent_raw_redis_first_then_fallback(self):
        from langgraph_agents.nodes.memory import _load_recent_raw
        with patch("langgraph_agents.nodes.memory._load_recent_raw_redis") as mock_redis:
            mock_redis.return_value = [{"role": "user", "content": "from redis"}]
            result = await _load_recent_raw("test-session")
            assert len(result) == 1
            assert result[0]["content"] == "from redis"


@pytest.mark.unit
class TestMemoryNode:
    """Test memory_node orchestration (mocked)."""

    @pytest.mark.asyncio
    async def test_memory_node_assembles_context(self):
        from langgraph_agents.nodes.memory import memory_node
        from langchain_core.runnables import RunnableConfig

        config = RunnableConfig(configurable={
            "user_id": "test-user", "session_id": "test-sess",
            "query": "xin chao", "persona_id": "eca_default",
            "request_id": "req-001",
        })
        state: AgentState = {"messages": [], "errors": [], "total_tokens": 0,
                             "retry_count": 0}

        with patch("langgraph_agents.nodes.memory._load_user_facts") as mock_facts, \
             patch("langgraph_agents.nodes.memory._load_summary_chunks") as mock_summaries, \
             patch("langgraph_agents.nodes.memory._load_recent_raw") as mock_recent:
            mock_facts.return_value = ["65 tuoi"]
            mock_summaries.return_value = ["Session: squat"]
            mock_recent.return_value = []

            result = await memory_node(state, config)
            assert "messages" in result
            assert len(result["messages"]) >= 1  # at least SystemMessage

    @pytest.mark.asyncio
    async def test_memory_node_empty_context_returns_empty(self):
        from langgraph_agents.nodes.memory import memory_node
        from langchain_core.runnables import RunnableConfig

        config = RunnableConfig(configurable={
            "user_id": "test-user", "session_id": "test-sess",
            "query": "xin chao", "persona_id": "eca_default",
            "request_id": "req-001",
        })
        state: AgentState = {"messages": [], "errors": [], "total_tokens": 0,
                             "retry_count": 0}

        with patch("langgraph_agents.nodes.memory._load_user_facts") as mock_facts, \
             patch("langgraph_agents.nodes.memory._load_summary_chunks") as mock_summaries, \
             patch("langgraph_agents.nodes.memory._load_recent_raw") as mock_recent:
            mock_facts.return_value = []
            mock_summaries.return_value = []
            mock_recent.return_value = []

            result = await memory_node(state, config)
            assert result == {} or result.get("messages", []) == []
