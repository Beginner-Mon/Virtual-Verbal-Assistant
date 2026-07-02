"""Unit tests for P1/P2/P3 fixes — retrieval perf & web_search toggle.

P1: embedding offline load (local_files_only)
P2: hard cap retriever loop at MAX_RETRIEVER_ROUNDS=2
P3: conditional prompt omits search_medical when off; execution guard blocks it
"""

import pytest
from unittest.mock import AsyncMock, MagicMock, patch


# ─────────────────────────────────────────────────────────────────────────────
# P1 — Embedding loads offline (local_files_only)
# ─────────────────────────────────────────────────────────────────────────────

@pytest.mark.unit
class TestP1EmbeddingOffline:
    """Verify SentenceTransformer is loaded with local_files_only by default."""

    def test_embedding_service_loads_offline_by_default(self, monkeypatch):
        """When EMBEDDING_ALLOW_DOWNLOAD is unset, local_files_only=True is passed."""
        monkeypatch.delenv("EMBEDDING_ALLOW_DOWNLOAD", raising=False)

        captured = {}

        class FakeST:
            def __init__(self, model_name, local_files_only=False, **kwargs):
                captured["local_files_only"] = local_files_only
                captured["model_name"] = model_name

        import langgraph_agents.shared.embedding as emb_module
        # Reset any cached model on the singleton
        if emb_module._embedding_instance is not None:
            emb_module._embedding_instance._model = None

        with patch.dict("sys.modules", {"sentence_transformers": MagicMock(SentenceTransformer=FakeST)}):
            from langgraph_agents.shared.embedding import E5EmbeddingService
            svc = E5EmbeddingService()
            _ = svc.model  # triggers lazy load

        assert captured["local_files_only"] is True, (
            "Default (no EMBEDDING_ALLOW_DOWNLOAD) must load local cache only"
        )

    def test_embedding_service_allows_download_when_env_set(self, monkeypatch):
        """When EMBEDDING_ALLOW_DOWNLOAD=1, local_files_only=False is passed."""
        monkeypatch.setenv("EMBEDDING_ALLOW_DOWNLOAD", "1")

        captured = {}

        class FakeST:
            def __init__(self, model_name, local_files_only=False, **kwargs):
                captured["local_files_only"] = local_files_only

        with patch.dict("sys.modules", {"sentence_transformers": MagicMock(SentenceTransformer=FakeST)}):
            from langgraph_agents.shared.embedding import E5EmbeddingService
            svc = E5EmbeddingService()
            _ = svc.model

        assert captured["local_files_only"] is False, (
            "EMBEDDING_ALLOW_DOWNLOAD=1 must allow download (local_files_only=False)"
        )

    def test_resource_guard_loads_offline_by_default(self, monkeypatch):
        """shared_embedder() passes local_files_only=True when env not set."""
        monkeypatch.delenv("EMBEDDING_ALLOW_DOWNLOAD", raising=False)

        captured = {}

        class FakeST:
            def __init__(self, model_name, local_files_only=False, **kwargs):
                captured["local_files_only"] = local_files_only

        import langgraph_agents.core.resource_guard as rg_module
        # Clear cache so load runs again
        rg_module.clear_embedder_cache()

        with patch.dict("sys.modules", {"sentence_transformers": MagicMock(SentenceTransformer=FakeST)}):
            rg_module.shared_embedder("test-model")

        assert captured["local_files_only"] is True

    def test_resource_guard_allows_download_when_env_set(self, monkeypatch):
        """shared_embedder() passes local_files_only=False when EMBEDDING_ALLOW_DOWNLOAD=1."""
        monkeypatch.setenv("EMBEDDING_ALLOW_DOWNLOAD", "1")

        captured = {}

        class FakeST:
            def __init__(self, model_name, local_files_only=False, **kwargs):
                captured["local_files_only"] = local_files_only

        import langgraph_agents.core.resource_guard as rg_module
        rg_module.clear_embedder_cache()

        with patch.dict("sys.modules", {"sentence_transformers": MagicMock(SentenceTransformer=FakeST)}):
            rg_module.shared_embedder("test-model-dl")

        assert captured["local_files_only"] is False


# ─────────────────────────────────────────────────────────────────────────────
# P2 — Hard cap retriever loop at 2 rounds
# ─────────────────────────────────────────────────────────────────────────────

@pytest.mark.unit
class TestP2RetrieverRoundCap:
    """Verify route_after_retriever enforces MAX_RETRIEVER_ROUNDS=2."""

    def _make_state(self, retriever_rounds: int, has_tool_calls: bool = True, needs_motion: bool = False):
        from langchain_core.messages import AIMessage
        from langgraph_agents.state import AgentState

        tool_calls = [{"name": "kb_search", "args": {}, "id": "1"}] if has_tool_calls else []
        msg = AIMessage(content="", tool_calls=tool_calls)
        state: AgentState = {
            "messages": [msg],
            "errors": [],
            "retry_count": 0,
            "total_tokens": 0,
            "retriever_rounds": retriever_rounds,
            "needs_motion": needs_motion,
        }
        return state

    def test_round_1_with_tool_calls_routes_to_tools(self):
        """Round 1, tool calls present → tools (under cap)."""
        from langgraph_agents.routing import route_after_retriever
        state = self._make_state(retriever_rounds=1, has_tool_calls=True)
        assert route_after_retriever(state) == "tools"

    def test_round_2_with_tool_calls_forces_synthesizer(self):
        """Round 2 (== MAX), tool calls present → synthesizer (cap enforced)."""
        from langgraph_agents.routing import route_after_retriever
        state = self._make_state(retriever_rounds=2, has_tool_calls=True)
        assert route_after_retriever(state) == "synthesizer"

    def test_round_3_with_tool_calls_forces_synthesizer(self):
        """Round 3 (> MAX), tool calls present → synthesizer (cap enforced)."""
        from langgraph_agents.routing import route_after_retriever
        state = self._make_state(retriever_rounds=3, has_tool_calls=True)
        assert route_after_retriever(state) == "synthesizer"

    def test_round_2_with_motion_forces_kimodo(self):
        """Round 2 (at cap), needs_motion=True → kimodo (not synthesizer)."""
        from langgraph_agents.routing import route_after_retriever
        state = self._make_state(retriever_rounds=2, has_tool_calls=True, needs_motion=True)
        assert route_after_retriever(state) == "kimodo"

    def test_round_0_no_tool_calls_to_synthesizer(self):
        """Round 0, no tool calls → synthesizer (normal completion)."""
        from langgraph_agents.routing import route_after_retriever
        state = self._make_state(retriever_rounds=0, has_tool_calls=False)
        assert route_after_retriever(state) == "synthesizer"

    def test_retriever_node_increments_round_counter(self):
        """retriever_agent_node returns retriever_rounds = previous + 1."""
        import asyncio
        from langchain_core.messages import AIMessage
        from langchain_core.runnables import RunnableConfig
        from langgraph_agents.nodes.retriever_agent import retriever_agent_node
        from langgraph_agents.state import AgentState

        config = RunnableConfig(configurable={
            "user_id": "u1", "session_id": "s1",
            "query": "test", "persona_id": "eca_default",
            "request_id": "r1", "web_search": False,
        })
        state: AgentState = {
            "messages": [], "errors": [], "retry_count": 0,
            "total_tokens": 0, "required_outputs": [],
            "resolved_query": "test", "retriever_rounds": 1,
        }

        fake_ai = AIMessage(content="done", tool_calls=[])

        async def run():
            with patch("langgraph_agents.nodes.retriever_agent.get_chat_model") as mock_llm:
                mock_llm.return_value.bind_tools.return_value.ainvoke = AsyncMock(return_value=fake_ai)
                result = await retriever_agent_node(state, config)
            return result

        result = asyncio.run(run())
        assert result["retriever_rounds"] == 2, "Should increment from 1 → 2"

    def test_missing_retriever_rounds_defaults_to_zero(self):
        """State without retriever_rounds (backward compat): treats as 0, allows tools."""
        from langgraph_agents.routing import route_after_retriever
        from langchain_core.messages import AIMessage
        from langgraph_agents.state import AgentState

        msg = AIMessage(content="", tool_calls=[{"name": "kb_search", "args": {}, "id": "1"}])
        state: AgentState = {
            "messages": [msg], "errors": [], "retry_count": 0, "total_tokens": 0
            # retriever_rounds intentionally absent
        }
        # 0 < MAX_RETRIEVER_ROUNDS(2) → tools
        assert route_after_retriever(state) == "tools"


# ─────────────────────────────────────────────────────────────────────────────
# P3 — Enforce web_search toggle at all layers
# ─────────────────────────────────────────────────────────────────────────────

@pytest.mark.unit
class TestP3WebSearchPrompt:
    """Verify _build_retriever_system_prompt omits/includes search_medical block."""

    def test_prompt_omits_search_medical_when_off(self):
        """With web_search_enabled=False, 'search_medical' must not appear in prompt."""
        from langgraph_agents.nodes.retriever_agent import _build_retriever_system_prompt
        prompt = _build_retriever_system_prompt(
            web_search_enabled=False,
            retry_note="",
            required_outputs="test",
            resolved_query="thời tiết hôm nay",
        )
        assert "search_medical" not in prompt, (
            "Prompt must NOT mention search_medical when web_search is off"
        )

    def test_prompt_includes_search_medical_when_on(self):
        """With web_search_enabled=True, 'search_medical' must appear in prompt."""
        from langgraph_agents.nodes.retriever_agent import _build_retriever_system_prompt
        prompt = _build_retriever_system_prompt(
            web_search_enabled=True,
            retry_note="",
            required_outputs="test",
            resolved_query="thời tiết hôm nay",
        )
        assert "search_medical" in prompt, (
            "Prompt MUST mention search_medical when web_search is on"
        )

    def test_prompt_always_includes_kb_search(self):
        """kb_search must be in prompt regardless of web_search flag."""
        from langgraph_agents.nodes.retriever_agent import _build_retriever_system_prompt
        for flag in (True, False):
            prompt = _build_retriever_system_prompt(
                web_search_enabled=flag,
                retry_note="",
                required_outputs="ex",
                resolved_query="bài tập squat",
            )
            assert "kb_search" in prompt, f"kb_search missing when web_search_enabled={flag}"

    def test_prompt_omits_real_time_decision_rule_when_off(self):
        """The 'Real-time/external' decision rule line must be absent when web off."""
        from langgraph_agents.nodes.retriever_agent import _build_retriever_system_prompt
        prompt = _build_retriever_system_prompt(
            web_search_enabled=False,
            retry_note="",
            required_outputs="",
            resolved_query="weather",
        )
        assert "Real-time/external" not in prompt


@pytest.mark.unit
class TestP3WebSearchGuard:
    """Verify the execution guard blocks search_medical when web_search=False."""

    def _make_guarded_node(self):
        """Build a guarded tools node with a mock ToolNode."""
        from langgraph_agents.graph import _make_guarded_tools_node
        mock_tool_node = MagicMock()
        mock_tool_node.ainvoke = AsyncMock(return_value={"messages": []})
        return _make_guarded_tools_node(mock_tool_node), mock_tool_node

    def _make_state_with_tool_call(self, tool_name: str):
        from langchain_core.messages import AIMessage
        tc = {"name": tool_name, "args": {"query": "test"}, "id": "tc1"}
        msg = AIMessage(content="", tool_calls=[tc])
        return {"messages": [msg], "errors": [], "retry_count": 0, "total_tokens": 0}

    @pytest.mark.asyncio
    async def test_guard_blocks_search_medical_when_web_off(self):
        """search_medical tool call is short-circuited when web_search=False."""
        from langchain_core.messages import ToolMessage
        guarded, mock_tn = self._make_guarded_node()
        state = self._make_state_with_tool_call("search_medical")
        config = {"configurable": {"web_search": False, "request_id": "t1"}}

        result = await guarded(state, config)

        # ToolNode should NOT have been called
        mock_tn.ainvoke.assert_not_called()
        # Synthetic blocked message must be returned
        msgs = result.get("messages", [])
        assert len(msgs) == 1
        assert isinstance(msgs[0], ToolMessage)
        assert "web_search_disabled" in msgs[0].content
        assert msgs[0].tool_call_id == "tc1"

    @pytest.mark.asyncio
    async def test_guard_passes_kb_search_through(self):
        """kb_search is NOT blocked even when web_search=False."""
        guarded, mock_tn = self._make_guarded_node()
        state = self._make_state_with_tool_call("kb_search")
        config = {"configurable": {"web_search": False, "request_id": "t2"}}

        await guarded(state, config)

        # ToolNode should have been called normally
        mock_tn.ainvoke.assert_called_once()

    @pytest.mark.asyncio
    async def test_guard_passes_search_medical_when_web_on(self):
        """search_medical is NOT blocked when web_search=True."""
        guarded, mock_tn = self._make_guarded_node()
        state = self._make_state_with_tool_call("search_medical")
        config = {"configurable": {"web_search": True, "request_id": "t3"}}

        await guarded(state, config)

        # ToolNode should have been called normally
        mock_tn.ainvoke.assert_called_once()

    @pytest.mark.asyncio
    async def test_guard_default_web_off(self):
        """When web_search key is absent from config, default is off → blocks search_medical."""
        from langchain_core.messages import ToolMessage
        guarded, mock_tn = self._make_guarded_node()
        state = self._make_state_with_tool_call("search_medical")
        # No "web_search" key in configurable
        config = {"configurable": {"request_id": "t4"}}

        result = await guarded(state, config)

        mock_tn.ainvoke.assert_not_called()
        msgs = result.get("messages", [])
        assert len(msgs) == 1 and isinstance(msgs[0], ToolMessage)

    @pytest.mark.asyncio
    async def test_guard_mixed_calls_blocks_only_search_medical(self):
        """Mixed tool_calls: search_medical blocked, kb_search passed through."""
        from langchain_core.messages import AIMessage, ToolMessage
        guarded, mock_tn = self._make_guarded_node()
        mock_tn.ainvoke = AsyncMock(return_value={"messages": [
            ToolMessage(content="kb result", tool_call_id="tc2", name="kb_search")
        ]})

        tc_web = {"name": "search_medical", "args": {}, "id": "tc1"}
        tc_kb = {"name": "kb_search", "args": {"query": "squat"}, "id": "tc2"}
        msg = AIMessage(content="", tool_calls=[tc_web, tc_kb])
        state = {"messages": [msg], "errors": [], "retry_count": 0, "total_tokens": 0}
        config = {"configurable": {"web_search": False, "request_id": "t5"}}

        result = await guarded(state, config)

        # ToolNode invoked once (for kb_search)
        mock_tn.ainvoke.assert_called_once()
        msgs = result.get("messages", [])
        # Should have synthetic (web blocked) + kb_search result
        assert len(msgs) == 2
        blocked_msgs = [m for m in msgs if isinstance(m, ToolMessage) and "web_search_disabled" in m.content]
        kb_msgs = [m for m in msgs if isinstance(m, ToolMessage) and m.tool_call_id == "tc2"]
        assert len(blocked_msgs) == 1
        assert len(kb_msgs) == 1

    @pytest.mark.asyncio
    async def test_retriever_node_default_web_off_no_search_medical_in_tools(self):
        """retriever_agent_node with web_search absent from config defaults to False.
        search_medical must not be in bound tools."""
        import asyncio
        from langchain_core.messages import AIMessage
        from langchain_core.runnables import RunnableConfig
        from langgraph_agents.nodes.retriever_agent import retriever_agent_node
        from langgraph_agents.state import AgentState

        config = RunnableConfig(configurable={
            "user_id": "u1", "session_id": "s1",
            "query": "weather", "persona_id": "eca_default",
            "request_id": "r2",
            # "web_search" intentionally absent — should default False
        })
        state: AgentState = {
            "messages": [], "errors": [], "retry_count": 0,
            "total_tokens": 0, "required_outputs": [],
            "resolved_query": "weather", "retriever_rounds": 0,
        }

        fake_ai = AIMessage(content="done", tool_calls=[])
        bound_tools_captured = []

        def fake_get_chat_model(role):
            m = MagicMock()
            def bind_tools(tools):
                bound_tools_captured.extend(tools)
                inner = MagicMock()
                inner.ainvoke = AsyncMock(return_value=fake_ai)
                return inner
            m.bind_tools = bind_tools
            return m

        with patch("langgraph_agents.nodes.retriever_agent.get_chat_model", side_effect=fake_get_chat_model):
            await retriever_agent_node(state, config)

        tool_names = [getattr(t, "name", None) for t in bound_tools_captured]
        assert "search_medical" not in tool_names, (
            "search_medical must NOT be in bound tools when web_search defaults to False"
        )
