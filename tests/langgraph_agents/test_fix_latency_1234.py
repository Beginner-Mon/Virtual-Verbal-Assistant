"""Unit tests for latency-optimization fixes #1-#4.

Spec: docs/fixes/latency-optimization-1234.md

  #1 — DeepSeek prompt-cache hit/miss telemetry surfaced in node_complete logs
  #2 — timeout/max_retries + one-shot Gemini fallback for planner/synthesizer
  #3 — max_tokens bound on the LLM client + tightened synthesizer word-limit
  #4 — dedupe identical (name+args) tool_calls before ToolNode execution

All LLM calls are mocked — no live DeepSeek/Gemini calls are made. Tests that
exercise the *_fallback* code path always patch get_fallback_chat_model
explicitly (never rely on it being None) because a real GEMINI_API_KEYS is
present in the test .env (agenticRAG/agentic_rag_gemini/.env, auto-loaded by
conftest.py) — an un-mocked fallback would otherwise attempt a live call.
"""

import pytest
from unittest.mock import AsyncMock, MagicMock, patch

from langchain_core.runnables import RunnableConfig
from langchain_core.messages import AIMessage


# ─────────────────────────────────────────────────────────────────────────────
# #1 — extract_cache_tokens helper (llm.py)
# ─────────────────────────────────────────────────────────────────────────────

@pytest.mark.unit
class TestExtractCacheTokens:
    def test_reads_response_metadata_token_usage(self):
        from langgraph_agents.llm import extract_cache_tokens
        msg = AIMessage(content="hi")
        msg.response_metadata = {"token_usage": {
            "prompt_cache_hit_tokens": 120, "prompt_cache_miss_tokens": 30,
        }}
        hit, miss = extract_cache_tokens(msg)
        assert hit == 120
        assert miss == 30

    def test_falls_back_to_usage_metadata_when_response_metadata_absent(self):
        from langgraph_agents.llm import extract_cache_tokens
        msg = AIMessage(content="hi")
        msg.response_metadata = {}
        msg.usage_metadata = {"input_token_details": {"cache_read": 55}}
        hit, miss = extract_cache_tokens(msg)
        assert hit == 55
        assert miss is None

    def test_absent_telemetry_returns_none_not_error(self):
        """DeepSeek not returning cache fields is informative, not a bug (spec)."""
        from langgraph_agents.llm import extract_cache_tokens
        msg = AIMessage(content="hi")
        hit, miss = extract_cache_tokens(msg)
        assert hit is None
        assert miss is None

    def test_none_message_returns_none_none_without_raising(self):
        from langgraph_agents.llm import extract_cache_tokens
        hit, miss = extract_cache_tokens(None)
        assert hit is None
        assert miss is None


@pytest.mark.unit
class TestFix1NodeCompleteLogsCacheTokens:
    """Verify planner/retriever_agent/synthesizer surface cache_hit/miss in node_complete."""

    @pytest.mark.asyncio
    async def test_planner_logs_cache_tokens_from_raw_response(self):
        from langgraph_agents.nodes import planner as planner_mod
        from langgraph_agents.nodes.planner import PlanOutput

        fake_plan = PlanOutput(required_outputs=[], resolved_query="hi", needs_retrieval=False)
        fake_ai = AIMessage(content="{}")
        fake_ai.response_metadata = {"token_usage": {
            "prompt_cache_hit_tokens": 200, "prompt_cache_miss_tokens": 10,
        }}
        include_raw_result = {"raw": fake_ai, "parsed": fake_plan, "parsing_error": None}

        with patch.object(planner_mod, "get_chat_model") as mock_llm, \
             patch.object(planner_mod, "logger") as mock_logger:
            mock_llm.return_value.with_structured_output.return_value.ainvoke = AsyncMock(
                return_value=include_raw_result
            )
            config = RunnableConfig(configurable={
                "user_id": "u", "session_id": "s", "query": "xin chao",
                "persona_id": "eca_default", "request_id": "r1",
            })
            state = {"messages": [], "errors": [], "total_tokens": 0, "retry_count": 0}
            await planner_mod.planner_node(state, config)

        calls = [c for c in mock_logger.info.call_args_list if c.args[0] == "node_complete"]
        assert len(calls) == 1
        extra = calls[0].kwargs["extra"]
        assert extra["cache_hit_tokens"] == 200
        assert extra["cache_miss_tokens"] == 10
        assert extra["llm_fallback_used"] is False

    @pytest.mark.asyncio
    async def test_planner_logs_null_cache_tokens_when_absent(self):
        """Existing test style mocks ainvoke to return a bare PlanOutput (no raw
        AIMessage available) — cache fields must log as None, not crash."""
        from langgraph_agents.nodes import planner as planner_mod
        from langgraph_agents.nodes.planner import PlanOutput

        fake_plan = PlanOutput(required_outputs=[], resolved_query="hi", needs_retrieval=False)

        with patch.object(planner_mod, "get_chat_model") as mock_llm, \
             patch.object(planner_mod, "logger") as mock_logger:
            mock_llm.return_value.with_structured_output.return_value.ainvoke = AsyncMock(
                return_value=fake_plan
            )
            config = RunnableConfig(configurable={
                "user_id": "u", "session_id": "s", "query": "hi",
                "persona_id": "eca_default", "request_id": "r1b",
            })
            state = {"messages": [], "errors": [], "total_tokens": 0, "retry_count": 0}
            await planner_mod.planner_node(state, config)

        calls = [c for c in mock_logger.info.call_args_list if c.args[0] == "node_complete"]
        extra = calls[0].kwargs["extra"]
        assert extra["cache_hit_tokens"] is None
        assert extra["cache_miss_tokens"] is None

    @pytest.mark.asyncio
    async def test_retriever_agent_logs_cache_tokens(self):
        from langgraph_agents.nodes import retriever_agent as ra_mod

        fake_ai = AIMessage(content="", tool_calls=[])
        fake_ai.response_metadata = {"token_usage": {
            "prompt_cache_hit_tokens": 77, "prompt_cache_miss_tokens": 5,
        }}

        config = RunnableConfig(configurable={
            "user_id": "u", "session_id": "s", "query": "hello",
            "persona_id": "eca_default", "request_id": "r2", "web_search": False,
        })
        state = {
            "messages": [], "errors": [], "retry_count": 0, "total_tokens": 0,
            "required_outputs": [], "resolved_query": "hello",
        }

        with patch.object(ra_mod, "get_chat_model") as mock_llm, \
             patch.object(ra_mod, "logger") as mock_logger:
            mock_llm.return_value.bind_tools.return_value.ainvoke = AsyncMock(return_value=fake_ai)
            await ra_mod.retriever_agent_node(state, config)

        calls = [c for c in mock_logger.info.call_args_list if c.args[0] == "node_complete"]
        extra = calls[0].kwargs["extra"]
        assert extra["cache_hit_tokens"] == 77
        assert extra["cache_miss_tokens"] == 5

    @pytest.mark.asyncio
    async def test_synthesizer_logs_cache_tokens_non_streaming(self):
        from langgraph_agents.nodes import synthesizer as syn_mod

        fake_ai = AIMessage(content="cau tra loi")
        fake_ai.response_metadata = {"token_usage": {
            "prompt_cache_hit_tokens": 300, "prompt_cache_miss_tokens": 0,
        }}

        with patch.object(syn_mod, "get_chat_model") as mock_llm, \
             patch.object(syn_mod, "logger") as mock_logger:
            mock_llm.return_value.ainvoke = AsyncMock(return_value=fake_ai)
            state = {
                "messages": [], "resolved_query": "hello",
                "required_outputs": [], "needs_clarification": False,
                "total_tokens": 0,
            }
            config = {"configurable": {
                "request_id": "r3", "persona_id": "eca_default", "query": "hello",
            }}
            await syn_mod.synthesizer_node(state, config)

        calls = [c for c in mock_logger.info.call_args_list if c.args[0] == "node_complete"]
        extra = calls[0].kwargs["extra"]
        assert extra["cache_hit_tokens"] == 300
        assert extra["cache_miss_tokens"] == 0
        assert extra["llm_fallback_used"] is False


# ─────────────────────────────────────────────────────────────────────────────
# #2 — timeout/max_retries + Gemini fallback
# ─────────────────────────────────────────────────────────────────────────────

@pytest.mark.unit
class TestFix2LlmClientTimeoutRetries:
    def test_fast_role_timeout_and_retries(self, monkeypatch):
        monkeypatch.setenv("DEEPSEEK_API_KEY", "fake")
        from langgraph_agents import llm as llm_mod
        llm_mod.get_chat_model.cache_clear()
        m = llm_mod.get_chat_model("planner")
        assert m.request_timeout == 20.0
        assert m.max_retries == 1

    def test_heavy_role_timeout_and_retries(self, monkeypatch):
        monkeypatch.setenv("DEEPSEEK_API_KEY", "fake")
        from langgraph_agents import llm as llm_mod
        llm_mod.get_chat_model.cache_clear()
        m = llm_mod.get_chat_model("synthesizer")
        assert m.request_timeout == 35.0
        assert m.max_retries == 1


@pytest.mark.unit
class TestFix2GetFallbackChatModel:
    def test_returns_none_when_gemini_api_keys_unset(self, monkeypatch):
        monkeypatch.delenv("GEMINI_API_KEYS", raising=False)
        from langgraph_agents import llm as llm_mod
        llm_mod.get_fallback_chat_model.cache_clear()
        assert llm_mod.get_fallback_chat_model("planner") is None

    def test_builds_model_from_first_key_when_set(self, monkeypatch):
        monkeypatch.setenv("GEMINI_API_KEYS", "key-one,key-two")
        from langgraph_agents import llm as llm_mod
        llm_mod.get_fallback_chat_model.cache_clear()
        model = llm_mod.get_fallback_chat_model("planner")
        assert model is not None
        assert model.model.endswith("gemini-2.0-flash")

    def test_heavy_role_uses_pro_model(self, monkeypatch):
        monkeypatch.setenv("GEMINI_API_KEYS", "key-one")
        from langgraph_agents import llm as llm_mod
        llm_mod.get_fallback_chat_model.cache_clear()
        model = llm_mod.get_fallback_chat_model("synthesizer")
        assert model is not None
        assert model.model.endswith("gemini-2.5-pro")


@pytest.mark.unit
class TestFix2PlannerFallback:
    @pytest.mark.asyncio
    async def test_fallback_used_on_primary_timeout(self):
        from langgraph_agents.nodes import planner as planner_mod
        from langgraph_agents.nodes.planner import PlanOutput

        fallback_plan = PlanOutput(
            required_outputs=["scope_disclaimer"], resolved_query="fallback answer",
            needs_retrieval=False,
        )

        with patch.object(planner_mod, "get_chat_model") as mock_primary, \
             patch.object(planner_mod, "get_fallback_chat_model") as mock_fallback_factory, \
             patch.object(planner_mod, "logger") as mock_logger:
            mock_primary.return_value.with_structured_output.return_value.ainvoke = AsyncMock(
                side_effect=TimeoutError("deepseek timed out")
            )
            fallback_model = MagicMock()
            fallback_model.with_structured_output.return_value.ainvoke = AsyncMock(
                return_value={
                    "raw": AIMessage(content="{}"), "parsed": fallback_plan, "parsing_error": None,
                }
            )
            mock_fallback_factory.return_value = fallback_model

            config = RunnableConfig(configurable={
                "user_id": "u", "session_id": "s", "query": "bai tap",
                "persona_id": "eca_default", "request_id": "r4",
            })
            state = {"messages": [], "errors": [], "total_tokens": 0, "retry_count": 0}
            result = await planner_mod.planner_node(state, config)

        assert result["resolved_query"] == "fallback answer"
        assert "errors" not in result
        fb_calls = [c for c in mock_logger.info.call_args_list if c.args[0] == "llm_fallback_used"]
        assert len(fb_calls) == 1
        assert fb_calls[0].kwargs["extra"]["llm_fallback_used"] is True

    @pytest.mark.asyncio
    async def test_fallback_not_invoked_when_primary_succeeds(self):
        """No wasted fallback call when the primary DeepSeek call succeeds."""
        from langgraph_agents.nodes import planner as planner_mod
        from langgraph_agents.nodes.planner import PlanOutput

        plan = PlanOutput(required_outputs=[], resolved_query="ok", needs_retrieval=False)

        with patch.object(planner_mod, "get_chat_model") as mock_primary, \
             patch.object(planner_mod, "get_fallback_chat_model") as mock_fallback_factory:
            mock_primary.return_value.with_structured_output.return_value.ainvoke = AsyncMock(
                return_value=plan
            )
            config = RunnableConfig(configurable={
                "user_id": "u", "session_id": "s", "query": "hi",
                "persona_id": "eca_default", "request_id": "r5",
            })
            state = {"messages": [], "errors": [], "total_tokens": 0, "retry_count": 0}
            result = await planner_mod.planner_node(state, config)

        mock_fallback_factory.assert_not_called()
        assert result["resolved_query"] == "ok"

    @pytest.mark.asyncio
    async def test_fallback_also_failing_falls_through_to_existing_error_path(self):
        from langgraph_agents.nodes import planner as planner_mod

        with patch.object(planner_mod, "get_chat_model") as mock_primary, \
             patch.object(planner_mod, "get_fallback_chat_model") as mock_fallback_factory:
            mock_primary.return_value.with_structured_output.return_value.ainvoke = AsyncMock(
                side_effect=TimeoutError("primary down")
            )
            fallback_model = MagicMock()
            fallback_model.with_structured_output.return_value.ainvoke = AsyncMock(
                side_effect=RuntimeError("fallback also down")
            )
            mock_fallback_factory.return_value = fallback_model

            config = RunnableConfig(configurable={
                "user_id": "u", "session_id": "s", "query": "hi",
                "persona_id": "eca_default", "request_id": "r6",
            })
            state = {"messages": [], "errors": [], "total_tokens": 0, "retry_count": 0}
            result = await planner_mod.planner_node(state, config)

        assert result["needs_clarification"] is True
        assert result["errors"][0]["severity"] == "recoverable"


@pytest.mark.unit
class TestFix2SynthesizerFallback:
    @pytest.mark.asyncio
    async def test_fallback_used_on_primary_timeout(self):
        from langgraph_agents.nodes import synthesizer as syn_mod

        with patch.object(syn_mod, "get_chat_model") as mock_primary, \
             patch.object(syn_mod, "get_fallback_chat_model") as mock_fallback_factory, \
             patch.object(syn_mod, "logger") as mock_logger:
            mock_primary.return_value.ainvoke = AsyncMock(side_effect=TimeoutError("timeout"))

            fb_ai = AIMessage(content="Day la cau tra loi du phong.")
            fallback_model = MagicMock()
            fallback_model.ainvoke = AsyncMock(return_value=fb_ai)
            mock_fallback_factory.return_value = fallback_model

            state = {
                "messages": [], "resolved_query": "bai tap squat",
                "required_outputs": [], "needs_clarification": False,
                "total_tokens": 0,
            }
            config = {"configurable": {
                "request_id": "r7", "persona_id": "eca_default", "query": "bai tap squat",
            }}
            result = await syn_mod.synthesizer_node(state, config)

        assert result["final_answer"] == "Day la cau tra loi du phong."
        assert "errors" not in result
        fb_calls = [c for c in mock_logger.info.call_args_list if c.args[0] == "llm_fallback_used"]
        assert len(fb_calls) == 1

    @pytest.mark.asyncio
    async def test_fallback_not_invoked_when_primary_succeeds(self):
        from langgraph_agents.nodes import synthesizer as syn_mod

        with patch.object(syn_mod, "get_chat_model") as mock_primary, \
             patch.object(syn_mod, "get_fallback_chat_model") as mock_fallback_factory:
            fake_ai = AIMessage(content="ok answer")
            mock_primary.return_value.ainvoke = AsyncMock(return_value=fake_ai)

            state = {
                "messages": [], "resolved_query": "hello",
                "required_outputs": [], "needs_clarification": False,
                "total_tokens": 0,
            }
            config = {"configurable": {
                "request_id": "r8", "persona_id": "eca_default", "query": "hello",
            }}
            result = await syn_mod.synthesizer_node(state, config)

        mock_fallback_factory.assert_not_called()
        assert result["final_answer"] == "ok answer"

    @pytest.mark.asyncio
    async def test_fallback_unavailable_falls_through_to_critical_error(self):
        from langgraph_agents.nodes import synthesizer as syn_mod
        from langgraph_agents.state import ErrorSeverity

        with patch.object(syn_mod, "get_chat_model") as mock_primary, \
             patch.object(syn_mod, "get_fallback_chat_model", return_value=None):
            mock_primary.return_value.ainvoke = AsyncMock(side_effect=RuntimeError("down"))

            state = {
                "messages": [], "resolved_query": "hello",
                "required_outputs": [], "needs_clarification": False,
                "total_tokens": 0,
            }
            config = {"configurable": {
                "request_id": "r9", "persona_id": "eca_default", "query": "hello",
            }}
            result = await syn_mod.synthesizer_node(state, config)

        assert result["errors"][0]["severity"] == ErrorSeverity.CRITICAL
        assert "Xin lỗi" in result["final_answer"]


# ─────────────────────────────────────────────────────────────────────────────
# #3 — max_tokens bound + tightened synthesizer word-limit instruction
# ─────────────────────────────────────────────────────────────────────────────

@pytest.mark.unit
class TestFix3MaxTokens:
    def test_fast_role_max_tokens_512(self, monkeypatch):
        monkeypatch.setenv("DEEPSEEK_API_KEY", "fake")
        from langgraph_agents import llm as llm_mod
        llm_mod.get_chat_model.cache_clear()
        assert llm_mod.get_chat_model("planner").max_tokens == 512
        assert llm_mod.get_chat_model("retriever").max_tokens == 512
        assert llm_mod.get_chat_model("conversation").max_tokens == 512

    def test_heavy_role_max_tokens_1024(self, monkeypatch):
        monkeypatch.setenv("DEEPSEEK_API_KEY", "fake")
        from langgraph_agents import llm as llm_mod
        llm_mod.get_chat_model.cache_clear()
        assert llm_mod.get_chat_model("synthesizer").max_tokens == 1024

    def test_synthesize_prompt_tightened_to_350_words(self):
        from langgraph_agents.nodes.synthesizer import _SYNTHESIZE_TASK
        assert "under 350 words" in _SYNTHESIZE_TASK
        assert "500 words" not in _SYNTHESIZE_TASK
        assert "Do not pad or repeat safety disclaimers" in _SYNTHESIZE_TASK


# ─────────────────────────────────────────────────────────────────────────────
# #4 — dedupe identical (name+args) tool_calls before ToolNode execution
# ─────────────────────────────────────────────────────────────────────────────

@pytest.mark.unit
class TestFix4DedupeToolCalls:
    def test_dedupe_helper_identical_args_kept_once(self):
        from langgraph_agents.nodes.retriever_agent import _dedupe_tool_calls
        tool_calls = [
            {"name": "kb_search", "args": {"query": "X"}, "id": "1"},
            {"name": "kb_search", "args": {"query": "X"}, "id": "2"},
            {"name": "kb_search", "args": {"query": "Y"}, "id": "3"},
        ]
        deduped, removed = _dedupe_tool_calls(tool_calls)
        assert removed == 1
        assert len(deduped) == 2
        assert sorted(tc["args"]["query"] for tc in deduped) == ["X", "Y"]

    def test_dedupe_helper_different_tool_names_both_kept(self):
        from langgraph_agents.nodes.retriever_agent import _dedupe_tool_calls
        tool_calls = [
            {"name": "kb_search", "args": {"query": "X"}, "id": "1"},
            {"name": "memory_search", "args": {"query": "X"}, "id": "2"},
        ]
        deduped, removed = _dedupe_tool_calls(tool_calls)
        assert removed == 0
        assert len(deduped) == 2

    def test_dedupe_helper_no_tool_calls_noop(self):
        from langgraph_agents.nodes.retriever_agent import _dedupe_tool_calls
        deduped, removed = _dedupe_tool_calls([])
        assert deduped == []
        assert removed == 0

    @pytest.mark.asyncio
    async def test_retriever_node_dedupes_before_returning(self):
        """Acceptance criteria: 2 identical kb_search(X) + 1 distinct kb_search(Y)
        → returned message has 2 tool_calls (X once, Y once), not 3."""
        from langgraph_agents.nodes.retriever_agent import retriever_agent_node

        fake_ai = AIMessage(content="", tool_calls=[
            {"name": "kb_search", "args": {"query": "X"}, "id": "1"},
            {"name": "kb_search", "args": {"query": "X"}, "id": "2"},
            {"name": "kb_search", "args": {"query": "Y"}, "id": "3"},
        ])

        config = RunnableConfig(configurable={
            "user_id": "u", "session_id": "s", "query": "bai tap squat",
            "persona_id": "eca_default", "request_id": "r10", "web_search": False,
        })
        state = {
            "messages": [], "errors": [], "retry_count": 0, "total_tokens": 0,
            "required_outputs": ["exercise_protocol"], "resolved_query": "bai tap squat",
        }

        with patch("langgraph_agents.nodes.retriever_agent.get_chat_model") as mock_llm:
            mock_llm.return_value.bind_tools.return_value.ainvoke = AsyncMock(return_value=fake_ai)
            result = await retriever_agent_node(state, config)

        returned_msg = result["messages"][0]
        assert len(returned_msg.tool_calls) == 2
        assert sorted(tc["args"]["query"] for tc in returned_msg.tool_calls) == ["X", "Y"]

    @pytest.mark.asyncio
    async def test_retriever_node_logs_tool_calls_deduped_count(self):
        from langgraph_agents.nodes import retriever_agent as ra_mod

        fake_ai = AIMessage(content="", tool_calls=[
            {"name": "kb_search", "args": {"query": "X"}, "id": "1"},
            {"name": "kb_search", "args": {"query": "X"}, "id": "2"},
        ])

        config = RunnableConfig(configurable={
            "user_id": "u", "session_id": "s", "query": "q",
            "persona_id": "eca_default", "request_id": "r11", "web_search": False,
        })
        state = {
            "messages": [], "errors": [], "retry_count": 0, "total_tokens": 0,
            "required_outputs": [], "resolved_query": "q",
        }

        with patch.object(ra_mod, "get_chat_model") as mock_llm, \
             patch.object(ra_mod, "logger") as mock_logger:
            mock_llm.return_value.bind_tools.return_value.ainvoke = AsyncMock(return_value=fake_ai)
            await ra_mod.retriever_agent_node(state, config)

        calls = [c for c in mock_logger.info.call_args_list if c.args[0] == "node_complete"]
        assert len(calls) == 1
        assert calls[0].kwargs["extra"]["tool_calls_deduped"] == 1

    @pytest.mark.asyncio
    async def test_retriever_node_no_dedupe_key_when_nothing_removed(self):
        """tool_calls_deduped should only appear in the log when count > 0."""
        from langgraph_agents.nodes import retriever_agent as ra_mod

        fake_ai = AIMessage(content="", tool_calls=[
            {"name": "kb_search", "args": {"query": "X"}, "id": "1"},
        ])

        config = RunnableConfig(configurable={
            "user_id": "u", "session_id": "s", "query": "q",
            "persona_id": "eca_default", "request_id": "r12", "web_search": False,
        })
        state = {
            "messages": [], "errors": [], "retry_count": 0, "total_tokens": 0,
            "required_outputs": [], "resolved_query": "q",
        }

        with patch.object(ra_mod, "get_chat_model") as mock_llm, \
             patch.object(ra_mod, "logger") as mock_logger:
            mock_llm.return_value.bind_tools.return_value.ainvoke = AsyncMock(return_value=fake_ai)
            await ra_mod.retriever_agent_node(state, config)

        calls = [c for c in mock_logger.info.call_args_list if c.args[0] == "node_complete"]
        assert "tool_calls_deduped" not in calls[0].kwargs["extra"]
