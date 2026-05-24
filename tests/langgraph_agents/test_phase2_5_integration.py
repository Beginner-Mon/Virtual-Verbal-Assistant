"""Integration tests for Phase 2.5 graph — requires GEMINI_API_KEY."""

import os
import pytest

from langgraph_agents.graph import build_graph_async
from langgraph_agents.state import AgentState

HAS_LLM_KEY = bool(os.getenv("DEEPSEEK_API_KEY"))


def _base_invoke_args(**overrides):
    """Returns (initial_state, config) for graph.ainvoke."""
    state = {
        "messages": [],
        "errors": [],
        "retry_count": 0,
        "total_tokens": 0,
    }
    config = {
        "configurable": {
            "user_id": "test-user",
            "session_id": "test-session",
            "query": overrides.pop("query", "Xin chao"),
            "persona_id": "eca_default",
            "output_mode": "text",
            "request_id": "test-001",
            "token_limit": None,
        }
    }
    state.update(overrides)
    return state, config


@pytest.mark.integration
@pytest.mark.skipif(not HAS_LLM_KEY, reason="DEEPSEEK_API_KEY not set")
@pytest.mark.asyncio
async def test_full_graph_v24_conversation_path():
    graph = await build_graph_async()
    state, config = _base_invoke_args(query="Xin chao")
    result = await graph.ainvoke(state, config=config)
    assert result["final_answer"], "final_answer should not be empty"
    assert result.get("intent") == "conversation"


@pytest.mark.integration
@pytest.mark.skipif(not HAS_LLM_KEY, reason="DEEPSEEK_API_KEY not set")
@pytest.mark.asyncio
async def test_full_graph_v24_exercise_full_pipeline():
    graph = await build_graph_async()
    state, config = _base_invoke_args(query="Bai tap cho dau lung")
    result = await graph.ainvoke(state, config=config)
    assert result["final_answer"], "final_answer should not be empty"
    assert result.get("intent") in ("exercise_recommendation", "knowledge_query"), \
        f"Expected exercise/knowledge intent, got {result.get('intent')}"


@pytest.mark.integration
@pytest.mark.skipif(not HAS_LLM_KEY, reason="DEEPSEEK_API_KEY not set")
@pytest.mark.asyncio
async def test_full_graph_v24_grader_retry_loop():
    graph = await build_graph_async()
    # Use a query that expects exercise_recommendation — grader will check for markers
    state, config = _base_invoke_args(query="Bai tap cho co")
    result = await graph.ainvoke(state, config=config)
    assert result["final_answer"], "final_answer should not be empty"
    # grader_result should be set
    assert result.get("grader_result") in ("pass", "pass_with_warning", None)
