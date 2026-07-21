"""Tests for circuit breakers on LLM + MCP (Phase 6 P0.3)."""

import asyncio
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from langgraph_agents.core.circuit_breaker import CircuitBreaker


def _reset_llm_breaker(role: str = "test"):
    """Create a fresh breaker and inject it into llm.py's _cb_map.
    Also clears lru_cache on get_chat_model so new breaker takes effect."""
    from langgraph_agents.llm import _cb_map, get_chat_model
    breaker = CircuitBreaker(
        name=f"llm:{role}",
        failure_threshold=3,
        cool_down_seconds=30.0,
    )
    _cb_map[role] = breaker
    get_chat_model.cache_clear()
    return breaker


def _reset_mcp_breaker():
    """Reset the MCP breaker to a clean state."""
    import langgraph_agents.mcp.client as mod
    mod._mcp_breaker._state.failures = 0
    mod._mcp_breaker._state.state = "closed"
    mod._mcp_breaker._state.opened_at = 0.0
    # Also clear cache
    mod._mcp_tools = []
    mod._mcp_client = None


# ── LLM breaker unit tests ──────────────────────────────────────────


@pytest.mark.unit
@pytest.mark.asyncio
async def test_llm_breaker_opens_after_3_failures():
    """3 consecutive failures → breaker opens → 4th call raises CircuitBreakerOpenError."""
    breaker = _reset_llm_breaker("test_open")
    # Force 3 failures
    for i in range(3):
        assert breaker.allow() is True
        breaker.record_failure()
    assert breaker.state == "open"
    assert breaker.allow() is False


@pytest.mark.unit
@pytest.mark.asyncio
async def test_llm_breaker_closes_on_success_after_cooldown():
    """After cooldown, a successful probe call closes the breaker."""
    breaker = _reset_llm_breaker("test_recover")
    # Force open
    for _ in range(3):
        breaker.allow()
        breaker.record_failure()
    assert breaker.state == "open"

    # Simulate cooldown expired by manually advancing state
    breaker._state.state = "half_open"
    breaker._state.failures = 0

    assert breaker.allow() is True  # half_open allows probe
    breaker.record_success()
    assert breaker.state == "closed"


@pytest.mark.unit
@pytest.mark.asyncio
async def test_llm_breaker_opens_again_on_failed_probe():
    """If probe fails in half_open state, breaker re-opens."""
    breaker = _reset_llm_breaker("test_reopen")
    for _ in range(3):
        breaker.allow()
        breaker.record_failure()
    assert breaker.state == "open"

    breaker._state.state = "half_open"
    assert breaker.allow() is True
    breaker.record_failure()
    assert breaker.state == "open"


@pytest.mark.unit
def test_breaker_snapshot_includes_all_fields():
    breaker = CircuitBreaker(name="test_snap", failure_threshold=5, cool_down_seconds=10.0)
    snap = breaker.snapshot()
    assert snap["name"] == "test_snap"
    assert snap["state"] == "closed"
    assert snap["failures"] == 0
    assert snap["failure_threshold"] == 5
    assert snap["cool_down_seconds"] == 10.0


# ── MCP breaker unit tests ──────────────────────────────────────────


@pytest.mark.unit
@pytest.mark.asyncio
async def test_mcp_breaker_returns_empty_when_open():
    """When MCP breaker is open, get_mcp_tools() returns [] immediately."""
    import langgraph_agents.mcp.client as mod
    _reset_mcp_breaker()

    # Force breaker open
    for _ in range(2):
        mod._mcp_breaker.record_failure()
    mod._mcp_breaker._state.state = "open"
    assert mod._mcp_breaker.allow() is False

    tools = await mod.get_mcp_tools()
    assert tools == []


@pytest.mark.unit
@pytest.mark.asyncio
async def test_mcp_breaker_records_success_and_closes():
    """Successful MCP discovery after failure records success."""
    import langgraph_agents.mcp.client as mod
    _reset_mcp_breaker()

    # Record 1 failure (breaker still closed)
    mod._mcp_breaker.record_failure()
    assert mod._mcp_breaker.state == "closed"

    # Successful discovery
    fake_tool = MagicMock()
    fake_tool.name = "pgvector_search"
    ok_client = MagicMock()
    ok_client.get_tools = AsyncMock(return_value=[fake_tool])

    with patch("langgraph_agents.mcp.client.MultiServerMCPClient", return_value=ok_client):
        with patch("langgraph_agents.mcp.client._load_mcp_config", return_value={"x": {}}):
            tools = await mod.get_mcp_tools()
            assert len(tools) == 1
            assert mod._mcp_breaker.state == "closed"


# ── Planner integration: BreakerOpen → clarify ──────────────────────


@pytest.mark.unit
@pytest.mark.asyncio
async def test_planner_handles_breaker_open_as_recoverable():
    """When planner LLM breaker is open, planner returns clarify + recoverable error."""
    breaker = _reset_llm_breaker("planner")
    # Force breaker open
    for _ in range(3):
        breaker.record_failure()
    breaker._state.state = "open"

    from langgraph_agents.nodes.planner import planner_node

    state: AgentState = {"messages": [], "errors": [], "retry_count": 0,
                          "total_tokens": 0}
    config: RunnableConfig = {"configurable": {
        "query": "Bài tập cho đau lưng",
        "user_id": "u1",
        "session_id": "s1",
        "request_id": "r-test",
        "persona_id": "eca_default",
    }}

    # Gemini fallback (fix #2) is disabled here so this test stays a pure
    # unit test — a real GEMINI_API_KEYS may be present in .env, but we must
    # not let a unit test attempt a live fallback call.
    with patch("langgraph_agents.nodes.planner.get_fallback_chat_model", return_value=None):
        result = await planner_node(state, config)

    assert result["needs_clarification"] is True
    errors = result.get("errors", [])
    assert len(errors) == 1
    assert errors[0]["severity"] in ("recoverable", "RECOVERABLE")
    assert "planner" in errors[0]["node"]
