"""Tests for error_handler node (Task 2.5.10)."""

import pytest

from langgraph_agents.state import ErrorSeverity, AgentState


@pytest.mark.unit
@pytest.mark.asyncio
async def test_error_handler_writes_reasoning_output():
    from langgraph_agents.nodes.error_handler import error_handler_node

    state: AgentState = {
        "messages": [],
        "errors": [{"node": "test", "severity": ErrorSeverity.CRITICAL, "message": "fail"}],
        "final_answer": "",
    }
    result = await error_handler_node(state)
    assert "reasoning_output" in result
    assert result["reasoning_output"]  # non-empty


@pytest.mark.unit
@pytest.mark.asyncio
async def test_error_handler_no_error():
    from langgraph_agents.nodes.error_handler import error_handler_node

    state: AgentState = {
        "messages": [],
        "errors": [],
        "final_answer": "",
    }
    result = await error_handler_node(state)
    assert "reasoning_output" in result


@pytest.mark.unit
@pytest.mark.asyncio
async def test_error_handler_recoverable_only():
    from langgraph_agents.nodes.error_handler import error_handler_node

    state: AgentState = {
        "messages": [],
        "errors": [{"node": "test", "severity": ErrorSeverity.RECOVERABLE, "message": "minor"}],
        "final_answer": "",
    }
    result = await error_handler_node(state)
    # Recoverable-only should produce the soft message, not the critical apology.
    assert "sự cố" not in result["reasoning_output"]
    assert "lỗi nhỏ" in result["reasoning_output"]
