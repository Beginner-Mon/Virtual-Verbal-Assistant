"""Tests for error_handler node (Task 2.5.10)."""

import pytest

from langgraph_agents.state import ErrorSeverity, AgentState
from langgraph_agents.nodes._persona_loader import get_ui_string


@pytest.mark.unit
@pytest.mark.asyncio
async def test_error_handler_writes_final_answer():
    from langgraph_agents.nodes.error_handler import error_handler_node

    state: AgentState = {
        "messages": [],
        "errors": [{"node": "test", "severity": ErrorSeverity.CRITICAL, "message": "fail"}],
        "final_answer": "",
    }
    result = await error_handler_node(state)
    assert "final_answer" in result
    assert result["final_answer"]  # non-empty


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
    assert "final_answer" in result


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
    # Recoverable-only takes the soft branch, not the critical apology.
    #
    # Asserted against the persona's own copy rather than the Vietnamese
    # substrings this used to look for ("lỗi nhỏ" / "sự cố"): the wording now
    # comes from the character's `## UI Strings`, and the default persona
    # (eca_default / Seele) speaks English. Comparing to get_ui_string keeps the
    # branch check while staying language-agnostic.
    assert result["final_answer"] == get_ui_string("eca_default", "error_partial")
    assert result["final_answer"] != get_ui_string("eca_default", "error_system")


@pytest.mark.unit
@pytest.mark.asyncio
async def test_error_handler_uses_selected_character_copy():
    """The error line belongs to whoever the user picked.

    This is the one reply a user is guaranteed to read on a bad day, and until
    now it was the only message in the product that belonged to nobody.
    """
    from langgraph_agents.nodes.error_handler import error_handler_node

    state: AgentState = {
        "messages": [],
        "errors": [{"node": "test", "severity": ErrorSeverity.CRITICAL, "message": "fail"}],
        "final_answer": "",
    }
    config = {"configurable": {"persona_id": "anne"}}

    result = await error_handler_node(state, config)
    assert result["final_answer"] == get_ui_string("anne", "error_system")
    assert result["final_answer"] != get_ui_string("bronya", "error_system")
