"""Tests for retriever_agent node (Task 2.5.6)."""

import pytest

from langgraph_agents.nodes.retriever_agent import RETRIEVER_TOOLS, _RETRIEVER_TOOLS


@pytest.mark.unit
def test_retriever_tools_list():
    assert len(RETRIEVER_TOOLS) == 1
    assert RETRIEVER_TOOLS[0].name == "pgvector_search"


@pytest.mark.unit
def test_retriever_tools_backwards_compat_alias():
    """`_RETRIEVER_TOOLS` is an alias kept for older imports."""
    assert _RETRIEVER_TOOLS is RETRIEVER_TOOLS


@pytest.mark.unit
def test_retriever_tools_are_callable_via_tool_interface():
    """Tool must expose async invoke contract used by ToolNode."""
    t = RETRIEVER_TOOLS[0]
    assert hasattr(t, "ainvoke")
    assert callable(t.ainvoke)


@pytest.mark.unit
def test_route_after_retriever_errors_first():
    """CRITICAL error in state must short-circuit to error_handler before tool/end check."""
    from langgraph_agents.routing import route_after_retriever
    from langgraph_agents.state import ErrorSeverity

    state = {
        "messages": [],
        "errors": [{"node": "retriever_agent", "severity": ErrorSeverity.CRITICAL, "message": "x"}],
    }
    assert route_after_retriever(state) == "error_handler"


@pytest.mark.unit
def test_route_after_retriever_no_tool_calls_goes_to_synthesizer():
    from langgraph_agents.routing import route_after_retriever
    from langchain_core.messages import AIMessage

    state = {"messages": [AIMessage(content="done, no tools needed")], "errors": []}
    assert route_after_retriever(state) == "synthesizer"


@pytest.mark.unit
def test_route_after_retriever_with_tool_calls_goes_to_tools():
    from langgraph_agents.routing import route_after_retriever
    from langchain_core.messages import AIMessage

    msg = AIMessage(
        content="",
        tool_calls=[{"name": "pgvector_search", "args": {"query": "x"}, "id": "1"}],
    )
    state = {"messages": [msg], "errors": []}
    assert route_after_retriever(state) == "tools"
