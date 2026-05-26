"""Tests for retriever_agent node (Task 2.5.6 + Phase 6.7 web toggle)."""

from unittest.mock import AsyncMock, MagicMock, patch

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


@pytest.mark.unit
@pytest.mark.asyncio
async def test_retriever_drops_web_tool_when_disabled():
    """When config web_search=False, search_medical must not be in bound tools."""
    from langgraph_agents.nodes.retriever_agent import retriever_agent_node

    mock_web_tool = MagicMock()
    mock_web_tool.name = "search_medical"
    mock_pgvector = MagicMock()
    mock_pgvector.name = "pgvector_search"

    captured_tools = []

    mock_llm = MagicMock()
    mock_llm.bind_tools = MagicMock(return_value=mock_llm)
    mock_llm.ainvoke = AsyncMock(return_value=MagicMock(tool_calls=[], usage_metadata={}))

    def capture_bind_tools(tools):
        captured_tools.extend(tools)
        return mock_llm

    mock_llm.bind_tools = MagicMock(side_effect=capture_bind_tools)

    config = {"configurable": {
        "query": "back pain", "request_id": "test", "web_search": False,
    }}
    state = {"plan": {}, "expanded_query": "back pain", "memory_context": {}}

    with patch("langgraph_agents.nodes.retriever_agent._build_tools",
               new_callable=AsyncMock) as mock_build:
        mock_build.return_value = [mock_pgvector, mock_web_tool]
        with patch("langgraph_agents.nodes.retriever_agent.get_chat_model",
                   return_value=mock_llm):
            await retriever_agent_node(state, config)

    tool_names = [t.name for t in captured_tools]
    assert "pgvector_search" in tool_names
    assert "search_medical" not in tool_names


@pytest.mark.unit
@pytest.mark.asyncio
async def test_retriever_includes_web_tool_when_enabled():
    """When config web_search=True, search_medical IS in bound tools."""
    from langgraph_agents.nodes.retriever_agent import retriever_agent_node

    mock_web_tool = MagicMock()
    mock_web_tool.name = "search_medical"
    mock_pgvector = MagicMock()
    mock_pgvector.name = "pgvector_search"

    captured_tools = []

    mock_llm = MagicMock()
    mock_llm.ainvoke = AsyncMock(return_value=MagicMock(tool_calls=[], usage_metadata={}))

    def capture_bind_tools(tools):
        captured_tools.extend(tools)
        return mock_llm

    mock_llm.bind_tools = MagicMock(side_effect=capture_bind_tools)

    config = {"configurable": {
        "query": "back pain", "request_id": "test", "web_search": True,
    }}
    state = {"plan": {}, "expanded_query": "back pain", "memory_context": {}}

    with patch("langgraph_agents.nodes.retriever_agent._build_tools",
               new_callable=AsyncMock) as mock_build:
        mock_build.return_value = [mock_pgvector, mock_web_tool]
        with patch("langgraph_agents.nodes.retriever_agent.get_chat_model",
                   return_value=mock_llm):
            await retriever_agent_node(state, config)

    tool_names = [t.name for t in captured_tools]
    assert "pgvector_search" in tool_names
    assert "search_medical" in tool_names
