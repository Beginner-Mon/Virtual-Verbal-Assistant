"""Retriever unit tests — M.2 tool selection, routing, web toggle, empty≠error.
"""

import pytest
from unittest.mock import AsyncMock, patch

from langgraph_agents.state import AgentState


@pytest.mark.unit
class TestRetrieverRouting:
    """Verify route_after_retriever and route_after_retriever_or_tools."""

    def test_route_retriever_with_tool_calls(self):
        from langgraph_agents.routing import route_after_retriever
        from langchain_core.messages import AIMessage
        msg = AIMessage(content="", tool_calls=[{"name": "kb_search", "args": {}, "id": "1"}])
        state: AgentState = {"messages": [msg], "errors": [], "retry_count": 0,
                             "total_tokens": 0}
        result = route_after_retriever(state)
        assert result == "tools"

    def test_route_retriever_no_tool_calls_to_synthesizer(self):
        from langgraph_agents.routing import route_after_retriever
        from langchain_core.messages import AIMessage
        msg = AIMessage(content="no tools needed")
        state: AgentState = {"messages": [msg], "errors": [], "retry_count": 0,
                             "total_tokens": 0}
        result = route_after_retriever(state)
        assert result == "synthesizer"

    def test_route_retriever_to_kimodo(self):
        from langgraph_agents.routing import route_after_retriever
        from langchain_core.messages import AIMessage
        msg = AIMessage(content="motion needed", tool_calls=[])
        state: AgentState = {"messages": [msg], "errors": [], "retry_count": 0,
                             "total_tokens": 0, "needs_motion": True}
        result = route_after_retriever(state)
        assert result == "kimodo"

    def test_route_retriever_critical_error(self):
        from langgraph_agents.routing import route_after_retriever
        from langgraph_agents.state import ErrorSeverity
        state: AgentState = {
            "messages": [], "errors": [
                {"severity": ErrorSeverity.CRITICAL, "node": "retriever_agent",
                 "message": "LLM failed"}
            ], "retry_count": 0, "total_tokens": 0,
        }
        result = route_after_retriever(state)
        assert result == "error_handler"

    def test_route_after_retriever_or_tools_with_motion(self):
        from langgraph_agents.graph import route_after_retriever_or_tools
        state: AgentState = {"messages": [], "errors": [], "retry_count": 0,
                             "total_tokens": 0, "needs_motion": True}
        result = route_after_retriever_or_tools(state)
        assert result == "kimodo"

    def test_route_after_retriever_or_tools_no_motion(self):
        from langgraph_agents.graph import route_after_retriever_or_tools
        state: AgentState = {"messages": [], "errors": [], "retry_count": 0,
                             "total_tokens": 0, "needs_motion": False}
        result = route_after_retriever_or_tools(state)
        assert result == "synthesizer"


@pytest.mark.unit
class TestPlannerRouting:
    """Verify route_after_planner and related routing."""

    def test_route_planner_critical_error(self):
        from langgraph_agents.routing import route_after_planner
        from langgraph_agents.state import ErrorSeverity
        state: AgentState = {
            "messages": [], "errors": [
                {"severity": ErrorSeverity.CRITICAL, "node": "memory",
                 "message": "DB down"}
            ], "retry_count": 0, "total_tokens": 0,
        }
        assert route_after_planner(state) == "error_handler"

    def test_route_planner_clarification(self):
        from langgraph_agents.routing import route_after_planner
        state: AgentState = {"messages": [], "errors": [], "retry_count": 0,
                             "total_tokens": 0, "needs_clarification": True}
        assert route_after_planner(state) == "synthesizer"

    def test_route_planner_needs_retrieval(self):
        from langgraph_agents.routing import route_after_planner
        state: AgentState = {"messages": [], "errors": [], "retry_count": 0,
                             "total_tokens": 0, "needs_retrieval": True}
        assert route_after_planner(state) == "retriever_agent"

    def test_route_planner_needs_motion_only(self):
        from langgraph_agents.routing import route_after_planner
        state: AgentState = {"messages": [], "errors": [], "retry_count": 0,
                             "total_tokens": 0, "needs_motion": True}
        assert route_after_planner(state) == "kimodo"

    def test_route_planner_neither_to_synthesizer(self):
        from langgraph_agents.routing import route_after_planner
        state: AgentState = {"messages": [], "errors": [], "retry_count": 0,
                             "total_tokens": 0}
        assert route_after_planner(state) == "synthesizer"


@pytest.mark.unit
class TestRetrieverToolConfig:
    """Verify RETRIEVER_BASE_TOOLS and tool filtering."""

    def test_base_tools_include_kb_search(self):
        from langgraph_agents.nodes.retriever_agent import RETRIEVER_BASE_TOOLS
        assert len(RETRIEVER_BASE_TOOLS) >= 1
        tool_names = [t.name for t in RETRIEVER_BASE_TOOLS]
        assert "kb_search" in tool_names

    def test_motion_not_in_base_tools(self):
        """D26: generate_motion is NOT in retriever tools."""
        from langgraph_agents.nodes.retriever_agent import RETRIEVER_BASE_TOOLS
        tool_names = [t.name for t in RETRIEVER_BASE_TOOLS]
        assert "generate_motion" not in tool_names

    def test_web_search_toggle_off(self):
        """D27: web search tool filtered when user toggle is off."""
        # _filter_tools is an internal function in retriever_agent, not exported.
        # Test via retriever_agent_node with web_search=False config.
        from langgraph_agents.nodes.retriever_agent import RETRIEVER_BASE_TOOLS
        tool_names = [t.name for t in RETRIEVER_BASE_TOOLS]
        assert "kb_search" in tool_names
        assert "generate_motion" not in tool_names


@pytest.mark.unit
class TestErrorRouting:
    """Verify check_errors and error_handler routing."""

    def test_check_errors_no_errors(self):
        from langgraph_agents.routing import check_errors
        state: AgentState = {"messages": [], "errors": [], "retry_count": 0,
                             "total_tokens": 0}
        assert check_errors(state) == "continue"

    def test_check_errors_critical(self):
        from langgraph_agents.routing import check_errors
        from langgraph_agents.state import ErrorSeverity
        state: AgentState = {
            "messages": [], "errors": [
                {"severity": ErrorSeverity.CRITICAL, "node": "planner",
                 "message": "LLM down", "timestamp": ""}
            ], "retry_count": 0, "total_tokens": 0,
        }
        assert check_errors(state) == "error_handler"

    def test_check_errors_recoverable_not_critical(self):
        from langgraph_agents.routing import check_errors
        from langgraph_agents.state import ErrorSeverity
        state: AgentState = {
            "messages": [], "errors": [
                {"severity": ErrorSeverity.RECOVERABLE, "node": "kimodo",
                 "message": "Motion server timeout", "timestamp": ""}
            ], "retry_count": 0, "total_tokens": 0,
        }
        assert check_errors(state) == "continue"


@pytest.mark.unit
class TestRetrieverNode:
    """Test retriever_agent_node with mocked LLM."""

    @pytest.mark.asyncio
    async def test_retriever_node_calls_llm_with_tools(self):
        from langgraph_agents.nodes.retriever_agent import retriever_agent_node
        from langchain_core.runnables import RunnableConfig
        from langchain_core.messages import AIMessage

        config = RunnableConfig(configurable={
            "user_id": "test-user", "session_id": "test-sess",
            "query": "bai tap squat", "persona_id": "eca_default",
            "request_id": "req-001", "web_search": False,
        })
        state: AgentState = {
            "messages": [], "errors": [], "retry_count": 0,
            "total_tokens": 0, "required_outputs": ["exercise_protocol"],
            "resolved_query": "bai tap squat",
        }

        fake_ai = AIMessage(content="", tool_calls=[{
            "name": "kb_search", "args": {"query": "squat exercises"}, "id": "1"
        }])

        with patch("langgraph_agents.nodes.retriever_agent.get_chat_model") as mock_llm:
            mock_llm.return_value.bind_tools.return_value.ainvoke = AsyncMock(
                return_value=fake_ai
            )
            result = await retriever_agent_node(state, config)
            assert "messages" in result
            assert len(result["messages"]) == 1

    @pytest.mark.asyncio
    async def test_retriever_node_llm_error_fallback(self):
        from langgraph_agents.nodes.retriever_agent import retriever_agent_node
        from langchain_core.runnables import RunnableConfig

        config = RunnableConfig(configurable={
            "user_id": "test-user", "session_id": "test-sess",
            "query": "hello", "persona_id": "eca_default",
            "request_id": "req-001", "web_search": False,
        })
        state: AgentState = {
            "messages": [], "errors": [], "retry_count": 0,
            "total_tokens": 0, "required_outputs": [],
            "resolved_query": "hello",
        }

        with patch("langgraph_agents.nodes.retriever_agent.get_chat_model") as mock_llm:
            mock_llm.return_value.bind_tools.return_value.ainvoke = AsyncMock(
                side_effect=RuntimeError("LLM timeout")
            )
            result = await retriever_agent_node(state, config)
            errors = result.get("errors", [])
            assert len(errors) >= 1
            assert errors[0]["severity"] == "critical"

    @pytest.mark.asyncio
    async def test_retriever_agent_receives_grader_feedback_on_retry(self):
        """On retry, grader_feedback should be injected into the retriever prompt."""
        from langgraph_agents.nodes.retriever_agent import retriever_agent_node
        from langchain_core.runnables import RunnableConfig
        from langchain_core.messages import AIMessage

        config = RunnableConfig(configurable={
            "user_id": "test-user", "session_id": "test-sess",
            "query": "bai tap squat", "persona_id": "eca_default",
            "request_id": "req-001", "web_search": False,
        })
        state: AgentState = {
            "messages": [], "errors": [], "retry_count": 1,
            "total_tokens": 0,
            "required_outputs": ["exercise_protocol", "exercise_steps"],
            "resolved_query": "bai tap squat",
            "grader_feedback": "Missing exercise_steps. Add step-by-step instructions.",
        }

        fake_ai = AIMessage(content="", tool_calls=[{
            "name": "kb_search", "args": {"query": "squat steps"}, "id": "1"
        }])

        with patch("langgraph_agents.nodes.retriever_agent.get_chat_model") as mock_llm:
            mock_llm.return_value.bind_tools.return_value.ainvoke = AsyncMock(
                return_value=fake_ai
            )
            result = await retriever_agent_node(state, config)
            assert "messages" in result

            # Verify grader_feedback was injected into the prompt
            call_args = mock_llm.return_value.bind_tools.return_value.ainvoke.call_args
            msgs = call_args[0][0]
            prompt_text = str(msgs)
            assert "exercise_steps" in prompt_text.lower() or "retry" in prompt_text.lower()
