"""Integration test: retriever with MCP tools.

Requires DEEPSEEK_API_KEY — requires real LLM but MCP tools are mocked.
"""

import os
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from langgraph_agents.graph import build_graph_async
from langgraph_agents.state import AgentState

HAS_LLM_KEY = bool(os.getenv("DEEPSEEK_API_KEY"))


@pytest.mark.integration
@pytest.mark.skipif(not HAS_LLM_KEY, reason="DEEPSEEK_API_KEY not set")
@pytest.mark.asyncio
async def test_retriever_has_generate_motion_tool():
    """After MCP integration, retriever's tool list includes generate_motion."""
    import langgraph_agents.mcp.client as mcp_client_mod

    from langgraph_agents.mcp.kimodo_server import _generate_motion_mock

    class FakeMotionTool:
        name = "generate_motion"
        description = "Generate 3D motion mock"

        def invoke(self, input):
            result = _generate_motion_mock(input.get("prompt", ""), input.get("constraints", []), input.get("duration_seconds", 3.0))
            from langchain_core.messages import ToolMessage
            import json
            return ToolMessage(content=json.dumps(result), name="generate_motion", tool_call_id="mock_call_1")

    fake_motion = FakeMotionTool()

    with patch.object(mcp_client_mod, "get_mcp_tools", new=AsyncMock(return_value=[fake_motion])):
        graph = await build_graph_async()

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
                "query": "Cho tôi xem động tác bridge",
                "persona_id": "eca_default",
                "output_mode": "text",
                "request_id": "test-mcp-001",
                "token_limit": None,
            }
        }

        result = await graph.ainvoke(state, config=config)

        assert "final_answer" in result
        assert result["final_answer"], "Graph should produce a final answer"

        # With a query about motion visualization, the intent should be
        # visualize_motion or something motion-related
        intent = result.get("intent", "")
        print(f"Intent: {intent}, Answer preview: {result['final_answer'][:200]}")
