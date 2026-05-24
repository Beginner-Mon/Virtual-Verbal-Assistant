"""Unit tests for MCP client wrapper."""

from unittest.mock import patch, AsyncMock, MagicMock

import pytest


@pytest.mark.unit
@pytest.mark.asyncio
async def test_get_mcp_tools_caches():
    fake_tool = MagicMock()
    fake_tool.name = "fake_tool"

    mock_client = MagicMock()
    mock_client.get_tools = AsyncMock(return_value=[fake_tool])

    with patch("langgraph_agents.mcp.client.MultiServerMCPClient", return_value=mock_client):
        with patch("langgraph_agents.mcp.client._load_mcp_config", return_value={"test": {}}):
            # Clear cache
            import langgraph_agents.mcp.client as mcp_client_mod
            mcp_client_mod._mcp_tools = []
            mcp_client_mod._mcp_client = None

            tools1 = await mcp_client_mod.get_mcp_tools()
            tools2 = await mcp_client_mod.get_mcp_tools()

            assert len(tools1) == 1
            assert tools1 is tools2  # same list object = cached
            mock_client.get_tools.assert_awaited_once()


@pytest.mark.unit
@pytest.mark.asyncio
async def test_get_mcp_tools_empty_config():
    with patch("langgraph_agents.mcp.client._load_mcp_config", return_value={}):
        import langgraph_agents.mcp.client as mcp_client_mod
        mcp_client_mod._mcp_tools = []
        mcp_client_mod._mcp_client = None

        tools = await mcp_client_mod.get_mcp_tools()
        assert tools == []
