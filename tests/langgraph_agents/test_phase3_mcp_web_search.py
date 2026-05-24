"""Integration tests for DuckDuckGo web search MCP server."""

from unittest.mock import patch, MagicMock

import pytest

from langgraph_agents.mcp.web_search_server import _search, list_tools


@pytest.mark.integration
@pytest.mark.asyncio
async def test_web_search_list_tools():
    tools = await list_tools()
    names = [t.name for t in tools]
    assert "search_medical" in names


@pytest.mark.integration
def test_web_search_returns_results():
    try:
        from duckduckgo_search import DDGS
    except ImportError:
        pytest.skip("duckduckgo-search not installed")

    results = _search("back pain exercises", max_results=2)
    assert isinstance(results, list)
    assert len(results) >= 1
    assert "title" in results[0]
    assert "url" in results[0]


@pytest.mark.unit
def test_web_search_handles_failure():
    try:
        from duckduckgo_search import DDGS
    except ImportError:
        pytest.skip("duckduckgo-search not installed")

    with patch("duckduckgo_search.DDGS") as mock_ddgs:
        mock_ddgs.return_value.__enter__.return_value.text.side_effect = RuntimeError("network down")
        results = _search("back pain exercises")
        assert len(results) == 1
        assert "error" in results[0]
        assert "network down" in results[0]["error"]
