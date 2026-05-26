"""Integration tests for SearXNG web search MCP server."""

import json
from unittest.mock import AsyncMock, MagicMock, patch

import httpx
import pytest

from langgraph_agents.mcp.web_search_server import _search_searxng, list_tools


@pytest.mark.integration
@pytest.mark.asyncio
async def test_web_search_list_tools():
    tools = await list_tools()
    names = [t.name for t in tools]
    assert "search_medical" in names


@pytest.mark.unit
@pytest.mark.asyncio
async def test_web_search_returns_results():
    """Patch SearXNG response with 3 fake hits, verify normalized output shape."""
    fake_response = MagicMock()
    fake_response.raise_for_status = MagicMock()
    fake_response.json.return_value = {
        "results": [
            {"title": "Back Pain Exercises", "content": "Gentle stretches for lower back.", "url": "https://pubmed.ncbi.nlm.nih.gov/1", "engine": "pubmed"},
            {"title": "Physical Therapy Guide", "content": "Evidence-based PT for back pain.", "url": "https://example.com/pt", "engine": "google"},
            {"title": "WebMD Back Pain", "content": "Common causes and treatments.", "url": "https://webmd.com/back", "engine": "bing"},
        ]
    }

    with patch("httpx.AsyncClient.get", new_callable=AsyncMock) as mock_get:
        mock_get.return_value = fake_response
        results = await _search_searxng("back pain exercises", max_results=2)

    assert isinstance(results, list)
    assert len(results) == 2
    assert results[0]["title"] == "Back Pain Exercises"
    assert results[0]["snippet"] == "Gentle stretches for lower back."
    assert results[0]["url"] == "https://pubmed.ncbi.nlm.nih.gov/1"
    assert results[0]["source_domain"] == "pubmed"


@pytest.mark.unit
@pytest.mark.asyncio
async def test_web_search_handles_searxng_down():
    """When SearXNG is unreachable, return [] (graceful degradation)."""
    with patch("httpx.AsyncClient.get", new_callable=AsyncMock) as mock_get:
        mock_get.side_effect = httpx.ConnectError("connection refused")
        results = await _search_searxng("back pain exercises")

    assert isinstance(results, list)
    assert len(results) == 0


@pytest.mark.unit
@pytest.mark.asyncio
async def test_web_search_handles_http_error():
    """When SearXNG returns 5xx, return []."""
    with patch("httpx.AsyncClient.get", new_callable=AsyncMock) as mock_get:
        mock_get.side_effect = httpx.HTTPStatusError(
            "server error",
            request=MagicMock(),
            response=MagicMock(status_code=502),
        )
        results = await _search_searxng("back pain exercises")

    assert isinstance(results, list)
    assert len(results) == 0
