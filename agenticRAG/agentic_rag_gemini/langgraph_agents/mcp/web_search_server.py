"""Web search MCP server (SearXNG backend).

Self-hosted metasearch aggregator (Google + Bing + DDG + Wikipedia).
No API key required. Used by retriever_agent as fallback when pgvector
returns low-quality results or when planner explicitly requests web search.

Run standalone: python -m langgraph_agents.mcp.web_search_server
Run via stdio (default in tests): spawned by MultiServerMCPClient
"""

import asyncio
import json
import os
from typing import Any

import httpx
from mcp.server import Server
from mcp.server.stdio import stdio_server
from mcp.types import Tool, TextContent

from langgraph_agents.shared.logging import get_logger

logger = get_logger("langgraph.mcp.web_search")

SEARXNG_URL = os.getenv("SEARXNG_URL", "http://localhost:6666")
DEFAULT_TIMEOUT = 10.0

server = Server("web-search")


@server.list_tools()
async def list_tools() -> list[Tool]:
    return [
        Tool(
            name="search_medical",
            description=(
                "Web search via SearXNG metasearch (Google+Bing+DDG+Wikipedia). "
                "Use as fallback when internal knowledge base (pgvector) is "
                "insufficient. Returns up to max_results items."
            ),
            inputSchema={
                "type": "object",
                "properties": {
                    "query": {"type": "string"},
                    "max_results": {"type": "integer", "default": 3, "minimum": 1, "maximum": 10},
                    "domain_filter": {"type": "string", "description": "optional site: filter, e.g. site:pubmed.ncbi.nlm.nih.gov"},
                },
                "required": ["query"],
            },
        )
    ]


@server.call_tool()
async def call_tool(name: str, arguments: dict[str, Any]) -> list[TextContent]:
    if name != "search_medical":
        raise ValueError(f"Unknown tool: {name}")
    results = await _search_searxng(**arguments)
    return [TextContent(type="text", text=json.dumps(results))]


async def _search_searxng(query: str, max_results: int = 3, domain_filter: str | None = None) -> list[dict]:
    """Query self-hosted SearXNG, return normalized result list.

    Output shape preserved from DDG version so retriever/synthesizer
    code stays unchanged.
    """
    full_query = f"{query} {domain_filter}" if domain_filter else query
    try:
        async with httpx.AsyncClient(timeout=DEFAULT_TIMEOUT) as client:
            resp = await client.get(
                f"{SEARXNG_URL}/search",
                params={
                    "q": full_query,
                    "format": "json",
                    "categories": "general,science",
                },
            )
        resp.raise_for_status()
        data = resp.json()
        return [
            {
                "title": r.get("title", ""),
                "snippet": r.get("content", ""),
                "url": r.get("url", ""),
                "source_domain": r.get("engine", "unknown"),
            }
            for r in data.get("results", [])[:max_results]
        ]
    except httpx.ConnectError:
        logger.warning("search_failed", extra={"error": "searxng_connect_refused", "url": SEARXNG_URL})
        return []
    except httpx.HTTPStatusError as exc:
        logger.warning("search_failed", extra={"error": f"searxng_http_{exc.response.status_code}", "url": SEARXNG_URL})
        return []
    except Exception as exc:
        logger.warning("search_failed", extra={"error": str(exc)})
        return []


async def main():
    async with stdio_server() as (r, w):
        await server.run(r, w, server.create_initialization_options())


if __name__ == "__main__":
    asyncio.run(main())
