"""Web search MCP server (DuckDuckGo backend).

Free, no API key. Used by retriever_agent as fallback when pgvector returns
low-quality results or when planner explicitly requests web search.

Run standalone: python -m langgraph_agents.mcp.web_search_server
Run via stdio (default in tests): spawned by MultiServerMCPClient
"""

import asyncio
import json
from typing import Any
from urllib.parse import urlparse

from mcp.server import Server
from mcp.server.stdio import stdio_server
from mcp.types import Tool, TextContent


server = Server("web-search")


@server.list_tools()
async def list_tools() -> list[Tool]:
    return [
        Tool(
            name="search_medical",
            description=(
                "Web search via DuckDuckGo. Use as fallback when internal knowledge "
                "base (pgvector) is insufficient. Returns up to max_results items."
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
    results = await asyncio.to_thread(_search, **arguments)
    return [TextContent(type="text", text=json.dumps(results))]


def _search(query: str, max_results: int = 3, domain_filter: str | None = None) -> list[dict]:
    from duckduckgo_search import DDGS
    full_query = f"{query} {domain_filter}" if domain_filter else query
    out = []
    try:
        with DDGS() as ddgs:
            for r in ddgs.text(full_query, max_results=max_results):
                domain = urlparse(r.get("href", "")).netloc
                out.append({
                    "title": r.get("title", ""),
                    "snippet": r.get("body", ""),
                    "url": r.get("href", ""),
                    "source_domain": domain,
                })
    except Exception as exc:
        out.append({"error": f"DuckDuckGo search failed: {exc}"})
    return out


async def main():
    async with stdio_server() as (r, w):
        await server.run(r, w, server.create_initialization_options())


if __name__ == "__main__":
    asyncio.run(main())
