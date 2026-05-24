"""MultiServerMCPClient wrapper — lazy singleton.

Loads config/mcp_servers.yaml on first call, builds the client, returns the
list of LangChain tools discovered from all MCP servers. Cached afterwards.
"""

import asyncio
import logging
from pathlib import Path

import yaml

from langchain_mcp_adapters.client import MultiServerMCPClient

logger = logging.getLogger("langgraph.mcp.client")

_mcp_client = None
_mcp_tools: list = []
_init_lock = asyncio.Lock()


def _load_mcp_config() -> dict:
    config_path = Path(__file__).resolve().parents[4] / "config" / "mcp_servers.yaml"
    if not config_path.exists():
        return {}
    with open(config_path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f).get("mcp_servers", {})


async def get_mcp_tools() -> list:
    """Discover tools from all configured MCP servers (idempotent, cached)."""
    global _mcp_client, _mcp_tools
    if _mcp_tools:
        return _mcp_tools

    async with _init_lock:
        if _mcp_tools:
            return _mcp_tools

        cfg = _load_mcp_config()
        if not cfg:
            return []

        _mcp_client = MultiServerMCPClient(cfg)
        _mcp_tools = await _mcp_client.get_tools()
        return _mcp_tools


async def close_mcp_client():
    global _mcp_client, _mcp_tools
    if _mcp_client is not None:
        for attr in ("aclose", "close"):
            fn = getattr(_mcp_client, attr, None)
            if fn:
                try:
                    res = fn()
                    if asyncio.iscoroutine(res):
                        await res
                    break
                except Exception:
                    pass
        _mcp_client = None
        _mcp_tools = []
