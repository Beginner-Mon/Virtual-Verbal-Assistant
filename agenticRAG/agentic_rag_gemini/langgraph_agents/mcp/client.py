"""MultiServerMCPClient wrapper — lazy singleton.

Loads config/mcp_servers.yaml on first call, builds the client, returns the
list of LangChain tools discovered from all MCP servers. Cached afterwards.

Two production-critical concerns handled here:

1. **stdio subprocess Python interpreter** — the YAML config uses
   `command: python` for portability, but bare `python` resolves against the
   process PATH and frequently picks the system Python (which lacks
   `langgraph_agents`). We rewrite the command to `sys.executable` and inject
   `PYTHONPATH` pointing at `agenticRAG/agentic_rag_gemini/` so the subprocess
   can import the server modules.

2. **Graceful degradation** — if MCP discovery fails (subprocess crash, network
   timeout, package missing), the graph still builds with only in-process tools
   (`pgvector_search`). Retriever skips MCP-only intents (e.g. visualize_motion);
   grader catches the missing ToolMessage and routes to clarify.
"""

import asyncio
import logging
import os
import sys
from pathlib import Path

import yaml

from langchain_mcp_adapters.client import MultiServerMCPClient

logger = logging.getLogger("langgraph.mcp.client")

_mcp_client = None
_mcp_tools: list = []
_init_lock = asyncio.Lock()

# Env vars worth inheriting into MCP subprocesses (avoid leaking everything).
_ENV_PASSTHROUGH = (
    "PATH", "HOME", "USERPROFILE", "TEMP", "TMP", "SYSTEMROOT", "APPDATA",
    "LOCALAPPDATA", "HF_TOKEN",
)


def _package_root() -> str:
    """Return absolute path of `agenticRAG/agentic_rag_gemini/` for PYTHONPATH."""
    return str(Path(__file__).resolve().parents[3])


def _normalize_stdio_config(server_cfg: dict) -> None:
    """Mutate one server config dict in place:
       - rewrite `python` → `sys.executable`
       - inject PYTHONPATH so subprocess can import langgraph_agents
       - pass through critical env vars from parent process
    """
    if server_cfg.get("transport") != "stdio":
        return
    if server_cfg.get("command") in ("python", "python3"):
        server_cfg["command"] = sys.executable
    env = server_cfg.setdefault("env", {})
    env.setdefault("PYTHONPATH", _package_root())
    for var in _ENV_PASSTHROUGH:
        if var in os.environ:
            env.setdefault(var, os.environ[var])


def _load_mcp_config() -> dict:
    config_path = Path(__file__).resolve().parents[4] / "config" / "mcp_servers.yaml"
    if not config_path.exists():
        return {}
    with open(config_path, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f).get("mcp_servers", {}) or {}
    for server_cfg in cfg.values():
        _normalize_stdio_config(server_cfg)
    return cfg


async def get_mcp_tools() -> list:
    """Discover tools from all configured MCP servers (idempotent, cached).

    Never raises — on failure, returns [] and logs a warning. Caller (retriever
    + graph) treats MCP as best-effort; in-process tools always work.
    """
    global _mcp_client, _mcp_tools
    if _mcp_tools:
        return _mcp_tools

    async with _init_lock:
        if _mcp_tools:
            return _mcp_tools

        cfg = _load_mcp_config()
        if not cfg:
            return []

        try:
            _mcp_client = MultiServerMCPClient(cfg)
            _mcp_tools = await _mcp_client.get_tools()
            logger.info("MCP discovery OK: %d tool(s) loaded from %d server(s)",
                        len(_mcp_tools), len(cfg))
            return _mcp_tools
        except Exception as exc:
            logger.warning(
                "MCP discovery failed (graph will run with in-process tools only): %s",
                exc,
            )
            _mcp_client = None
            _mcp_tools = []
            return []


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
