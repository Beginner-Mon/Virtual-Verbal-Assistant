"""MultiServerMCPClient wrapper — lazy singleton.

Loads config/mcp_servers.yaml on first call, builds the client, returns the
list of LangChain tools discovered from all MCP servers. Cached afterwards.

Two production-critical concerns handled here:

1. **stdio subprocess Python interpreter** — the YAML config uses
   `command: python` for portability, but bare `python` resolves against the
   process PATH and frequently picks the system Python (which lacks
   `langgraph_agents`). We rewrite the command to `sys.executable` and inject
   `PYTHONPATH` pointing at `agenticRAG/` so the subprocess can import the
   server modules. (It used to say `agenticRAG/agentic_rag_gemini/`; the code
   stopped doing that when the MCP servers moved into this package, but the
   comment did not.)

2. **Graceful degradation** — if MCP discovery fails (subprocess crash, network
   timeout, package missing), the graph still builds with only in-process tools
   (`pgvector_search`). Retriever skips MCP-only intents (e.g. visualize_motion);
   grader catches the missing ToolMessage and routes to clarify.
"""

import asyncio
import os
import sys
from pathlib import Path

import yaml

from langchain_mcp_adapters.client import MultiServerMCPClient
from langgraph_agents.core.circuit_breaker import CircuitBreaker

from langgraph_agents.shared.logging import get_logger

logger = get_logger("langgraph.mcp.client")

_mcp_client = None
_mcp_tools: list = []
_init_lock = asyncio.Lock()

_mcp_breaker = CircuitBreaker(
    name="mcp_discovery",
    failure_threshold=2,
    cool_down_seconds=60,
)

# Env vars worth inheriting into MCP subprocesses (avoid leaking everything).
_ENV_PASSTHROUGH = (
    "PATH", "HOME", "USERPROFILE", "TEMP", "TMP", "SYSTEMROOT", "APPDATA",
    "LOCALAPPDATA", "HF_TOKEN", "SEARXNG_URL",
)


def _package_root() -> str:
    """Return absolute path of agenticRAG for PYTHONPATH.
    
    Since langgraph_agents is now fully independent, we only need agenticRAG.
    """
    return str(Path(__file__).resolve().parents[2])


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


def mcp_enabled() -> bool:
    """Whether to discover MCP servers at all. Default on, so local is unchanged.

    The deployed agent sets ENABLE_MCP=false, and the reason is cold-start cost
    rather than tidiness. Discovery runs inside build_graph_async(), i.e. in the
    FastAPI lifespan, i.e. inside Lambda's 10-second INIT budget — and both
    configured servers use `transport: stdio`, so it spawns TWO Python
    subprocesses there. On the first cloud deployment neither can do anything:
    mcp/kimodo_server.py is a mock returning mock:// URLs, and
    web_search_server.py wants SearXNG on localhost:6666. Paying seconds of every
    cold start to start two processes that cannot work is worse than not having
    them.

    Turning Kimodo on later is this flag plus a `streamable_http` entry in
    config/mcp_servers.yaml pointing at the tunnel — no code change here.
    """
    return os.getenv("ENABLE_MCP", "true").strip().lower() != "false"


def _load_mcp_config() -> dict:
    if not mcp_enabled():
        # Empty config, which get_mcp_tools() already short-circuits on. Reusing
        # that path rather than adding a second early return means the "no tools"
        # case has one shape, and the graph keeps building with its in-process
        # tools exactly as it does when discovery fails.
        logger.info("mcp_disabled", extra={"reason": "ENABLE_MCP=false"})
        return {}

    config_path = Path(__file__).resolve().parents[3] / "config" / "mcp_servers.yaml"
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

        if not _mcp_breaker.allow():
            logger.warning("mcp_discovery_skipped_breaker_open")
            return []

        cfg = _load_mcp_config()
        if not cfg:
            return []

        try:
            _mcp_client = MultiServerMCPClient(cfg)
            _mcp_tools = await _mcp_client.get_tools()
            _mcp_breaker.record_success()
            logger.info("mcp_discovery_ok", extra={
                "tool_count": len(_mcp_tools), "server_count": len(cfg),
            })
            return _mcp_tools
        except Exception as exc:
            _mcp_breaker.record_failure()
            logger.warning("mcp_discovery_failed", extra={"error": str(exc)})
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
