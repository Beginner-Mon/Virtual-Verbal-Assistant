"""Unit tests for MCP client wrapper."""

import sys
from unittest.mock import patch, AsyncMock, MagicMock

import pytest


def _reset_cache():
    import langgraph_agents.mcp.client as mod
    mod._mcp_tools = []
    mod._mcp_client = None


@pytest.mark.unit
@pytest.mark.asyncio
async def test_get_mcp_tools_caches():
    fake_tool = MagicMock()
    fake_tool.name = "fake_tool"

    mock_client = MagicMock()
    mock_client.get_tools = AsyncMock(return_value=[fake_tool])

    with patch("langgraph_agents.mcp.client.MultiServerMCPClient", return_value=mock_client):
        with patch("langgraph_agents.mcp.client._load_mcp_config", return_value={"test": {}}):
            import langgraph_agents.mcp.client as mcp_client_mod
            _reset_cache()

            tools1 = await mcp_client_mod.get_mcp_tools()
            tools2 = await mcp_client_mod.get_mcp_tools()

            assert len(tools1) == 1
            assert tools1 is tools2  # same list object = cached
            mock_client.get_tools.assert_awaited_once()


@pytest.mark.unit
@pytest.mark.asyncio
async def test_get_mcp_tools_empty_config():
    with patch("langgraph_agents.mcp.client._load_mcp_config", return_value={}):
        _reset_cache()
        from langgraph_agents.mcp.client import get_mcp_tools
        tools = await get_mcp_tools()
        assert tools == []


# ── Regression: subprocess Python interpreter substitution ─────────────────

@pytest.mark.unit
def test_normalize_stdio_substitutes_python_with_sys_executable():
    """Bare `python` command must be rewritten to current interpreter so the
    MCP subprocess inherits the venv (firstconda), not whatever first python is
    on PATH (often system Python lacking langgraph_agents)."""
    from langgraph_agents.mcp.client import _normalize_stdio_config

    cfg = {
        "transport": "stdio",
        "command": "python",
        "args": ["-m", "langgraph_agents.mcp.kimodo_server"],
    }
    _normalize_stdio_config(cfg)
    assert cfg["command"] == sys.executable, "command must be replaced with sys.executable"


@pytest.mark.unit
def test_normalize_stdio_substitutes_python3():
    """Also rewrite `python3`."""
    from langgraph_agents.mcp.client import _normalize_stdio_config
    cfg = {"transport": "stdio", "command": "python3", "args": []}
    _normalize_stdio_config(cfg)
    assert cfg["command"] == sys.executable


@pytest.mark.unit
def test_normalize_stdio_injects_pythonpath():
    """Subprocess needs PYTHONPATH=agenticRAG/agentic_rag_gemini so `python -m
    langgraph_agents.mcp.X` resolves."""
    from langgraph_agents.mcp.client import _normalize_stdio_config, _package_root
    cfg = {"transport": "stdio", "command": "python", "args": []}
    _normalize_stdio_config(cfg)
    assert "env" in cfg
    assert cfg["env"]["PYTHONPATH"] == _package_root()


@pytest.mark.unit
def test_normalize_stdio_preserves_custom_command():
    """Don't touch non-python commands or non-stdio transports."""
    from langgraph_agents.mcp.client import _normalize_stdio_config

    custom_stdio = {"transport": "stdio", "command": "/custom/path/bin", "args": []}
    _normalize_stdio_config(custom_stdio)
    assert custom_stdio["command"] == "/custom/path/bin"

    http_cfg = {"transport": "streamable_http", "url": "http://localhost:5001/mcp"}
    _normalize_stdio_config(http_cfg)
    assert "command" not in http_cfg  # untouched
    assert "env" not in http_cfg


@pytest.mark.unit
def test_normalize_stdio_does_not_override_explicit_env():
    """Caller can pre-set env values; defaults must not clobber them."""
    from langgraph_agents.mcp.client import _normalize_stdio_config
    cfg = {
        "transport": "stdio",
        "command": "python",
        "args": [],
        "env": {"PYTHONPATH": "/explicit/path"},
    }
    _normalize_stdio_config(cfg)
    assert cfg["env"]["PYTHONPATH"] == "/explicit/path"


# ── Regression: graceful fallback when MCP discovery fails ─────────────────

@pytest.mark.unit
@pytest.mark.asyncio
async def test_get_mcp_tools_falls_back_to_empty_on_error():
    """If MultiServerMCPClient.get_tools() raises (subprocess crash, McpError),
    the function must log a warning and return [] so the graph still builds
    with in-process tools only."""
    failing_client = MagicMock()
    failing_client.get_tools = AsyncMock(side_effect=RuntimeError("subprocess crashed"))

    with patch("langgraph_agents.mcp.client.MultiServerMCPClient", return_value=failing_client):
        with patch("langgraph_agents.mcp.client._load_mcp_config", return_value={"x": {}}):
            _reset_cache()
            from langgraph_agents.mcp.client import get_mcp_tools
            tools = await get_mcp_tools()
            assert tools == []


@pytest.mark.unit
@pytest.mark.asyncio
async def test_get_mcp_tools_recovers_after_failure():
    """After a failed discovery, the next call should retry (cache stays empty)."""
    import langgraph_agents.mcp.client as mod

    fail_client = MagicMock()
    fail_client.get_tools = AsyncMock(side_effect=RuntimeError("fail"))

    fake_tool = MagicMock()
    fake_tool.name = "ok_tool"
    ok_client = MagicMock()
    ok_client.get_tools = AsyncMock(return_value=[fake_tool])

    with patch("langgraph_agents.mcp.client._load_mcp_config", return_value={"x": {}}):
        # First call: fails
        with patch("langgraph_agents.mcp.client.MultiServerMCPClient", return_value=fail_client):
            _reset_cache()
            t1 = await mod.get_mcp_tools()
            assert t1 == []

        # Second call: succeeds (cache is empty after failure, retries)
        with patch("langgraph_agents.mcp.client.MultiServerMCPClient", return_value=ok_client):
            t2 = await mod.get_mcp_tools()
            assert len(t2) == 1
            assert t2[0].name == "ok_tool"
