"""ENABLE_MCP — the switch that lets the agent deploy before Kimodo exists.

Owner's scope, 21-08: get langgraph-agent into production first; TTS, Kimodo and
the cache are supporting features that can follow on another branch. That only
works if each of them can be switched off without the agent noticing, and MCP
was the one seam that had no switch.

The cost it removes is not hypothetical. `build_graph_async()` calls
`get_mcp_tools()` unconditionally, from the FastAPI lifespan — which on Lambda is
inside a 10-second INIT budget. Both configured servers are `transport: stdio`,
so discovery spawns TWO Python subprocesses there, and on the first cloud deploy
neither can do anything: kimodo_server is a mock returning mock:// URLs and
web_search_server wants SearXNG on localhost.

Default is ON so local development is unchanged; the Lambda image sets it off.
"""

from __future__ import annotations

import pytest

from langgraph_agents.mcp import client as mcp_client


@pytest.mark.unit
def test_enabled_by_default(monkeypatch):
    """An unset variable must keep local development working as before."""
    monkeypatch.delenv("ENABLE_MCP", raising=False)
    assert mcp_client.mcp_enabled() is True


@pytest.mark.unit
@pytest.mark.parametrize("value,expected", [
    ("false", False),
    ("FALSE", False),
    ("  false  ", False),
    ("true", True),
    ("", True),
    ("0", True),          # see the docstring below
])
def test_only_false_switches_it_off(monkeypatch, value, expected):
    """Only the word "false" disables it — "0" does not.

    The opposite convention to ENABLE_GDPR_ROUTES, which accepts only "true",
    and deliberately so: these two fail in opposite directions. Reading a GDPR
    flag too generously exposes an irreversible delete; reading this one too
    generously silently removes tool discovery from a deployment that wanted it.
    Each errs toward the state that is safe for it — GDPR toward off, MCP toward
    on.
    """
    monkeypatch.setenv("ENABLE_MCP", value)
    assert mcp_client.mcp_enabled() is expected


@pytest.mark.unit
def test_disabled_yields_empty_config_without_touching_the_yaml(monkeypatch):
    """No config means no subprocess — that IS the saving.

    Asserted through _load_mcp_config rather than by counting processes: it is
    the function that decides, and get_mcp_tools() already returns [] for an
    empty config, which is the path a discovery failure takes too.
    """
    monkeypatch.setenv("ENABLE_MCP", "false")
    assert mcp_client._load_mcp_config() == {}


@pytest.mark.unit
def test_enabled_still_reads_the_real_servers(monkeypatch):
    """The switch must be reversible, or it is just a deletion.

    Turning Kimodo on later should be this flag plus a streamable_http entry —
    so the enabled path has to keep loading config/mcp_servers.yaml. Without
    this test the other three would pass just as well if _load_mcp_config always
    returned {}.
    """
    monkeypatch.setenv("ENABLE_MCP", "true")
    cfg = mcp_client._load_mcp_config()
    assert cfg, "config/mcp_servers.yaml produced no servers with MCP enabled"
    # kimodo_motion was removed from config/mcp_servers.yaml in Task 8 — the
    # LLM never chooses the motion tool (D26: needs_motion is a hard edge),
    # so MCP's tool-discovery bought nothing on that path. web_search is the
    # entry that survives: the LLM does choose that one.
    assert "web_search" in cfg
