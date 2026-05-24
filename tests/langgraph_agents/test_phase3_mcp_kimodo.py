"""Unit tests for Kimodo MCP server (mock mode)."""

import json

import pytest

from langgraph_agents.mcp.kimodo_server import _generate_motion_mock, list_tools


@pytest.mark.unit
@pytest.mark.asyncio
async def test_kimodo_list_tools():
    tools = await list_tools()
    names = [t.name for t in tools]
    assert "generate_motion" in names


@pytest.mark.unit
def test_kimodo_mock_returns_url():
    result = _generate_motion_mock("bridge pose", None, 3.0)
    assert result["_mock"] is True
    assert result["video_url"].startswith("mock://motion/")
    assert result["video_url"].endswith(".mp4")
    assert result["format"] == "mp4"
    assert result["duration_sec"] == 3.0


@pytest.mark.unit
def test_kimodo_constraints_passthrough():
    constraints = [
        {"joint": "spine", "angle": 45},
        {"joint": "hip", "angle": 30},
    ]
    result = _generate_motion_mock("bridge pose", constraints, 3.0)
    assert result["joints_used"] == ["spine", "hip"]


@pytest.mark.unit
def test_kimodo_empty_constraints():
    result = _generate_motion_mock("hello world", [], 5.0)
    assert result["joints_used"] == []
    assert result["duration_sec"] == 5.0
