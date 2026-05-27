"""Kimodo MCP server — mock mode for Phase 3.

Plug NVIDIA Kimodo real inference behind _generate_motion_real() when GPU is ready.

Run standalone (HTTP): python -m langgraph_agents.mcp.kimodo_server
Run via stdio (default in tests): spawned by MultiServerMCPClient
"""

import asyncio
import hashlib
import json
from typing import Any

from mcp.server import Server
from mcp.server.stdio import stdio_server
from mcp.types import Tool, TextContent


_USE_REAL = False  # flip when NVIDIA Kimodo wrapper is ready


server = Server("kimodo-motion")


@server.list_tools()
async def list_tools() -> list[Tool]:
    return [
        Tool(
            name="generate_motion",
            description=(
                "Generate a 3D motion animation from a natural language description "
                "with optional joint constraints. Use ONLY for visualize_motion intent."
            ),
            inputSchema={
                "type": "object",
                "properties": {
                    "prompt": {"type": "string", "description": "motion description"},
                    "constraints": {
                        "type": "array",
                        "items": {
                            "type": "object",
                            "properties": {
                                "joint": {"type": "string"},
                                "angle": {"type": "number"},
                            },
                        },
                        "default": [],
                    },
                    "duration_seconds": {"type": "number", "default": 3.0},
                },
                "required": ["prompt"],
            },
        )
    ]


@server.call_tool()
async def call_tool(name: str, arguments: dict[str, Any]) -> list[TextContent]:
    if name != "generate_motion":
        raise ValueError(f"Unknown tool: {name}")
    result = await _generate_motion(**arguments)
    return [TextContent(type="text", text=json.dumps(result))]


async def _generate_motion(prompt: str, constraints: list[dict] | None = None,
                            duration_seconds: float = 3.0) -> dict:
    if _USE_REAL:
        return await _generate_motion_real(prompt, constraints, duration_seconds)
    return _generate_motion_mock(prompt, constraints, duration_seconds)


def _generate_motion_mock(prompt: str, constraints, duration_seconds) -> dict:
    digest = hashlib.sha1(prompt.encode("utf-8")).hexdigest()[:12]
    joints = [c.get("joint") for c in (constraints or []) if c.get("joint")]
    return {
        "video_url": f"mock://motion/{digest}.mp4",
        "duration_sec": float(duration_seconds),
        "format": "mp4",
        "joints_used": joints,
        "_mock": True,
        "_prompt_echo": prompt[:200],
    }


async def _generate_motion_real(prompt, constraints, duration_seconds) -> dict:
    raise NotImplementedError("Set _USE_REAL=True after wiring NVIDIA Kimodo inference here.")


async def main():
    async with stdio_server() as (read_stream, write_stream):
        await server.run(
            read_stream, write_stream,
            server.create_initialization_options(),
        )


if __name__ == "__main__":
    asyncio.run(main())
