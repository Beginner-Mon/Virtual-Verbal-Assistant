"""Tests for kimodo_node — the wire, not the thing at the far end of it.

Until 21-08 this node had no tests at all. test_phase3_mcp_kimodo.py exercised
the mock MCP server directly, which is a different piece of code: it called
`_generate_motion_mock(prompt=...)` and passed, while the node called
`ainvoke({"query": ...})` and failed schema validation on every single request.
The failure was swallowed into a RECOVERABLE error, so motion silently never ran
and nothing went red.

That is why the fixture below builds a REAL StructuredTool with a real pydantic
schema instead of a MagicMock. A mock accepts any keyword you hand it, so a mock
based test would have passed for the whole time the bug existed — it would have
asserted the node's opinion of the contract rather than the contract.

The schema here mirrors both servers:
  * mcp/kimodo_server.py            inputSchema.required = ["prompt"]
  * text-to-motion/kimodo/mcp_server.py   def generate_motion(prompt: str)
"""

from __future__ import annotations

import json
from unittest.mock import AsyncMock, patch

import pytest
from langchain_core.tools import StructuredTool
from pydantic import BaseModel, Field

# Imported at module scope, not inside the tests, and that is load-bearing:
# pytest.ini sets `filterwarnings = error` and langgraph's import chain emits a
# LangChainPendingDeprecationWarning, which subclasses PendingDeprecationWarning
# and so is not covered by the DeprecationWarning ignore. Imported inside a test
# the warning is raised as an error and every assertion below fails for a reason
# that has nothing to do with Kimodo.
from langgraph_agents.nodes.kimodo import kimodo_node
from langgraph_agents.state import ErrorSeverity


_MOTION_RESULT = {"status": "success", "files": ["motion_abc.npz"]}


class _GenerateMotionArgs(BaseModel):
    """The argument contract both Kimodo MCP servers declare."""

    prompt: str = Field(description="motion description")


def _make_motion_tool(calls: list) -> StructuredTool:
    """A generate_motion tool that ENFORCES the real schema."""

    async def _run(prompt: str) -> str:
        calls.append(prompt)
        return json.dumps(_MOTION_RESULT)

    return StructuredTool.from_function(
        coroutine=_run,
        name="generate_motion",
        description="Generate a 3D motion animation from a description.",
        args_schema=_GenerateMotionArgs,
    )


def _patch_client(tools: list):
    """Patch MCP discovery — the same seam test_phase3_retriever_with_mcp uses.

    Patched on the module rather than on the node, because the node imports the
    function inside its try block: a patch of `nodes.kimodo.get_mcp_tools` would
    bind nothing and the real discovery would run.
    """
    return patch(
        "langgraph_agents.mcp.client.get_mcp_tools",
        new=AsyncMock(return_value=tools),
    )


def _config(query: str = "động tác squat"):
    return {"configurable": {"query": query, "request_id": "test-req"}}


# ── The regression this file exists for ───────────────────────────────────────


@pytest.mark.unit
@pytest.mark.asyncio
async def test_node_calls_tool_with_the_argument_the_schema_declares():
    """The whole point. A regression to {"query": ...} fails schema validation.

    Note what is NOT asserted: that the payload key is the literal string
    "prompt". Hardcoding that on both sides would only restate the node's own
    choice. The schema does the judging, so this test tracks the servers'
    contract rather than a copy of it.
    """
    calls: list = []
    with _patch_client([_make_motion_tool(calls)]):
        result = await kimodo_node({}, _config("động tác squat"))

    assert calls == ["động tác squat"], (
        f"Tool was invoked {len(calls)} time(s) with {calls}. The node must pass "
        f"the resolved query as the argument generate_motion's schema requires."
    )
    assert "errors" not in result, f"Unexpected error path: {result.get('errors')}"
    assert "messages" in result and len(result["messages"]) == 1
    assert "motion_abc.npz" in result["messages"][0].content


@pytest.mark.unit
@pytest.mark.asyncio
async def test_node_prefers_resolved_query_over_raw_query():
    """resolved_query is the planner's rewrite — the raw query is the fallback.

    Worth pinning: the planner exists partly to turn "bài tập cổ đã hỏi tuần
    trước" into something searchable, and sending the raw text instead would
    quietly waste that.
    """
    calls: list = []
    with _patch_client([_make_motion_tool(calls)]):
        await kimodo_node({"resolved_query": "squat có tạ"}, _config("cái đó"))

    assert calls == ["squat có tạ"]


# ── Degradation: motion is optional, the answer is not ────────────────────────


@pytest.mark.unit
@pytest.mark.asyncio
async def test_missing_tool_degrades_instead_of_raising():
    """Kimodo down (or not deployed) must not take the graph with it.

    This is the everyday case right now: Kimodo runs on a GPU box that is off
    unless someone is demoing.
    """
    with _patch_client([]):                      # discovery returned no tools
        result = await kimodo_node({}, _config())

    assert "errors" in result and len(result["errors"]) == 1
    assert result["errors"][0]["severity"] == ErrorSeverity.RECOVERABLE
    assert result["errors"][0]["node"] == "kimodo"


@pytest.mark.unit
@pytest.mark.asyncio
async def test_tool_exception_degrades_instead_of_escaping():
    """A GPU OOM or a timeout inside Kimodo is still a RECOVERABLE error here.

    If this escaped, one failed motion render would abort a turn that already
    has a usable text answer waiting in the synthesizer.
    """
    async def _boom(prompt: str) -> str:
        raise RuntimeError("CUDA out of memory")

    exploding = StructuredTool.from_function(
        coroutine=_boom,
        name="generate_motion",
        description="always fails",
        args_schema=_GenerateMotionArgs,
    )

    with _patch_client([exploding]):
        result = await kimodo_node({}, _config())

    assert "errors" in result
    assert result["errors"][0]["severity"] == ErrorSeverity.RECOVERABLE
    assert "CUDA out of memory" in result["errors"][0]["message"]


@pytest.mark.unit
@pytest.mark.asyncio
async def test_wrong_argument_name_would_be_caught():
    """Proof that the fixture actually judges — a guard on the guard.

    If StructuredTool silently accepted extra keys, every assertion above would
    pass no matter what the node sent, and this file would be decoration. This
    test fails loudly if that ever becomes true.
    """
    calls: list = []
    tool = _make_motion_tool(calls)

    with pytest.raises(Exception):
        await tool.ainvoke({"query": "động tác squat"})

    assert calls == [], "The tool ran despite a payload missing its required field"
