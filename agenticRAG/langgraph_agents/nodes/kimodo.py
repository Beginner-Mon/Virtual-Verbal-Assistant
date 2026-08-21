"""Kimodo node — standalone motion generation (D26).

Decisions encoded:
  D3:   needs_motion = HARD GATE (edge cứng, not LLM tool-choice)
  D26:  Motion = node riêng, parallel to retriever, NOT in retriever bind_tools
        MCP = transport, edge = control (2 independent layers)
  D26 revised: synthesizer does NOT receive motion flag.
        Coherence via tag motion_descriptor + UI combines text+video.

The Kimodo node:
  - Only runs when needs_motion=true (hard edge from planner)
  - Calls Kimodo MCP server's generate_motion tool
  - Runs in parallel with retriever (not dependent on retrieval results)
  - Output: ToolMessage with motion result → UI consumes directly
"""

from __future__ import annotations

import time

from langchain_core.runnables import RunnableConfig

from langgraph_agents.state import AgentState, ErrorSeverity
from langgraph_agents.shared.logging import get_logger

logger = get_logger("langgraph.kimodo")


async def kimodo_node(state: AgentState, config: RunnableConfig) -> dict:
    """Kimodo motion generation node.

    Called via hard edge when planner sets needs_motion=true.
    Runs in parallel with retriever — not dependent on retrieval results.

    Input: resolved_query (for motion description)
    Output: ToolMessage with motion generation result
    """
    t0 = time.perf_counter()
    request_id = config["configurable"].get("request_id", "-")
    resolved_query = state.get("resolved_query") or config["configurable"]["query"]

    logger.info("node_start", extra={
        "node": "kimodo", "request_id": request_id,
        "query_preview": resolved_query[:80],
    })

    try:
        # `get_mcp_tools()`, awaited — the same call graph.py:165,
        # retriever_agent.py:55 and health.py:100 all make.
        #
        # This used to be `from ... import get_mcp_client` then
        # `client.get_tools()`. No such function has ever existed in
        # mcp/client.py, which exports get_mcp_tools() and close_mcp_client().
        # So the import raised ImportError on the very first line of this try
        # block, was caught below, and became a RECOVERABLE error — meaning this
        # node has never once reached the Kimodo server. Nothing went red because
        # nothing tested the node; see test_kimodo_node.py.
        from langgraph_agents.mcp.client import get_mcp_tools

        tools = await get_mcp_tools()

        # Find generate_motion tool
        motion_tool = None
        for t in tools:
            if t.name == "generate_motion":
                motion_tool = t
                break

        if motion_tool is None:
            logger.warning("kimodo_tool_not_found", extra={
                "request_id": request_id,
                "available_tools": [t.name for t in tools],
            })
            elapsed_ms = round((time.perf_counter() - t0) * 1000)
            return {
                "errors": [{
                    "node": "kimodo",
                    "severity": ErrorSeverity.RECOVERABLE,
                    "message": "generate_motion tool not available (Kimodo MCP server may be down)",
                    "timestamp": __import__("datetime").datetime.now(
                        __import__("datetime").timezone.utc
                    ).isoformat(),
                }],
            }

        # Call Kimodo — the MCP tool handles the actual motion generation.
        #
        # The key is "prompt", not "query", and that is not a cosmetic detail:
        # both MCP servers declare `prompt` as the required argument —
        # mcp/kimodo_server.py's inputSchema and text-to-motion/kimodo/
        # mcp_server.py's `def generate_motion(prompt: str)`. Sending "query"
        # left the required field missing, so the call failed schema validation,
        # fell into the except below and was logged as a RECOVERABLE error.
        # Motion silently never ran.
        #
        # It survived because the tests exercised the mock MCP server directly
        # and nothing exercised this node — the thing at the far end of the wire
        # was checked, the wire was not. See test_kimodo_node.py.
        result = await motion_tool.ainvoke({
            "prompt": resolved_query,
        })

        elapsed_ms = round((time.perf_counter() - t0) * 1000)
        logger.info("node_complete", extra={
            "node": "kimodo", "request_id": request_id,
            "elapsed_ms": elapsed_ms,
            "has_result": bool(result),
        })

        # Return as a ToolMessage so LangGraph can route it
        from langchain_core.messages import ToolMessage
        return {
            "messages": [ToolMessage(
                content=str(result),
                tool_call_id="kimodo_motion",
                name="generate_motion",
            )],
        }

    except Exception as exc:
        elapsed_ms = round((time.perf_counter() - t0) * 1000)
        logger.error("node_failed", extra={
            "node": "kimodo", "request_id": request_id,
            "elapsed_ms": elapsed_ms, "error": str(exc),
        }, exc_info=True)
        return {
            "errors": [{
                "node": "kimodo",
                "severity": ErrorSeverity.RECOVERABLE,
                "message": f"Kimodo motion generation failed: {exc}",
                "timestamp": __import__("datetime").datetime.now(
                    __import__("datetime").timezone.utc
                ).isoformat(),
            }],
        }
