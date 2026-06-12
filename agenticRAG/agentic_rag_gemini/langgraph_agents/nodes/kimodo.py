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
        from langgraph_agents.mcp.client import get_mcp_client

        client = get_mcp_client()
        tools = client.get_tools()

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

        # Call Kimodo — the MCP tool handles the actual motion generation
        result = await motion_tool.ainvoke({
            "query": resolved_query,
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
