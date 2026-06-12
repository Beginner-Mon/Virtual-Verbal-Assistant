"""LangGraph StateGraph — M.2 topology with 2 independent gates (D2, D15).

Build:  M.9 step order — memory → planner → retriever → kimodo → synthesizer → grader.
Gates:  Cổng RETRIEVER ⟸ needs_retrieval; Cổng GRADER ⟸ required_outputs != [] (D15).
        Motion = Kimodo node riêng, hard edge (D3, D26).
        Kimodo runs after retriever (sequential — both write to messages independently;
        true parallelism via Send() is a future optimization for Phase 7).

Nodes (8 total):
  memory          — assemble context (M.5)
  planner         — 3-axis intent (M.1)
  retriever_agent — self-choose tools, parallel tool calls (M.2)
  tools           — ToolNode (kb_search, search_medical, memory_search)
  kimodo          — Kimodo MCP motion gen (D26)
  synthesizer     — universal responder (M.3b)
  grader          — tag-driven quality check (M.3)
  error_handler   — graceful degradation
"""

import asyncio

from langgraph.graph import StateGraph, END, START
from langgraph.prebuilt import ToolNode

from langgraph_agents.state import AgentState
from langgraph_agents.nodes.memory import memory_node
from langgraph_agents.nodes.planner import planner_node
from langgraph_agents.nodes.retriever_agent import retriever_agent_node, RETRIEVER_BASE_TOOLS
from langgraph_agents.nodes.kimodo import kimodo_node
from langgraph_agents.nodes.synthesizer import synthesizer_node
from langgraph_agents.nodes.grader import grader_node
from langgraph_agents.nodes.error_handler import error_handler_node
from langgraph_agents.routing import (
    route_after_memory,
    route_after_retriever,
    route_after_synthesizer,
    route_after_grader,
    check_errors,
)

_RECURSION_LIMIT = 20  # planner→retriever⇄tools→kimodo→synth→grader (retry once)


# ── Single routing function from planner (one conditional edge only) ────

def route_after_planner(state: AgentState) -> str:
    """Planner → retriever_agent | kimodo | synthesizer | error_handler.

    Priority (single path — LangGraph supports one conditional edge per node):
      1. CRITICAL error → error_handler
      2. needs_clarification → synthesizer (skip all)
      3. needs_retrieval → retriever_agent (then chain → kimodo if needed)
      4. needs_motion (only, no retrieval) → kimodo
      5. neither → synthesizer (chat/greeting/safety-only)
    """
    if check_errors(state) == "error_handler":
        return "error_handler"

    if state.get("needs_clarification"):
        return "synthesizer"

    if state.get("needs_retrieval"):
        return "retriever_agent"

    if state.get("needs_motion"):
        return "kimodo"

    # No retrieval, no motion → chat / safety-only / clarify
    return "synthesizer"


# ── After retriever: chain to kimodo if motion needed ──────────────────

def route_after_retriever_or_tools(state: AgentState) -> str:
    """After retriever completes (or tools→retriever loop): go to kimodo if needed.

    This ensures Kimodo runs after retrieval but before synthesizer.
    Both write to messages independently → synthesizer reads combined results.
    """
    if check_errors(state) == "error_handler":
        return "error_handler"

    if state.get("needs_motion"):
        return "kimodo"
    return "synthesizer"


async def build_graph_async():
    """Build the LangGraph state graph with MCP tools discovered at startup.

    Must be called within a running event loop (FastAPI startup, test fixtures).
    """
    from langgraph_agents.mcp.client import get_mcp_tools

    mcp_tools = await get_mcp_tools()

    # Filter: motion tools NOT in retriever's ToolNode (D26)
    # generate_motion is called by kimodo node, not retriever
    retriever_mcp_tools = [
        t for t in mcp_tools
        if t.name not in ("generate_motion",)
    ]
    all_tools = [*RETRIEVER_BASE_TOOLS, *retriever_mcp_tools]

    g = StateGraph(AgentState)

    # ── Nodes ─────────────────────────────────────────────────────────
    g.add_node("memory", memory_node)
    g.add_node("planner", planner_node)
    g.add_node("retriever_agent", retriever_agent_node)
    g.add_node("tools", ToolNode(all_tools))
    g.add_node("kimodo", kimodo_node)
    g.add_node("synthesizer", synthesizer_node)
    g.add_node("grader", grader_node)
    g.add_node("error_handler", error_handler_node)

    # ── Edges ─────────────────────────────────────────────────────────

    # START → memory (always first — D9)
    g.add_edge(START, "memory")

    # memory → planner (or error_handler on CRITICAL)
    g.add_conditional_edges("memory", route_after_memory, {
        "planner": "planner",
        "error_handler": "error_handler",
    })

    # planner → retriever_agent | kimodo | synthesizer | error_handler
    g.add_conditional_edges("planner", route_after_planner, {
        "retriever_agent": "retriever_agent",
        "kimodo": "kimodo",
        "synthesizer": "synthesizer",
        "error_handler": "error_handler",
    })

    # ── Retriever ⇄ tools loop (max 2 rounds) ────────────────────────
    # After loop: → kimodo (if needs_motion) or → synthesizer
    g.add_conditional_edges("retriever_agent", route_after_retriever, {
        "tools": "tools",
        "kimodo": "kimodo",
        "synthesizer": "synthesizer",
        "error_handler": "error_handler",
    })
    g.add_edge("tools", "retriever_agent")

    # ── Kimodo → synthesizer ──────────────────────────────────────────
    g.add_edge("kimodo", "synthesizer")

    # ── Synthesizer → grader gate (D15: Cổng GRADER ⟸ required_outputs) ──
    g.add_conditional_edges("synthesizer", route_after_synthesizer, {
        "grader": "grader",
        "end": END,
        "error_handler": "error_handler",
    })

    # ── Grader → retry (retriever_agent) | END ────────────────────────
    g.add_conditional_edges("grader", route_after_grader, {
        "retriever_agent": "retriever_agent",
        "end": END,
    })

    # ── Error handler → END ───────────────────────────────────────────
    g.add_edge("error_handler", END)

    return g.compile().with_config(recursion_limit=_RECURSION_LIMIT)


def build_graph():
    """Sync convenience wrapper for use outside event loops (tests, scripts)."""
    return asyncio.run(build_graph_async())
