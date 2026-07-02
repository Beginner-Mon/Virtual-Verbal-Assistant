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

from langchain_core.messages import ToolMessage
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
from langgraph_agents.shared.logging import get_logger

_logger = get_logger("langgraph.graph")

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


def _make_guarded_tools_node(base_tool_node: ToolNode):
    """P3 execution guard: wraps ToolNode to block search_medical when web_search=False.

    If web_search is off and the last AIMessage contains a search_medical tool_call,
    that call is short-circuited with a synthetic ToolMessage (content:
    '{"blocked": "web_search_disabled"}') so the graph keeps flowing. All other
    tool_calls are delegated to the underlying ToolNode unchanged.

    This is the second layer of P3 defense (execution guard). The first layer
    (conditional prompt) should prevent the LLM from calling search_medical at all
    when web_search=False; this guard is a hard backstop in case it still does.
    """
    async def _guarded_tools(state: AgentState, config) -> dict:
        from langchain_core.messages import AIMessage

        cfg = config.get("configurable", {}) if isinstance(config, dict) else getattr(config, "configurable", {}) or {}
        web_search_on = cfg.get("web_search", False)

        if web_search_on:
            # Fast path: delegate entirely to ToolNode
            return await base_tool_node.ainvoke(state, config)

        # Web search is OFF — scan last AIMessage for blocked calls
        messages = state.get("messages", [])
        last_msg = messages[-1] if messages else None
        if not (isinstance(last_msg, AIMessage) and getattr(last_msg, "tool_calls", None)):
            return await base_tool_node.ainvoke(state, config)

        # Partition tool_calls: blocked vs allowed
        blocked = [tc for tc in last_msg.tool_calls if tc.get("name") == "search_medical"]
        allowed = [tc for tc in last_msg.tool_calls if tc.get("name") != "search_medical"]

        if not blocked:
            return await base_tool_node.ainvoke(state, config)

        _logger.warning("web_search_guard_blocked", extra={
            "tool": "search_medical",
            "blocked_count": len(blocked),
            "request_id": cfg.get("request_id", "-"),
        })

        # Build synthetic ToolMessages for blocked calls
        synthetic = [
            ToolMessage(
                content='{"blocked": "web_search_disabled"}',
                tool_call_id=tc["id"],
                name=tc["name"],
            )
            for tc in blocked
        ]

        if not allowed:
            # All calls blocked — return synthetic messages directly
            return {"messages": synthetic}

        # Some calls are allowed: rebuild AIMessage with only allowed tool_calls,
        # run ToolNode, then prepend synthetic blocked results.
        import copy
        pruned_msg = copy.copy(last_msg)
        pruned_msg.tool_calls = allowed
        pruned_state = {**state, "messages": [*messages[:-1], pruned_msg]}
        tool_result = await base_tool_node.ainvoke(pruned_state, config)
        tool_messages = tool_result.get("messages", [])
        return {"messages": [*synthetic, *tool_messages]}

    return _guarded_tools


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
    g.add_node("tools", _make_guarded_tools_node(ToolNode(all_tools)))
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
