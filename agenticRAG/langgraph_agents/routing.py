"""Routing — M.2 HAI CỔNG ĐỘC LẬP (D2, D15).

Decisions encoded:
  D2:   Routing ⟸ needs_retrieval; grader ⟸ required_outputs (2 independent gates)
  D2b:  Manager says WHAT, dev chooses HOW — no tool flags in planner
  D15:  NEVER merge gates → safety bug (no-retrieval + safety tag → skip grader)
  D22:  Clarify = multi-turn, no loop in graph
  D24:  Retry ONLY for grader quality fail, NOT for empty results
"""

from langgraph_agents.state import AgentState, ErrorSeverity


def check_errors(state: AgentState) -> str:
    """After each node: route to error_handler if CRITICAL error exists."""
    for err in state.get("errors", []):
        if err.get("severity") == ErrorSeverity.CRITICAL:
            return "error_handler"
    return "continue"


# ── After memory ──────────────────────────────────────────────────────────

def route_after_memory(state: AgentState) -> str:
    """Memory → planner (always, unless CRITICAL error)."""
    if check_errors(state) == "error_handler":
        return "error_handler"
    return "planner"


# ── After planner — TWO INDEPENDENT PATHS ─────────────────────────────────
# Path A: retriever gate (⟸ needs_retrieval)
# Path B: Kimodo gate (⟸ needs_motion, hard edge)
# Both can run in parallel (LangGraph fan-out)

def route_after_planner(state: AgentState) -> str:
    """Planner → retriever_agent | kimodo | synthesizer | error_handler.

    Cổng RETRIEVER ⟸ needs_retrieval (D2).
    Kimodo hard edge ⟸ needs_motion (D3, D26).
    Does NOT read required_outputs — those are for the grader gate (D15).

    Priority (single path — one conditional edge per node):
      1. CRITICAL error → error_handler
      2. needs_clarification → synthesizer (skip all)
      3. needs_retrieval → retriever_agent (may chain to kimodo after)
      4. needs_motion (only) → kimodo
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

    return "synthesizer"


# ── After retriever ───────────────────────────────────────────────────────

_MAX_TOOL_ROUNDS = 2  # Hard cap on retriever⇄tools iterations


def route_after_retriever(state: AgentState) -> str:
    """Retriever → tools (more calls) | kimodo (if needs_motion) | synthesizer | error_handler.

    Hard-caps tool call rounds to prevent infinite loops (D24: retry only
    for service errors, not empty results).
    After retrieval done: chain to kimodo if needs_motion (D26: motion after retrieval).
    """
    if check_errors(state) == "error_handler":
        return "error_handler"

    from langchain_core.messages import AIMessage
    messages = state.get("messages", [])

    # Count tool-call rounds
    tool_rounds = sum(
        1 for m in messages
        if isinstance(m, AIMessage) and getattr(m, "tool_calls", None)
    )

    last_msg = messages[-1] if messages else None
    last_has_tool_calls = bool(
        last_msg and getattr(last_msg, "tool_calls", None)
    )

    if last_has_tool_calls and tool_rounds <= _MAX_TOOL_ROUNDS:
        return "tools"

    # Retrieval done — chain to kimodo if motion needed (D26)
    if state.get("needs_motion"):
        return "kimodo"
    return "synthesizer"


# ── After synthesizer — GRADER GATE ───────────────────────────────────────
# Cổng GRADER ⟸ required_outputs != [] (D2, D15)
# Independent from retriever gate — MUST run even when retriever was skipped.
# This closes the safety bug: "đau ngực" (no retrieval, safety tag) still gets
# grader enforcement.

def route_after_synthesizer(state: AgentState) -> str:
    """Synthesizer → grader | END.

    Cổng GRADER ⟸ required_outputs != [] (D2, D15).
    Empty tags → fast-path END (D8: chat/general/clarify no contract).
    """
    if check_errors(state) == "error_handler":
        return "error_handler"

    required_outputs = state.get("required_outputs", [])
    if required_outputs:
        return "grader"
    return "end"


# ── After grader ──────────────────────────────────────────────────────────

def route_after_grader(state: AgentState) -> str:
    """Grader → retriever_agent (retry once) | END.

    Retry only for quality fails (D6: safety fails get template cứng, no retry).
    Retry count max 1 (D24).
    """
    result = state.get("grader_result", "pass")
    if result == "retry":
        return "retriever_agent"
    return "end"
