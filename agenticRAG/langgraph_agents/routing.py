from langgraph_agents.state import AgentState, ErrorSeverity


def check_errors(state: AgentState) -> str:
    """After each node: route to error_handler if CRITICAL error exists."""
    for err in state.get("errors", []):
        if err.get("severity") == ErrorSeverity.CRITICAL:
            return "error_handler"
    return "continue"


def route_after_memory(state: AgentState) -> str:
    """Memory always goes to planner unless CRITICAL error."""
    if check_errors(state) == "error_handler":
        return "error_handler"
    return "planner"


def route_after_planner(state: AgentState) -> str:
    """Planner → synthesizer (direct, for greeting/clarify) | retriever_agent | error_handler.

    After Phase 6.9: conversation node deleted. Synthesizer is the universal
    response generator and handles greeting + clarify modes directly without
    going through retriever (no tools needed for those intents).
    """
    if check_errors(state) == "error_handler":
        return "error_handler"

    if state.get("needs_clarification"):
        return "synthesizer"

    intent = state.get("intent", "conversation")
    if intent in ("knowledge_query", "exercise_recommendation", "visualize_motion"):
        return "retriever_agent"
    # conversation, clarify, or unknown → synthesizer direct (chat mode)
    return "synthesizer"


_MAX_TOOL_ROUNDS = 2  # hard cap on retriever⇄tools iterations within one pipeline run


def route_after_retriever(state: AgentState) -> str:
    """Retriever → tools (more calls) | synthesizer (done) | error_handler.

    Replaces langgraph.prebuilt.tools_condition so we can:
      1. Check CRITICAL errors first
      2. Hard-cap the retriever⇄tools loop (prevents infinite tool-call loops
         when the LLM keeps requesting more searches with empty results)
    """
    if check_errors(state) == "error_handler":
        return "error_handler"

    from langchain_core.messages import AIMessage
    messages = state.get("messages", [])
    tool_rounds = sum(
        1 for m in messages
        if isinstance(m, AIMessage) and getattr(m, "tool_calls", None)
    )

    last_has_tool_calls = bool(messages and getattr(messages[-1], "tool_calls", None))

    if last_has_tool_calls and tool_rounds <= _MAX_TOOL_ROUNDS:
        return "tools"
    return "synthesizer"


def route_after_grader(state: AgentState) -> str:
    """Grader → retriever_agent (retry once) | END (pass / pass_with_warning).

    After Phase 6.9: synthesizer already produced final_answer with persona
    voice. Grader appends warning to final_answer on pass_with_warning path
    (no extra node/LLM needed).
    """
    result = state.get("grader_result", "pass")
    if result == "retry":
        return "retriever_agent"
    return "end"
