"""Retriever Agent — executes planner's plan via tool calls.

Pattern: ChatModel.bind_tools() → invoke → if tool_calls → ToolNode → loop
Phase 3: pgvector_search (in-process) + MCP tools (generate_motion, search_medical).
"""

from datetime import datetime, timezone
from langchain_core.messages import SystemMessage, HumanMessage

from langgraph_agents.state import AgentState, ErrorSeverity
from langgraph_agents.llm import get_chat_model
from langgraph_agents.tools.pgvector_tool import pgvector_search


RETRIEVER_TOOLS = [pgvector_search]
_RETRIEVER_TOOLS = RETRIEVER_TOOLS


async def _build_tools() -> list:
    """In-process + MCP. Called per node invocation; tool list cached by mcp/client."""
    from langgraph_agents.mcp.client import get_mcp_tools
    mcp_tools = await get_mcp_tools()
    return [pgvector_search, *mcp_tools]


_RETRIEVER_SYSTEM_PROMPT = """You are a tool execution agent. Execute the plan from the planner.

Rules:
- For knowledge_query and exercise_recommendation: ALWAYS call pgvector_search first
- Use the planner's expanded_query as the search query
- You may call tools in parallel via multiple tool_calls in one response
- Do NOT generate the final answer — only retrieve evidence and let synthesizer compose it
- If a tool fails, note in your response and continue

{retry_note}

Plan from planner:
{plan}
"""


async def retriever_agent_node(state: AgentState, config) -> dict:
    """Run one tool-calling step. ToolNode handles execution."""
    plan = state.get("plan", {})
    expanded = state.get("expanded_query") or config["configurable"]["query"]

    retry_note = ""
    feedback = state.get("grader_feedback")
    if feedback:
        retry_note = f"## Retry context\nPrevious attempt was rejected. Grader feedback:\n{feedback}\nTry a different query or call additional tools."

    system = _RETRIEVER_SYSTEM_PROMPT.format(
        retry_note=retry_note,
        plan=plan,
    )

    tools = await _build_tools()
    llm = get_chat_model("retriever").bind_tools(tools)

    try:
        ai_msg = await llm.ainvoke([
            SystemMessage(content=system),
            HumanMessage(content=f"Execute plan for query: {expanded}"),
        ])
    except Exception as exc:
        return {
            "errors": [{
                "node": "retriever_agent",
                "severity": ErrorSeverity.CRITICAL,
                "message": f"Retriever LLM failed: {exc}",
                "timestamp": datetime.now(timezone.utc).isoformat(),
            }],
        }

    tokens = (ai_msg.usage_metadata or {}).get("total_tokens", 0) if hasattr(ai_msg, "usage_metadata") else 0

    return {
        "messages": [ai_msg],
        "total_tokens": tokens,
    }
