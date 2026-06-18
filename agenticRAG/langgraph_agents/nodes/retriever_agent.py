"""Retriever Agent — M.2 (dev tự chọn tool, song song).

Decisions encoded:
  D2b:   Retriever tự chọn kb/web/memory — planner only says WHAT
  D16:   kb_search + memory_search gọi SONG SONG
  D20:   NO memory_context input (retrieval ≠ personalize)
  D21:   resolved_query keeps tool-selection cues
  D23:   Empty {found:[]} ≠ error {error}
  D25:   Out-of-scope → referral_advice (not severity enum)
  D26:   Motion NOT in retriever (Kimodo = separate node)
  D27:   Web-off + real-time = instance of no-source
  D28:   Top-k caps: kb=5, memory=3, web=3
"""

from __future__ import annotations

import time
from datetime import datetime, timezone

from langchain_core.messages import SystemMessage, HumanMessage
from langchain_core.runnables import RunnableConfig

from langgraph_agents.state import AgentState, ErrorSeverity
from langgraph_agents.llm import get_chat_model
from langgraph_agents.shared.logging import get_logger

logger = get_logger("langgraph.retriever")


# ── Tool list (imported once, MCP tools discovered at startup) ────────────

# In-process tools
from langgraph_agents.tools.pgvector_tool import (
    kb_search as _kb_search,
    memory_search as _memory_search,
    resume_last_session as _resume_last_session,
    youtube_transcript as _youtube_transcript,
)

RETRIEVER_BASE_TOOLS = [_kb_search, _memory_search, _resume_last_session, _youtube_transcript]


async def _build_tools(web_search_enabled: bool = True) -> list:
    """Assemble tool list: kb_search + memory_search + resume_last_session + MCP tools.

    memory_search + resume_last_session always available.
    web_search (search_medical) is filtered by user toggle (D27).
    """
    from langgraph_agents.mcp.client import get_mcp_tools
    mcp_tools = await get_mcp_tools()

    tools = [*RETRIEVER_BASE_TOOLS]

    for t in mcp_tools:
        # Filter: motion tools NOT in retriever (D26)
        if t.name in ("generate_motion",):
            continue
        # Filter: web search only if user toggle allows (D27)
        if t.name == "search_medical" and not web_search_enabled:
            continue
        tools.append(t)

    return tools


# ── System prompt ─────────────────────────────────────────────────────────

_RETRIEVER_SYSTEM_PROMPT = """You are a KNOWLEDGE RETRIEVER for a physical therapy & wellness AI assistant.

## YOUR ROLE (Dev metaphor)
You are a DEVELOPER — you decide HOW to find information. The planner (manager) told you WHAT
is needed (tags) and gave you a resolved question. You choose which tools to use and what
to search for. Do NOT write the final answer — that's the synthesizer's job.

## TOOLS AVAILABLE
- **kb_search(query, top_k=5)**: Search the internal PT/wellness knowledge base.
  Use for: exercises, stretches, anatomy, physiotherapy techniques, health facts.
- **search_medical(query)**: Search the web via metasearch engine.
  Use for: real-time info (news, prices, current events), general facts outside PT domain.
- **memory_search(query, since_days=None, top_k=3)**: Search user's past conversation summaries.
  Use for: "lần trước", "tuần trước", "tôi đã hỏi", "nhắc lại".
  Scope is automatically scoped to the current user — no need to pass user_id.
- **resume_last_session(since_days=None)**: Resume the most recent past session.
  Use for: "tiếp tục", "làm tiếp bài hôm trước", "quay lại bài tập".
  Returns both summary chunks and recent messages from that session.
- **youtube_transcript(url)**: Fetch the spoken transcript of a YouTube video.
  Use ONLY when the user's message contains a YouTube link (youtube.com/watch?v= or youtu.be/).
  Pass the URL verbatim from the user's message. Speech-only — does NOT understand visuals.

## DECISION RULES
1. **PT/wellness topic** (exercises, stretches, anatomy, physiotherapy) → kb_search FIRST
2. **Real-time/external** (news, prices, weather, general non-PT facts) → search_medical
3. **Past conversation recall** ("lần trước", "như đã nói", "tiếp tục") → memory_search
4. **YouTube link in message** (youtube.com/watch or youtu.be) → call `youtube_transcript(url)`
   with the exact URL from the user's message; use the returned transcript to answer.
5. **Multiple needs** → call tools IN PARALLEL (multiple tool_calls in one response)
6. **NOT SURE which tool** → call kb_search (default, most common)

## SEARCH QUERY TIPS
- Use the resolved_query as base, enrich with relevant keywords from required_outputs tags
- For Vietnamese queries: include both accented and unaccented variants
- For exercises: add "bài tập", "hướng dẫn", "vật lý trị liệu"
- For anatomy: add "giải phẫu", "cơ", "xương khớp"

## EMPTY vs ERROR
- If a tool returns results → pass them to synthesizer
- If a tool returns empty (no results found) → that's OK, synthesizer will handle no-source
- If a tool returns an ERROR → try an alternative tool or different query
- Do NOT loop infinitely — max 2 rounds of tool calls

## RETRY CONTEXT
{retry_note}

## WHAT IS NEEDED (from planner)
Required outputs: {required_outputs}
Resolved query: {resolved_query}
"""


# ── Node ──────────────────────────────────────────────────────────────────

async def retriever_agent_node(state: AgentState, config: RunnableConfig) -> dict:
    """Retriever node — self-chooses tools, calls in parallel, returns evidence.

    Input: resolved_query + required_outputs (+ grader_feedback on retry)
    Output: messages (AIMessage with tool_calls → ToolNode executes → ToolMessages)
    """
    t0 = time.perf_counter()
    request_id = config["configurable"].get("request_id", "-")

    resolved_query = state.get("resolved_query") or config["configurable"]["query"]
    required_outputs = state.get("required_outputs", [])
    feedback = state.get("grader_feedback")

    # Web search toggle (D27): user config overrides retriever choice
    web_search_enabled = config["configurable"].get("web_search", True)

    retry_note = ""
    if feedback:
        retry_note = (
            "## RETRY — Previous attempt rejected\n"
            f"Grader feedback: {feedback}\n"
            "Try different search queries or additional tools."
        )

    is_retry = bool(feedback)
    logger.info("node_start", extra={
        "node": "retriever_agent", "request_id": request_id,
        "query_preview": resolved_query[:80],
        "tags": required_outputs,
        "is_retry": is_retry,
        "web_search": web_search_enabled,
    })

    tools = await _build_tools(web_search_enabled=web_search_enabled)

    system = _RETRIEVER_SYSTEM_PROMPT.format(
        retry_note=retry_note,
        required_outputs=", ".join(required_outputs) if required_outputs else "(none — general/chat)",
        resolved_query=resolved_query,
    )

    llm = get_chat_model("retriever").bind_tools(tools)

    try:
        ai_msg = await llm.ainvoke([
            SystemMessage(content=system),
            HumanMessage(content=f"Find information for: {resolved_query}"),
        ])
    except Exception as exc:
        elapsed_ms = round((time.perf_counter() - t0) * 1000)
        logger.error("node_failed", extra={
            "node": "retriever_agent", "request_id": request_id,
            "elapsed_ms": elapsed_ms, "error": str(exc),
        }, exc_info=True)
        return {
            "errors": [{
                "node": "retriever_agent",
                "severity": ErrorSeverity.CRITICAL,
                "message": f"Retriever LLM failed: {exc}",
                "timestamp": datetime.now(timezone.utc).isoformat(),
            }],
        }

    tokens = 0
    if hasattr(ai_msg, "usage_metadata") and ai_msg.usage_metadata:
        tokens = ai_msg.usage_metadata.get("total_tokens", 0)

    tool_calls = getattr(ai_msg, "tool_calls", []) or []
    elapsed_ms = round((time.perf_counter() - t0) * 1000)

    logger.info("node_complete", extra={
        "node": "retriever_agent", "request_id": request_id,
        "elapsed_ms": elapsed_ms, "tokens": tokens,
        "tool_calls": len(tool_calls),
        "tool_names": [tc.get("name") for tc in tool_calls],
    })

    return {
        "messages": [ai_msg],
        "total_tokens": tokens,
    }
