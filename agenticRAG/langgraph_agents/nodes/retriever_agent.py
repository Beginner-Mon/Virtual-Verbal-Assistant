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
  D34:   Tag KHÔNG có red_flag_screen/referral_advice + web_search bật → gọi kb_search VÀ
         search_medical SONG SONG (round 1), không chờ kb rỗng rồi mới thử web ở round 2 —
         P2 hard-cap (route_after_retriever) drop tool_calls của round 2 vô điều kiện, nên
         "kb rỗng → thử web" tuần tự KHÔNG BAO GIỜ chạy được trong ngân sách 2 round hiện có.
         (Owner 23/07: đừng refuse câu hỏi PT phổ thông chỉ vì KB rỗng — vẫn giữ toggle làm
         cổng chính, không bật web ngoài ý muốn user; không đổi MAX_RETRIEVER_ROUNDS)
"""

from __future__ import annotations

import time
from datetime import datetime, timezone

from langchain_core.messages import SystemMessage, HumanMessage
from langchain_core.runnables import RunnableConfig

from langgraph_agents.state import AgentState, ErrorSeverity
from langgraph_agents.llm import get_chat_model, extract_cache_tokens
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


# ── System prompt (P3: conditional web search block) ─────────────────────

# Web-search tool description + decision rule — injected only when web_search_enabled.
_WEB_SEARCH_TOOL_BLOCK = """\
- **search_medical(query)**: Search the web via metasearch engine.
  Use for: real-time info (news, prices, current events), general facts outside PT domain.
"""

_WEB_SEARCH_RULE_LINE = """\
2. **Real-time/external** (news, prices, weather, general non-PT facts) → search_medical
"""

# D34: injected only when web_search_enabled AND allow_web_fallback (no high-safety tag).
# MUST call both tools in the SAME round — route_after_retriever force-drops any tool_calls
# requested on round 2, so there is no second chance to fall back after seeing kb empty.
_PT_WEB_FALLBACK_RULE_LINE = """\
1b. **PT/wellness topic, this query has no safety tag** → call kb_search AND search_medical
   TOGETHER in this same round (parallel), do NOT wait to see if kb_search is empty first —
   there is no second round to fall back in. If kb_search has real results, synthesizer will
   prefer them; search_medical is just insurance against a sparse/empty internal KB.
"""

# Static sections (always present regardless of web_search flag)
_RETRIEVER_PROMPT_BASE = """\
You are a KNOWLEDGE RETRIEVER for a physical therapy & wellness AI assistant.

## YOUR ROLE (Dev metaphor)
You are a DEVELOPER — you decide HOW to find information. The planner (manager) told you WHAT
is needed (tags) and gave you a resolved question. You choose which tools to use and what
to search for. Do NOT write the final answer — that's the synthesizer's job.

## TOOLS AVAILABLE
- **kb_search(query, top_k=5)**: Search the internal PT/wellness knowledge base.
  Use for: exercises, stretches, anatomy, physiotherapy techniques, health facts.
{web_search_tool_block}\
- **memory_search(query, since_days=None, top_k=3)**: Search user's past conversation summaries.
  Use when the user refers back to an earlier conversation — recalling what they asked
  before, naming a past time ("last week", "yesterday"), or asking you to repeat something.
  Scope is automatically scoped to the current user — no need to pass user_id.
- **resume_last_session(since_days=None)**: Resume the most recent past session.
  Use when the user wants to CONTINUE where they left off rather than recall a fact —
  picking the previous session's work back up, carrying on with it.
  Returns both summary chunks and recent messages from that session.
- **youtube_transcript(url)**: Fetch the spoken transcript of a YouTube video.
  Use ONLY when the user's message contains a YouTube link (youtube.com/watch?v= or youtu.be/).
  Pass the URL verbatim from the user's message. Speech-only — does NOT understand visuals.

## DECISION RULES
1. **PT/wellness topic** (exercises, stretches, anatomy, physiotherapy) → kb_search FIRST
{pt_fallback_rule_line}\
{web_search_rule_line}\
2. **Past conversation recall** (the user refers to something said earlier, or asks to
   continue previous work) → memory_search
3. **YouTube link in message** (youtube.com/watch or youtu.be) → call `youtube_transcript(url)`
   with the exact URL from the user's message; use the returned transcript to answer.
4. **Multiple needs** → call tools IN PARALLEL (multiple tool_calls in one response)
5. **NOT SURE which tool** → call kb_search (default, most common)

## SEARCH QUERY TIPS
- Use the resolved_query as base, enrich with relevant keywords from required_outputs tags
- The knowledge base is written in English. When the user asks in another language,
  search in ENGLISH — use the English clinical term for what they described. The
  embedding model is multilingual, but an English query matches English documents best.
- Enrich with the domain words that fit the question: for a movement, the exercise or
  stretch name; for anatomy, the muscle or joint. Do not pad with generic words.

## EMPTY vs ERROR
{empty_handling}\

## RETRY CONTEXT
{retry_note}

## WHAT IS NEEDED (from planner)
Required outputs: {required_outputs}
Resolved query: {resolved_query}
"""

# D34: default empty-handling (no fallback — either web is off, or query carries a
# high-safety tag where we deliberately do NOT want ungated web content).
_EMPTY_HANDLING_DEFAULT = """\
- If a tool returns results → pass them to synthesizer
- If a tool returns empty (no results found) → that's OK, synthesizer will handle no-source
- If a tool returns an ERROR → try an alternative tool or different query
- Do NOT loop infinitely — max 2 rounds of tool calls
"""

# D34: injected only when web_search_enabled AND allow_web_fallback. Reminds the agent it
# already called search_medical alongside kb_search per rule 1b — this is NOT a "wait and
# see" note, since there is no round left to react to an empty kb_search result in.
_EMPTY_HANDLING_WITH_WEB_FALLBACK = """\
- If a tool returns results → pass them to synthesizer
- If kb_search comes back empty but search_medical (called in parallel per rule 1b) has
  results → use those, and note the source; synthesizer needs a citation for web-sourced info
- If both kb_search and search_medical are empty → that's OK, synthesizer will handle no-source
- If a tool returns an ERROR → try an alternative tool or different query
- Do NOT loop infinitely — max 2 rounds of tool calls
"""


def _dedupe_tool_calls(tool_calls: list[dict]) -> tuple[list[dict], int]:
    """Drop exact-duplicate tool_calls emitted in the same round (fix #4).

    A duplicate = same tool ``name`` AND identical ``args``. Calls with the
    same name but different args (e.g. two different kb_search queries) are
    kept as-is — that's a legitimate parallel search, not waste. Benchmark
    showed rounds like [kb_search, kb_search] with identical args wasting a
    full ToolNode execution.

    Returns (deduped_list, removed_count).
    """
    seen: set[tuple] = set()
    deduped: list[dict] = []
    for tc in tool_calls:
        name = tc.get("name")
        args = tc.get("args") or {}
        try:
            key = (name, frozenset(args.items()))
        except TypeError:
            # Unhashable arg value (nested list/dict) — fall back to a stable
            # repr so exact repeats still dedupe without crashing.
            key = (name, repr(sorted(args.items())))
        if key in seen:
            continue
        seen.add(key)
        deduped.append(tc)
    return deduped, len(tool_calls) - len(deduped)


def _build_retriever_system_prompt(
    *,
    web_search_enabled: bool,
    allow_web_fallback: bool,
    retry_note: str,
    required_outputs: str,
    resolved_query: str,
) -> str:
    """Build the retriever system prompt.

    When web_search_enabled=False, the search_medical tool description and its
    decision rule are omitted entirely — the LLM is never told web search exists.
    This is the first layer of P3 defense (conditional prompt).

    allow_web_fallback (D34): only meaningful when web_search_enabled=True — tells the LLM
    to call kb_search + search_medical TOGETHER in round 1 for PT/wellness topics (rule 1b),
    since round 2's tool_calls get force-dropped by the P2 hard cap (route_after_retriever)
    regardless of content — there is no "wait for kb empty, then fall back" round available.
    False for high-safety tags (red_flag_screen/referral_advice): hard no-source refusal only.
    """
    use_fallback = web_search_enabled and allow_web_fallback
    empty_handling = _EMPTY_HANDLING_WITH_WEB_FALLBACK if use_fallback else _EMPTY_HANDLING_DEFAULT
    return _RETRIEVER_PROMPT_BASE.format(
        web_search_tool_block=_WEB_SEARCH_TOOL_BLOCK if web_search_enabled else "",
        pt_fallback_rule_line=_PT_WEB_FALLBACK_RULE_LINE if use_fallback else "",
        web_search_rule_line=_WEB_SEARCH_RULE_LINE if web_search_enabled else "",
        empty_handling=empty_handling,
        retry_note=retry_note,
        required_outputs=required_outputs,
        resolved_query=resolved_query,
    )


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

    # Hard-cap round counter (P2): increment each node execution
    current_rounds = state.get("retriever_rounds", 0) + 1

    # Web search toggle (D27): user config overrides retriever choice
    web_search_enabled = config["configurable"].get("web_search", False)

    # D34: high-safety tags never get the kb-empty→web fallback, regardless of toggle.
    _HIGH_SAFETY_TAGS = {"red_flag_screen", "referral_advice"}
    allow_web_fallback = not (_HIGH_SAFETY_TAGS & set(required_outputs))

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
        "web_fallback_allowed": web_search_enabled and allow_web_fallback,
    })

    tools = await _build_tools(web_search_enabled=web_search_enabled)

    system = _build_retriever_system_prompt(
        web_search_enabled=web_search_enabled,
        allow_web_fallback=allow_web_fallback,
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
            "retriever_rounds": current_rounds,
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

    # Dedupe identical (name+args) tool_calls before ToolNode executes them (fix #4).
    deduped_tool_calls, deduped_count = _dedupe_tool_calls(tool_calls)
    if deduped_count > 0:
        try:
            ai_msg.tool_calls = deduped_tool_calls
        except Exception:
            object.__setattr__(ai_msg, "tool_calls", deduped_tool_calls)
        tool_calls = deduped_tool_calls

    cache_hit_tokens, cache_miss_tokens = extract_cache_tokens(ai_msg)
    elapsed_ms = round((time.perf_counter() - t0) * 1000)

    log_extra = {
        "node": "retriever_agent", "request_id": request_id,
        "elapsed_ms": elapsed_ms, "tokens": tokens,
        "tool_calls": len(tool_calls),
        "tool_names": [tc.get("name") for tc in tool_calls],
        "cache_hit_tokens": cache_hit_tokens,
        "cache_miss_tokens": cache_miss_tokens,
    }
    if deduped_count > 0:
        log_extra["tool_calls_deduped"] = deduped_count
    logger.info("node_complete", extra=log_extra)

    return {
        "messages": [ai_msg],
        "total_tokens": tokens,
        "retriever_rounds": current_rounds,
    }
