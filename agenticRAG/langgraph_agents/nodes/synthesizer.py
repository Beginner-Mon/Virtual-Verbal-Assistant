"""Synthesizer node — M.3b universal responder, persona-styled.

Decisions encoded:
  D29:  Mode EMERGES from signals, NOT enum response_mode
  D30:  Persona applies to ALL modes (including refuse/clarify)
  D32:  Safety warning = FIRST in output (synthesizer writes);
        unverified disclaimer = LAST (grader appends)
  D33:  Danger detection = PLANNER only (1 place);
        synthesizer executes tags, does NOT re-evaluate danger
  D26:  Motion coherence via tag motion_descriptor;
        synthesizer does NOT receive motion flag

Mode derivation (D29 — derive, don't store):
  needs_clarification OR tool ambiguous → CLARIFY
  clinical tag + all tools empty (no-source)  → REFUSE
  Has ToolMessage non-empty                   → SYNTHESIZE
  No tools + no tags                          → CHAT
"""

from __future__ import annotations

import asyncio
import time
from datetime import datetime, timezone

from langchain_core.messages import SystemMessage, HumanMessage, AIMessage, ToolMessage
from langchain_core.runnables import RunnableConfig

from langgraph.config import get_stream_writer
from langgraph_agents.state import AgentState, ErrorSeverity
from langgraph_agents.llm import get_chat_model, get_fallback_chat_model, extract_cache_tokens
from langgraph_agents.nodes._persona_loader import get_persona, build_persona_prompt
from langgraph_agents.shared.logging import get_logger

logger = get_logger("langgraph.synthesizer")


# ── Language rule (shared across all modes) ──────────────────────────────

_LANGUAGE_RULE = """## LANGUAGE RULE (overrides persona defaults)
- Supported: ENGLISH and VIETNAMESE only.
- Detect the user's query language:
    * Mostly Vietnamese → respond entirely in Vietnamese
    * Mostly English → respond entirely in English
- CRITICAL: the persona rules are WRITTEN in Vietnamese. That is the language of
  the instructions, NOT the language of your reply. Obey what they MEAN, then
  write the reply in the user's language. An English query gets an English reply
  — including clinical terms, exercise names and safety warnings.
- Do NOT mix languages. Do NOT write preambles like "I'll answer in...".
- Start your response DIRECTLY with the answer content.
"""


# ── Safety warning prefix (D32: safety = ĐẦU output, mọi persona) ─────

_SAFETY_PREFIX_RULES = """
## SAFETY RULES (applied BEFORE everything else — D32, D33)

IF the required_outputs contain `red_flag_screen`:
  Your FIRST sentence MUST be a clear warning that the user's symptom
  (chest pain, numbness, dizziness, loss of control) could be serious.
  Example: "⚠️ Đau ngực khi tập thể dục có thể là dấu hiệu nghiêm trọng.
  Bạn nên NGỪNG tập ngay và đi khám bác sĩ."

IF the required_outputs contain `referral_advice`:
  Your response MUST include a statement that the user should consult
  a medical professional. Use your persona voice but keep the message clear.
  Example: "Tôi khuyên bạn nên gặp bác sĩ chuyên khoa để được chẩn đoán chính xác."

IF the required_outputs contain `scope_disclaimer`:
  Include a brief note that this is wellness advice, not a substitute for
  clinical examination. Place it near the end (grader appends boilerplate too).
"""


# ── Mode-specific prompts ────────────────────────────────────────────────

_SYNTHESIZE_TASK = """You are a physical therapy & wellness AI assistant.
Stay in the persona voice defined above.

{language_rule}
{safety_rules}

## Required deliverables (tags)
{required_outputs}

## Retrieved evidence
{tool_results}

## User's question (cleaned, coreferences resolved)
{resolved_query}

Instructions:
- Cover ALL required_outputs tags in your response
- Use the persona voice (tone, formatting from persona block)
- Base your answer on the retrieved evidence — cite sources when available
- For exercise_protocol: include sets, reps, frequency (e.g. "3 hiệp × 10 lần, 2-3 lần/tuần")
- For exercise_steps: provide ≥2 ordered steps
- For contraindication: list conditions where the exercise should NOT be done
- For motion_descriptor: describe the movement + joints involved clearly
- For evidence_citation: mention sources (document title, web source)
- Keep under 350 words. Do not pad or repeat safety disclaimers — state each once.
"""

_REFUSE_TASK = """You are a physical therapy & wellness AI assistant.
Stay in the persona voice defined above.

{language_rule}
{safety_rules}

## Situation
The user asked a question that is OUTSIDE your wellness advisory scope
and/or no reliable sources were found. You MUST NOT fabricate an answer.

## Required deliverables (tags)
{required_outputs}

## User's question
{resolved_query}

Instructions:
- Be honest: explain WHY you cannot answer (out of scope / no sources)
- If referral_advice tag is present: strongly recommend seeing a medical professional
- If no sources were found: state this clearly, suggest the user rephrase or ask a professional
- Keep it brief but compassionate (persona voice)
- Do NOT invent exercises, diagnoses, or medical advice
"""

_CLARIFY_TASK = """The user's query needs clarification before you can give a useful answer.
Stay in the persona voice defined above.

{language_rule}

## User's question
{resolved_query}

## Context (tool results may contain ambiguity candidates)
{tool_results}

Instructions:
- Ask the user for the specific missing information in your persona voice
- Be warm and professional — explain WHY you need this info
- Keep concise (1-3 sentences)
- If tool results contain candidates (multiple matching sessions/articles), list 2-3 briefly for the user to choose
"""

_CHAT_TASK = """You are responding to a casual conversational message.
Stay in the persona voice defined above.

{language_rule}

## User's question
{resolved_query}

Instructions:
- Respond naturally in your persona voice (warm, brief, friendly)
- Keep under 50 words for greetings, under 100 for follow-up chat
- Do NOT add clinical advice unless the user explicitly asks
- You may offer PT/wellness help in 1 short line if natural
"""


# ── Helpers ──────────────────────────────────────────────────────────────

def _extract_tool_results(messages: list) -> str:
    """Format ToolMessage content from retriever tool calls."""
    parts = []
    for i, m in enumerate(messages, 1):
        if isinstance(m, ToolMessage):
            content = str(m.content)[:1500]  # Truncate per-message for context window
            parts.append(f"[Tool {i}: {m.name}]\n{content}")
    return "\n\n".join(parts) if parts else ""


def _has_tool_results(messages: list) -> bool:
    """Check if any ToolMessage has non-empty, non-error results."""
    for m in messages:
        if isinstance(m, ToolMessage):
            content = str(m.content)
            # Empty result (D23: {found: false} or [])
            if content in ("", "[]", "{}", '{"found": false}'):
                continue
            # Error result
            if '"error"' in content or '"error":' in content:
                continue
            return True
    return False


def _check_tool_ambiguous(messages: list) -> bool:
    """Check if any tool returned ambiguity metadata (D22: dynamic clarify)."""
    import json
    for m in messages:
        if isinstance(m, ToolMessage):
            try:
                data = json.loads(str(m.content))
                if isinstance(data, dict) and data.get("ambiguous"):
                    return True
            except (json.JSONDecodeError, TypeError):
                pass
    return False


# ── Mode derivation (D29: emerge from signals, no enum) ─────────────────

def _derive_mode(state: AgentState) -> str:
    """Derive synthesizer mode from state signals.

    Returns one of: 'clarify', 'refuse', 'synthesize', 'chat'
    """
    needs_clarification = state.get("needs_clarification", False)
    required_outputs = state.get("required_outputs", [])
    messages = state.get("messages", [])

    # 1. CLARIFY: static (planner-detected) or dynamic (tool ambiguous)
    if needs_clarification or _check_tool_ambiguous(messages):
        return "clarify"

    has_results = _has_tool_results(messages)

    # 2. REFUSE: clinical/safety tags + no sources (D25 — clinical-no-source)
    safety_tags = {"red_flag_screen", "referral_advice"}
    clinical_tags = {"exercise_protocol", "exercise_steps", "contraindication",
                     "evidence_citation", "scope_disclaimer"}
    has_clinical = bool(set(required_outputs) & (safety_tags | clinical_tags))

    if has_clinical and not has_results:
        return "refuse"

    # 3. SYNTHESIZE: has tool results (regardless of tags)
    if has_results:
        return "synthesize"

    # 4. CHAT: no tools, no tags (greeting/general/empty)
    return "chat"


# ── Node ─────────────────────────────────────────────────────────────────

async def synthesizer_node(state: AgentState, config: RunnableConfig) -> dict:
    """Synthesizer node — universal responder (M.3b).

    Derives mode from state signals (D29), applies persona voice (D30),
    writes safety warning FIRST (D32).
    """
    t0 = time.perf_counter()
    request_id = config["configurable"].get("request_id", "-")
    persona_id = config["configurable"].get("persona_id", "eca_default")
    resolved_query = state.get("resolved_query") or config["configurable"]["query"]
    required_outputs = state.get("required_outputs", [])

    mode = _derive_mode(state)

    logger.info("node_start", extra={
        "node": "synthesizer", "request_id": request_id,
        "mode": mode, "persona_id": persona_id,
        "tags": required_outputs,
        "query_preview": resolved_query[:80],
    })

    # ── Build prompts ─────────────────────────────────────────────────
    tool_results = _extract_tool_results(state.get("messages", []))
    tags_str = ", ".join(required_outputs) if required_outputs else "(none — free response)"

    # Safety rules: only inject when safety tags present (D32, D33)
    has_safety = bool({"red_flag_screen", "referral_advice", "scope_disclaimer"} & set(required_outputs))
    safety_rules = _SAFETY_PREFIX_RULES if has_safety else ""

    if mode == "clarify":
        task_system = _CLARIFY_TASK.format(
            language_rule=_LANGUAGE_RULE,
            resolved_query=resolved_query,
            tool_results=tool_results or "(no tool results — static clarification)",
        )
    elif mode == "refuse":
        task_system = _REFUSE_TASK.format(
            language_rule=_LANGUAGE_RULE,
            safety_rules=safety_rules,
            required_outputs=tags_str,
            resolved_query=resolved_query,
        )
    elif mode == "synthesize":
        task_system = _SYNTHESIZE_TASK.format(
            language_rule=_LANGUAGE_RULE,
            safety_rules=safety_rules,
            required_outputs=tags_str,
            tool_results=tool_results or "(no evidence)",
            resolved_query=resolved_query,
        )
    else:  # chat
        task_system = _CHAT_TASK.format(
            language_rule=_LANGUAGE_RULE,
            resolved_query=resolved_query,
        )

    # Persona prompt (D30: applies to ALL modes)
    persona = get_persona(persona_id)
    persona_system = build_persona_prompt(persona, mode)
    system = f"{persona_system}\n\n---\n\n{task_system}"

    llm = get_chat_model("synthesizer")

    try:
        writer = get_stream_writer()
    except RuntimeError:
        writer = None

    # Include prior conversation (loaded by memory node into state messages)
    # so the model has context for follow-ups ("what did I just say"). Keep
    # plain user/assistant turns only — drop the memory SystemMessage, tool-call
    # AIMessages, and ToolMessages (tool evidence is already in the system prompt).
    history = [
        m for m in state.get("messages", [])
        if isinstance(m, HumanMessage)
        or (isinstance(m, AIMessage) and not getattr(m, "tool_calls", None))
    ]
    msgs = [SystemMessage(content=system), *history, HumanMessage(content=resolved_query)]

    ai_msg = None  # kept for prompt-cache telemetry (fix #1)
    used_fallback = False

    try:
        if writer is not None:
            final = ""
            tokens = 0
            async for chunk in llm.astream(msgs):
                content = chunk.content if hasattr(chunk, "content") else str(chunk)
                if content:
                    final += content
                    writer({"content": content})
                    # LangGraph's "custom" stream mode only drains this node's
                    # writer() queue when a sibling "waiter" task gets scheduled
                    # by asyncio (see PregelRunner.atick's asyncio.wait(...,
                    # FIRST_COMPLETED) race). This loop's own awaits (network
                    # reads from llm.astream) keep resuming THIS task fast enough
                    # that the waiter never gets a turn — so every token silently
                    # queues up and only flushes to the SSE client in one burst
                    # right as the node finishes. sleep(0) forces one real event-
                    # loop tick per token, giving the waiter a chance to run and
                    # actually deliver tokens as they're generated.
                    await asyncio.sleep(0)
                if hasattr(chunk, "usage_metadata") and chunk.usage_metadata:
                    tokens = chunk.usage_metadata.get("total_tokens", 0)
                    ai_msg = chunk
                elif (getattr(chunk, "response_metadata", None) or {}).get("token_usage"):
                    ai_msg = chunk
        else:
            ai_msg = await llm.ainvoke(msgs)
            final = ai_msg.content
            tokens = 0
            if hasattr(ai_msg, "usage_metadata") and ai_msg.usage_metadata:
                tokens = ai_msg.usage_metadata.get("total_tokens", 0)
    except Exception as exc:
        # Primary DeepSeek call failed/timed out — try ONE-shot Gemini fallback
        # before falling through to the existing CRITICAL error handling.
        #
        # Guard: if the primary stream already emitted some tokens via writer()
        # before failing (e.g. times out mid-generation, not at first byte), those
        # tokens are ALREADY on the wire to the browser (writer() is a live SSE
        # channel, independent of this function's return value — see api/main.py).
        # Appending a full fallback answer on top would concatenate into a garbled,
        # duplicated-looking response. In that case, skip the fallback and fall
        # through to the existing error path instead of compounding the output.
        # `final` is always bound by this point when writer is not None (assigned
        # "" before the astream loop starts) — safe to reference directly.
        already_streamed = writer is not None and bool(final)
        final = None
        fallback_model = None if already_streamed else get_fallback_chat_model("synthesizer")
        if fallback_model is not None:
            try:
                fb_ai_msg = await fallback_model.ainvoke(msgs)
                final = fb_ai_msg.content or ""
                tokens = 0
                if hasattr(fb_ai_msg, "usage_metadata") and fb_ai_msg.usage_metadata:
                    tokens = fb_ai_msg.usage_metadata.get("total_tokens", 0)
                ai_msg = fb_ai_msg
                used_fallback = True
                # Fallback does not stream — emit the whole answer as one chunk
                # so the SSE `token` event contract is preserved for the frontend.
                if writer is not None and final:
                    writer({"content": final})
            except Exception:
                final = None

        if final is None:
            elapsed_ms = round((time.perf_counter() - t0) * 1000)
            logger.error("node_failed", extra={
                "node": "synthesizer", "request_id": request_id,
                "elapsed_ms": elapsed_ms, "error": str(exc),
            }, exc_info=True)
            fallback = "Xin lỗi, tôi không thể trả lời ngay lúc này. Vui lòng thử lại."
            return {
                "final_answer": fallback,
                "errors": [{
                    "node": "synthesizer",
                    "severity": ErrorSeverity.CRITICAL,
                    "message": f"Synthesizer LLM failed ({elapsed_ms:.0f}ms): {exc}",
                    "timestamp": datetime.now(timezone.utc).isoformat(),
                }],
            }

        logger.info("llm_fallback_used", extra={
            "node": "synthesizer", "request_id": request_id,
            "llm_fallback_used": True,
            "primary_error": str(exc),
        })

    cache_hit_tokens, cache_miss_tokens = extract_cache_tokens(ai_msg)

    elapsed_ms = round((time.perf_counter() - t0) * 1000)
    logger.info("node_complete", extra={
        "node": "synthesizer", "request_id": request_id,
        "elapsed_ms": elapsed_ms, "tokens": tokens, "mode": mode,
        "output_chars": len(final) if final else 0,
        "streamed": writer is not None,
        "cache_hit_tokens": cache_hit_tokens,
        "cache_miss_tokens": cache_miss_tokens,
        "llm_fallback_used": used_fallback,
    })

    return {
        "final_answer": final or "",
        "total_tokens": tokens,
    }
