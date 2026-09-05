"""Planner node — M.1 3-axis intent model.

Replaces old 6-enum intent system. Decisions encoded:
  D1:  3-axis (required_outputs / resolved_query / routing bits) replaces 6-enum
  D2b: Manager says WHAT (tags + query), NOT HOW (which tools)
  D9:  resolved_query with coreference resolution (memory ran first)
  D18: required_outputs = list[str] tag thuần — NOT list[{tag, scope}]
  D21: resolved_query keeps tool-selection cues (temporal/source/topic)
  D33: Danger detection = PLANNER (1 place), not synthesizer

Metaphor: Planner = manager — assigns deliverables (WHAT).
         Retriever = dev — chooses tools (HOW).
"""

from __future__ import annotations

import time
from datetime import datetime, timezone
from typing import Optional

from pydantic import BaseModel, Field

from langchain_core.runnables import RunnableConfig

from langgraph_agents.state import AgentState, ErrorSeverity
from langgraph_agents.llm import (
    get_chat_model, get_fallback_chat_model, get_gemini_cached_chat_model,
    extract_cache_tokens,
)
from langgraph_agents.shared.logging import get_logger

logger = get_logger("langgraph.planner")


# ── TAG_RULES vocabulary (M.3) — single source of truth (D7) ──────────────

# Tags planner can emit. Must match TAG_RULES in grader.py exactly.
# startup assertion in grader.py ensures planner_tags ⊆ TAG_RULES keys (D7).
PLANNER_TAGS = frozenset({
    # Safety tags (thiếu → template cứng, no retry — D6)
    "red_flag_screen",    # cảnh báo dấu hiệu nguy hiểm (đau ngực, tê, mất kiểm soát)
    "referral_advice",    # khuyên đi gặp bác sĩ/chuyên gia khi vượt scope wellness
    "scope_disclaimer",   # "đây là tư vấn wellness, không thay khám lâm sàng"

    # Quality tags (thiếu → retry max 1 — D6)
    "exercise_protocol",  # bài tập phải có sets + reps + tần suất
    "exercise_steps",     # hướng dẫn thực hiện ≥2 bước
    "contraindication",   # nêu trường hợp KHÔNG nên tập
    "evidence_citation",  # knowledge query phải có nguồn/citation
    "motion_descriptor",  # mô tả động tác + khớp (đi kèm Kimodo)
})


# ── PlanOutput (3-axis — M.1) ─────────────────────────────────────────────

class PlanOutput(BaseModel):
    """Planner output — 3 independent axes. No intent enum, no scope dicts.

    required_outputs  = deliverables (WHAT) — grader reads this
    resolved_query    = cleaned question — synthesizer reads this
    needs_retrieval   = gate for retriever node
    needs_motion      = hard gate for Kimodo node (D3, D26)
    """

    # TRỤC 1 — required_outputs: DELIVERABLE checklist
    #   Tags ∈ PLANNER_TAGS. Empty = no contract → grader skip (D8).
    required_outputs: list[str] = Field(
        default_factory=list,
        description="Delivery checklist tags. Must be from: "
                    + ", ".join(sorted(PLANNER_TAGS)),
    )

    # TRỤC 2 — resolved_query: coreference resolved, tool-selection cues kept (D21)
    #   "đào sâu về nó" → "chi tiết bài bird-dog". GIỮ SẠCH (chỉ là câu hỏi).
    resolved_query: str = Field(
        default="",
        description="User's question with coreferences resolved. Keep tool-selection "
                    "cues (temporal/source/topic), only replace pronouns.",
    )

    # TRỤC 3 — routing bits
    needs_retrieval: bool = Field(
        default=False,
        description="Does answering need external knowledge lookup? "
                    "Retriever decides which tools (kb/web/memory).",
    )
    needs_motion: bool = Field(
        default=False,
        description="Does the user want to SEE/visualize a movement? "
                    "Hard gate for Kimodo GPU node (D3).",
    )

    # ── Clarify (static — planner knows before retrieval) ─────────────
    needs_clarification: bool = Field(
        default=False,
        description="True if query is missing critical info the planner can detect. "
                    "Dynamic clarify (tool ambiguous) is handled by synthesizer (D22).",
    )


# ── System prompt ─────────────────────────────────────────────────────────

_PLANNER_SYSTEM_PROMPT = """You are the PLANNER for a physical therapy & wellness AI assistant.
Your job: analyze the user query + conversation context and produce a structured plan.

## YOUR ROLE (Manager metaphor)
You are a MANAGER — you assign DELIVERABLES (WHAT), not methods (HOW).
The RETRIEVER (dev) decides which tools to use (kb/web/memory).
You do NOT specify tools. You only say WHAT needs to be delivered and WHETHER lookup is needed.

## THE 3 AXES

### 1. required_outputs — deliverable checklist (tags)
Tags you can assign (ONLY these, no inventing):
  SAFETY (critical, hard enforcement):
    red_flag_screen   — user mentions dangerous symptoms (chest pain, numbness, dizziness, loss of control)
    referral_advice   — question is out of wellness scope, needs medical professional
    scope_disclaimer  — any clinical/exercise answer needs wellness disclaimer
  QUALITY (checked, retry if missing):
    exercise_protocol — specific exercise recommendation needs sets+reps+frequency
    exercise_steps    — movement instructions need ordered steps (≥2)
    contraindication  — exercise has risks for certain conditions, must list warnings
    evidence_citation — knowledge/explanation needs sources
    motion_descriptor — motion visualization needs movement+joint description

Rules:
- Empty list [] = casual chat/greeting/general (no contract, grader skipped — D8)
- Clinical answer ALWAYS needs at least [scope_disclaimer] (safety)
- Chest pain, numbness, dizziness, loss of bladder/bowel control, fainting
  → MUST include [red_flag_screen, referral_advice]
- Exercise recommendation → [scope_disclaimer, exercise_protocol, exercise_steps, contraindication]
- A request to SEE a movement → [motion_descriptor] + needs_motion=true
- Out of wellness scope (diagnosis, medication, test interpretation) → [referral_advice]

### 2. resolved_query — cleaned question (1 sentence)
- Resolve pronouns using conversation context (a pronoun or "that one" → the subject it refers to)
- KEEP tool-selection cues: temporal (last week, yesterday), source (latest), topic keywords
- DO NOT add search instructions or scope notes — keep it clean
- If no coreference to resolve, use the original query as-is

### 3. routing bits
- needs_retrieval=true: question needs external knowledge (KB, web, or memory search)
  Examples: PT exercises, health facts, news, real-time info, recalling past sessions
- needs_retrieval=false: greeting, casual chat, or static safety response (red_flag needs no lookup)
- needs_motion=true: user explicitly asks to SEE or VISUALIZE a movement — to be shown it,
  to have it demonstrated, simulated, animated, or rendered in 3D

### Clarify (static)
- needs_clarification=true ONLY when planner can detect missing critical info WITHOUT querying:
  "exercises" with no body region, "it hurts" with no location, "medication" with no question
- Do NOT clarify for: greetings, red-flag symptoms (answer with safety warning instead)

## EXAMPLES

These show HOW TO ASSIGN TAGS, not what language to work in. The user writes in
whatever language they like and the plan you return is the same either way.
Keep `resolved_query` in the user's own language: it is their question tidied
up, not a translation of it.

Query: "hello"
-> {"required_outputs":[],"resolved_query":"hello","needs_retrieval":false,"needs_motion":false,"needs_clarification":false}

Query: "i get chest pain when i exercise"
-> {"required_outputs":["red_flag_screen","referral_advice"],"resolved_query":"chest pain during exercise","needs_retrieval":false,"needs_motion":false,"needs_clarification":false}

Query: "exercises for an L4-L5 disc herniation"
-> {"required_outputs":["scope_disclaimer","exercise_protocol","exercise_steps","contraindication"],"resolved_query":"physiotherapy exercises for an L4-L5 disc herniation","needs_retrieval":true,"needs_motion":false,"needs_clarification":false}

Query: "exercises" (missing body region - critical)
-> {"required_outputs":[],"resolved_query":"exercises","needs_retrieval":false,"needs_motion":false,"needs_clarification":true}

Query: "gold price today" (outside wellness, but answerable from sources)
-> {"required_outputs":["evidence_citation"],"resolved_query":"gold price today","needs_retrieval":true,"needs_motion":false,"needs_clarification":false}

Query: "show me the squat movement"
-> {"required_outputs":["scope_disclaimer","motion_descriptor","exercise_steps"],"resolved_query":"squat movement","needs_retrieval":true,"needs_motion":true,"needs_clarification":false}

Query: "i asked about neck exercises last week, remind me" (recall past session)
-> {"required_outputs":["scope_disclaimer","exercise_protocol","exercise_steps"],"resolved_query":"neck exercises asked about last week","needs_retrieval":true,"needs_motion":false,"needs_clarification":false}

Query: "can you prescribe something for the pain" (outside wellness scope)
-> {"required_outputs":["referral_advice"],"resolved_query":"medication for pain","needs_retrieval":false,"needs_motion":false,"needs_clarification":false}

## INPUT NOTE
Users write in any language, and may omit diacritics, accents or tone marks -
they often type an unaccented spelling of an accented word. Infer intent from
the unaccented form, and do NOT downgrade confidence because accents are absent.

Respond as a single JSON object matching the schema."""


# ── Node ──────────────────────────────────────────────────────────────────

async def planner_node(state: AgentState, config: RunnableConfig) -> dict:
    """Planner node — classify intent into 3-axis PlanOutput (M.1).

    Reads: messages (context assembled by memory node) + config.query
    Outputs: required_outputs, resolved_query, needs_retrieval, needs_motion,
             needs_clarification
    """
    t0 = time.perf_counter()
    request_id = config["configurable"].get("request_id", "-")
    query = config["configurable"]["query"]

    logger.info("node_start", extra={
        "node": "planner", "request_id": request_id,
        "query_preview": query[:80],
    })

    llm = get_chat_model("planner")

    # get_chat_model returns None when no provider can be built at all — no
    # DeepSeek key AND no Gemini fallback. That is reachable in production, not
    # just in tests: llm.py reads both credentials from SSM SecureStrings and
    # deliberately returns None rather than raising when a fetch fails, so that a
    # credential problem degrades one provider instead of killing the request.
    #
    # Without this guard the very next line does `None.with_structured_output`
    # and the AttributeError escapes planner_node entirely — the graph raises
    # instead of degrading, which defeats the whole point of that design three
    # files away. The `try` below starts AFTER this line, so it cannot catch it.
    #
    # Returning the same shape the LLM-failure path returns: ask the user to
    # rephrase, and report RECOVERABLE so the graph carries on.
    if llm is None:
        elapsed_ms = round((time.perf_counter() - t0) * 1000)
        logger.error("node_failed", extra={
            "node": "planner", "request_id": request_id,
            "elapsed_ms": elapsed_ms,
            "error": "no chat model available (no DeepSeek key and no Gemini fallback)",
        })
        return {
            "required_outputs": [],
            "resolved_query": query,
            "needs_retrieval": False,
            "needs_motion": False,
            "needs_clarification": True,
            "errors": [{
                "node": "planner",
                "severity": ErrorSeverity.RECOVERABLE,
                "message": "No LLM provider is configured or reachable",
                "timestamp": datetime.now(timezone.utc).isoformat(),
            }],
        }

    # include_raw=True so we can read response_metadata/usage_metadata for
    # DeepSeek prompt-cache telemetry (fix #1) alongside the parsed plan.
    structured_llm = llm.with_structured_output(PlanOutput, method="json_mode", include_raw=True)

    # Build user message: query + context hint from recent messages
    messages = state.get("messages", [])
    context_hint = ""
    if len(messages) > 1:
        # Last few messages for context (exclude system messages)
        recent = [m for m in messages[-6:] if hasattr(m, "content") and m.content]
        if recent:
            context_hint = "\n\nRecent context:\n" + "\n".join(
                f"[{getattr(m, 'type', 'unknown')}]: {str(m.content)[:200]}"
                for m in recent[-4:]
            )

    user_msg = query + context_hint

    prompt_msgs = [
        ("system", _PLANNER_SYSTEM_PROMPT),
        ("user", user_msg),
    ]

    def _unpack(result):
        """include_raw=True returns {"raw": AIMessage, "parsed": PlanOutput|None, ...}.
        Existing unit tests mock ainvoke to return a bare PlanOutput directly —
        handle both shapes."""
        if isinstance(result, dict):
            return result.get("raw"), result.get("parsed")
        return None, result

    ai_msg = None
    used_fallback = False

    try:
        raw_result = await structured_llm.ainvoke(prompt_msgs)
        ai_msg, plan = _unpack(raw_result)
    except Exception as exc:
        # Primary DeepSeek call failed/timed out — try ONE-shot Gemini fallback
        # before falling through to the existing safe-fallback error handling.
        plan = None

        # Cached-context Gemini fallback: only taken if a cache is ALREADY warm
        # (in-memory lookup, no network call — see llm.get_gemini_cached_chat_model).
        # Nothing warms the cache today, so this is currently always a no-op and
        # falls through to the regular fallback below — kept ready for when
        # something calls llm.warm_gemini_cache("planner", ...) (e.g. a startup
        # hook, once Gemini is on a tier that actually allows caching).
        cached_model = get_gemini_cached_chat_model("planner")
        if cached_model is not None:
            try:
                cached_structured = cached_model.with_structured_output(
                    PlanOutput, method="json_mode", include_raw=True,
                )
                # cached_content already carries the system prompt — sending it
                # again would violate the API's "no system_instruction with
                # cached_content" constraint, so send only the user turn.
                fb_result = await cached_structured.ainvoke([("user", user_msg)])
                ai_msg, plan = _unpack(fb_result)
                used_fallback = plan is not None
            except Exception:
                plan = None

        if plan is None:
            fallback_model = get_fallback_chat_model("planner")
            if fallback_model is not None:
                try:
                    fallback_structured = fallback_model.with_structured_output(
                        PlanOutput, method="json_mode", include_raw=True,
                    )
                    fb_result = await fallback_structured.ainvoke(prompt_msgs)
                    ai_msg, plan = _unpack(fb_result)
                    used_fallback = plan is not None
                except Exception:
                    plan = None

        if plan is None:
            elapsed_ms = round((time.perf_counter() - t0) * 1000)
            logger.warning("node_failed", extra={
                "node": "planner", "request_id": request_id,
                "elapsed_ms": elapsed_ms, "error": str(exc),
            })
            # Safe fallback: assume chat, no tags, no retrieval
            return {
                "required_outputs": [],
                "resolved_query": query,
                "needs_retrieval": False,
                "needs_motion": False,
                "needs_clarification": True,
                "errors": [{
                    "node": "planner",
                    "severity": ErrorSeverity.RECOVERABLE,
                    "message": f"LLM call failed ({elapsed_ms:.0f}ms): {exc}",
                    "timestamp": datetime.now(timezone.utc).isoformat(),
                }],
            }

        logger.info("llm_fallback_used", extra={
            "node": "planner", "request_id": request_id,
            "llm_fallback_used": True,
            "primary_error": str(exc),
        })

    # ── Handle None plan (circuit breaker open / chain returns None) ───
    if plan is None:
        elapsed_ms = round((time.perf_counter() - t0) * 1000)
        logger.warning("node_failed", extra={
            "node": "planner", "request_id": request_id,
            "elapsed_ms": elapsed_ms, "error": "LLM chain returned None (breaker open?)",
        })
        return {
            "required_outputs": [],
            "resolved_query": query,
            "needs_retrieval": False,
            "needs_motion": False,
            "needs_clarification": True,
            "errors": [{
                "node": "planner",
                "severity": ErrorSeverity.RECOVERABLE,
                "message": f"LLM chain returned None ({elapsed_ms:.0f}ms)",
                "timestamp": datetime.now(timezone.utc).isoformat(),
            }],
        }

    # ── Post-validate ─────────────────────────────────────────────────
    # Filter tags to known vocabulary (D7: no invented tags)
    valid_tags = [t for t in plan.required_outputs if t in PLANNER_TAGS]
    if len(valid_tags) != len(plan.required_outputs):
        unknown = set(plan.required_outputs) - PLANNER_TAGS
        logger.warning("unknown_tags_filtered", extra={
            "request_id": request_id, "unknown": list(unknown),
        })

    # Safety override: red_flag_screen always gets referral_advice too (D33)
    if "red_flag_screen" in valid_tags and "referral_advice" not in valid_tags:
        valid_tags.append("referral_advice")

    # If needs_clarification, don't set tags — clarify is its own path
    needs_clarification = plan.needs_clarification
    if needs_clarification:
        valid_tags = []

    # Resolved query fallback
    resolved_query = plan.resolved_query.strip() or query

    cache_hit_tokens, cache_miss_tokens = extract_cache_tokens(ai_msg)

    elapsed_ms = round((time.perf_counter() - t0) * 1000)
    logger.info("node_complete", extra={
        "node": "planner", "request_id": request_id,
        "elapsed_ms": elapsed_ms,
        "tags": valid_tags,
        "needs_retrieval": plan.needs_retrieval,
        "needs_motion": plan.needs_motion,
        "needs_clarification": needs_clarification,
        "cache_hit_tokens": cache_hit_tokens,
        "cache_miss_tokens": cache_miss_tokens,
        "llm_fallback_used": used_fallback,
    })

    return {
        "required_outputs": valid_tags,
        "resolved_query": resolved_query,
        "needs_retrieval": plan.needs_retrieval and not needs_clarification,
        "needs_motion": plan.needs_motion and not needs_clarification,
        "needs_clarification": needs_clarification,
    }
