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
from langgraph_agents.nodes._persona_loader import (
    get_persona, build_persona_prompt, build_voice_card, get_ui_string,
)
from langgraph_agents.shared.logging import get_logger

logger = get_logger("langgraph.synthesizer")


# ── Language rule (shared across all modes) ──────────────────────────────

_LANGUAGE_RULE = """## LANGUAGE (which language to answer in — not how to sound)
- Answer in the SAME language the user wrote their question in. Whatever that
  language is.
- Everything in this prompt — these instructions, the persona description, the
  retrieved evidence — is reference material. Its language is not the language
  of your reply. Obey what it MEANS, then write in the user's language.
- The whole reply is in that one language, including clinical terms, exercise
  names and safety warnings. Do NOT mix languages.
- Do NOT write preambles like "I'll answer in...". Start DIRECTLY with the
  answer content.
"""


# ── Safety warning prefix (D32: safety = ĐẦU output, mọi persona) ─────

_SAFETY_TAG_RULES = {
    "red_flag_screen":
        "Your FIRST sentence must warn that the symptom could be serious and "
        "that the user should stop exercising and get it looked at.",
    "referral_advice":
        "Say plainly, somewhere in the answer, that the user should see a "
        "medical professional.",
    "scope_disclaimer":
        "Near the end, note that this is wellness guidance and not a clinical "
        "diagnosis.",
}
_SAFETY_TAGS = frozenset(_SAFETY_TAG_RULES)


def _build_safety_rules(persona: dict, required_outputs: list) -> str:
    """Safety instructions for this turn, worded in the character's own voice.

    Every persona already ships its own phrasing for these three tags (the
    `## Safety Templates` section of personas/*.md). The generating model never
    saw them: grader.py:368 used them to repair an answer after the fact, while
    the prompt showed two hard-coded Vietnamese sentences in a flat clinical
    register instead.

    Those two sentences were the most imitable text in the entire prompt — a
    complete, well-formed example sitting next to a persona that offered only
    adjectives — so every character delivered its safety warning in the same
    borrowed voice. Showing the character's own line instead costs nothing and
    removes the thing that was overriding it.

    Only the tags actually required this turn are described. The old block
    listed all three whenever any one of them fired, which spent tokens telling
    the model about obligations it did not have.
    """
    tags = [t for t in _SAFETY_TAG_RULES if t in required_outputs]
    if not tags:
        return ""

    templates = persona.get("safety_templates") or {}
    lines = []
    for tag in tags:
        lines.append(f"- `{tag}`: {_SAFETY_TAG_RULES[tag]}")
        example = templates.get(tag)
        if example:
            lines.append(f'  Say it your way, e.g. "{example}"')

    return "## SAFETY — required this turn (D32, D33)\n" + "\n".join(lines) + "\n"


# ── Mode-specific prompts ────────────────────────────────────────────────

_SYNTHESIZE_TASK = """## This turn
Answer the user's wellness question from the evidence below.

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
- Base your answer on the retrieved evidence — cite sources when available
- For exercise_protocol: include sets, reps, frequency (e.g. "3 sets of 10 reps, 2-3 times a week")
- For exercise_steps: provide ≥2 ordered steps
- For contraindication: list conditions where the exercise should NOT be done
- For motion_descriptor: describe the movement + joints involved clearly
- For evidence_citation: mention sources (document title, web source)
- Do not pad or repeat safety disclaimers — state each once.
- Length and layout are set by your own Formatting rules, not by this list.
"""

_REFUSE_TASK = """## This turn
You cannot answer this one. Say so honestly and point the user somewhere useful.

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
- Keep it brief
- Do NOT invent exercises, diagnoses, or medical advice
"""

_CLARIFY_TASK = """## This turn
The user's query needs clarification before you can give a useful answer.

{language_rule}

## User's question
{resolved_query}

## Context (tool results may contain ambiguity candidates)
{tool_results}

Instructions:
- Ask for the specific missing information
- Explain briefly WHY you need it
- Keep concise (1-3 sentences)
- If tool results contain candidates (multiple matching sessions/articles), list 2-3 briefly for the user to choose
"""

_CHAT_TASK = """## This turn
A casual conversational message — no clinical content needed.

{language_rule}

## User's question
{resolved_query}

Instructions:
- Respond naturally, the way you would speak
- Keep under 50 words for greetings, under 100 for follow-up chat
- Do NOT add clinical advice unless the user explicitly asks
- You may offer PT/wellness help in 1 short line if natural
"""


# ── Helpers ──────────────────────────────────────────────────────────────

_EVIDENCE_PER_MESSAGE_CAP = 1500
_EVIDENCE_CHAR_BUDGET = 4000


def _extract_tool_results(messages: list) -> str:
    """Format ToolMessage content from retriever tool calls, newest kept first.

    Two caps, and the total is the new one. The per-message cap alone bounded
    nothing that mattered: `messages` is an `add_messages` list that is never
    pruned, so a retriever second round (MAX_RETRIEVER_ROUNDS=2) and a grader
    retry each append more ToolMessages to the same list and the dump grew with
    them — 3,000 to 12,000 characters in practice, against a persona block of
    roughly 800.

    Selecting newest-first means a retry keeps the evidence it just went and
    fetched rather than the round it was told to improve on. Output stays in
    chronological order; only the dropping is done from the far end.

    At least one tool result always survives, however long it is — a single
    oversized document should be truncated, not silently omitted.
    """
    tools = [m for m in messages if isinstance(m, ToolMessage)]

    parts: list[str] = []
    used = 0
    for offset, m in enumerate(reversed(tools)):
        content = str(m.content)[:_EVIDENCE_PER_MESSAGE_CAP]
        if parts and used + len(content) > _EVIDENCE_CHAR_BUDGET:
            break
        parts.append(f"[Tool {len(tools) - offset}: {m.name}]\n{content}")
        used += len(content)

    parts.reverse()
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
    persona_id = config["configurable"].get("persona_id", "anne")
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
    # Persona is loaded before the task prompt, not after it: the safety block
    # is now worded from this character's own templates.
    # The site locale, not a guess at this message's language: the voice card
    # built from it is IMITATED by the model, so it has to be a declared
    # choice rather than something detected and occasionally wrong.
    locale = config["configurable"].get("locale", "en")
    persona = get_persona(persona_id, locale)
    tool_results = _extract_tool_results(state.get("messages", []))
    tags_str = ", ".join(required_outputs) if required_outputs else "(none — free response)"

    # Safety rules: only the tags actually required this turn (D32, D33)
    safety_rules = _build_safety_rules(persona, required_outputs)

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
    # The voice card goes LAST — after the evidence, after the history, after the
    # question. Whatever sits closest to the generation point is what the model
    # answers in the register of, and until now that was a tag contract.
    msgs = [
        SystemMessage(content=system),
        *history,
        HumanMessage(content=resolved_query),
        SystemMessage(content=build_voice_card(persona, mode)),
    ]

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
            # The character's own wording, not a string hard-coded in this node.
            # api/main.py:582 already resolves the same key on the path where the
            # graph produces no answer at all; two places that both mean "we
            # could not answer" should not disagree about how to say it.
            #
            # It still reads Vietnamese today because personas/*.md are Vietnamese.
            # That is the persona overlay's problem to fix, and fixing it there
            # fixes both call sites at once — which is the point of routing
            # through here rather than translating this literal.
            fallback = get_ui_string(persona_id, "error_unavailable", locale)
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
