"""Synthesizer node — universal response generator with persona voice.

Replaces both reasoning.py and conversation.py (after Phase 6.9 refactor).
Handles 3 modes in a single LLM call:
  - CLINICAL: tool results present → generate evidence-based response
  - CLARIFY:  needs_clarification → ask styled clarification question
  - CHAT:     intent=conversation (greeting/casual) → respond naturally

All 3 modes carry the persona voice (loaded from persona MD file).
"""

import time
from datetime import datetime, timezone
from langchain_core.messages import SystemMessage, HumanMessage, ToolMessage

from langgraph.config import get_stream_writer
from langgraph_agents.state import AgentState, ErrorSeverity
from langgraph_agents.llm import get_chat_model
from langgraph_agents.nodes._persona_loader import get_persona, build_persona_prompt
from langgraph_agents.shared.logging import get_logger

logger = get_logger("langgraph.synthesizer")


def _extract_tool_results(messages: list) -> str:
    """Format ToolMessage content from retriever_agent's tool calls."""
    parts = []
    for i, m in enumerate(messages, 1):
        if isinstance(m, ToolMessage):
            parts.append(f"[Tool {i}: {m.name}]\n{m.content}")
    return "\n\n".join(parts) if parts else ""


def _format_memory(memory: dict) -> str:
    parts = []
    stm = memory.get("short_term") or []
    if stm:
        parts.append("Recent Q&A:\n" + "\n".join(f"- Q: {p['q']}\n  A: {p['a']}" for p in stm))
    profile = memory.get("user_profile") or {}
    if profile:
        parts.append(f"User profile: {profile}")
    return "\n\n".join(parts) if parts else ""


# ── Mode-specific task prompts ──────────────────────────────────────

# Language rule shared across all modes — placed first to make it priority.
_LANGUAGE_RULE = """CRITICAL LANGUAGE RULE (overrides any persona default):
- Supported languages: ENGLISH and VIETNAMESE only.
- Detect the user's query language:
    * If query is mostly Vietnamese → respond entirely in Vietnamese
    * If query is mostly English (or any other language) → respond entirely in English
- Do NOT mix languages within the response.
- Do NOT write any preamble like "I'll answer in English" or "Tôi sẽ trả lời...".
- Start your response DIRECTLY with the answer content.
"""

_CLINICAL_TASK = """You are an expert physical therapist AI assistant.
Stay in the persona voice defined above.

{language_rule}

## Plan requirements
{required_outputs}
{constraints}
{notes}

## Retrieved evidence
{tool_results}

## Patient memory
{memory}

Instructions:
- Cover ALL required_outputs from the plan
- Respond in the persona voice (tone, formatting from persona block above)
- Include safety warnings for exercise recommendations
- Cite sources when available
- Keep under 500 words unless topic requires detail
"""

_CHAT_TASK = """You are responding to a casual conversational message from the user.
Stay in the persona voice defined above.

{language_rule}

## Patient memory
{memory}

Instructions:
- Respond naturally in your persona voice (warm, brief, friendly)
- Keep under 50 words for greetings, under 100 for follow-up chat
- Do NOT add clinical advice unless the user explicitly asks
"""

_CLARIFY_TASK = """The user's query needs clarification before you can give a useful answer.
Stay in the persona voice defined above.

{language_rule}

## Required clarification question
{clarification_question}

## Patient memory
{memory}

Instructions:
- Rephrase the clarification question in your persona voice (warm, professional)
- Keep concise (1-3 sentences)
- Briefly explain WHY you need this info (so user understands)
"""


# ── Mode detection + prompt build ──────────────────────────────────

def _select_mode_and_prompt(state, query, memory_str) -> tuple[str, str]:
    """Returns (mode_label, task_system_prompt) based on state.

    Mode priority:
      1. needs_clarification → CLARIFY
      2. Has ToolMessages in state.messages → CLINICAL
      3. Otherwise (intent=conversation or empty) → CHAT
    """
    plan = state.get("plan", {})
    needs_clarification = state.get("needs_clarification", False)
    has_tools = any(isinstance(m, ToolMessage) for m in state.get("messages", []))

    if needs_clarification and plan.get("clarification_question"):
        return "clarify", _CLARIFY_TASK.format(
            language_rule=_LANGUAGE_RULE,
            clarification_question=plan["clarification_question"],
            memory=memory_str or "(none)",
        )

    if has_tools:
        tool_results = _extract_tool_results(state.get("messages", []))
        required = ", ".join(plan.get("required_outputs", []))
        constraints = ", ".join(plan.get("constraints_detected", []))
        notes = plan.get("notes", "")
        return "clinical", _CLINICAL_TASK.format(
            language_rule=_LANGUAGE_RULE,
            required_outputs=("Required: " + required) if required else "",
            constraints=("Constraints: " + constraints) if constraints else "",
            notes=("Notes: " + notes) if notes else "",
            tool_results=tool_results or "(no results)",
            memory=memory_str or "(none)",
        )

    # Default: chat/greeting
    return "chat", _CHAT_TASK.format(language_rule=_LANGUAGE_RULE, memory=memory_str or "(none)")


async def synthesizer_node(state: AgentState, config) -> dict:
    t0 = time.perf_counter()
    request_id = config["configurable"].get("request_id", "-")
    persona_id = config["configurable"].get("persona_id", "eca_default")
    intent = state.get("intent", "")
    memory = state.get("memory_context", {})
    query = config["configurable"]["query"]

    memory_str = _format_memory(memory)
    mode, task_system = _select_mode_and_prompt(state, query, memory_str)

    logger.info("node_start", extra={
        "node": "synthesizer", "request_id": request_id,
        "intent": intent, "mode": mode, "persona_id": persona_id,
    })

    persona = get_persona(persona_id)
    persona_system = build_persona_prompt(persona, intent)
    system = f"{persona_system}\n\n---\n\n{task_system}"

    llm = get_chat_model("synthesizer")

    try:
        writer = get_stream_writer()
    except RuntimeError:
        writer = None

    try:
        msgs = [SystemMessage(content=system), HumanMessage(content=query)]
        if writer is not None:
            final = ""
            tokens = 0
            async for chunk in llm.astream(msgs):
                content = chunk.content if hasattr(chunk, "content") else str(chunk)
                if content:
                    final += content
                    writer({"content": content})
                if hasattr(chunk, "usage_metadata") and chunk.usage_metadata:
                    tokens = chunk.usage_metadata.get("total_tokens", 0)
        else:
            ai_msg = await llm.ainvoke(msgs)
            final = ai_msg.content
            tokens = (ai_msg.usage_metadata or {}).get("total_tokens", 0) if hasattr(ai_msg, "usage_metadata") else 0
    except Exception as exc:
        elapsed_ms = round((time.perf_counter() - t0) * 1000)
        logger.error("node_failed", extra={
            "node": "synthesizer", "request_id": request_id,
            "elapsed_ms": elapsed_ms, "error": str(exc),
        }, exc_info=True)
        fallback = "Xin lỗi, tôi không thể trả lời ngay lúc này. Vui lòng thử lại."
        return {
            "reasoning_output": fallback,
            "final_answer": fallback,
            "errors": [{
                "node": "synthesizer",
                "severity": ErrorSeverity.CRITICAL,
                "message": f"Synthesizer LLM failed ({elapsed_ms:.0f}ms): {exc}",
                "timestamp": datetime.now(timezone.utc).isoformat(),
            }],
        }

    elapsed_ms = round((time.perf_counter() - t0) * 1000)
    logger.info("node_complete", extra={
        "node": "synthesizer", "request_id": request_id,
        "elapsed_ms": elapsed_ms, "tokens": tokens, "mode": mode,
        "output_chars": len(final) if final else 0,
        "streamed": writer is not None,
    })

    # Set both reasoning_output (legacy/grader compat) and final_answer
    return {
        "reasoning_output": final,
        "final_answer": final,
        "total_tokens": tokens,
    }
