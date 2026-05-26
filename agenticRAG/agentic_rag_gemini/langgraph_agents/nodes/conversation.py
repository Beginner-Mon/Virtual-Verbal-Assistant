"""Conversation node — dual mode: styling (restyle reasoning) or generation (greeting/clarify/fallback).

Phase 5: uses get_stream_writer for token-by-token streaming via astream_events custom events.
Phase 2.5: voice_path removed — TTS moves to FastAPI layer.
"""

import time
from datetime import datetime, timezone
from langchain_core.messages import SystemMessage, HumanMessage

from langgraph.config import get_stream_writer
from langgraph_agents.state import AgentState, ErrorSeverity
from langgraph_agents.llm import get_chat_model
from langgraph_agents.nodes._persona_loader import get_persona, build_persona_prompt
from langgraph_agents.shared.logging import get_logger

logger = get_logger("langgraph.conversation")


_STYLING_INSTRUCTION = """Restyle the following clinical response to match your personality and formatting rules.
- Do NOT add new medical information — only restyle.
- Preserve all safety warnings (rephrase in your tone).
- Respond in the same language as the input."""

_GENERATION_INSTRUCTION_CONVERSATION = """Respond naturally to the user's greeting or casual message in your persona.
- Stay in character.
- Be concise (under 50 words for greetings)."""

_GENERATION_INSTRUCTION_CLARIFY = """The planner detected that the user's query needs clarification.
Style the clarification question naturally in your persona.

Clarification question: {question}"""

_GENERATION_INSTRUCTION_FALLBACK = """The system could not produce a clinical response. Respond to the user politely in character,
explaining you need more information or that you cannot help with this specific query right now."""


async def conversation_node(state: AgentState, config) -> dict:
    t0 = time.perf_counter()
    request_id = config["configurable"].get("request_id", "-")
    persona_id = config["configurable"].get("persona_id", "eca_default")
    persona = get_persona(persona_id)

    reasoning = (state.get("reasoning_output") or "").strip()
    intent = state.get("intent", "conversation")
    needs_clarification = state.get("needs_clarification", False)
    grader_warning = state.get("grader_warning")
    query = config["configurable"].get("query", "")
    plan = state.get("plan", {})

    mode = "styling" if (reasoning and not needs_clarification) else "generation"
    logger.info("node_start", extra={
        "node": "conversation", "request_id": request_id,
        "mode": mode, "intent": intent,
    })

    # Fast path: skip LLM restyle when synthesizer already produced content.
    # Synthesizer now applies persona itself + streams tokens via writer (L1+L4
    # changes), so conversation has nothing to add for styling mode. Just
    # propagate reasoning_output as final_answer. UI already received tokens
    # during synthesizer streaming — do NOT push again here (would duplicate).
    if mode == "styling":
        final = reasoning
        if grader_warning:
            final = f"{final}\n\n_{grader_warning}_"
            # Append warning via writer (one extra chunk) so it appears in UI
            try:
                writer = get_stream_writer()
                if writer is not None:
                    writer({"content": f"\n\n_{grader_warning}_"})
            except RuntimeError:
                pass
        elapsed_ms = round((time.perf_counter() - t0) * 1000)
        logger.info("node_complete", extra={
            "node": "conversation", "request_id": request_id,
            "elapsed_ms": elapsed_ms, "mode": "styling_skipped",
            "output_chars": len(final),
        })
        return {"final_answer": final, "total_tokens": 0}

    persona_system = build_persona_prompt(persona, intent)

    # Mode selection — generation only (styling fast-pathed above)
    if needs_clarification and plan.get("clarification_question"):
        # GENERATION MODE — clarify
        user_msg = _GENERATION_INSTRUCTION_CLARIFY.format(question=plan["clarification_question"])
    elif intent == "conversation":
        # GENERATION MODE — greeting/chat
        user_msg = f"{_GENERATION_INSTRUCTION_CONVERSATION}\n\nUser said: {query}"
    else:
        # GENERATION MODE — fallback (empty reasoning, no clarification question)
        user_msg = _GENERATION_INSTRUCTION_FALLBACK

    llm = get_chat_model("conversation")
    messages = [
        SystemMessage(content=persona_system),
        HumanMessage(content=user_msg),
    ]

    try:
        writer = get_stream_writer()
    except RuntimeError:
        writer = None  # astream() without custom mode, or graph invoked via ainvoke

    try:
        if writer is not None:
            # Streaming path: astream + push tokens via StreamWriter
            final = ""
            tokens = 0
            async for chunk in llm.astream(messages):
                content = chunk.content if hasattr(chunk, "content") else str(chunk)
                if content:
                    final += content
                    writer({"content": content})
                if hasattr(chunk, "usage_metadata") and chunk.usage_metadata:
                    tokens = chunk.usage_metadata.get("total_tokens", 0)
        else:
            ai_msg = await llm.ainvoke(messages)
            final = ai_msg.content
            tokens = (ai_msg.usage_metadata or {}).get("total_tokens", 0) if hasattr(ai_msg, "usage_metadata") else 0
    except Exception as exc:
        final = reasoning or "Xin lỗi, tôi không thể trả lời lúc này."
        if grader_warning:
            final = f"{final}\n\n_{grader_warning}_"
        elapsed_ms = round((time.perf_counter() - t0) * 1000)
        logger.warning("node_failed", extra={
            "node": "conversation", "request_id": request_id,
            "elapsed_ms": elapsed_ms, "error": str(exc),
        })
        return {
            "final_answer": final,
            "errors": [{
                "node": "conversation",
                "severity": ErrorSeverity.RECOVERABLE,
                "message": f"Persona styling failed: {exc}",
                "timestamp": datetime.now(timezone.utc).isoformat(),
            }],
        }

    elapsed_ms = round((time.perf_counter() - t0) * 1000)
    logger.info("node_complete", extra={
        "node": "conversation", "request_id": request_id,
        "elapsed_ms": elapsed_ms, "tokens": tokens,
        "output_chars": len(final),
    })

    return {
        "final_answer": final,
        "total_tokens": tokens,
    }
