"""Conversation node — dual mode: styling (restyle reasoning) or generation (greeting/clarify/fallback).

Phase 2.5: voice_path removed — TTS moves to FastAPI layer.
"""

from datetime import datetime, timezone
from langchain_core.messages import SystemMessage, HumanMessage

from langgraph_agents.state import AgentState, ErrorSeverity
from langgraph_agents.llm import get_chat_model
from langgraph_agents.nodes._persona_loader import get_persona, build_persona_prompt


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
    persona_id = config["configurable"].get("persona_id", "eca_default")
    persona = get_persona(persona_id)

    reasoning = (state.get("reasoning_output") or "").strip()
    intent = state.get("intent", "conversation")
    needs_clarification = state.get("needs_clarification", False)
    grader_warning = state.get("grader_warning")
    query = config["configurable"].get("query", "")
    plan = state.get("plan", {})

    persona_system = build_persona_prompt(persona, intent)

    # Mode selection
    if reasoning and not needs_clarification:
        # STYLING MODE
        user_msg = f"{_STYLING_INSTRUCTION}\n\n---\n{reasoning}"
    elif needs_clarification and plan.get("clarification_question"):
        # GENERATION MODE — clarify
        user_msg = _GENERATION_INSTRUCTION_CLARIFY.format(question=plan["clarification_question"])
    elif intent == "conversation":
        # GENERATION MODE — greeting/chat
        user_msg = f"{_GENERATION_INSTRUCTION_CONVERSATION}\n\nUser said: {query}"
    else:
        # GENERATION MODE — fallback (empty reasoning, no clarification question)
        user_msg = _GENERATION_INSTRUCTION_FALLBACK

    llm = get_chat_model("conversation")

    try:
        ai_msg = await llm.ainvoke([
            SystemMessage(content=persona_system),
            HumanMessage(content=user_msg),
        ])
        final = ai_msg.content
        tokens = (ai_msg.usage_metadata or {}).get("total_tokens", 0) if hasattr(ai_msg, "usage_metadata") else 0
    except Exception as exc:
        final = reasoning or "Xin lỗi, tôi không thể trả lời lúc này."
        if grader_warning:
            final = f"{final}\n\n_{grader_warning}_"
        return {
            "final_answer": final,
            "errors": [{
                "node": "conversation",
                "severity": ErrorSeverity.RECOVERABLE,
                "message": f"Persona styling failed: {exc}",
                "timestamp": datetime.now(timezone.utc).isoformat(),
            }],
        }

    return {
        "final_answer": final,
        "total_tokens": tokens,
    }
