"""Synthesizer node — composes clinical response from retrieved evidence.

Replaces reasoning.py. Reads ToolMessages from retriever_agent instead
of retrieval_results from state. Uses LangChain ChatModel via llm.py.
"""

import time
from datetime import datetime, timezone
from langchain_core.messages import SystemMessage, HumanMessage, ToolMessage

from langgraph_agents.state import AgentState, ErrorSeverity
from langgraph_agents.llm import get_chat_model


def _extract_tool_results(messages: list) -> str:
    """Format ToolMessage content from retriever_agent's tool calls."""
    parts = []
    for i, m in enumerate(messages, 1):
        if isinstance(m, ToolMessage):
            parts.append(f"[Tool {i}: {m.name}]\n{m.content}")
    return "\n\n".join(parts) if parts else "No tool results."


def _format_memory(memory: dict) -> str:
    parts = []
    stm = memory.get("short_term") or []
    if stm:
        parts.append("Recent Q&A:\n" + "\n".join(f"- Q: {p['q']}\n  A: {p['a']}" for p in stm))
    profile = memory.get("user_profile") or {}
    if profile:
        parts.append(f"User profile: {profile}")
    return "\n\n".join(parts) if parts else "No memory."


_SYNTH_SYSTEM_PROMPT = """You are an expert physical therapist AI assistant.

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
- Use Vietnamese if user query is in Vietnamese
- Include safety warnings for exercise recommendations
- Cite sources when available
- Keep under 500 words unless topic requires detail
"""


async def synthesizer_node(state: AgentState, config) -> dict:
    plan = state.get("plan", {})
    messages = state.get("messages", [])
    memory = state.get("memory_context", {})
    query = config["configurable"]["query"]

    tool_results = _extract_tool_results(messages)
    memory_str = _format_memory(memory)

    required = ", ".join(plan.get("required_outputs", []))
    constraints = ", ".join(plan.get("constraints_detected", []))
    notes = plan.get("notes", "")

    system = _SYNTH_SYSTEM_PROMPT.format(
        required_outputs="Required: " + required if required else "",
        constraints=("Constraints: " + constraints) if constraints else "",
        notes=("Notes: " + notes) if notes else "",
        tool_results=tool_results,
        memory=memory_str,
    )

    llm = get_chat_model("synthesizer")

    t0 = time.perf_counter()
    try:
        ai_msg = await llm.ainvoke([
            SystemMessage(content=system),
            HumanMessage(content=query),
        ])
    except Exception as exc:
        elapsed_ms = (time.perf_counter() - t0) * 1000
        return {
            "reasoning_output": "",
            "errors": [{
                "node": "synthesizer",
                "severity": ErrorSeverity.CRITICAL,
                "message": f"Synthesizer LLM failed ({elapsed_ms:.0f}ms): {exc}",
                "timestamp": datetime.now(timezone.utc).isoformat(),
            }],
        }

    tokens = (ai_msg.usage_metadata or {}).get("total_tokens", 0) if hasattr(ai_msg, "usage_metadata") else 0

    return {
        "reasoning_output": ai_msg.content,
        "total_tokens": tokens,
    }
