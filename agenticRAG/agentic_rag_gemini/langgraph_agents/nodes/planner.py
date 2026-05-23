"""Planner node — intent classification + query expansion + structured plan output.

Replaces manager.py. Uses LangChain ChatModel with with_structured_output()
for typed PlanOutput via Pydantic.
"""

import time
from datetime import datetime, timezone
from typing import Optional

from pydantic import BaseModel, Field

from langgraph_agents.state import AgentState, ErrorSeverity
from langgraph_agents.llm import get_chat_model


class PlanOutput(BaseModel):
    """Defaults are intentionally permissive — DeepSeek (thinking model in
    json_mode) frequently omits fields. Planner_node post-processes for the
    rest. Only `intent` is mandatory in spirit; we tolerate omission and
    fall back at the node level."""
    intent: str = Field(
        default="clarify",
        description="conversation | knowledge_query | exercise_recommendation | visualize_motion | clarify",
    )
    confidence: float = Field(default=0.8, ge=0.0, le=1.0)
    expanded_query: str = ""
    needs_clarification: bool = False
    clarification_question: Optional[str] = None
    required_outputs: list[str] = Field(default_factory=list)
    search_strategy: list[str] = Field(default_factory=list)
    constraints_detected: list[str] = Field(default_factory=list)
    notes: Optional[str] = None


_PLANNER_SYSTEM_PROMPT = """You are the planning brain for a physical therapy AI assistant.

Analyze the user query + memory context and produce a structured plan.

Intents:
- conversation            : greetings, follow-ups, no exercise content
- knowledge_query         : explanation, facts, non-motion advice
- exercise_recommendation : exercises, stretches, workouts
- visualize_motion        : asks to SEE / animate a specific movement
- clarify                 : query too vague to route

Required outputs per intent:
- knowledge_query         : ["answer", "sources"]
- exercise_recommendation : ["exercise_name", "description", "sets_reps", "safety_warnings"]
- visualize_motion        : ["motion_description", "joint_constraints"]
- conversation            : ["greeting_response"]
- clarify                 : ["clarification_question"]

Search strategy (suggest tools for retriever):
- "pgvector_search"           : internal knowledge base
- "web_search_if_low_quality" : fallback to web
- "generate_motion"           : Kimodo motion synthesis (visualize_motion only)

Rules:
- confidence < 0.5 → needs_clarification = true + provide clarification_question
- Detect missing critical info (e.g. exercise without body region) → needs_clarification
- expanded_query: add anatomical/physiotherapy synonyms
- For greetings, set intent=conversation, required_outputs=["greeting_response"], search_strategy=[]

Respond strictly as a single valid JSON object matching the required schema."""

_VALID_INTENTS = {"conversation", "knowledge_query", "exercise_recommendation", "visualize_motion", "clarify"}


async def planner_node(state: AgentState, config) -> dict:
    """Intent classification + query expansion + structured plan output."""
    llm = get_chat_model("planner")
    # DeepSeek (v4-pro thinking model) constraints:
    #   - json_schema  → "response_format type unavailable"
    #   - function_calling → "thinking mode does not support tool_choice"
    # → use json_mode: schema is injected into the prompt and `response_format=
    #   {"type": "json_object"}` is set; Pydantic validates the returned JSON.
    structured_llm = llm.with_structured_output(PlanOutput, method="json_mode")

    query = config["configurable"]["query"]
    memory = state.get("memory_context", {})

    # Build context snippet from memory
    stm = memory.get("short_term") or []
    history_snippet = ""
    if stm:
        history_snippet = "\n\nRecent Q&A:\n" + "\n".join(
            f"Q: {p['q']}\nA: {p['a']}" for p in stm[-3:]
        )

    profile = memory.get("user_profile") or {}
    profile_snippet = f"\n\nUser profile: {profile}" if profile else ""

    ltm = memory.get("long_term") or {}
    ltm_snippet = ""
    if ltm.get("ambiguous"):
        ltm_snippet = "\n\nNote: Multiple past sessions matched recall — ask user for clarification."
    elif ltm.get("results"):
        ltm_snippet = "\n\nRelevant past context found in memory."

    user_msg = query + history_snippet + profile_snippet + ltm_snippet

    t0 = time.perf_counter()
    try:
        plan: PlanOutput = await structured_llm.ainvoke([
            ("system", _PLANNER_SYSTEM_PROMPT),
            ("user", user_msg),
        ])
        elapsed_ms = (time.perf_counter() - t0) * 1000
    except Exception as exc:
        elapsed_ms = (time.perf_counter() - t0) * 1000
        return {
            "intent": "clarify",
            "confidence": 0.3,
            "expanded_query": query,
            "plan": {},
            "needs_clarification": True,
            "errors": [{
                "node": "planner",
                "severity": ErrorSeverity.RECOVERABLE,
                "message": f"LLM call failed ({elapsed_ms:.0f}ms): {exc}",
                "timestamp": datetime.now(timezone.utc).isoformat(),
            }],
        }

    # Validate intent
    intent = plan.intent if plan.intent in _VALID_INTENTS else "clarify"

    # Hard rule (Plan v2.4 §5.4): ambiguous LTM → force clarification
    # LLM may ignore the prompt hint, so enforce here.
    needs_clarification = plan.needs_clarification
    plan_dict = plan.model_dump()
    if ltm.get("ambiguous"):
        needs_clarification = True
        if not plan_dict.get("clarification_question"):
            plan_dict["clarification_question"] = (
                "Tôi thấy nhiều phiên trò chuyện trước đó. Bạn có thể nhớ thêm chi tiết "
                "(chủ đề, thời gian gần đúng, hoặc bài tập cụ thể) để tôi tìm chính xác hơn không?"
            )

    return {
        "intent": intent,
        "confidence": plan.confidence,
        "expanded_query": plan.expanded_query or query,
        "plan": plan_dict,
        "needs_clarification": needs_clarification,
    }
