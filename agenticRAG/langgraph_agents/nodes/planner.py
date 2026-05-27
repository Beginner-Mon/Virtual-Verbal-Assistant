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
from langgraph_agents.shared.logging import get_logger

logger = get_logger("langgraph.planner")


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
- conversation            : greetings, thanks, follow-ups, casual chat, no clinical content
- knowledge_query         : explanation, facts, non-motion advice
- exercise_recommendation : exercises, stretches, workouts
- visualize_motion        : asks to SEE / animate a specific movement
- clarify                 : query mentions an action/topic but is missing critical specifics

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

Vietnamese input note:
Users frequently type Vietnamese WITHOUT diacritics (mobile typing, no IME).
Treat "xin chao" === "xin chào", "cam on" === "cảm ơn", "bai tap" === "bài tập",
"dau lung" === "đau lưng", etc. Do NOT downgrade confidence just because diacritics
are missing — recognize the intent from the unaccented form.

Rules:
- Clear greetings / thanks / casual replies → intent=conversation, confidence ≥ 0.8,
  needs_clarification=false. NEVER route greetings to clarify.
- clarify ONLY when the query points at an action or topic but omits a critical
  specific (e.g. "bài tập" without body region, "đau" without location).
  Confidence-alone is not sufficient grounds for clarify.
- expanded_query: add anatomical / physiotherapy synonyms (for non-conversation intents).
- For greetings: required_outputs=["greeting_response"], search_strategy=[].

Examples:

Query: "Xin chào"
→ {"intent":"conversation","confidence":0.95,"expanded_query":"Xin chào","needs_clarification":false,"required_outputs":["greeting_response"],"search_strategy":[]}

Query: "Xin chao" (no diacritics)
→ {"intent":"conversation","confidence":0.9,"expanded_query":"Xin chào","needs_clarification":false,"required_outputs":["greeting_response"],"search_strategy":[]}

Query: "Hi" / "Hello" / "Cảm ơn nhé"
→ {"intent":"conversation","confidence":0.95,"needs_clarification":false,"required_outputs":["greeting_response"],"search_strategy":[]}

Query: "Bài tập cho đau lưng"
→ {"intent":"exercise_recommendation","confidence":0.9,"expanded_query":"bài tập vật lý trị liệu cho đau thắt lưng (lower back pain)","needs_clarification":false,"required_outputs":["exercise_name","description","sets_reps","safety_warnings"],"search_strategy":["pgvector_search","web_search_if_low_quality"]}

Query: "Bài tập" (missing body region)
→ {"intent":"clarify","confidence":0.4,"needs_clarification":true,"clarification_question":"Bạn muốn bài tập cho vùng nào (lưng, cổ, vai, đầu gối...)?","required_outputs":["clarification_question"]}

Query: "Cho tôi xem động tác giơ tay phải lên 90 độ"
→ {"intent":"visualize_motion","confidence":0.95,"needs_clarification":false,"required_outputs":["motion_description","joint_constraints"],"search_strategy":["generate_motion"]}

Query: "Đau lưng là gì?"
→ {"intent":"knowledge_query","confidence":0.9,"expanded_query":"định nghĩa đau lưng (lower back pain) nguyên nhân triệu chứng","needs_clarification":false,"required_outputs":["answer","sources"],"search_strategy":["pgvector_search","web_search_if_low_quality"]}

Respond strictly as a single valid JSON object matching the required schema."""

_VALID_INTENTS = {"conversation", "knowledge_query", "exercise_recommendation", "visualize_motion", "clarify"}


async def planner_node(state: AgentState, config) -> dict:
    """Intent classification + query expansion + structured plan output."""
    t0 = time.perf_counter()
    request_id = config["configurable"].get("request_id", "-")
    query = config["configurable"]["query"]

    logger.info("node_start", extra={
        "node": "planner", "request_id": request_id,
        "query_preview": query[:80],
    })

    llm = get_chat_model("planner")
    # DeepSeek (v4-pro thinking model) constraints:
    #   - json_schema  → "response_format type unavailable"
    #   - function_calling → "thinking mode does not support tool_choice"
    # → use json_mode: schema is injected into the prompt and `response_format=
    #   {"type": "json_object"}` is set; Pydantic validates the returned JSON.
    structured_llm = llm.with_structured_output(PlanOutput, method="json_mode")

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

    try:
        plan: PlanOutput = await structured_llm.ainvoke([
            ("system", _PLANNER_SYSTEM_PROMPT),
            ("user", user_msg),
        ])
        llm_elapsed_ms = round((time.perf_counter() - t0) * 1000)
    except Exception as exc:
        elapsed_ms = round((time.perf_counter() - t0) * 1000)
        logger.warning("node_failed", extra={
            "node": "planner", "request_id": request_id,
            "elapsed_ms": elapsed_ms, "error": str(exc),
        })
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

    elapsed_ms = round((time.perf_counter() - t0) * 1000)
    logger.info("node_complete", extra={
        "node": "planner", "request_id": request_id,
        "elapsed_ms": elapsed_ms, "intent": intent,
        "confidence": plan.confidence, "needs_clarification": needs_clarification,
    })

    return {
        "intent": intent,
        "confidence": plan.confidence,
        "expanded_query": plan.expanded_query or query,
        "plan": plan_dict,
        "needs_clarification": needs_clarification,
    }
