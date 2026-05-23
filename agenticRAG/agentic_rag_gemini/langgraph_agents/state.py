from __future__ import annotations
import operator
from enum import Enum
from typing import Annotated, Optional, TypedDict

from langgraph.graph.message import add_messages


class ErrorSeverity(str, Enum):
    CRITICAL = "critical"
    RECOVERABLE = "recoverable"
    IGNORABLE = "ignorable"


class AgentState(TypedDict):
    # ── LangGraph message passing (retriever_agent ToolNode) ─────────
    messages: Annotated[list, add_messages]

    # ── Planner output ────────────────────────────────────────────────
    intent: str                         # conversation | knowledge_query | exercise_recommendation | visualize_motion | clarify
    confidence: float
    expanded_query: str
    plan: dict                          # PlanOutput.model_dump()
    needs_clarification: bool

    # ── Memory output ────────────────────────────────────────────────
    memory_context: dict                # {short_term: [...], long_term: {...}, user_profile: {...}}

    # ── Synthesizer output ───────────────────────────────────────────
    reasoning_output: str               # clinical response OR error msg (from error_handler)

    # ── Grader output ────────────────────────────────────────────────
    grader_result: str                  # "pass" | "retry" | "pass_with_warning"
    grader_warning: Optional[str]
    grader_feedback: Optional[str]      # injected into retriever on retry
    retry_count: int                    # 0 → max 1

    # ── Conversation output ──────────────────────────────────────────
    final_answer: str

    # ── Token tracking (reducer auto-accumulates) ────────────────────
    total_tokens: Annotated[int, operator.add]

    # ── Error tracking (append-only) ─────────────────────────────────
    errors: Annotated[list[dict], operator.add]
