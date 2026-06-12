"""VVA Agent State — REUPDATE_PLAN.md §M.1 (3-axis model).

Replaces old 6-enum intent system. Key changes (33 decisions D1-D33):
  - D1:  Intent 6-enum → 3-axis (required_outputs / resolved_query / routing bits)
  - D18: required_outputs = list[str] (tag thuần), NOT list[{tag, scope}]
  - D19: messages + summaries have NO user_id (session_id only)
  - D15: 2 independent gates: retriever⟸needs_retrieval, grader⟸required_outputs
  - D29: Synthesizer mode EMERGES from signals, no response_mode enum
"""

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
    """LangGraph agent state — 3-axis model per M.1.

    Fields are overwritten by default unless annotated with a reducer.
    """

    # ── LangGraph message passing (retriever_agent + ToolNode) ─────────
    # Carries: system prompt, user queries, assistant replies, tool calls,
    # tool results (ToolMessage). Memory node prepends context here.
    messages: Annotated[list, add_messages]

    # ── Planner output (3-axis — M.1) ──────────────────────────────────
    # TRỤC 1: required_outputs — DELIVERABLE checklist (tag ∈ TAG_RULES)
    #   Grader reads this to select rules. Empty list = no contract → skip grader.
    required_outputs: list[str]

    # TRỤC 2: resolved_query — coreference resolved, tool-selection cues kept (D21)
    #   "đào sâu về nó" → "chi tiết bài bird-dog". Synthesizer uses this as "đề bài".
    resolved_query: str

    # TRỤC 3: routing bits
    #   needs_retrieval → retriever gate (1 bit, not 3 tool flags — D2b)
    #   needs_motion → Kimodo hard gate (D3, D26)
    needs_retrieval: bool
    needs_motion: bool

    # ── Clarify (static — planner detects missing info) ────────────────
    # Dynamic clarify (tool ambiguous) is handled by synthesizer — M.2b/D22
    needs_clarification: bool

    # ── Grader output (tag-driven, rule-based — M.3) ───────────────────
    grader_result: str               # "pass" | "retry" | "pass_with_warning"
    grader_feedback: Optional[str]   # injected into retriever on retry
    retry_count: int                 # 0 → max 1 (D6: safety no retry, quality retry 1)

    # ── Synthesizer / error_handler output ─────────────────────────────
    final_answer: str

    # ── Token tracking (reducer auto-accumulates) ──────────────────────
    total_tokens: Annotated[int, operator.add]

    # ── Error tracking (append-only) ───────────────────────────────────
    errors: Annotated[list[dict], operator.add]
