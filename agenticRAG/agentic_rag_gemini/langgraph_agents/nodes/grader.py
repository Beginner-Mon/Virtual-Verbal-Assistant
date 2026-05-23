"""Grader node — rule-based quality check. No LLM. Retry max 1.

Replaces validator.py. Three outcomes: pass | retry | pass_with_warning.
"""

import re

from langchain_core.messages import ToolMessage

from langgraph_agents.state import AgentState


_GENERIC_FAIL_FEEDBACK = "The response is too short or lacks content. Search for more evidence and write a more detailed answer."
_EXERCISE_FAIL_FEEDBACK = (
    "Missing specific exercise guidance. The response must include either rep/set counts "
    "(\"lần\", \"hiệp\", \"reps\", \"sets\") or numbered step-by-step instructions. "
    "Add concrete actionable detail."
)
_MOTION_FAIL_FEEDBACK = "No generate_motion tool result found. Call that tool."

_WARNING_MSG = "This answer may be incomplete. Please consult a physical therapist."

# Stricter than the v2.2 draft: require either a concrete count keyword OR an
# enumerated list of >=2 steps. A lone "-" bullet or "1." line should NOT pass.
_COUNT_KEYWORDS_RE = re.compile(
    r"\b(\d+\s*(?:lần|hiệp|reps?|sets?|giây|seconds?|phút|minutes?))",
    re.IGNORECASE,
)
_ENUMERATED_STEPS_RE = re.compile(r"(?m)^\s*[1-9]\.\s+\S")


def _has_exercise_guidance(text: str) -> bool:
    if _COUNT_KEYWORDS_RE.search(text):
        return True
    # Need at least 2 enumerated steps to count as guidance
    return len(_ENUMERATED_STEPS_RE.findall(text)) >= 2


def _check_rules(state: dict) -> tuple[bool, str | None]:
    """Run rule checks. Returns (pass: bool, fail_feedback: str|None)."""
    reasoning = (state.get("reasoning_output") or "").strip()
    intent = state.get("intent", "conversation")
    messages = state.get("messages", [])

    # Rule 1: non-empty
    if not reasoning:
        return False, _GENERIC_FAIL_FEEDBACK

    # Rule 2: intent-specific checks
    if intent == "knowledge_query":
        if len(reasoning) < 50:
            return False, _GENERIC_FAIL_FEEDBACK

    if intent == "exercise_recommendation":
        if not _has_exercise_guidance(reasoning):
            return False, _EXERCISE_FAIL_FEEDBACK

    if intent == "visualize_motion":
        has_motion = any(
            isinstance(m, ToolMessage) and m.name == "generate_motion"
            for m in messages
        )
        if not has_motion:
            return False, _MOTION_FAIL_FEEDBACK

    return True, None


async def grader_node(state: AgentState) -> dict:
    """Rule-based quality check. Retry max 1 time."""
    passed, feedback = _check_rules(state)
    retry_count = state.get("retry_count", 0)

    if passed:
        return {"grader_result": "pass"}

    if retry_count == 0:
        return {
            "grader_result": "retry",
            "retry_count": 1,
            "grader_feedback": feedback,
        }

    # retry_count >= 1 — fail-safe pass with warning
    return {
        "grader_result": "pass_with_warning",
        "grader_warning": _WARNING_MSG,
    }
