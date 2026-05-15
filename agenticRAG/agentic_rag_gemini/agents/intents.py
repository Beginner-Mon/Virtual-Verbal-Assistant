"""Single source of truth for Intent and Action vocabulary.

Both ``OrchestratorAgent`` (Gemini path) and ``LocalOrchestrator`` (Ollama path)
import their enums and validators from this module.  Any new intent or action MUST
be declared here first and never duplicated in router modules.

Design notes:
    - ``IntentType`` enumerates the canonical, UI-facing intents (4 values).
    - ``ActionType`` enumerates the executable actions the Plan Executor can take.
    - ``LOCAL_INTENT_ALIASES`` maps the richer Qwen vocabulary onto canonical intents,
      so a Qwen "greeting" maps to ``IntentType.CONVERSATION``.
    - ``LOCAL_AGENT_NAMES`` is the set of agent strings the Qwen prompt is allowed to
      emit; kept here so prompt updates and validator updates stay in sync.
    - Helper ``parse_intent``/``parse_action``/``parse_local_intent`` return the
      canonical enum or a safe default; callers never raise on bad LLM output.
"""

from __future__ import annotations

from enum import Enum
from typing import FrozenSet, Optional


class IntentType(str, Enum):
    """Canonical intents exposed to the UI and used by Plan Executor."""

    CONVERSATION = "conversation"
    KNOWLEDGE_QUERY = "knowledge_query"
    EXERCISE_RECOMMENDATION = "exercise_recommendation"
    VISUALIZE_MOTION = "visualize_motion"


class ActionType(str, Enum):
    """High-level actions the orchestrator can take after intent classification."""

    RETRIEVE_MEMORY = "retrieve_memory"
    CALL_LLM = "call_llm"
    GENERATE_MOTION = "generate_motion"
    HYBRID = "hybrid"
    CLARIFY = "clarify"


# ── Local (Ollama) router vocabulary ───────────────────────────────────────────
# The Qwen prompt is trained against a richer set of strings; we keep that
# extended vocabulary documented here and map it back to the canonical
# ``IntentType`` so downstream consumers never see drift.
LOCAL_INTENT_ALIASES: dict[str, IntentType] = {
    # canonical first
    "conversation": IntentType.CONVERSATION,
    "knowledge_query": IntentType.KNOWLEDGE_QUERY,
    "exercise_recommendation": IntentType.EXERCISE_RECOMMENDATION,
    "visualize_motion": IntentType.VISUALIZE_MOTION,
    # local-only aliases
    "greeting": IntentType.CONVERSATION,
    "followup_question": IntentType.CONVERSATION,
    "resume_conversation": IntentType.CONVERSATION,
    "ask_exercise_info": IntentType.KNOWLEDGE_QUERY,
    "general_fitness_question": IntentType.KNOWLEDGE_QUERY,
    "clear_context": IntentType.CONVERSATION,
    "unknown": IntentType.KNOWLEDGE_QUERY,
}

LOCAL_VALID_INTENT_STRINGS: FrozenSet[str] = frozenset(LOCAL_INTENT_ALIASES.keys())

LOCAL_AGENT_NAMES: FrozenSet[str] = frozenset(
    {
        "retrieval_agent",
        "motion_agent",
        "web_search_agent",
        "memory_agent",
    }
)


# ── Action map (Plan Executor consults this) ───────────────────────────────────
ACTION_DEFAULT_TOOLS: dict[ActionType, tuple[str, ...]] = {
    ActionType.RETRIEVE_MEMORY: ("memory",),
    ActionType.CALL_LLM: ("memory", "documents", "web_search"),
    ActionType.GENERATE_MOTION: tuple(),
    ActionType.HYBRID: ("memory", "documents", "web_search"),
    ActionType.CLARIFY: tuple(),
}

# Intents that should not run heavy retrieval tools even when the action says so.
INTENT_SKIP_CONTENT_TOOLS: FrozenSet[IntentType] = frozenset(
    {IntentType.VISUALIZE_MOTION, IntentType.CONVERSATION}
)


# ── Parsers (defensive — never raise on bad LLM output) ────────────────────────
def parse_intent(value: Optional[str], default: IntentType = IntentType.KNOWLEDGE_QUERY) -> IntentType:
    """Parse a free-form intent string into the canonical enum.

    Accepts any value present in ``LOCAL_INTENT_ALIASES`` and gracefully degrades
    to ``default`` for unknown strings or ``None``.
    """
    if not value:
        return default
    value = value.strip().lower()
    if value in LOCAL_INTENT_ALIASES:
        return LOCAL_INTENT_ALIASES[value]
    return default


def parse_action(value: Optional[str], default: ActionType = ActionType.CALL_LLM) -> ActionType:
    """Parse a free-form action string into the canonical enum."""
    if not value:
        return default
    value = value.strip().lower()
    try:
        return ActionType(value)
    except ValueError:
        return default


def parse_local_intent(value: Optional[str]) -> IntentType:
    """Convenience alias used by ``LocalOrchestrator`` for clarity."""
    return parse_intent(value)


def is_known_local_intent(value: Optional[str]) -> bool:
    """Whether ``value`` is a known string in the Qwen prompt vocabulary."""
    if not value:
        return False
    return value.strip().lower() in LOCAL_VALID_INTENT_STRINGS


def is_known_local_agent(value: Optional[str]) -> bool:
    """Whether ``value`` is a recognised local-router agent name."""
    if not value:
        return False
    return value.strip().lower() in LOCAL_AGENT_NAMES


__all__ = [
    "ActionType",
    "IntentType",
    "ACTION_DEFAULT_TOOLS",
    "INTENT_SKIP_CONTENT_TOOLS",
    "LOCAL_AGENT_NAMES",
    "LOCAL_INTENT_ALIASES",
    "LOCAL_VALID_INTENT_STRINGS",
    "is_known_local_agent",
    "is_known_local_intent",
    "parse_action",
    "parse_intent",
    "parse_local_intent",
]
