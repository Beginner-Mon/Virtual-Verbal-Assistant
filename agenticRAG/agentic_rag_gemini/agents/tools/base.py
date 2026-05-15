"""Tool ABC and shared types for the Agentic RAG Plan Executor.

Every retrieval/generation utility the orchestrator can invoke MUST inherit from
``Tool``.  This gives us:

- A single, named interface (``run``) regardless of underlying service.
- Declarative intent gating (``applicable_intents``) so the orchestrator does
  not need a hard-coded ``_ACTION_TOOL_MAP``.
- Uniform error handling: any tool can fail without aborting the rest.

The legacy concrete classes (``MemoryTool``, ``DocumentRetrievalTool`` …) keep
their existing public methods for callers that haven't migrated yet; we only
add a ``Tool``-compliant wrapper around them in ``registry.py``.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any, Dict, FrozenSet, Optional

from agents.intents import IntentType


@dataclass
class ToolContext:
    """Bundle passed to every tool on each invocation.

    Attributes:
        query:                The (possibly expanded) user query string.
        user_id:              Stable user identifier; used for per-user collections.
        intent:               Canonical intent that triggered this run.
        request_id:           Trace correlation ID propagated from the gateway.
        conversation_history: Last N turns, may be empty.
        extras:               Free-form bag for tool-specific overrides.
    """

    query: str
    user_id: str = "default"
    intent: Optional[IntentType] = None
    request_id: Optional[str] = None
    conversation_history: Optional[list] = None
    extras: Dict[str, Any] = field(default_factory=dict)


@dataclass
class ToolResult:
    """Uniform return shape for every tool."""

    name: str
    payload: Any = None
    success: bool = True
    error: Optional[str] = None
    elapsed_ms: float = 0.0
    metadata: Dict[str, Any] = field(default_factory=dict)


class Tool(ABC):
    """Abstract tool interface.

    Subclasses must declare:
        - ``name``               : unique identifier used by the registry.
        - ``applicable_intents`` : the set of intents for which this tool runs.

    And implement ``run(ctx) -> ToolResult``.
    """

    #: Unique tool name; must match registry key.
    name: str = ""

    #: Set of intents this tool wants to participate in.  An empty set means
    #: "always opt-in unless the intent is explicitly excluded by the planner".
    applicable_intents: FrozenSet[IntentType] = frozenset()

    @abstractmethod
    def run(self, ctx: ToolContext) -> ToolResult:
        """Execute the tool and return a ``ToolResult``.

        Implementations MUST NOT raise — they should swallow exceptions and
        return ``ToolResult(success=False, error=...)`` instead.  This keeps the
        Plan Executor's parallel run loop simple.
        """

    # Optional capability declarations. Override in subclasses if applicable.
    requires_user_id: bool = False
    """If True, the executor will skip running when ``ctx.user_id`` is unset."""

    is_io_bound: bool = True
    """All current tools are I/O bound; CPU-bound tools should set this False."""


__all__ = ["Tool", "ToolContext", "ToolResult"]
