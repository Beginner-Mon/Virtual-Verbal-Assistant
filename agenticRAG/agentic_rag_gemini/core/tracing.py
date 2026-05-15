"""Lightweight per-request agent tracing.

The goal is to make agent decisions inspectable without dragging in OpenTelemetry
or any external tracing system.  An ``AgentTrace`` is a plain dataclass that:

- Is created once per HTTP request (in the gateway / FastAPI router).
- Is passed by reference into the router, plan executor, and tools.
- Accumulates ordered ``decisions`` and timing-only ``timings_ms`` slots.
- Is serialised back into the API response when ``AGENTIC_TRACE=1``.

Why not use ``contextvars``?
    Contextvars survive across ``asyncio`` await boundaries but get lost in
    ``ThreadPoolExecutor`` runs unless explicitly bridged.  Passing the trace
    explicitly is annoying but makes the data flow obvious in code review and
    matches what we want to draw in the thesis sequence diagram.

Environment:
    ``AGENTIC_TRACE=1`` enables attaching the trace to API responses.  When
    disabled, traces still collect (overhead is negligible) but are not
    surfaced to the user.
"""

from __future__ import annotations

import os
import time
import uuid
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional


def _epoch_ms() -> int:
    return int(time.time() * 1000)


def is_tracing_enabled() -> bool:
    """Whether agent traces should be exposed to API responses."""
    return os.getenv("AGENTIC_TRACE", "0").strip().lower() in {"1", "true", "yes", "on"}


@dataclass
class AgentDecision:
    """A single decision emitted by some component during a request."""

    stage: str
    payload: Dict[str, Any] = field(default_factory=dict)
    epoch_ms: int = field(default_factory=_epoch_ms)


@dataclass
class AgentTrace:
    """Ordered, attachable record of everything a single request did."""

    request_id: str = field(default_factory=lambda: f"req_{uuid.uuid4().hex[:12]}")
    user_id: Optional[str] = None
    intent: Optional[str] = None
    action: Optional[str] = None
    started_at_ms: int = field(default_factory=_epoch_ms)
    ended_at_ms: Optional[int] = None

    decisions: List[AgentDecision] = field(default_factory=list)
    tools_run: List[str] = field(default_factory=list)
    llm_calls: int = 0
    timings_ms: Dict[str, float] = field(default_factory=dict)

    # ── Mutators ───────────────────────────────────────────────────────────
    def add_decision(self, stage: str, **payload: Any) -> None:
        self.decisions.append(AgentDecision(stage=stage, payload=dict(payload)))

    def record_tool(self, tool_name: str, elapsed_ms: float, success: bool, error: Optional[str] = None) -> None:
        self.tools_run.append(tool_name)
        self.timings_ms[f"tool.{tool_name}"] = round(float(elapsed_ms), 2)
        self.add_decision(
            stage=f"tool.{tool_name}",
            elapsed_ms=round(float(elapsed_ms), 2),
            success=bool(success),
            error=error,
        )

    def record_timing(self, label: str, elapsed_ms: float) -> None:
        self.timings_ms[label] = round(float(elapsed_ms), 2)

    def increment_llm_calls(self, by: int = 1) -> None:
        self.llm_calls += int(by)

    def finish(self, intent: Optional[str] = None, action: Optional[str] = None) -> "AgentTrace":
        if intent is not None:
            self.intent = intent
        if action is not None:
            self.action = action
        self.ended_at_ms = _epoch_ms()
        return self

    # ── Serialisation ──────────────────────────────────────────────────────
    def to_dict(self) -> Dict[str, Any]:
        return {
            "request_id": self.request_id,
            "user_id": self.user_id,
            "intent": self.intent,
            "action": self.action,
            "started_at_ms": self.started_at_ms,
            "ended_at_ms": self.ended_at_ms,
            "duration_ms": (
                self.ended_at_ms - self.started_at_ms
                if self.ended_at_ms is not None
                else None
            ),
            "tools_run": list(self.tools_run),
            "llm_calls": self.llm_calls,
            "timings_ms": dict(self.timings_ms),
            "decisions": [
                {"stage": d.stage, "payload": d.payload, "epoch_ms": d.epoch_ms}
                for d in self.decisions
            ],
        }


# ── Stage timer helper (context manager) ──────────────────────────────────────
class TraceStage:
    """Context manager that records elapsed time of a labelled block.

    Example::

        with TraceStage(trace, "router.classify"):
            decision = router.classify_intent_and_analyze(query)
    """

    __slots__ = ("trace", "label", "_t0")

    def __init__(self, trace: Optional[AgentTrace], label: str) -> None:
        self.trace = trace
        self.label = label
        self._t0 = 0.0

    def __enter__(self) -> "TraceStage":
        self._t0 = time.perf_counter()
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        if self.trace is None:
            return
        elapsed_ms = (time.perf_counter() - self._t0) * 1000.0
        self.trace.record_timing(self.label, elapsed_ms)


def new_trace(request_id: Optional[str] = None, user_id: Optional[str] = None) -> AgentTrace:
    """Factory for a fresh trace, optionally seeded with caller-provided IDs."""
    trace = AgentTrace()
    if request_id:
        trace.request_id = request_id
    if user_id:
        trace.user_id = user_id
    return trace


__all__ = [
    "AgentDecision",
    "AgentTrace",
    "TraceStage",
    "is_tracing_enabled",
    "new_trace",
]
