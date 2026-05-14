"""Tiny in-memory circuit breaker.

Used by ``PipelineOrchestrator`` to short-circuit calls to a flaky downstream
(e.g. DART) instead of waiting for the full timeout on every request.

Design choice:
    - In-memory only.  We do not persist breaker state across processes; for a
      single-machine deployment this is fine and avoids dragging in Redis just
      for this concern.
    - Three states: ``closed`` (healthy), ``open`` (failing), ``half_open``
      (probe).  After ``cool_down_seconds`` an open breaker becomes
      ``half_open``; the very next call decides whether to close or stay open.
    - Thread-safe via a simple ``Lock``.
"""

from __future__ import annotations

import threading
import time
from dataclasses import dataclass
from typing import Optional


@dataclass
class _BreakerState:
    failures: int = 0
    opened_at: float = 0.0
    state: str = "closed"  # closed | open | half_open


class CircuitBreaker:
    """Simple consecutive-failure circuit breaker.

    Args:
        name:                    Identifier used in log messages.
        failure_threshold:       Number of consecutive failures that trip the breaker.
        cool_down_seconds:       Time the breaker stays ``open`` before allowing a probe.
    """

    def __init__(
        self,
        name: str = "default",
        failure_threshold: int = 3,
        cool_down_seconds: float = 60.0,
    ) -> None:
        self.name = name
        self.failure_threshold = int(failure_threshold)
        self.cool_down_seconds = float(cool_down_seconds)
        self._state = _BreakerState()
        self._lock = threading.Lock()

    @property
    def state(self) -> str:
        with self._lock:
            self._maybe_transition_to_half_open_locked()
            return self._state.state

    def allow(self) -> bool:
        """Return True if the caller should attempt the protected operation."""
        with self._lock:
            self._maybe_transition_to_half_open_locked()
            return self._state.state in {"closed", "half_open"}

    def record_success(self) -> None:
        with self._lock:
            self._state.failures = 0
            self._state.opened_at = 0.0
            self._state.state = "closed"

    def record_failure(self) -> None:
        with self._lock:
            self._state.failures += 1
            if self._state.failures >= self.failure_threshold:
                self._state.state = "open"
                self._state.opened_at = time.time()

    def _maybe_transition_to_half_open_locked(self) -> None:
        if (
            self._state.state == "open"
            and (time.time() - self._state.opened_at) >= self.cool_down_seconds
        ):
            self._state.state = "half_open"

    def snapshot(self) -> dict:
        with self._lock:
            return {
                "name": self.name,
                "state": self._state.state,
                "failures": self._state.failures,
                "opened_at": self._state.opened_at,
                "failure_threshold": self.failure_threshold,
                "cool_down_seconds": self.cool_down_seconds,
            }


__all__ = ["CircuitBreaker"]
