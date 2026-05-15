"""Process-wide resource guards for CPU-only / 16 GB RAM deployments.

The constraint we are designing against:
    - 1 user-class machine (CPU-only or modest GPU), 16 GB RAM, Windows + WSL split.
    - Up to ~5 concurrent demo users.
    - Multiple Python processes (AgenticRAG, Coordinator, DART, optional Celery).

Without explicit guards, each FastAPI request can spawn its own
``ThreadPoolExecutor``, each ``MotionCandidateRetriever`` loads its own copy of
``all-MiniLM-L6-v2`` (~80 MB), and each ``RAGPipeline`` instantiates its own
``RateLimiter``.  At 5 users this leaks 400+ MB plus tens of threads.

This module centralises three things:

1. ``shared_tool_executor()`` — a single bounded ``ThreadPoolExecutor`` that all
   tool-call sites should use instead of spawning their own.
2. ``shared_embedder(model_name)`` — singleton per model name; subsequent calls
   reuse the same object.
3. ``ResourceBudget`` — read-only snapshot of the configured limits, useful for
   ``/health`` reporting.

Defaults can be overridden via environment variables:
    ``AGENTIC_MAX_TOOL_CONCURRENCY`` (default ``min(4, cpu//2)``)
    ``AGENTIC_EMBEDDING_CACHE_SIZE``  (default 2000)
"""

from __future__ import annotations

import os
import threading
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from typing import Any, Optional

from utils.logger import get_logger

logger = get_logger(__name__)


def _cpu_count_safe() -> int:
    try:
        return max(1, os.cpu_count() or 1)
    except Exception:
        return 1


def _env_int(name: str, default: int) -> int:
    value = os.getenv(name)
    if value is None:
        return default
    try:
        return int(value)
    except ValueError:
        logger.warning("Invalid integer for %s=%r — using default %d", name, value, default)
        return default


@dataclass(frozen=True)
class ResourceBudget:
    max_tool_concurrency: int
    embedding_cache_size: int
    cpu_count: int


def _compute_budget() -> ResourceBudget:
    cpu = _cpu_count_safe()
    default_conc = max(2, min(4, cpu // 2 or 2))
    return ResourceBudget(
        max_tool_concurrency=_env_int("AGENTIC_MAX_TOOL_CONCURRENCY", default_conc),
        embedding_cache_size=_env_int("AGENTIC_EMBEDDING_CACHE_SIZE", 2000),
        cpu_count=cpu,
    )


_BUDGET = _compute_budget()


def get_resource_budget() -> ResourceBudget:
    """Read-only view of the active resource budget."""
    return _BUDGET


# ── Shared ThreadPoolExecutor ─────────────────────────────────────────────────
_executor_lock = threading.Lock()
_shared_executor: Optional[ThreadPoolExecutor] = None


def shared_tool_executor() -> ThreadPoolExecutor:
    """Return the process-wide ``ThreadPoolExecutor`` used by all tool runs.

    Lazily created on first access.  The pool size is capped by
    ``ResourceBudget.max_tool_concurrency``.  Callers MUST NOT call
    ``executor.shutdown()`` — the pool is owned by this module and outlives
    individual requests.
    """
    global _shared_executor
    if _shared_executor is not None:
        return _shared_executor
    with _executor_lock:
        if _shared_executor is None:
            _shared_executor = ThreadPoolExecutor(
                max_workers=_BUDGET.max_tool_concurrency,
                thread_name_prefix="agentic-tool",
            )
            logger.info(
                "ResourceGuard: shared tool executor created max_workers=%d (cpu=%d)",
                _BUDGET.max_tool_concurrency,
                _BUDGET.cpu_count,
            )
    return _shared_executor


def shutdown_shared_executor(wait: bool = False) -> None:
    """Shutdown the shared executor — typically only used in tests."""
    global _shared_executor
    with _executor_lock:
        if _shared_executor is not None:
            _shared_executor.shutdown(wait=wait)
            _shared_executor = None


# ── Shared sentence-transformer embedder ──────────────────────────────────────
_embedder_lock = threading.Lock()
_embedder_cache: dict[str, Any] = {}


def shared_embedder(model_name: str = "sentence-transformers/all-MiniLM-L6-v2") -> Any:
    """Return a singleton ``SentenceTransformer`` keyed by ``model_name``.

    Imports ``sentence_transformers`` lazily so this module stays cheap to
    import.  Returns ``None`` if the package is unavailable; callers should
    handle that gracefully (the existing fallback paths already do).
    """
    if model_name in _embedder_cache:
        return _embedder_cache[model_name]
    with _embedder_lock:
        if model_name in _embedder_cache:
            return _embedder_cache[model_name]
        try:
            from sentence_transformers import SentenceTransformer  # type: ignore

            embedder = SentenceTransformer(model_name)
            _embedder_cache[model_name] = embedder
            logger.info("ResourceGuard: loaded shared embedder '%s'", model_name)
            return embedder
        except Exception as exc:  # noqa: BLE001
            logger.warning(
                "ResourceGuard: failed to load embedder '%s' — %s", model_name, exc
            )
            return None


def clear_embedder_cache() -> None:
    """Drop all cached embedders — typically only used in tests."""
    with _embedder_lock:
        _embedder_cache.clear()


__all__ = [
    "ResourceBudget",
    "clear_embedder_cache",
    "get_resource_budget",
    "shared_embedder",
    "shared_tool_executor",
    "shutdown_shared_executor",
]
