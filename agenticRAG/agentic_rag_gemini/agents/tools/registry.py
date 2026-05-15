"""Tool registry — single place that knows which tools exist and when to run them.

Why a registry?
    Adding a new tool used to require touching four locations in ``api_orchestrator``
    (the action map, the gating logic, the executor branch, and the constructor).
    With this registry, a tool is added once via ``register(MyTool())`` and the
    orchestrator iterates ``registry.tools_for(intent)`` without further changes.

Adapter pattern:
    The legacy concrete tool classes (``MemoryTool``, ``DocumentRetrievalTool``,
    ``WebSearchTool``) expose bespoke method names.  Rather than rewrite them,
    we wrap each in a small ``Tool`` adapter so registry consumers get a uniform
    ``run(ctx)``  → ``ToolResult`` interface.

Thread safety:
    Registration is expected to happen at startup, so we do not lock.  Reads
    (``tools_for``, ``get``) are safe for concurrent use because the underlying
    dict is only mutated during init.
"""

from __future__ import annotations

import time
from typing import Any, Dict, FrozenSet, Iterable, List, Optional

from agents.intents import INTENT_SKIP_CONTENT_TOOLS, IntentType
from agents.tools.base import Tool, ToolContext, ToolResult
from agents.tools.document_retrieval_tool import DocumentRetrievalTool
from agents.tools.memory_tool import MemoryTool
from agents.tools.web_search_tool import WebSearchTool
from utils.logger import get_logger

logger = get_logger(__name__)


# ── Adapters around the legacy concrete tools ─────────────────────────────────


class MemoryToolAdapter(Tool):
    name = "memory"
    applicable_intents: FrozenSet[IntentType] = frozenset(
        {
            IntentType.CONVERSATION,
            IntentType.KNOWLEDGE_QUERY,
            IntentType.EXERCISE_RECOMMENDATION,
        }
    )
    requires_user_id = True

    def __init__(self, memory_tool: MemoryTool) -> None:
        self._tool = memory_tool

    def run(self, ctx: ToolContext) -> ToolResult:
        t0 = time.perf_counter()
        try:
            payload = self._tool.retrieve_memory(
                user_id=ctx.user_id,
                query=ctx.query,
                top_k=int(ctx.extras.get("top_k", 5)),
                memory_types=ctx.extras.get("memory_types"),
            )
            return ToolResult(
                name=self.name,
                payload=payload,
                elapsed_ms=(time.perf_counter() - t0) * 1000.0,
            )
        except Exception as exc:  # noqa: BLE001 — Tool contract requires no raise
            logger.error("[MemoryToolAdapter] %s", exc)
            return ToolResult(
                name=self.name,
                success=False,
                error=str(exc),
                elapsed_ms=(time.perf_counter() - t0) * 1000.0,
            )


class DocumentRetrievalAdapter(Tool):
    name = "documents"
    applicable_intents: FrozenSet[IntentType] = frozenset(
        {IntentType.KNOWLEDGE_QUERY, IntentType.EXERCISE_RECOMMENDATION}
    )

    def __init__(self, doc_tool: DocumentRetrievalTool) -> None:
        self._tool = doc_tool

    def run(self, ctx: ToolContext) -> ToolResult:
        t0 = time.perf_counter()
        try:
            payload = self._tool.search_documents(
                query=ctx.query,
                user_id=ctx.user_id,
                top_k=int(ctx.extras.get("top_k", 5)),
                max_chunks_per_document=int(ctx.extras.get("max_chunks_per_document", 3)),
                search_method=str(ctx.extras.get("search_method", "vector")),
            )
            return ToolResult(
                name=self.name,
                payload=payload,
                elapsed_ms=(time.perf_counter() - t0) * 1000.0,
            )
        except Exception as exc:  # noqa: BLE001
            logger.error("[DocumentRetrievalAdapter] %s", exc)
            return ToolResult(
                name=self.name,
                success=False,
                error=str(exc),
                elapsed_ms=(time.perf_counter() - t0) * 1000.0,
            )


class WebSearchAdapter(Tool):
    name = "web_search"
    applicable_intents: FrozenSet[IntentType] = frozenset(
        {IntentType.KNOWLEDGE_QUERY, IntentType.EXERCISE_RECOMMENDATION}
    )

    def __init__(self, web_tool: WebSearchTool) -> None:
        self._tool = web_tool

    def run(self, ctx: ToolContext) -> ToolResult:
        t0 = time.perf_counter()
        try:
            max_results = ctx.extras.get("max_results")
            payload = self._tool.search_web(
                query=ctx.query,
                max_results=int(max_results) if max_results is not None else None,
            )
            return ToolResult(
                name=self.name,
                payload=payload,
                elapsed_ms=(time.perf_counter() - t0) * 1000.0,
            )
        except Exception as exc:  # noqa: BLE001
            logger.error("[WebSearchAdapter] %s", exc)
            return ToolResult(
                name=self.name,
                success=False,
                error=str(exc),
                elapsed_ms=(time.perf_counter() - t0) * 1000.0,
            )


# ── Registry ──────────────────────────────────────────────────────────────────


class ToolRegistry:
    """Holds all registered tools and resolves which ones to run for an intent."""

    def __init__(self) -> None:
        self._tools: Dict[str, Tool] = {}

    def register(self, tool: Tool) -> None:
        if not getattr(tool, "name", None):
            raise ValueError(f"Tool {type(tool).__name__} has no .name attribute")
        if tool.name in self._tools:
            logger.warning("Tool '%s' already registered — overwriting", tool.name)
        self._tools[tool.name] = tool
        logger.info(
            "ToolRegistry: registered '%s' for intents=%s",
            tool.name,
            sorted(i.value for i in tool.applicable_intents) if tool.applicable_intents else ["*"],
        )

    def get(self, name: str) -> Optional[Tool]:
        return self._tools.get(name)

    def names(self) -> List[str]:
        return list(self._tools.keys())

    def tools_for(self, intent: Optional[IntentType], allowed: Optional[Iterable[str]] = None) -> List[Tool]:
        """Return tools that should run for ``intent``.

        ``allowed`` is an optional whitelist (used by the orchestrator's gates) —
        only tools whose ``name`` appears in this set will be returned.

        ``CONVERSATION`` and ``VISUALIZE_MOTION`` intents short-circuit to
        memory-only retrieval per ``INTENT_SKIP_CONTENT_TOOLS``.
        """
        allowed_set = set(allowed) if allowed is not None else None
        out: List[Tool] = []
        skip_content = intent in INTENT_SKIP_CONTENT_TOOLS

        for tool in self._tools.values():
            if allowed_set is not None and tool.name not in allowed_set:
                continue
            if tool.applicable_intents and intent and intent not in tool.applicable_intents:
                continue
            if skip_content and tool.name in {"documents", "web_search"}:
                continue
            out.append(tool)
        return out


# ── Convenience builders ──────────────────────────────────────────────────────


def build_default_registry(
    memory_tool: Optional[MemoryTool] = None,
    document_tool: Optional[DocumentRetrievalTool] = None,
    web_search_tool: Optional[WebSearchTool] = None,
    extra_tools: Optional[Iterable[Tool]] = None,
) -> ToolRegistry:
    """Wire the legacy concrete tools into the registry."""
    registry = ToolRegistry()
    if memory_tool is not None:
        registry.register(MemoryToolAdapter(memory_tool))
    if document_tool is not None:
        registry.register(DocumentRetrievalAdapter(document_tool))
    if web_search_tool is not None:
        registry.register(WebSearchAdapter(web_search_tool))
    if extra_tools:
        for tool in extra_tools:
            registry.register(tool)
    return registry


__all__ = [
    "DocumentRetrievalAdapter",
    "MemoryToolAdapter",
    "ToolRegistry",
    "WebSearchAdapter",
    "build_default_registry",
]
