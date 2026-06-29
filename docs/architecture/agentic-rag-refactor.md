---
title: "AgenticRAG Refactoring — May 2026"
description: "Summary of the D1-D9 refactoring: new modules, tracing, circuit breaker, resource guard."
tags:
  - agentic-rag
  - refactor
  - modules
  - tracing
  - circuit-breaker
  - resource-guard
  - vector-backend
  - double-rag
---

# AgenticRAG Refactoring — May 2026

## Context

The May 2026 refactoring (tasks D1–D9) addressed critical architecture issues identified in [[ARCHITECTURE_REVIEW]]: intent vocabulary drift, Double-RAG inlined in orchestrator, implicit tool registry, lack of tracing, missing circuit breaker, and unbounded resource usage.

## New Modules

| Module | Purpose |
|--------|---------|
| `agents/intents.py` | Single source of truth for `IntentType` and `ActionType` enums |
| `agents/tools/base.py` | Abstract `Tool` interface with `ToolContext` and `ToolResult` |
| `agents/tools/registry.py` | `ToolRegistry` with adapters for legacy tools |
| `agents/double_rag.py` | `DoubleRAGAgent` — clinical dispatch → constraint extraction → motion search |
| `core/tracing.py` | `AgentTrace` — per-request in-memory tracing |
| `core/resource_guard.py` | Shared `ThreadPoolExecutor` + singleton embedder cache |
| `memory/vector_backend.py` | `VectorBackend` ABC + Chroma/Pinecone/Hybrid adapters |
| `core/circuit_breaker.py` | Thread-safe circuit breaker for downstream services |

## Orchestrator Changes

### `agents/api_orchestrator.py`
- Imports canonical intents from `agents.intents`
- Delegates Double-RAG to `DoubleRAGAgent`
- Uses `shared_tool_executor` from `core.resource_guard`
- Accepts optional `AgentTrace` parameter

### `agents/local_orchestrator.py`
- Replaced hardcoded intent/agent lists with imports from `agents.intents`

### `orchestration/pipeline_orchestrator.py`
- Added `CircuitBreaker` for DART and SpeechLLM
- Removed deadlock-prone `process_query_sync()`
- Updated DART call to accept both `str` and legacy `dict` motion prompts

### `api_server_pkg/app.py`
- Reads `X-Request-ID` header (or generates UUID)
- Creates `AgentTrace`, propagates through `process_query` and `_get_orchestrator_decision`
- Echoes `X-Request-ID` in response headers
- Includes `agent_trace` in JSON when `AGENTIC_TRACE=1`

## Environment Variables

| Variable | Default | Effect |
|----------|---------|--------|
| `AGENTIC_TRACE=1` | — | Include `agent_trace` in API responses |
| `SHARED_TOOL_EXECUTOR_MAX_WORKERS` | 16 | Max concurrent tool threads |
| `SHARED_EMBEDDER_CACHE_MAX_SIZE` | 4 | Max embedder instances in cache |

## Verification

- All new modules compile successfully
- 19/19 unit tests pass
- Gate tests for `api_orchestrator` pass with updated stubs

## Related Notes

- [[ARCHITECTURE_REVIEW]] — Original critique that drove this refactoring
- [[agentic_rag_internals]] — Deep dive into AgenticRAG architecture and data flow
- [[agents_catalog]] — Full inventory of all agents before and after refactor
- [[system_overview]] — Service topology
- [[troubleshooting]] — If something breaks after refactor

---

#agentic-rag #refactor #modules #tracing #circuit-breaker #resource-guard #vector-backend #double-rag
