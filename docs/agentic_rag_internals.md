---
title: "AgenticRAG Internals"
description: "Deep dive into AgenticRAG architecture: agents, pipeline, data flow, and module interactions."
tags:
  - agentic-rag
  - internals
  - pipeline
  - agents
  - orchestrator
  - rag
  - memory
  - chromadb
  - gemini
  - ollama
---

# AgenticRAG Internals

> Location: `agenticRAG/agentic_rag_gemini/`  
> Entry points: `api_server.py` (port 8000), `main_api.py` (port 8080)

## High-Level Flow

```
User Query
    │
    ▼
┌─────────────────────────────────────────────┐
│ 1. ROUTER (Orchestrator)                     │
│    ├─ LocalOrchestrator (Ollama qwen:0.5b)   │
│    └─ OrchestratorAgent (Gemini 2.5 Flash)   │
│    Output: intent, action, confidence        │
└───────────────┬─────────────────────────────┘
                │
    ┌───────────┼──────────────┐
    │           │              │
    ▼           ▼              ▼
RETRIEVE    CALL_LLM      HYBRID
DOCUMENT/   (direct)      (retrieve +
MEMORY                     generate)
    │           │              │
    └───────────┴──────────────┘
                │
                ▼
┌─────────────────────────────────────────────┐
│ 2. RAG PIPELINE                              │
│    ├─ Query expansion (optional)            │
│    ├─ Hybrid retrieval (memory + docs)       │
│    ├─ Quality assessment                     │
│    ├─ Web search fallback (if needed)        │
│    ├─ Prompt building                          │
│    ├─ LLM response generation (Gemini)        │
│    └─ Iterative reflection (optional)        │
└───────────────┬─────────────────────────────┘
                │
                ▼
┌─────────────────────────────────────────────┐
│ 3. POST-PROCESSING                           │
│    ├─ Memory storage (ChromaDB)              │
│    ├─ Session persistence (JSON/Firestore)   │
│    └─ Motion prompt extraction               │
└───────────────┬─────────────────────────────┘
                │
                ▼
           JSON Response
```

## Agent Ecosystem

### Router Layer

| Agent | File | Model | Latency | Responsibility |
|-------|------|-------|---------|---------------|
| **LocalOrchestrator** | `agents/local_orchestrator.py` | Ollama `qwen:0.5b` | ~300ms | Regex pre-router + JSON intent classification |
| **OrchestratorAgent** | `agents/api_orchestrator.py` | Gemini 2.5 Flash | ~5s | Cloud fallback for complex/low-confidence queries |

Both routers now import canonical intents from `agents/intents.py` (D2 fix).

### Retrieval Agents

| Agent | File | Responsibility |
|-------|------|---------------|
| **KnowledgeLibrarian** | `agents/knowledge_librarian.py` | Multi-collection retrieval (memory, docs, humanml3d, MedQuAD), entity tagging, confidence gating |
| **DoubleRAGAgent** | `agents/double_rag.py` | Clinical dispatch → constraint extraction → conditioned motion search |
| **SemanticBridgeService** | `agents/semantic_bridge.py` | HyDE generation + clinical→motion constraint mapping |

### Utility Agents

| Agent | File | Responsibility |
|-------|------|---------------|
| **KeywordExtractor** | `agents/keyword_extractor.py` | Extract action verbs from queries for motion matching |
| **QueryTransformer** | `agents/query_transform.py` | Query rewriting / HyDE document generation |
| **SafetyFilter** | `agents/safety_filter.py` | Output safety checks |
| **SummarizeAgent** | `agents/summarize_agent.py` | Condense chat sessions into vector summaries |

## Key Modules

### Memory Layer

| Module | File | Purpose |
|--------|------|---------|
| **MemoryManager** | `memory/memory_manager.py` | CRUD for conversation-level memory |
| **DocumentStore** | `memory/document_store.py` | Chunked document storage & semantic search |
| **VectorStore** | `memory/vector_store.py` | ChromaDB wrapper (3 collections) |
| **SessionStore** | `memory/session_store.py` | JSON-on-disk / Firestore chat persistence |
| **EmbeddingService** | `memory/embedding_service.py` | `all-MiniLM-L6-v2` embeddings + Redis cache |
| **VectorBackend** | `memory/vector_backend.py` | Unified backend abstraction (Chroma/Pinecone/Hybrid) — D6 |

### RAG Pipeline

| Module | File | Purpose |
|--------|------|---------|
| **RAGPipeline** | `retrieval/rag_pipeline.py` | Core response generation with agentic loops |
| **GeminiClient** | `utils/gemini_client.py` | OpenAI-compatible Gemini wrapper with key rotation |
| **WebSearch** | `utils/web_search.py` | DuckDuckGo fallback |
| **RateLimiter** | Inside `rag_pipeline.py` | Thread-safe throttling for Gemini API |

### Tool Layer

| Module | File | Purpose |
|--------|------|---------|
| **Tool (ABC)** | `agents/tools/base.py` | Uniform tool interface — D3 |
| **ToolRegistry** | `agents/tools/registry.py` | Registration & selection by intent — D3 |
| **MemoryTool** | `agents/tools/memory_tool.py` | Conversation memory retrieval |
| **DocumentRetrievalTool** | `agents/tools/document_retrieval_tool.py` | Document search |
| **WebSearchTool** | `agents/tools/web_search_tool.py` | Web search adapter |
| **MotionCandidateRetriever** | `agents/tools/motion_candidate_retriever.py` | HumanML3D motion search |
| **MotionReranker** | `agents/tools/motion_reranker.py` | Motion candidate reranking |
| **MotionGenerationTool** | `agents/tools/motion_generation_tool.py` | DART /generate caller |

### Infrastructure

| Module | File | Purpose |
|--------|------|---------|
| **AgentTrace** | `core/tracing.py` | Per-request in-memory tracing — D5 |
| **ResourceGuard** | `core/resource_guard.py` | Shared executor + embedder cache — D7 |
| **CircuitBreaker** | `core/circuit_breaker.py` | Downstream failure protection — D8 |

## Data Flow: `/query` Endpoint

1. **FastAPI** (`api_server_pkg/app.py`) receives `POST /query`
2. Extracts `X-Request-ID`, creates `AgentTrace`
3. Calls `api_instance.process_query(..., trace=agent_trace)`
4. **OrchestratorAgent** classifies intent via Gemini (or LocalOrchestrator fallback)
5. If intent benefits from structured retrieval, calls **KnowledgeLibrarian** pre-fetch
6. If `GENERATE_MOTION` or `needs_rag`, calls **DoubleRAGAgent**
7. **ToolRegistry** selects tools by intent; **ResourceGuard** executor runs them in parallel
8. **RAGPipeline** generates response using retrieved context
9. **MemoryManager** stores interaction; **SessionStore** persists turn
10. Response includes `request_id`, `text_answer`, `exercises`, `motion_prompt`

## Configuration

Key settings in `config/config.yaml`:

```yaml
orchestrator:
  model: "gemini-2.5-flash"
  temperature: 0.1

llm:
  model: "gemini-2.5-flash"
  temperature: 0.7
  max_tokens: 1000

embedding:
  model: "sentence-transformers/all-MiniLM-L6-v2"
  dimension: 384

rag:
  top_k_documents: 8
  similarity_threshold: 0.3
  enable_query_expansion: false
  enable_iterative_reflection: false
```

## Related Notes

- [[agents_catalog]] — Full list of all agents in the system
- [[agentic_rag_refactor]] — May 2026 refactoring details
- [[system_overview]] — Where AgenticRAG fits in the full stack
- [[api_contract]] — `/query` request/response schemas
- [[troubleshooting]] — Common AgenticRAG issues

---

#agentic-rag #internals #pipeline #agents #orchestrator #rag #memory #chromadb #gemini #ollama
