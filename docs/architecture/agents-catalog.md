---
title: "Agents Catalog"
description: "Complete inventory of all agents, services, and specialized modules in the Virtual Verbal Assistant system."
tags:
  - agents
  - catalog
  - inventory
  - agentic-rag
  - dart
  - speechllm
  - orchestrator
  - router
  - retrieval
---

# Agents Catalog

## Orchestrator / Router Layer

### 1. OrchestratorAgent
- **File**: `agents/api_orchestrator.py`
- **Type**: Cloud LLM router
- **Model**: Gemini 2.5 Flash
- **Latency**: ~5s
- **Responsibilities**:
  - Classify user intent (`IntentType` enum)
  - Determine action (`ActionType` enum)
  - Build action plan with parameters
  - Delegate Double-RAG to `DoubleRAGAgent`
  - Run tools concurrently via `ToolRegistry`
  - Support per-request `AgentTrace` telemetry
- **Key methods**: `classify_intent_and_analyze()`, `process_query()`, `_run_tools()`

### 2. LocalOrchestrator
- **File**: `agents/local_orchestrator.py`
- **Type**: Local LLM router
- **Model**: Ollama `qwen:0.5b`
- **Latency**: ~300ms
- **Responsibilities**:
  - Regex pre-router for fast-path intents (greeting, knowledge, visualize, clear-context)
  - JSON intent classification via Ollama `format: "json"`
  - Fallback to `OrchestratorAgent` on low confidence
- **Key methods**: `_pre_route()`, `_parse_response()`

### 3. PipelineOrchestrator
- **File**: `orchestration/pipeline_orchestrator.py`
- **Type**: Multi-service coordinator
- **Responsibilities**:
  - HTTP fanout to AgenticRAG, DART, SpeechLLm
  - Circuit breaker for DART and SpeechLLM (D8)
  - Async task coordination with timeout handling
- **Note**: Removed `process_query_sync()` (deadlock anti-pattern) in D8

## Retrieval Agents

### 4. KnowledgeLibrarian
- **File**: `agents/knowledge_librarian.py`
- **Type**: Multi-collection retrieval agent
- **Architecture**: Hub & Spoke (keyword routing + SLM classification spoke)
- **Collections searched**:
  - `humanml3d_library` — kinematic motion descriptions
  - `user_{id}_documents` — uploaded PDFs/DOCX
  - `user_{id}_collection` — conversation memory
  - `medquad` — medical QA (optional, lazy-loaded)
- **Responsibilities**:
  - Entity tag extraction (deterministic + SLM spoke)
  - Confidence gating (`RAG_CONFIDENCE_THRESHOLD = 0.60`)
  - Fact summarization (`MAX_FACTS = 5`)
  - Motion direct-match short-circuit (exact HumanML3D hit bypasses DART)
- **Key methods**: `retrieve()`, `_classify_with_slm()`, `_apply_entity_tag_rerank()`

### 5. DoubleRAGAgent
- **File**: `agents/double_rag.py`
- **Type**: Multi-stage RAG pipeline
- **Created**: May 2026 (D4)
- **Stages**:
  1. **Clinical Dispatch** — search `clinical_knowledge` for safety/anatomy
  2. **Constraint Extraction** — LLM extracts constraints (e.g., "gentle movement")
  3. **Conditioned Motion Search** — HyDE + constraints → `humanml3d_library`
- **Responsibilities**:
  - Encapsulate Double-RAG logic extracted from `api_orchestrator.py`
  - Return structured `DoubleRAGResult` with clinical docs, constraints, motion candidates
- **Key methods**: `run()`

### 6. SemanticBridgeService
- **File**: `agents/semantic_bridge.py`
- **Type**: Constraint mapping agent
- **Responsibilities**:
  - HyDE document generation
  - Clinical constraint → motion search filter mapping
  - Natural language → DART-ready prompt translation

## Utility / Support Agents

### 7. KeywordExtractor
- **File**: `agents/keyword_extractor.py`
- **Type**: Lightweight extractor
- **Responsibilities**:
  - Extract action verbs from queries (e.g., "walk", "jump", "stretch")
  - Used for motion keyword matching and prompt resolution

### 8. QueryTransformer
- **File**: `agents/query_transform.py`
- **Type**: Query rewriting agent
- **Responsibilities**:
  - Query expansion (optional, disabled by default)
  - HyDE document generation for motion retrieval

### 9. SafetyFilter
- **File**: `agents/safety_filter.py`
- **Type**: Output validator
- **Responsibilities**:
  - Flag unsafe keywords (diagnosis, treatment plan, prescription)
  - Note: validation currently disabled in config for latency

### 10. SummarizeAgent
- **File**: `agents/summarize_agent.py`
- **Type**: Session summarizer
- **Responsibilities**:
  - Condense chat transcripts into 3–5 sentence summaries
  - Embed summaries into ChromaDB `chat_summaries` collection
- **Trigger**: `ui.py._summarize_current_session()` when switching sessions

### 11. ResponseTemplates
- **File**: `agents/response_templates.py`
- **Type**: Structured response formatter
- **Responsibilities**:
  - Pre-built response patterns for common intents
  - Consistent formatting across router outputs

## Tool Layer (via ToolRegistry)

### 12. MemoryTool
- **File**: `agents/tools/memory_tool.py`
- **Purpose**: Retrieve past conversation turns from `kinetichat_memory`

### 13. DocumentRetrievalTool
- **File**: `agents/tools/document_retrieval_tool.py`
- **Purpose**: Search uploaded document chunks from `kinetichat_memory_documents`

### 14. WebSearchTool
- **File**: `agents/tools/web_search_tool.py`
- **Purpose**: DuckDuckGo fallback when local context is insufficient

### 15. MotionCandidateRetriever
- **File**: `agents/tools/motion_candidate_retriever.py`
- **Purpose**: Search `humanml3d_library` for motion candidates (supports ChromaDB, Pinecone, JSONL)

### 16. MotionReranker
- **File**: `agents/tools/motion_reranker.py`
- **Purpose**: Rerank motion candidates by relevance

### 17. MotionGenerationTool
- **File**: `agents/tools/motion_generation_tool.py`
- **Purpose**: Call DART `/generate` endpoint with `duration_seconds` first-class contract

## External Services (Not Agents, but in Pipeline)

| Service | Port | Role |
|---------|------|------|
| **DART** | 5001 | Motion synthesis (diffusion + SMPL-X) |
| **SpeechLLm** | 5000 | Voice I/O, STT, TTS, emotion detection |
| **Redis** | 6379 | Embedding cache, Celery broker |
| **ChromaDB** | 8100 | Vector store for memory/documents/motion |

## Summary Table

| Agent | Layer | Model | Key Output |
|-------|-------|-------|------------|
| OrchestratorAgent | Router | Gemini 2.5 Flash | Intent, Action, Parameters |
| LocalOrchestrator | Router | Ollama qwen:0.5b | Intent, Confidence |
| PipelineOrchestrator | Coordinator | — | HTTP fanout, circuit breaker |
| KnowledgeLibrarian | Retrieval | Keyword + SLM | Facts, Confidence, Direct Match |
| DoubleRAGAgent | Retrieval | Gemini | Clinical Constraints, Motion Candidates |
| SemanticBridgeService | Translation | Gemini | HyDE, DART-ready prompt |
| KeywordExtractor | Utility | Regex/Heuristic | Action verb |
| QueryTransformer | Utility | Gemini (optional) | Expanded query, HyDE |
| SafetyFilter | Validator | Rule-based | Safety flags |
| SummarizeAgent | Utility | Gemini | Session summary |
| ResponseTemplates | Formatter | Template | Structured response |

## Related Notes

- [[agentic-rag-internals]] — Deep dive into AgenticRAG architecture and data flow
- [[agentic-rag-refactor]] — May 2026 refactoring that introduced Tool ABC, Registry, DoubleRAGAgent
- [[system-overview]] — Where these agents fit in the full system
- [[api-contract]] — How agents communicate via gateway endpoints

---

#agents #catalog #inventory #agentic-rag #dart #speechllm #orchestrator #router #retrieval
