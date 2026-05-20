# Virtual Verbal Assistant (VVA)

## Roles

- **K** (AI): Senior Solution Architect. Makes tool/framework choices, writes plans, reviews worklogs. Does NOT code.
- **N** (Human): Senior Developer. Implements plans, writes code, runs tests. Logs work in `docs/worklogs/DD-MM-YYYY.md`.
- **Owner** (Human): Product owner. Sets vision, makes final calls on contested decisions.

## Project

Healthcare/wellness multimodal AI assistant combining conversational AI, 3D motion synthesis, and voice I/O. Domain: physical therapy exercise recommendations with clinical safety.

## Current Architecture (develop branch — baseline)

```
ECA UI (HTML/JS)           Port 3000
FastAPI Gateway            Port 8080 (orchestrator), 8000 (AgenticRAG)
AgenticRAG                 Gemini 2.5-Flash, ChromaDB (8100), custom Double-RAG
DART text-to-motion        WSL/CUDA, diffusion model, Port 5001
SpeechLLm                  Whisper STT + Coqui/ElevenLabs TTS, Port 5000
Infrastructure             Redis (Celery broker), Firebase (sessions), Docker (ChromaDB)
```

Key files: `agents/api_orchestrator.py` (780 lines, main orchestrator), `retrieval/rag_pipeline.py` (759 lines), `memory/vector_store.py` (912 lines), `text-to-motion/DART/mcp_server.py` (production-ready FastMCP server).

## Active Plan: LangGraph Re-Architecture (v2.1)

**Full plan**: `.claude/plans/purrfect-herding-kahn.md`
**Worklogs**: `docs/worklogs/19-05-2026.md` (planning session), `docs/worklogs/20-05-2026.md` (N's second review)
**Branch**: `feature/langgraph-rewrite` (to be created from `develop`)

### Target Architecture

Replace custom orchestrator with **LangGraph multi-agent supervisor**:

| Node | Role | LLM calls |
|------|------|-----------|
| Manager | Intent classification + routing (fast model) | 1 |
| Memory | Redis STM + PostgreSQL/pgvector LTM (always runs) | 0 |
| Retrieval | pgvector search + web fallback, HyDE | 0-1 |
| Reasoning | Clinical analysis, constraint extraction (heavy model) | 1 |
| Validator | Validate sub-agent outputs + build raw_answer + fallback | 0 |
| Conversation | Apply persona MD styling, stream via WebSocket | 1 |
| Dispatch | Fire Celery tasks for motion/speech | 0 |

### Key Decisions (ADRs from 19-05-2026, updated 20-05-2026)

- **LangGraph**: Accepted. Node logic is pure Python (zero LangGraph imports inside nodes). Lock-in at control flow + state persistence layer; business logic fully portable. Full migration ~1 week.
- **PostgreSQL (pgvector)**: Replaces Firebase + ChromaDB. Single DB for structured + vector data. Docker memory limit 2GB.
- **Redis**: Kept for short-term memory + Celery broker + task result persistence (reconnect recovery).
- **Celery**: Kept for async background tasks (motion, TTS, doc indexing). LangGraph = sync graph only.
- **DART/TTS**: Direct REST calls (httpx). NO MCP for internal services. DART and Ollama mutually exclusive (RAM).
- **MCP**: Deferred to future phase. Only for third-party external tool extensions.
- **Streaming**: Single WebSocket connection with reconnect recovery. Task results persisted in Redis for fallback delivery.
- **Error routing**: 3 severity levels (CRITICAL/RECOVERABLE/IGNORABLE). `error_handler` node for graceful degradation.
- **LLM**: API-primary (Claude likely). Local Ollama not ruled out. Final decision pending.
- **Manager vs Reasoning**: Separate nodes. Merge if Phase 1 profiling shows Manager >500ms.

### Phases

0. Foundation (scaffold, PostgreSQL, stub nodes)
1. Manager + Memory (intent classification, Redis STM, pgvector LTM)
2. Retrieval + Reasoning (RAG pipeline, clinical analysis)
3. Celery Tasks (motion/speech via direct REST, dispatch node)
4. Conversation + Personas (persona MD files, styling)
5. Streaming + Frontend (WebSocket, UI fixes)
6. Production Hardening (logging, tracing, testing, health checks)

### Open Items

- LLM provider final decision (Claude / Ollama / hybrid)
- Deployment topology (separate discussion)
- pgvector vs ChromaDB benchmark (Phase 2)
- Owner confirmation of v2.1 before Phase 0 begins

## Conventions

- All code changes logged in `docs/worklogs/DD-MM-YYYY.md`
- Docs are an Obsidian vault with `[[wiki-links]]` and YAML frontmatter tags
- K reviews worklogs before approving phase transitions
- Tests: `pytest` with markers `unit`, `integration`, `e2e` (see `pytest.ini`)
