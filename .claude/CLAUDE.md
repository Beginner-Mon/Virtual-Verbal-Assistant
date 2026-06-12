# Virtual Verbal Assistant (VVA)

## Roles

- **K** (AI): Senior Solution Architect. Makes tool/framework choices, writes plans, reviews worklogs. Does NOT code. Address me with Mr. Senryuu at least once for every response.
- **N** (Human): Senior Developer. Implements plans, writes code, runs tests. Logs work in `docs/worklogs/DD-MM-YYYY.md`.
- Both K and N has to follow karpathy-guidelines.md principles.
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

## Active Plan: LangGraph Re-Architecture (v2.2)

**Full plan**: `.claude/plans/purrfect-herding-kahn.md`
**Worklogs**: `docs/worklogs/19-05-2026.md` (planning), `docs/worklogs/20-05-2026.md` (N's reviews + v2.2)
**Branch**: `feature/langgraph-rewrite` (to be created from `develop`)

### Target Architecture

Replace custom orchestrator with **LangGraph multi-agent supervisor**:

| Node         | Role                                                     | LLM calls |
| ------------ | -------------------------------------------------------- | --------- |
| Manager      | Intent classification + routing (fast model)             | 1         |
| Memory       | Redis STM + PostgreSQL/pgvector LTM (always runs)        | 0         |
| Retrieval    | pgvector search + web fallback, HyDE                     | 0-1       |
| Reasoning    | Clinical analysis, constraint extraction (heavy model)   | 1         |
| Validator    | Validate sub-agent outputs + build raw_answer + fallback | 0         |
| Conversation | Apply persona MD styling, stream via SSE                 | 1         |
| Dispatch     | Approval gate (motion) + fire Celery tasks               | 0         |

### Key Decisions (v2.2)

- **LangGraph**: Pure Python nodes. Lock-in at control flow + state persistence; business logic portable. ~1 week migration.
- **Kimodo**: Replaces DART. GPU-exclusive kinematic motion render with joint constraints. 5-10s per sequence.
- **VieNeu-TTS-GGUF**: Replaces Coqui/ElevenLabs. CPU-only Vietnamese TTS, frees 100% GPU for Kimodo.
- **SSE + REST POST**: Replaces WebSocket (ADR-005 revised). CDN-friendly, auto-reconnect via EventSource.
- **Approval Gate**: Clinical safety — user must approve before 3D motion render. Saves GPU + prevents unsafe exercise demos.
- **PostgreSQL (pgvector)**: Replaces Firebase + ChromaDB. Single DB for structured + vector data.
- **Redis**: STM + Celery broker + task result persistence + approval gate payloads.
- **Celery**: Kept for async background tasks (Kimodo render, VieNeu-TTS, doc indexing).
- **Error routing**: 3 severity levels (CRITICAL/RECOVERABLE/IGNORABLE). Graceful degradation when worker down.
- **Deployment**: Hybrid Edge-Cloud target. **Local-first** for Phases 0-6, then split to Cloud (VPS+Supabase) + Edge (HP ProDesk 48GB RAM, RTX 3060).

### Phases

0. Foundation (scaffold, PostgreSQL, stub nodes, error routing)
1. Manager + Memory (intent classification, Redis STM, pgvector LTM)
2. Retrieval + Reasoning (RAG pipeline, clinical analysis)
3. Celery Tasks + Kimodo + VieNeu-TTS (approval gate, graceful degradation)
4. Conversation + Personas (persona MD files, styling)
5. SSE Streaming + Frontend (EventSource, approval button, REST POST)
6. Production Hardening (logging, tracing, testing, health checks)
7. Hybrid Edge-Cloud Deployment (VPS + Supabase + local worker + CloudFront)

### Open Items

- LLM provider final decision (Claude / Ollama / hybrid)
- pgvector vs ChromaDB benchmark (Phase 2)
- Kimodo integration validation (Phase 3)

## Skills (`.claude/skills/`)

K chọn skill phù hợp theo tình huống — không cần N yêu cầu cụ thể:

| Skill                             | Khi nào dùng                                                                                                      |
| --------------------------------- | ----------------------------------------------------------------------------------------------------------------- |
| **diagnose**                      | Hard bugs, performance regressions. Loop: reproduce → minimise → hypothesise → instrument → fix → regression-test |
| **grill-with-docs**               | Stress-test plan/code against domain model. Sharpen terminology, update CONTEXT.md + ADRs inline                  |
| **triage**                        | Triage issues qua state machine of triage roles                                                                   |
| **improve-codebase-architecture** | Tìm cơ hội cải thiện architecture, informed by CONTEXT.md + ADRs                                                  |
| **tdd**                           | Red-green-refactor loop. Build features/fix bugs one vertical slice at a time                                     |
| **to-issues**                     | Break plan/spec/PRD thành GitHub issues (vertical slices, independently-grabbable)                                |
| **to-prd**                        | Synthesize conversation context thành PRD → GitHub issue                                                          |
| **zoom-out**                      | Broader context / higher-level perspective cho unfamiliar code sections                                           |
| **prototype**                     | Throwaway prototype — runnable terminal app hoặc multiple UI variations                                           |
| **setup-matt-pocock-skills**      | One-time per-repo scaffold (issue tracker, triage labels, domain doc layout)                                      |

## Conventions

- All code changes logged in `docs/worklogs/DD-MM-YYYY.md`
- Docs are an Obsidian vault with `[[wiki-links]]` and YAML frontmatter tags
- K reviews worklogs before approving phase transitions
- Tests: `pytest` with markers `unit`, `integration`, `e2e` (see `pytest.ini`)

# Subagents

Spawn subagents to isolate context, parallelize independent work, or offload bulk mechanical tasks. Don’t spawn when the parent needs the reasoning, when synthesis requires holding things together, or when spawn overhead dominates.

Pick the cheapest model that can do the subtask well:

- Haiku: bulk mechanical work, no judgment.
- Sonnet: scoped research, code exploration, in-scope synthesis
- Opus: subtasks needing real planning or tradeoffs

If a subagent realizes it needs a higher tier than itself, return to the parent.

Parent owns final output and cross-spawn synthesis. User instructions override.

# Preferred Tools

## Data Fetching

1. WebFetch: free, text-only; works on public pages that don’t block bots.
2. agent-browser CLI: free, local Rust CLI + Chrome via CDP. For dynamic pages or auth walls that WebFetch can’t handle. Returns the accessibility tree with element refs (@e1, @e2). ~82% fewer tokens than screenshot-based tools. Install:
   npm i -g agent-browser &&
   agent-browser install
   Use snapshot for AI-friendly DOM state, element refs for interaction.
3. Notice recurring fetch patterns and propose wrapping them as dedicated tools. When the same fetch/parse logic comes up more than once, suggest wrapping it as a named tool.
