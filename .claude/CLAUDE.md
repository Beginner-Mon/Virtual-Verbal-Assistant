# Virtual Verbal Assistant (VVA)

## Roles

- **K** (AI): Senior Solution Architect. Makes tool/framework choices, writes plans, reviews worklogs. Does NOT code. Address me with Mr. Senryuu at least once for every response.
- **N** (Human): Senior Developer. Implements plans, writes code, runs tests. Logs work in `docs/worklogs/DD-MM-YYYY.md`.
- Both K and N has to follow karpathy-guidelines.md principles.
- **Owner** (Human): Product owner. Sets vision, makes final calls on contested decisions.

## Project

Healthcare/wellness multimodal AI assistant combining conversational AI, 3D motion synthesis, and voice I/O. Domain: physical therapy exercise recommendations with clinical safety.

## Current Architecture (as built, `feature/langgraph-rewrite`)

```
ECA UI                     Vite 6 + React 19 + TS, Capacitor for mobile, Amplify hosting
LangGraph service          Port 8000, FastAPI + SSE (agenticRAG/langgraph_agents)
PostgreSQL + pgvector      Neon, us-east-1, PG 18.4, pgvector 0.8.6 (chuyển 17/08)
Redis                      STM + TTS task results — STILL LOCAL, not yet hosted
Kimodo text-to-motion      GPU MCP server (replaced DART); ECS deploy awaiting Owner
VieNeu-TTS                 CPU Vietnamese TTS (replaced Coqui/ElevenLabs)
Auth                       AWS Cognito via Amplify Gen 2
Infra                      AWS CDK — see infra/README.md for the two-track split
```

Key files: `agenticRAG/langgraph_agents/graph.py` (graph wiring),
`api/main.py` (FastAPI + SSE), `nodes/synthesizer.py` (persona applied here),
`db/postgres.py` (asyncpg client), `text-to-motion/DART/mcp_server.py`.

> Firebase, ChromaDB, DART, Celery and the `agentic_rag_gemini` package are all
> gone. `agents/api_orchestrator.py`, `retrieval/rag_pipeline.py` and
> `memory/vector_store.py` were deleted on 10/08 (71 files, 46 MB).

## Active Plan: LangGraph Re-Architecture

**Decisions**: ADR-001…005 in `docs/worklogs/19-05-2026.md`, D1–D33 in `docs/plans/reupdate-plan.md`, ADR-006 in `docs/worklogs/16-08-2026.md`
**Graph + persona reference**: `docs/architecture/langgraph-flow-persona.md`
**Status**: `docs/tracking/status.md`
**Branch**: `feature/langgraph-rewrite`

### Graph as built

The v2.2 node design below was **not** what shipped. Actual nodes
(`graph.py:180-235`):

| Node             | Role                                                       | LLM calls |
| ---------------- | ---------------------------------------------------------- | --------- |
| memory           | Redis STM + user facts + summary chunks; runs first        | 0         |
| planner          | 3-axis: required_outputs / resolved_query / routing bits   | 1         |
| retriever_agent  | Self-selects tools; loops with `tools`                     | 1         |
| tools            | ToolNode (pgvector, web search)                            | 0         |
| kimodo           | Motion generation via MCP, gated on `needs_motion`         | 0         |
| **synthesizer**  | **Universal responder — applies persona styling**          | 1         |
| grader           | Rule-based tag check + persona safety templates            | 0         |
| error_handler    | Writes `final_answer` directly                             | 0         |

The v2.2 `Manager`/`Reasoning`/`Validator`/`Dispatch`/`Conversation` nodes do
not exist. `conversation.py` was deleted in Phase 6.9 and its persona work moved
into `synthesizer.py`; `nodes/_persona_loader.py` is the surviving fragment.

### Key Decisions (v2.2)

- **LangGraph**: Pure Python nodes. Lock-in at control flow + state persistence; business logic portable. ~1 week migration.
- **Kimodo**: Replaces DART. GPU-exclusive kinematic motion render with joint constraints. 5-10s per sequence.
- **VieNeu-TTS-GGUF**: Replaces Coqui/ElevenLabs. CPU-only Vietnamese TTS, frees 100% GPU for Kimodo.
- **SSE + REST POST**: Replaces WebSocket (ADR-005 revised). CDN-friendly, auto-reconnect via EventSource.
- **Approval Gate**: Clinical safety — user must approve before 3D motion render. Saves GPU + prevents unsafe exercise demos.
- **PostgreSQL (pgvector)**: Replaces Firebase + ChromaDB. Single DB for structured + vector data. **Now hosted on Neon** (`VVA_PG_DSN`, direct endpoint — not `-pooler`; see `.claude/plans/neon-migration.md`).
- **Redis**: STM + Celery broker + task result persistence + approval gate payloads.
- ~~**Celery**: Kept for async background tasks.~~ **Reversed in v2.4.1** — `celery_app = None`, not in requirements. Background work uses `asyncio.create_task` with a strong-reference set (`main.py:69-73`).
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
