<p align="center">
  <h1 align="center">🧠 Virtual Verbal Assistant (VVA)</h1>
  <p align="center">
    <strong>Healthcare/wellness multimodal AI assistant — LangGraph multi-agent supervisor for physical therapy exercise recommendations with clinical safety.</strong>
  </p>
  <p align="center">
    <a href="docs/RUNBOOK.md"><img src="https://img.shields.io/badge/📘_Runbook-blue?style=for-the-badge" alt="Runbook"></a>
    <a href="docs/DEPLOYMENT.md"><img src="https://img.shields.io/badge/🚀_Deployment-4B32C3?style=for-the-badge" alt="Deployment"></a>
    <a href=".claude/plans/purrfect-herding-kahn.md"><img src="https://img.shields.io/badge/📐_Plan_v2.4.1-00B4D8?style=for-the-badge" alt="Plan"></a>
  </p>
</p>

---

## ✨ What it does

User asks *"Tôi bị đau lưng dưới khi ngồi lâu, có bài tập nào không?"* — VVA returns:

1. **Clinical recommendation** — exercise list with sets/reps and safety warnings, styled in selected persona (warm / clinical / friendly)
2. **Token-streamed response** — text streams to browser via SSE as the LLM generates it
3. **Optional 3D motion** — if user requests visualization, Kimodo renders the movement (5-10s GPU)
4. **Optional speech** — VieNeu-TTS synthesizes Vietnamese audio in-process (FastAPI BackgroundTasks)

Built on a **7-node LangGraph state machine** with structured plan/execute pattern, conditional long-term memory, rule-based quality grading, and graceful error routing.

---

## 🏗️ Architecture (v2.4.1)

```
┌─────────────────────────────────────────────────────────────┐
│  ECA UI (HTML/JS, Port 3000)                                │
│  EventSource SSE + REST POST                                │
└─────────────────────────────┬───────────────────────────────┘
                              │
┌─────────────────────────────▼───────────────────────────────┐
│  FastAPI (Port 8080)                                        │
│  POST /chat        → SSE stream                             │
│  GET  /sessions    → list user sessions                     │
│  POST /sessions/{id}/resume → reopen + populate STM         │
│  GET  /health      → liveness                               │
│  GET  /health/detailed → readiness (PG/Redis/MCP/LLM)       │
└─────────────────────────────┬───────────────────────────────┘
                              │
┌─────────────────────────────▼───────────────────────────────┐
│  LangGraph StateGraph (7 nodes)                             │
│                                                             │
│  memory ─► planner ─┬─► conversation (greeting/clarify)     │
│                     │                                       │
│                     └─► retriever_agent ─► synthesizer      │
│                              │                  │           │
│                       (pgvector @tool +         ▼           │
│                        MCP tools parallel)    grader        │
│                                              ↓ ↓ ↓          │
│                                       pass / retry / warn   │
│                                              │              │
│                                              ▼              │
│                                       conversation ─► END   │
│                                                             │
│  error_handler ─► conversation (any CRITICAL error)         │
└──┬──────────────────────────────────────────────────────────┘
   │
┌──▼────────────────────┬──────────────────────┬──────────────┐
│  Data Layer           │  External (MCP)      │  TTS         │
│  ├─ PostgreSQL 5432  │  ├─ Kimodo (5001)   │  ├─ VieNeu   │
│  │  + pgvector       │  │  generate_motion │  │  (FastAPI  │
│  └─ Redis 6379       │  └─ Web Search      │  │   bg task) │
│     STM + task_result│     (5020)          │  └─ Redis    │
│                      │                      │     persist  │
└──────────────────────┴──────────────────────┴──────────────┘
```

| Node | Role | LLM calls |
|------|------|-----------|
| `memory` | Redis STM (3 Q&A FIFO) + conditional pgvector LTM | 0 |
| `planner` | Intent classification + structured plan (Pydantic) | 1 (fast) |
| `retriever_agent` | Execute plan: pgvector @tool + MCP tools parallel | 1+ |
| `synthesizer` | Generate clinical response from tool results | 1 (heavy) |
| `grader` | Rule-based quality check, max 1 retry | 0 |
| `conversation` | Dual-mode: styling existing content OR generating greeting/clarify | 1 |
| `error_handler` | Graceful Vietnamese error → reasoning_output | 0 |

**Production hardening (Phase 6 P0)**: structured JSON logging with `request_id` correlation, parallel dependency health checks, per-role circuit breakers (DeepSeek + MCP), socket-timeout bounded Redis ops.

---

## 🚀 Quick Start

### Prerequisites

| Requirement | Notes |
|---|---|
| **Python 3.10+** | Conda `firstconda` recommended |
| **Docker Desktop** | For PostgreSQL + Redis + SearXNG containers |
| **DeepSeek API key** | Set in `agenticRAG/agentic_rag_gemini/.env` |

### Run in 4 commands

```powershell
# 1. Start PostgreSQL + Redis + SearXNG
docker compose -f docker-compose.langgraph.yml up -d

# 2. Activate env + install deps (first time only)
conda activate firstconda
pip install -r requirements-langgraph.txt

# 3. Start backend (Terminal 1)
cd agenticRAG
python -m uvicorn langgraph_agents.api.main:create_app --factory --port 8080

# 4. Start frontend (Terminal 2)
cd ECA_UI
python -m http.server 3000
```

Open: <http://localhost:3000/?api_base=http://localhost:8080>

### Verify

```powershell
# Liveness (instant, no dep checks)
curl http://localhost:8080/health

# Readiness (parallel checks: PG, Redis, MCP, LLM)
curl http://localhost:8080/health/detailed
```

Smoke chat:

```powershell
'{"query":"Xin chào"}' | curl -s -N -X POST http://localhost:8080/chat `
  -H "Content-Type: application/json" -d '@-'
```

Expected SSE events: `stage` (memory, planner, conversation) → `token` (...) → `done`.

### Tests

```powershell
pytest tests/langgraph_agents/ -m unit -v          # 100 tests, ~10s
pytest tests/langgraph_agents/ -m integration -v   # 6 tests, ~6min (live DeepSeek)
```

Expected: **100/100 unit + 6/6 integration**.

> For detailed setup, troubleshooting, and log analysis, see [docs/RUNBOOK.md](docs/RUNBOOK.md).

---

## 🔌 API Reference

| Endpoint | Method | Purpose |
|---|---|---|
| `/chat` | POST | Submit query, stream SSE response (`stage` / `token` / `speech_ready` / `done`) |
| `/sessions?user_id=X` | GET | List user sessions sorted by `updated_at`, with first-message preview |
| `/sessions/{id}/resume` | POST | Load session messages + populate Redis STM |
| `/tts/{task_id}/result` | GET | Poll TTS result (fallback when SSE missed) |
| `/health` | GET | Liveness — process alive |
| `/health/detailed` | GET | Readiness — parallel PG/Redis/MCP/LLM checks, breaker states |

### POST /chat request body

```json
{
  "query": "Bài tập cho đau lưng",
  "user_id": "anonymous",
  "session_id": "session-uuid",
  "persona_id": "eca_default",
  "output_mode": "text"
}
```

| Field | Default | Options |
|---|---|---|
| `persona_id` | `eca_default` | `eca_default` / `eca_friendly` / `eca_clinical` |
| `output_mode` | `text` | `text` / `speech` / `both` |

### SSE event types

```
event: stage           data: {"node": "planner", "status": "complete", "intent": "exercise_recommendation"}
event: token           data: {"content": "Bài "}
event: session_persisted   data: {"session_id": "..."}
event: speech_pending  data: {"task_id": "..."}
event: speech_ready    data: {"task_id": "...", "url": "..."}
event: done            data: {"request_id": "...", "total_tokens": 423, "intent": "..."}
```

---

## 📁 Project Structure

```
Virtual-Verbal-Assistant/
├── README.md                      ← you are here
├── docs/
│   ├── RUNBOOK.md                 # Step-by-step deploy + troubleshooting + log analysis
│   └── DEPLOYMENT.md              # Phase 7 hybrid edge-cloud (stub)
├── docker-compose.langgraph.yml   # PostgreSQL + Redis + SearXNG for local dev
├── requirements-langgraph.txt     # Python deps (pinned)
│
├── agenticRAG/agentic_rag_gemini/
│   └── langgraph_agents/
│       ├── api/
│       │   ├── main.py            # FastAPI app, /chat SSE, /sessions, /health
│       │   ├── health.py          # Parallel dep checks (PG, Redis, MCP, LLM)
│       │   ├── sse.py             # SSE encoding + StreamingResponse
│       │   └── schemas.py         # Pydantic request/response models
│       ├── nodes/                 # 7 graph nodes (pure async fn)
│       │   ├── memory.py          # STM (Redis) + conditional LTM (pgvector)
│       │   ├── planner.py         # Intent + structured plan (Pydantic)
│       │   ├── retriever_agent.py # LLM + ToolNode (pgvector @tool + MCP)
│       │   ├── synthesizer.py     # Heavy LLM → clinical response
│       │   ├── grader.py          # Rule-based quality check + retry
│       │   ├── conversation.py    # Persona styling OR generation
│       │   └── error_handler.py   # Graceful error message
│       ├── tools/
│       │   └── pgvector_tool.py   # @tool wrapper for in-process vector search
│       ├── mcp/                   # MCP server implementations + client
│       │   ├── client.py          # MultiServerMCPClient + circuit breaker
│       │   ├── kimodo_server.py   # Kimodo motion (mock for local dev)
│       │   └── web_search_server.py # SearXNG metasearch wrapper
│       ├── shared/
│       │   ├── logging.py         # JSON formatter + request_id ContextVar
│       │   └── __init__.py        # EmbeddingService + PostgresClient singletons
│       ├── db/                    # asyncpg client + vector backend + session store
│       ├── personas/              # MD files (eca_default, friendly, clinical)
│       ├── services/vieneu_tts/   # TTS client + BackgroundTask
│       ├── graph.py               # StateGraph construction
│       ├── routing.py             # Conditional edges + error routing
│       ├── llm.py                 # ChatOpenAI (DeepSeek) + circuit breakers per role
│       └── state.py               # AgentState TypedDict
│
├── ECA_UI/                        # Frontend
│   ├── index.html                 # Chat + persona + sessions UI
│   └── api.js                     # streamChat (SSE) + listSessions + resumeSession
│
└── tests/langgraph_agents/        # 100 unit + 6 integration tests
```

---

## 📚 Documentation

| Document | Description |
|---|---|
| [docs/RUNBOOK.md](docs/RUNBOOK.md) | Deploy step-by-step, common errors, log analysis (`jq` / PowerShell filters) |
| [docs/DEPLOYMENT.md](docs/DEPLOYMENT.md) | Phase 7 hybrid edge-cloud target (stub) |
| [PLAN-v2.4-DRAFT.md](PLAN-v2.4-DRAFT.md) | Active architecture plan (v2.4.1) |
| [PHASE-6-P0.md](PHASE-6-P0.md) | Production hardening implementation spec |
| [.claude/CLAUDE.md](.claude/CLAUDE.md) | Project conventions, roles (K = architect, N = dev) |

---

## 🛠️ Common Errors

| Symptom | Cause | Fix |
|---|---|---|
| `ConnectionRefusedError [WinError 1225]` | Docker Desktop not running | Start Docker Desktop, then `docker compose up -d` |
| `/health/detailed` returns 503 (`redis: timeout`) | Redis hanging at protocol level | Investigate via `redis-cli info memory`; restart container if needed |
| `CircuitBreakerOpenError` from planner/synthesizer | 3+ DeepSeek failures | Wait 30s for breaker cool-down, retry. Check `/health/detailed` for `breaker:planner` |
| `MCP discovery loaded 0 tools` | Subprocess crash (PYTHONPATH, import) | Graceful — graph runs without MCP. Check log `mcp_discovery_failed` for cause |
| Greeting classified as `clarify` | Old planner prompt | Verify `nodes/planner.py` has few-shot examples block |
| SearXNG returns 403 / empty results | `limiter: true` or `formats` missing `json` in settings.yml | See RUNBOOK §7 |

Full troubleshooting: [docs/RUNBOOK.md § 7](docs/RUNBOOK.md#7-common-errors).

---

## 🔧 Environment Toggles

| Variable | Default | Effect |
|---|---|---|
| `LOG_LEVEL` | `INFO` | `DEBUG` verbose, `WARNING` quiet |
| `LLM_HEALTHCHECK` | `0` | `1` = `/health/detailed` actually calls DeepSeek (costs API credits) |
| `DEEPSEEK_API_KEY` | — | Required for live LLM calls |
| `DEEPSEEK_MODEL` | `deepseek-v4-pro` | Override model name |
| `DEEPSEEK_BASE_URL` | `https://api.deepseek.com` | Override endpoint (e.g. proxy) |
| `SEARXNG_URL` | `http://localhost:6666` | SearXNG endpoint for web search tool |

---

## 🗺️ Roadmap

- ✅ **Phase 0-2.5** — Foundation, planner, memory, retriever, synthesizer (LangGraph v2.4)
- ✅ **Phase 3** — MCP servers (Kimodo mock, web search), FastAPI BackgroundTasks
- ✅ **Phase 3.5** — v2.4.1 simplification (drop Celery, drop session summary, drop token interrupt)
- ✅ **Phase 4** — Persona system (3 MD files, dual-mode conversation)
- ✅ **Phase 5** — SSE streaming, session reopen, frontend rework
- ✅ **Phase 6 P0** — Structured logging, health endpoints, circuit breakers, runbook
- ⏳ **Phase 6.5 P1** — LangSmith tracing, pgvector index tuning, expanded per-node tests (data-driven)
- ⏳ **Phase 7** — Hybrid edge-cloud (VPS + Supabase + edge worker GPU + AWS S3/CloudFront)

Active plan: [PLAN-v2.4-DRAFT.md](PLAN-v2.4-DRAFT.md). Per-phase specs: `PHASE-*.md` files at repo root.

---

<p align="center"><sub>Built with LangGraph · DeepSeek · pgvector · MCP · VieNeu-TTS · SSE</sub></p>
