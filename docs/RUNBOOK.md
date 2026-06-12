# VVA LangGraph — Runbook

Step-by-step guide for deploying the VVA LangGraph backend + ECA UI on a single machine.

## Prerequisites

| Requirement | Notes |
|---|---|
| **Python 3.10+** | Conda recommended (env name `vva` used below) |
| **Docker Desktop** | For PostgreSQL + Redis (canonical setup) |
| **Git** | To clone the repo |
| **DeepSeek API key** | Set in `agenticRAG/agentic_rag_gemini/.env` (gitignored) |

Local install of PostgreSQL/Redis works but isn't documented — Docker is the supported path.

## 1. Clone & environment

```bash
git clone <repo-url> vva
cd vva
git checkout feature/langgraph-rewrite
```

Create conda environment:

```bash
conda create -n vva python=3.10 -y
conda activate vva
pip install -r requirements-langgraph.txt
pip install sentence-transformers  # not pinned in requirements; needed by memory node
```

## 2. Configuration

Create `.env` at `agenticRAG/agentic_rag_gemini/.env` (no `.env.example` shipped):

```ini
# Required
DEEPSEEK_API_KEY=sk-...

# Optional — defaults shown
DEEPSEEK_BASE_URL=https://api.deepseek.com
DEEPSEEK_MODEL=deepseek-v4-pro
LOG_LEVEL=INFO
LLM_HEALTHCHECK=0
```

Voice output is optional and works out of the box via VieNeu-TTS in-process; no extra env needed.

## 3. Containers setup (Docker)

Start PostgreSQL (pgvector) + Redis + SearXNG containers:

```powershell
docker compose -f docker-compose.langgraph.yml up -d
docker ps  # verify vva-postgres (healthy) + vva-redis + vva-searxng are Up
```

The compose file creates:
- PostgreSQL: DB `vva`, user `vva`, password `vva_dev`, port 5432
- Redis: port 6379, 512MB maxmemory, LRU eviction
- SearXNG: port 6666, web search aggregator (Google + Bing + DDG + Wikipedia)

**First-time setup**: generate SearXNG secret and put in repo-root `.env`:

```bash
python -c "import secrets; print('SEARXNG_SECRET_KEY=' + secrets.token_hex(32))" >> .env
```

Verify `.env` is in `.gitignore`. Then `docker compose -f docker-compose.langgraph.yml up -d`
will auto-read `.env` and pass `SEARXNG_SECRET_KEY` to the container. If not set,
compose fails immediately with a clear error (intended — better than silent SearXNG crash).

Run schema migration (Alembic, M.4 schema):

```bash
cd agenticRAG/langgraph_agents
alembic upgrade head
```

Verify tables exist:

```bash
docker exec -it vva-postgres psql -U vva -d vva -c "\dt"
# Expected: users, conversations, messages, summaries, user_memory, documents, kb_embeddings
```

## 4. Start services

Terminal 1 — Backend (port 8080, logs to file):

```powershell
conda activate vva
cd agenticRAG
# Redirect stdout to vva.log so log-analysis commands in §8 work.
# Drop the `*> ..\vva.log` part if you prefer console output.
python -m uvicorn langgraph_agents.api.main:create_app --factory --port 8080 --host 0.0.0.0 *> ..\vva.log
```

Terminal 2 — Frontend (port 3000):

```bash
cd ECA_UI
python -m http.server 3000
```

## 5. Verify

Health checks:

```bash
# Liveness (no dependency checks, <10ms)
curl http://localhost:8080/health
# → {"status": "ok"}

# Readiness (parallel PG/Redis/MCP/LLM checks, 3s timeout each)
curl http://localhost:8080/health/detailed
# → {
#     "status": "ready",
#     "checks": {
#       "postgres":   {"ok": true, "latency_ms": 1.2},
#       "redis":      {"ok": true, "latency_ms": 0.5},
#       "graph":      {"ok": true, "latency_ms": 0.0},
#       "llm":        {"ok": true, "latency_ms": 0.0, "detail": "skipped"},
#       "mcp":        {"ok": true, "latency_ms": 1.8, "detail": "2 tool(s)"},
#       "breaker:planner":      "closed",
#       "breaker:synthesizer":  "closed",
#       "breaker:conversation": "closed"
#     }
#   }
# Status 503 if any check.ok=false (except mcp 0-tools which is graceful).
```

Smoke test (SSE chat):

```bash
echo '{"query":"Xin chao"}' | curl -s -N -X POST http://localhost:8080/chat \
  -H "Content-Type: application/json" -d @-
```

You should see SSE events: `stage:` (memory, planner, conversation), `token:`, `done:`.

Pytest smoke (unit ~10s, integration ~6min — live DeepSeek):

```powershell
pytest tests/langgraph_agents/ -m unit -v            # 100 tests, <30s
pytest tests/langgraph_agents/ -m integration -v     # 6 tests, ~6min
```

Expected: 100/100 unit pass, 6/6 integration pass.

## 6. Open the UI

Browser: `http://localhost:3000/?api_base=http://localhost:8080`

The `?api_base=` query parameter points the frontend at the Phase 5 backend.

## 7. Common errors

### `ConnectionRefusedError [WinError 1225]` (Windows)
Docker Desktop is not running. Start it from Start menu (or PowerShell):
```powershell
& "C:\Program Files\Docker\Docker\Docker Desktop.exe"
docker compose -f docker-compose.langgraph.yml up -d
docker ps  # verify vva-postgres + vva-redis are Up
```
Wait ~30s for the daemon to be ready before `docker compose up`.

### PostgreSQL connection refused
Container down or unhealthy:
```powershell
docker ps --format "{{.Names}}: {{.Status}}"            # should show vva-postgres healthy
docker logs vva-postgres --tail 30
docker compose -f docker-compose.langgraph.yml restart postgres
```

### `pgvector` extension missing
Only an issue if you run PostgreSQL outside the `pgvector/pgvector:pg16` image. The
docker-compose image ships the extension; init_schema creates it. To force-create:
```sql
CREATE EXTENSION IF NOT EXISTS vector;
```

### Redis connection refused
Container down:
```powershell
docker ps --format "{{.Names}}: {{.Status}}"            # should show vva-redis Up
docker compose -f docker-compose.langgraph.yml restart redis
```
Local-install fallback (only if you skipped Docker): `redis-server --port 6379`.

### `ModuleNotFoundError: No module named 'sentence_transformers'`
```bash
pip install sentence-transformers
```

### `ModuleNotFoundError: No module named 'langgraph_agents'` (in MCP subprocess)
MCP subprocess PYTHONPATH is wrong. Verify `mcp/client.py:_package_root()` returns
`parents[2]` (= `agentic_rag_gemini`), NOT `parents[3]` (= `agenticRAG`). Fixed in
commit 858e8a8. Symptom in log:
```json
{"msg":"mcp_discovery_failed","error":"No module named 'langgraph_agents'"}
```

### MCP discovery loaded 0 tools
Subprocess crashed during startup. Common causes:
- PYTHONPATH wrong (see above)
- Server file (`kimodo_server.py`, `web_search_server.py`) has import error
- Python interpreter mismatch: bare `python` resolves to system Python lacking project deps

The graph still runs (graceful degradation) with only in-process `pgvector_search`
tool. Retriever skips MCP-only intents (e.g. `visualize_motion`); grader routes to
clarify when expected tool output is missing.

### Greeting classified as `clarify` instead of `conversation`
Old planner prompt (no few-shot, no ASCII-Vietnamese rule). Verify `nodes/planner.py`
has the 6-example few-shot block including `"Xin chao"` (no diacritics) → conversation.
Fixed in Phase 5 wrap-up commit.

### `CircuitBreakerOpenError` from planner / synthesizer / conversation
DeepSeek API has failed 3+ consecutive calls. Symptom in log:
```json
{"msg":"node_failed","error":"LLM circuit breaker open for 'llm:planner'..."}
```
Resolution: wait 30s for breaker cool-down (half-open probe), then retry. The breaker
auto-closes after one successful call. Each role has its own breaker — one role
opening does not affect others.

Check current state via `/health/detailed`:
```json
{"checks":{"breaker:planner":"open", ...}}
```

### SearXNG returns 403 / empty results
- `limiter: true` in settings.yml → set to `false` for local dev
- `formats:` doesn't include `json` → add it, restart container
- Secret key still placeholder → regenerate as above

### `httpx.ConnectError` from search_medical tool
SearXNG container down or wrong port. Verify:
```powershell
docker ps --format "{{.Names}}: {{.Status}}" | grep searxng
curl -s "http://localhost:6666/search?q=test&format=json" | head
```

### `/health/detailed` returns 503 with `redis: timeout after 2s`
Redis is reachable at TCP level but hanging at protocol (memory pressure, AOF rewrite,
network blip). Liveness `/health` still 200. Investigate Redis health (`redis-cli info memory`).

### Frontend shows "API Error" or blank responses
Check browser console. Frontend defaults to port 8000 — use
`?api_base=http://localhost:8080` to point at the Phase 5 backend.

## 8. Log analysis

All logs are JSON-per-line (configured by `shared/logging.py` at FastAPI startup).
Each line has: `ts`, `lvl`, `logger`, `msg`, `request_id`, plus per-event extras.

**Capturing logs to a file**: uvicorn writes to stdout by default. Redirect when starting
backend (see §4 — `*> vva.log` on PowerShell, `> vva.log 2>&1` on bash/WSL). Without
redirect, the filter commands below have nothing to read.

### Filter by request_id (PowerShell)

```powershell
# Tail backend, parse JSON, filter one request
Get-Content vva.log -Wait | ForEach-Object {
  $obj = $_ | ConvertFrom-Json
  if ($obj.request_id -eq 'abc-123-def') { $obj }
}

# All ERRORs in last 100 lines
Get-Content vva.log -Tail 100 | ForEach-Object { $_ | ConvertFrom-Json } |
  Where-Object lvl -eq 'ERROR' | Format-List
```

### Filter by request_id (bash / WSL)

```bash
# All log lines for one request
tail -f vva.log | jq -c 'select(.request_id == "abc-123-def")'

# Errors with timing
jq -c 'select(.lvl == "ERROR") | {ts, msg, node, error}' vva.log

# Per-node duration distribution (last 1000 lines)
tail -1000 vva.log | jq -c 'select(.msg == "node_complete") | {node, elapsed_ms}'
```

### Tracing one request end-to-end

`POST /chat` generates a `request_id` (UUID4). Every node log line carries it via
`with_request_id()` ContextVar wrapping in `api/main.py`. To trace:

1. Capture `request_id` from `done` SSE event (UI shows or curl `-N` prints it):
   ```
   event: done
   data: {"request_id": "abc-123-def", ...}
   ```
2. Grep all logs for that ID. Order chronologically. You should see:
   `chat_start → node_start(memory) → node_complete(memory) → node_start(planner)
   → node_complete(planner) → node_start(...) → ... → chat_complete`

### Common log events

| `msg` | Where | When |
|---|---|---|
| `lifespan_start` / `lifespan_complete` | api/main.py | FastAPI startup |
| `chat_start` / `chat_complete` | api/main.py | Per request |
| `node_start` / `node_complete` | each node | Per node execution |
| `node_failed` | each node | Node caught exception |
| `error_handler_invoked` | error_handler.py | Routed to error handler |
| `mcp_discovery_ok` / `mcp_discovery_failed` | mcp/client.py | Tool discovery |
| `mcp_discovery_skipped_breaker_open` | mcp/client.py | MCP breaker tripped |

## Stopping

Press `Ctrl+C` in each terminal. No cleanup needed — all state is in PostgreSQL and Redis.

Container shutdown:
```bash
docker compose -f docker-compose.langgraph.yml down       # keep data
docker compose -f docker-compose.langgraph.yml down -v    # wipe volumes
```

## Port map

| Service   | Port | Notes                    |
|-----------|------|--------------------------|
| Backend   | 8080 | FastAPI + SSE streaming  |
| Frontend  | 3000 | Static HTTP server       |
| PostgreSQL| 5432 | Session + vector store   |
| Redis     | 6379 | STM + task results       |
| SearXNG   | 6666 | Self-hosted metasearch (Google+Bing+DDG+Wikipedia) |

## Environment toggles

| Variable | Default | Effect |
|---|---|---|
| `LOG_LEVEL` | `INFO` | Set to `DEBUG` for verbose, `WARNING` to suppress noise |
| `LLM_HEALTHCHECK` | `0` | Set to `1` to actually call LLM in `/health/detailed` (burns API credits) |
