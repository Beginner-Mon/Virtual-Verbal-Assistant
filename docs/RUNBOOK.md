# VVA LangGraph — Runbook

Step-by-step guide for deploying the VVA LangGraph backend + ECA UI on a single machine.

## Prerequisites

| Requirement | Notes |
|---|---|
| **Python 3.10+** | Conda recommended (env name `firstconda` used below) |
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
conda create -n firstconda python=3.10 -y
conda activate firstconda
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

Voice output is **optional** and requires the separate SpeechLLm/VieNeu-TTS service on
port 5000. If it is not running, `output_mode: text` works normally and speech requests
degrade gracefully (the `/health/detailed` `speechllm` check reports `ok: false` →
overall status `degraded`, which is expected — text chat is unaffected).

## 3. Containers setup (Docker)

Start PostgreSQL (pgvector) + Redis + SearXNG containers:

```powershell
docker compose -f docker-compose.langgraph.yml up -d
docker ps  # verify vva-postgres (healthy) + vva-redis + vva-searxng are Up
```

The compose file creates:
- PostgreSQL: DB `vva`, user `vva`, password `vva_dev`, **host port 5433** (mapped 5433→5432
  to avoid clashing with a local PostgreSQL on 5432). All DSNs use 5433.
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

Terminal 1 — Backend (port 8000, logs to file):

> ⚠️ **Port 8080 is reserved** on this machine for the Owner's Spring service — do NOT
> bind the VVA backend to 8080. Use **8000**.

```powershell
conda activate firstconda
cd agenticRAG
# Redirect stdout to vva.log so log-analysis commands in §8 work.
# Drop the `*> ..\vva.log` part if you prefer console output.
python -m uvicorn langgraph_agents.api.main:create_app --factory --port 8000 --host 0.0.0.0 *> ..\vva.log
```

**Auth:** by default `REQUIRE_AUTH=false` — the backend accepts the client-supplied
`user_id` (no login needed, correct for internal demo). When a valid Cognito **ID token**
is sent as `Authorization: Bearer <jwt>`, the backend ignores the client `user_id` and
uses the token's `sub`. Set `REQUIRE_AUTH=true` (+ `COGNITO_REGION` /
`COGNITO_USER_POOL_ID` / `COGNITO_APP_CLIENT_ID`) to **require** a valid token — this is
the production setting that closes the IDOR gap and MUST be on before any network exposure.

Terminal 2 — Frontend. There are two UIs:

**(a) Main UI — new React app (recommended for demo), port 5173:**

```bash
cd ECA_UI/frontend
npm install            # first time only
npm run dev            # Vite dev server → http://localhost:5173
```

It reads the backend URL from `VITE_API_BASE_URL` (default `http://localhost:8000`;
set in `ECA_UI/frontend/.env.local`).
Without `amplify_outputs.json` (i.e. you haven't run `npx ampx sandbox`) it runs in
**demo mode** — no login screen, chat works against the backend with a generated demo
user id. `npm run build` (production bundle) needs `amplify_outputs.json`; `npm run dev`
does not.

**(b) SSE test UI — old vanilla page (debugging), port 3000:**

```bash
cd ECA_UI/test-ui/sse-test
python -m http.server 3000
```

## 5. Verify

Health checks:

```bash
# Liveness (no dependency checks, <10ms)
curl http://localhost:8000/health
# → {"status": "ok"}

# Readiness (parallel checks, 3s timeout each). Use curl.exe on Windows to see the
# body even on HTTP 503 (PowerShell's Invoke-RestMethod throws and hides it).
curl.exe -s http://localhost:8000/health/detailed
# → {
#     "status": "ready",          # "degraded" if a non-critical dep is down
#     "checks": {
#       "redis":     {"ok": true},
#       "postgres":  {"ok": true},
#       "graph":     {"ok": true},
#       "llm":       {"ok": true, "detail": "skipped"},   # config check; not a live call unless LLM_HEALTHCHECK=1
#       "mcp":       {"ok": true, "detail": "2 tool(s)"},
#       "speechllm": {"ok": false},  # TTS :5000 not running → degraded, but text chat still works
#       "searxng":   {"ok": true}
#     }
#   }
# HTTP 503 when any check.ok=false (incl. speechllm down). "degraded" with only
# speechllm false is the EXPECTED state when you haven't started the TTS service.
```

Smoke test (SSE chat) — write the body to a file to avoid shell-quoting issues:

```bash
printf '%s' '{"query":"xin chao","user_id":"smoke","session_id":"smoke-1","output_mode":"text"}' > body.json
curl -s -N -X POST http://localhost:8000/chat -H "Content-Type: application/json" -d @body.json
```

You should see SSE events: `stage:` (memory → planner → synthesizer), a stream of
`token:` events, then `done:` with `total_tokens` + `required_outputs`.

Pytest (full suite ~3min; integration needs Docker PG/Redis + live DeepSeek key):

```powershell
python -m pytest tests/langgraph_agents/ -q          # 245 passed
python -m pytest tests/langgraph_agents/ -m unit -q  # fast subset, no live services
```

Expected: 245 passed (unit + integration on the running PostgreSQL/Redis). If you see
many `SKIPPED ... PostgreSQL not available on port 5433` + a few integration failures,
the Docker containers are down (postgres/redis have no restart policy) — bring them back
with `docker compose -f docker-compose.langgraph.yml up -d postgres redis`.

## 6. Open the UI

New React UI (demo): **`http://localhost:5173`** — talks to the backend on :8000 via
`VITE_API_BASE_URL` (default). No login needed in demo mode.

Old SSE test UI (debugging): `http://localhost:3000/?api_base=http://localhost:8000`
(served from `ECA_UI/test-ui/sse-test/`).

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
docker-compose image ships the extension; the Alembic migration (`alembic upgrade head`)
runs `CREATE EXTENSION IF NOT EXISTS vector`. To force-create:
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
MCP subprocess PYTHONPATH is wrong. After the package relevel, the package lives at
`agenticRAG/langgraph_agents/`, so `mcp/client.py:_package_root()` (`parents[2]`) must
resolve to `agenticRAG/`. A healthy startup logs `mcp_discovery_ok` with `tool_count: 2`.
Symptom of failure in log:
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

### Greeting classified as `clarify` instead of `chat`
Old planner prompt (no few-shot, no ASCII-Vietnamese rule). Verify `nodes/planner.py`
has the few-shot block including `"Xin chao"` (no diacritics) → `required_outputs: []`,
`needs_clarification: false` → synthesizer chat mode. (3-axis model — there is no
`conversation` node anymore; the synthesizer is the universal responder.)

### `CircuitBreakerOpenError` from planner / synthesizer
DeepSeek API has failed 3+ consecutive calls. Symptom in log:
```json
{"msg":"node_failed","error":"LLM circuit breaker open for 'llm:planner'..."}
```
Resolution: wait 30s for breaker cool-down (half-open probe), then retry. The breaker
auto-closes after one successful call. Each role has its own breaker — one role
opening does not affect others. (Breaker state is internal; it is not surfaced in
`/health/detailed`.)

### React UI (:5173) chat fails with `net::ERR_FAILED` / CORS blocked
`.env` `ALLOWED_ORIGINS` overrides the code default and must list the Vite origin.
Ensure `agenticRAG/agentic_rag_gemini/.env` has:
```ini
ALLOWED_ORIGINS=http://localhost:3000,http://localhost:8080,http://localhost:5173
```
Then restart the backend.

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

### Frontend shows "API Error" / "Sorry, something went wrong"
Check browser console for the failing request URL. The React UI defaults to
`http://localhost:8000` (via `VITE_API_BASE_URL`). A **404 from :8080** means the
request hit the Owner's Spring service, not the VVA backend — confirm the backend is
running on **:8000** (`curl http://localhost:8000/health`) and that
`ECA_UI/frontend/.env.local` sets `VITE_API_BASE_URL=http://localhost:8000`. The old
test UI can be pointed with `?api_base=http://localhost:8000`.

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
| Backend   | 8000 | FastAPI + SSE streaming (8080 reserved for Owner's Spring — do not use) |
| Frontend (React) | 5173 | Vite dev server — main UI for demo |
| Frontend (SSE test) | 3000 | Old vanilla page, `ECA_UI/test-ui/sse-test/` |
| PostgreSQL| 5433 | Session + vector store (host 5433 → container 5432) |
| SpeechLLm/TTS | 5000 | VieNeu TTS — optional, only for voice output |
| Redis     | 6379 | STM + task results       |
| SearXNG   | 6666 | Self-hosted metasearch (Google+Bing+DDG+Wikipedia) |

## Environment toggles

| Variable | Default | Effect |
|---|---|---|
| `LOG_LEVEL` | `INFO` | Set to `DEBUG` for verbose, `WARNING` to suppress noise |
| `LLM_HEALTHCHECK` | `0` | Set to `1` to actually call LLM in `/health/detailed` (burns API credits) |
| `REQUIRE_AUTH` | `false` | `true` = reject requests without a valid Cognito ID token (closes IDOR; production). Requires the 3 Cognito vars below. |
| `COGNITO_REGION` / `COGNITO_USER_POOL_ID` / `COGNITO_APP_CLIENT_ID` | — | Cognito user-pool identifiers used to verify the JWT (JWKS + audience + issuer). Only needed when `REQUIRE_AUTH=true`. |
| `ALLOWED_ORIGINS` | `localhost:3000,5173,8080` | CORS allow-list. The Vite UI origin `http://localhost:5173` is included by default. |
| `VITE_API_BASE_URL` (frontend) | `http://localhost:8000` | Where the React UI sends API calls. Set in `ECA_UI/frontend/.env.local` (gitignored) at build/dev time. |
| `EMBEDDING_ALLOW_DOWNLOAD` | `(unset)` | **Embedding model cache.** Default (unset): loads `intfloat/multilingual-e5-small` from `~/.cache/huggingface/hub/` only — no HF-Hub round-trips on restart. Set to `1` for a one-time download on a fresh machine without a local cache. Must be pre-cached before first run on the production machine; see `pip install sentence-transformers && python -c "from sentence_transformers import SentenceTransformer; SentenceTransformer('intfloat/multilingual-e5-small')"`. |
