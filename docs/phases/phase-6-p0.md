# PHASE 6 — Production Hardening (P0 only)

> Architect: K | Developer: N | Date: 2026-05-25
> Branch: `feature/langgraph-rewrite`
> Scope: **P0 — deploy-blocking only.** P1 (LangSmith tracing, pgvector index tuning, expanded per-node unit tests) tách thành phase-6.5 sau khi P0 ổn.

---

## Mục tiêu P0

Bốn hạng mục chặn-deploy. Sau Phase 6 P0, Owner phải:
- Trace 1 request qua 7 nodes bằng 1 `request_id` duy nhất trong log.
- Biết service nào down trong 5 giây qua 1 endpoint.
- LLM/MCP fail không kéo cả request treo > 10s.
- Deploy local (Docker compose up) bằng cách đọc 1 file runbook, không hỏi N.

Không scope: metric dashboards, distributed tracing, alerting (defer Phase 7).

---

## P0.1 — Structured logging + request_id correlation

### Vấn đề

Hiện tại mỗi node `logger = logging.getLogger("langgraph.<node>")` log văn bản, không có `request_id`. Khi 1 request fail, không thể grep ra log của riêng request đó từ stream multi-user.

### Spec

**File mới: `agenticRAG/agentic_rag_gemini/langgraph_agents/shared/logging.py`** (~60 LOC)

```python
"""Structured JSON logging with request_id correlation.

Usage in nodes:
    from langgraph_agents.shared.logging import get_logger, with_request_id

    logger = get_logger("langgraph.planner")

    async def planner_node(state, config):
        request_id = config["configurable"].get("request_id", "-")
        with with_request_id(request_id):
            logger.info("planner_start", extra={"intent_hint": ...})
            ...
"""
import contextvars
import json
import logging
from contextlib import contextmanager

_request_id_var: contextvars.ContextVar[str] = contextvars.ContextVar("request_id", default="-")


class JsonFormatter(logging.Formatter):
    def format(self, record: logging.LogRecord) -> str:
        payload = {
            "ts": self.formatTime(record, "%Y-%m-%dT%H:%M:%S.%fZ"),
            "lvl": record.levelname,
            "logger": record.name,
            "msg": record.getMessage(),
            "request_id": _request_id_var.get(),
        }
        # Merge extra fields (anything not standard)
        std_attrs = set(logging.LogRecord("", 0, "", 0, "", (), None).__dict__) | {"message", "asctime"}
        for k, v in record.__dict__.items():
            if k not in std_attrs:
                payload[k] = v
        if record.exc_info:
            payload["exc"] = self.formatException(record.exc_info)
        return json.dumps(payload, ensure_ascii=False, default=str)


@contextmanager
def with_request_id(request_id: str):
    token = _request_id_var.set(request_id or "-")
    try:
        yield
    finally:
        _request_id_var.reset(token)


def get_logger(name: str) -> logging.Logger:
    return logging.getLogger(name)


def configure_root_logger(level: str = "INFO") -> None:
    """Call once at FastAPI startup (lifespan)."""
    root = logging.getLogger()
    root.setLevel(level)
    # Replace any default handlers
    for h in root.handlers[:]:
        root.removeHandler(h)
    handler = logging.StreamHandler()
    handler.setFormatter(JsonFormatter())
    root.addHandler(handler)
```

**Changes to existing files:**

| File | Change |
|------|--------|
| `api/main.py` lifespan | Call `configure_root_logger(level=os.getenv("LOG_LEVEL","INFO"))` first thing |
| `api/main.py` chat endpoint | Wrap `event_generator()` body in `with with_request_id(request_id):` |
| `api/main.py` chat endpoint | Replace ad-hoc `print`/`logger.warning` with `logger.warning("event_name", extra={...})` style |
| Each node (`planner.py`, `memory.py`, `retriever_agent.py`, `synthesizer.py`, `grader.py`, `conversation.py`, `error_handler.py`) | At top: `request_id = config["configurable"].get("request_id","-"); ` log entry/exit with `extra={"node": ..., "elapsed_ms": ...}` |
| `mcp/client.py` | Replace existing `logger.warning(...)` with structured calls (no contextvar change — MCP runs in subprocess, request_id propagation N/A for now) |

**Convention for log events** (use snake_case event names, not free-text):

```python
logger.info("node_start", extra={"node": "planner"})
logger.info("node_complete", extra={"node": "planner", "elapsed_ms": 287, "intent": "conversation"})
logger.warning("llm_fallback", extra={"node": "planner", "reason": "json_parse_error"})
logger.error("graph_error", extra={"node": "memory", "severity": "critical"}, exc_info=True)
```

### Acceptance

1. `GET /health` returns immediately — log line is single JSON line with `request_id="-"`.
2. `POST /chat` with `request_id` X → every node log carries same X. Verify:
   ```powershell
   pytest tests/langgraph_agents/test_phase6_logging.py -v
   ```
3. Two concurrent requests → no `request_id` cross-contamination (uses `contextvars`, not module-global).
4. Logs are 1 JSON object per line, parseable by `json.loads(line)`.

### Tests to write

**`tests/langgraph_agents/test_phase6_logging.py`** (new, ~80 LOC):

- `test_log_format_is_json` — emit 1 log, assert JSON-parseable, has expected keys (ts, lvl, logger, msg, request_id).
- `test_request_id_propagates_in_context` — inside `with with_request_id("abc"):` emit log → record shows `"request_id":"abc"`.
- `test_request_id_isolated_across_async_tasks` — spawn 2 asyncio.tasks each with different request_id, emit logs, assert no leak.
- `test_extra_fields_merged` — `logger.info("x", extra={"foo":1})` → JSON contains `"foo":1`.

---

## P0.2 — Health endpoint with dependency checks

### Vấn đề

`GET /health` hiện chỉ trả `{"status":"ok", "graph_loaded": True}`. Không biết Redis/PG/MCP có sống không. Khi Owner deploy mà Redis container chết, nghĩ service OK → request đầu tiên mới phát hiện.

### Spec

**File mới: `agenticRAG/agentic_rag_gemini/langgraph_agents/api/health.py`** (~80 LOC):

```python
"""Dependency health checks.

Each check: returns (ok: bool, latency_ms: float, detail: str|None).
Aggregated /health/detailed endpoint times all checks in parallel and returns
overall status + per-dependency breakdown.
"""
import asyncio
import time
from dataclasses import dataclass


@dataclass
class CheckResult:
    name: str
    ok: bool
    latency_ms: float
    detail: str | None = None


async def check_redis(timeout: float = 2.0) -> CheckResult: ...
async def check_postgres(timeout: float = 2.0) -> CheckResult: ...
async def check_mcp_tools(timeout: float = 3.0) -> CheckResult: ...
async def check_llm(timeout: float = 5.0) -> CheckResult: ...
    # Cheap call: ainvoke with a 1-token "ok?" prompt. Skip if env LLM_HEALTHCHECK=0.


async def run_all_checks() -> list[CheckResult]:
    return await asyncio.gather(
        check_redis(),
        check_postgres(),
        check_mcp_tools(),
        check_llm(),
        return_exceptions=False,  # each check catches its own exceptions
    )
```

**Changes to `api/main.py`:**

```python
@application.get("/health")
async def health():
    """Liveness — does NOT check dependencies. Returns 200 if process alive."""
    return {"status": "ok", "graph_loaded": _graph is not None}


@application.get("/health/detailed")
async def health_detailed():
    """Readiness — checks all deps in parallel. Returns 200 if all OK, 503 if any fail."""
    results = await run_all_checks()
    overall_ok = all(r.ok for r in results)
    payload = {
        "status": "ok" if overall_ok else "degraded",
        "checks": [
            {"name": r.name, "ok": r.ok, "latency_ms": round(r.latency_ms, 1), "detail": r.detail}
            for r in results
        ],
    }
    status_code = 200 if overall_ok else 503
    return JSONResponse(payload, status_code=status_code)
```

### Check implementations

**Redis** — `_get_redis().ping()` in `asyncio.to_thread` with timeout.

**Postgres** — `await asyncpg.connect(...).fetchval("SELECT 1")`. Reuse existing connection helper in `db/postgres.py` if exists.

**MCP tools** — call `get_mcp_tools()` (cached). OK if returns non-empty list. Degraded but not fatal if empty (graceful degradation per Plan).

**LLM** — optional. Set `LLM_HEALTHCHECK=0` in env to skip (avoid burning DeepSeek credits on every health probe). Default off.

### Acceptance

1. `GET /health` returns 200 in <10ms even when Redis/PG down (liveness ≠ readiness).
2. `GET /health/detailed` with all deps up → 200 + all `ok:true`.
3. Stop Redis container → `/health/detailed` returns 503 + `redis.ok:false` within 3s (timeout enforced).
4. Stop PG → returns 503 + `postgres.ok:false`.
5. `LLM_HEALTHCHECK=0` env → `llm` check returns `ok:true, detail:"skipped"`.

### Tests

**`tests/langgraph_agents/test_phase6_health.py`** (new, ~120 LOC):

- `test_health_liveness_no_dep_check` — mock all deps to throw, `/health` still 200.
- `test_health_detailed_all_ok` — mock all check_* to return OK, status=200.
- `test_health_detailed_redis_down` — patch `check_redis` to return ok=False, status=503.
- `test_health_detailed_parallel_execution` — patch all 4 checks each sleeps 1s, total elapsed < 1.5s (not 4s).
- `test_health_detailed_check_timeout` — patch check to hang, verify timeout fires within 3s.
- `test_llm_health_skipped_when_env_zero` — set `LLM_HEALTHCHECK=0`, verify detail="skipped".

---

## P0.3 — Circuit breakers on LLM + MCP

### Vấn đề

DeepSeek hoặc MCP subprocess hang → mỗi request đợi 30s+ timeout. Cascading failure: nếu DeepSeek down 5 phút, 100 request đầu mỗi cái chờ 30s = service treo.

### Spec

Reuse `core/circuit_breaker.py` (đã có). Wrap 2 call sites:

#### A. DeepSeek LLM (in `llm.py`)

```python
# llm.py — add at top
from langgraph_agents.core.circuit_breaker import CircuitBreaker

_llm_breaker = CircuitBreaker(
    name="deepseek",
    failure_threshold=3,       # 3 consecutive fails → open
    cool_down_seconds=30,      # then probe after 30s
)


def _wrap_with_breaker(llm):
    """Wrap ChatOpenAI so .ainvoke()/.astream() check breaker first."""
    original_ainvoke = llm.ainvoke
    original_astream = llm.astream

    async def guarded_ainvoke(*args, **kwargs):
        _llm_breaker.before_call()  # raises BreakerOpen if open
        try:
            result = await original_ainvoke(*args, **kwargs)
            _llm_breaker.on_success()
            return result
        except Exception:
            _llm_breaker.on_failure()
            raise

    async def guarded_astream(*args, **kwargs):
        _llm_breaker.before_call()
        try:
            async for chunk in original_astream(*args, **kwargs):
                yield chunk
            _llm_breaker.on_success()
        except Exception:
            _llm_breaker.on_failure()
            raise

    llm.ainvoke = guarded_ainvoke
    llm.astream = guarded_astream
    return llm


@lru_cache(maxsize=8)
def get_chat_model(role, *, temperature=None):
    ...  # existing code
    return _wrap_with_breaker(ChatOpenAI(...))
```

**Note on `core/circuit_breaker.py` API**: verify it has `before_call() / on_success() / on_failure()` methods. If naming differs, adapt. If `BreakerOpen` exception doesn't exist, add it (subclass of `RuntimeError`).

**Node handling**: existing nodes already catch `Exception` and fall back to recoverable error (planner.py:110, synthesizer/conversation similar). `BreakerOpen` will bubble up the same path — no extra changes needed.

#### B. MCP tools (in `mcp/client.py`)

```python
_mcp_breaker = CircuitBreaker(name="mcp_discovery", failure_threshold=2, cool_down_seconds=60)

async def get_mcp_tools() -> list:
    global _mcp_client, _mcp_tools
    if _mcp_tools:
        return _mcp_tools

    async with _init_lock:
        if _mcp_tools:
            return _mcp_tools

        try:
            _mcp_breaker.before_call()
        except BreakerOpen:
            logger.warning("mcp_discovery_skipped_breaker_open")
            return []

        cfg = _load_mcp_config()
        if not cfg:
            return []

        try:
            _mcp_client = MultiServerMCPClient(cfg)
            _mcp_tools = await _mcp_client.get_tools()
            _mcp_breaker.on_success()
            ...
        except Exception as exc:
            _mcp_breaker.on_failure()
            ...
```

### Acceptance

1. 3 consecutive DeepSeek failures → 4th call raises `BreakerOpen` immediately (<10ms), no 30s wait.
2. After 30s cool-down → 1 probe call. If succeeds → breaker closed. If fails → re-open.
3. `BreakerOpen` from planner → recoverable error path → `intent="clarify"` (existing fallback).
4. MCP breaker open → `get_mcp_tools()` returns `[]` immediately, retriever skips MCP-only intents (existing graceful path).

### Tests

**`tests/langgraph_agents/test_phase6_circuit_breaker.py`** (new, ~100 LOC):

- `test_llm_breaker_opens_after_3_failures` — monkeypatch `ChatOpenAI.ainvoke` to raise, call get_chat_model().ainvoke() 3x, assert 4th raises BreakerOpen.
- `test_llm_breaker_closes_on_success_after_cooldown` — open breaker, sleep past cooldown, ainvoke succeeds → breaker state == closed.
- `test_mcp_breaker_returns_empty_when_open` — force breaker open, get_mcp_tools() returns [] within 10ms.
- `test_planner_handles_breaker_open_as_recoverable` — integration: open breaker, run planner_node, assert intent=="clarify" + recoverable error in state.

---

## P0.4 — Runbook + deployment doc

### Vấn đề

Plan v2.4 chưa có 1 file step-by-step để Owner deploy local từ zero. Hiện kiến thức ở N's head + worklogs rải rác.

### Spec

**File mới: `docs/RUNBOOK.md`** (~250 lines). Structure:

```markdown
# VVA LangGraph — Runbook

## 1. Prerequisites
- Windows/WSL with Docker Desktop
- Python 3.12 + miniconda env `firstconda`
- DeepSeek API key (.env file)

## 2. First-time setup (5 min)
- Clone repo, checkout feature/langgraph-rewrite
- Create .env: DEEPSEEK_API_KEY=..., REDIS_URL=..., POSTGRES_URL=...
- conda activate firstconda
- pip install -r requirements-langgraph.txt
- docker compose -f docker-compose.langgraph.yml up -d
- python agenticRAG/agentic_rag_gemini/langgraph_agents/db/init.py (run schema)

## 3. Start services (everyday)
- docker compose -f docker-compose.langgraph.yml up -d   # PG + Redis
- uvicorn langgraph_agents.api.main:create_app --factory --port 8000
- Open ECA_UI/index.html in browser (or live-server on :3000)

## 4. Verify health
- GET http://localhost:8000/health           → liveness
- GET http://localhost:8000/health/detailed  → all deps OK

## 5. Smoke test
- pytest tests/langgraph_agents/ -m unit -v             # 78 tests, <30s
- pytest tests/langgraph_agents/ -m integration -v      # 6 tests, ~6min (live DeepSeek)

## 6. Common errors

### "ConnectionRefusedError [WinError 1225]"
- Docker Desktop not running. Start it, then docker compose up -d.

### "ModuleNotFoundError: langgraph_agents"
- MCP subprocess PYTHONPATH wrong. Verify mcp/client.py _package_root() points
  at parents[2]=agentic_rag_gemini. Fixed in commit 858e8a8.

### "intent classified as clarify for greeting"
- Old planner prompt. Verify planner.py has few-shot examples (Phase 5 fix).

### "BreakerOpen exception from planner"
- DeepSeek down 3+ times. Wait 30s for cool-down probe.

### "MCP discovery loaded 0 tools"
- Subprocess crashed. Check log line "mcp_discovery_failed". Usually
  PYTHONPATH or missing package. Graph still works without MCP (graceful).

## 7. Log analysis

All logs are JSON-per-line. Filter by request_id:
  Get-Content vva.log | ConvertFrom-Json | Where-Object request_id -eq 'abc-123'

## 8. Shutdown
- Ctrl+C uvicorn
- docker compose -f docker-compose.langgraph.yml down  (keeps data)
- docker compose -f docker-compose.langgraph.yml down -v  (wipes volumes)
```

**File mới: `docs/DEPLOYMENT.md`** (~150 lines) — same content nhưng cho VPS (Phase 7 prep). Có thể stub cho giờ, ghi `// TODO: Phase 7` cho các section AWS/CloudFront.

### Acceptance

1. Owner read RUNBOOK.md từ đầu → service chạy + smoke test pass mà không hỏi N.
2. Mọi error message trong "Common errors" tái hiện được + có fix rõ ràng.
3. Log filter command (`ConvertFrom-Json | Where-Object request_id`) work với log thật.

### Tests

Manual only:
1. **Empty machine test** — N copy repo sang máy mới (hoặc tạo fresh Docker volumes), follow RUNBOOK section 2-5 word-for-word. Bất kỳ bước nào fail → fix RUNBOOK, không sửa code.
2. **Owner walkthrough** — Owner đọc RUNBOOK, deploy 1 mình. N ngồi cạnh nhưng không can thiệp, chỉ ghi lại chỗ Owner stuck → improve RUNBOOK.

---

## Implementation Order

Tier P0 chia 4 commit độc lập:

| # | Commit | Files | Est LOC | Tests |
|---|--------|-------|---------|-------|
| 1 | feat(phase-6): structured JSON logging + request_id ctxvar | shared/logging.py + 7 node updates + main.py | ~250 | test_phase6_logging.py (~80) |
| 2 | feat(phase-6): health/detailed endpoint with parallel dep checks | api/health.py + main.py | ~120 | test_phase6_health.py (~120) |
| 3 | feat(phase-6): circuit breakers on DeepSeek LLM + MCP discovery | llm.py + mcp/client.py + (maybe) core/circuit_breaker.py | ~80 | test_phase6_circuit_breaker.py (~100) |
| 4 | docs(phase-6): RUNBOOK + DEPLOYMENT stub | docs/RUNBOOK.md + docs/DEPLOYMENT.md | ~400 | manual walkthrough |

Total: ~850 LOC code + ~300 LOC tests + ~400 LOC docs. Estimated 2 days N's time.

**Suggested order**: 1 → 3 → 2 → 4. Logging first (debugging tool for the rest). Circuit breakers second (changes LLM behavior, want to test early). Health third (reuses logging). Docs last (when behavior frozen).

---

## Out of P0 scope (defer to phase-6.5)

- **LangSmith tracing**: just set env vars `LANGCHAIN_TRACING_V2=true`, `LANGCHAIN_API_KEY=...`. No code change needed. Document in RUNBOOK §3.
- **pgvector index tuning**: defer until Phase 7 prod data shows p95 > 500ms on vector search. Current IVFFlat lists=100 fine for <100k docs.
- **Per-node unit tests expansion**: integration tests cover happy path. Per-node unit tests valuable but not deploy-blocking.
- **Health UI badge in ECA_UI**: nice-to-have, defer.
- **Prometheus / Grafana metrics**: Phase 7 (cloud deploy).

---

## Acceptance gate for Phase 6 P0

Tôi (K) sẽ review code N theo checklist:

- [ ] All 4 commits land, individual tests pass.
- [ ] `pytest -m unit` total: 78 + ~30 new = ~108 tests, all green.
- [ ] `pytest -m integration` total: 6 existing, all green (no regression from circuit breaker wrap).
- [ ] Manual: `tail -f` uvicorn output → JSON-per-line. Stop Redis → `/health/detailed` returns 503 within 3s. Force DeepSeek failure (bad API key for 1 request) 3x → 4th request fails immediately, breaker recovers after 30s.
- [ ] RUNBOOK §5 smoke test reproducible on N's machine starting from `docker compose down -v`.
- [ ] Worklog `docs/worklogs/<date>.md` documents each commit + any spec deviations.

Pass → merge to `feature/langgraph-rewrite`, tag `phase-6-p0`, move to phase-6.5 P1.
