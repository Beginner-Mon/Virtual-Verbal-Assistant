# Phase 3.5 — Phase 3 Finalize + Simplification (v2.4 → v2.4.1)

**Architect**: K | **Developer**: N | **Date**: 2026-05-24
**Branch**: `feature/langgraph-rewrite` (continue from Phase 3 uncommitted work)
**Estimated time**: ~3-4h
**Reference**: [../plans/v2.4-plan.md](../plans/v2.4-plan.md) §17 (Phase 3.5), worklog [../worklogs/24-05-2026.md](../worklogs/24-05-2026.md)

---

## 0. Why this phase exists

Phase 3 code đã viết xong nhưng:
- **Chưa commit** (23 file untracked + modified)
- **3 test fail** ở `test_phase3_api.py` (mock bug, N báo nhầm là "FastAPI/starlette incompat")
- Plan v2.4 đã được **bump lên v2.4.1** với 3 quyết định kiến trúc mới:
  1. Bỏ Celery cho TTS → dùng FastAPI `BackgroundTasks`
  2. Bỏ session summary agent → reopen API đơn giản (timestamp + preview)
  3. Bỏ `interrupt()` token budget flow → chỉ tracking
- **Env packages chưa pin** → mỗi máy mới có thể bốc combo broken (fastapi/starlette mismatch xảy ra trong session vừa rồi)
- **Config v2.2 dead keys** chưa cleanup → misleading

Phase 3.5 = đóng sạch Phase 3 + áp dụng v2.4.1 simplification trước khi mở Phase 5.

---

## 1. Order of execution

```
3.5.1  Pin versions (env stability first)
3.5.2  Cleanup config/langgraph.yaml dead keys
3.5.3  REFACTOR TTS Celery → FastAPI BackgroundTasks   ← biggest task
3.5.4  Disable celery_app.py (skeleton reserved Phase 7)
3.5.5  Remove celery from requirements
3.5.6  Update api/main.py to use BackgroundTasks
3.5.7  Harden tts_result endpoint (defense-in-depth)
3.5.8  Fix 3 test_phase3_api.py mock bugs + rewrite test_phase3_tts_task.py
3.5.9  Verify all tests xanh
3.5.10 Commit Phase 3 + Phase 3.5 work
```

Commit từng task riêng (10 commits gọn) hoặc gom 2 commit lớn (Phase 3 work + Phase 3.5 refactor) — N tự quyết.

---

## 2. Task 3.5.1 — Pin versions

**File**: [requirements-langgraph.txt](requirements-langgraph.txt)

### Why

Session vừa rồi: pip resolver chọn `fastapi 0.128.0 + starlette 1.0.1` → `TypeError: Router.__init__() got an unexpected keyword argument 'on_startup'`. Fix bằng upgrade `fastapi → 0.135.1` thủ công. Máy mới của Owner/CI sẽ vấp lại nếu không pin.

### New content

```
# Core framework
langgraph>=0.2.0
langchain-core>=0.3.0

# LLM provider — DeepSeek via OpenAI-compatible
langchain-openai>=1.0.0,<2.0.0

# MCP Integration
langchain-mcp-adapters>=0.1.0
mcp>=1.0.0

# Database
asyncpg>=0.29.0
pgvector>=0.3.0
alembic>=1.13.0

# FastAPI stack — pin to avoid starlette compat breaks
fastapi>=0.130,<0.140
starlette>=1.0,<2.0
uvicorn[standard]>=0.30,<1.0
sse-starlette>=3.4,<4.0

# Redis — STM + task_result persistence (DB 0)
redis>=5.0.0,<6.0.0

# .env loader
python-dotenv>=1.0.0

# Web search (MCP backend)
duckduckgo-search>=6.0.0

# Pydantic
pydantic>=2.0

# Testing
pytest>=8.0.0
pytest-asyncio>=0.23.0
```

### Removed (v2.4.1)

```diff
- celery>=5.4.0
- langchain-google-genai>=2.0.0
- anthropic>=0.34.0
```

### Acceptance

- [ ] `pip install -r requirements-langgraph.txt` không có conflict warning
- [ ] `python -c "import fastapi, starlette, sse_starlette, langchain_openai; print('ok')"` chạy
- [ ] FastAPI `create_app()` không crash

---

## 3. Task 3.5.2 — Cleanup config/langgraph.yaml

**File**: [config/langgraph.yaml](config/langgraph.yaml)

### Hiện trạng — có nhiều key chết

```yaml
langgraph:
  llm:
    provider: "gemini"               # ← code dùng DeepSeek, không đọc field này
    manager_model: "gemini-2.5-flash"      # ← code dùng llm.py hardcoded
    reasoning_model: "gemini-2.5-flash"    # ← dead
    conversation_model: "gemini-2.5-flash" # ← dead
    temperature: {...}               # ← code dùng _DEFAULT_TEMPS trong llm.py
    max_tokens: {...}                # ← dead

  services:
    kimodo:                          # ← services/kimodo/ đã xóa
      url: "http://localhost:5001"
      ...
    vieneu_tts:                      # ← OK, vieneu_tts/client.py vẫn đọc
      url: "http://localhost:5000"
      ...

  celery:                            # ← v2.4.1: bỏ celery
    broker_url: "redis://localhost:6379/1"
    ...
```

### New content

```yaml
# LangGraph Agent Configuration (v2.4.1)
# Pruned: removed dead v2.2 keys (manager_model, services.kimodo, celery)

langgraph:
  # LLM defaults are in code (llm.py). Config here only for runtime overrides.
  # Set DEEPSEEK_MODEL / DEEPSEEK_BASE_URL env vars to override.

  memory:
    redis_url: "redis://localhost:6379"   # DB 0 (STM + task_result)
    stm_session_ttl: 7200                 # 2h
    ltm_top_k: 5

  retrieval:
    top_k: 5
    web_search_enabled: true
    web_search_max_results: 3

  services:
    vieneu_tts:
      url: "http://localhost:5000"
      endpoint: "/synthesize"
      timeout: 15
      circuit_breaker:
        failure_threshold: 3
        cool_down_seconds: 60

  postgres:
    dsn: "postgresql://vva:vva_dev@localhost:5432/vva"
    pool_min: 2
    pool_max: 10

  persona:
    default: "eca_default"
    personas_dir: "langgraph_agents/personas"
    voices_dir: "voices"
```

### Acceptance

- [ ] File chỉ còn key đang được code đọc
- [ ] `grep -rn "manager_model\|reasoning_model\|services.kimodo\|langgraph.celery" agenticRAG/agentic_rag_gemini/langgraph_agents/` rỗng (không reference)
- [ ] Smoke test: `python -m langgraph_agents.db.init_schema` vẫn chạy (postgres config OK)

---

## 4. Task 3.5.3 — REFACTOR TTS Celery → FastAPI BackgroundTasks

**File**: [services/vieneu_tts/tasks.py](agenticRAG/agentic_rag_gemini/langgraph_agents/services/vieneu_tts/tasks.py)

### Hiện tại (Celery)

```python
if celery_app is not None:
    @celery_app.task(name="langgraph.synthesize_speech", bind=True, ...)
    def synthesize_speech(self, text, request_id, session_id, voice_path=None):
        # sync httpx → VieNeu
        ...
```

### Rewrite — plain async function

```python
"""TTS task — async function fired by FastAPI BackgroundTasks (v2.4.1).

No Celery. Result persisted to Redis `task_result:{task_id}` with 1h TTL.
GET /tts/{task_id}/result polls this until populated (Phase 5: SSE push).
"""

from __future__ import annotations

import asyncio
import json
import logging

import redis.asyncio as aioredis

from langgraph_agents.services.exceptions import ServiceUnavailableError
from langgraph_agents.services.vieneu_tts.client import get_vieneu_tts_client

logger = logging.getLogger("langgraph.tasks.vieneu_tts")

_REDIS_URL = "redis://localhost:6379/0"   # v2.4.1: DB 0 (bo DB 1 broker)


async def synthesize_speech_async(
    text: str,
    task_id: str,
    voice_path: str | None = None,
) -> None:
    """Run VieNeu-TTS synthesize, persist result to Redis.

    Designed for FastAPI BackgroundTasks. Never raises (errors persisted as
    speech_failed payload). Caller does not await result — fire-and-forget.
    """
    client = get_vieneu_tts_client()

    try:
        result = await client.synthesize(text=text, voice_path=voice_path)
        payload = {
            "event": "speech_ready",
            "task_id": task_id,
            "url": result.get("audio_url", ""),
        }
    except ServiceUnavailableError as exc:
        logger.error("VieNeu-TTS unavailable: %s", exc)
        payload = {"event": "speech_failed", "task_id": task_id, "error": str(exc)}
    except Exception as exc:
        logger.exception("Unexpected TTS error")
        payload = {"event": "speech_failed", "task_id": task_id, "error": str(exc)}

    await _persist(task_id, payload)


async def _persist(task_id: str, payload: dict) -> None:
    """Best-effort Redis write — never raise back to caller."""
    r = aioredis.from_url(_REDIS_URL, socket_connect_timeout=2)
    try:
        await r.setex(f"task_result:{task_id}", 3600, json.dumps(payload))
    except Exception:
        logger.warning("Failed to persist TTS result to Redis (task_id=%s)", task_id)
    finally:
        close_fn = getattr(r, "aclose", None) or r.close
        try:
            await close_fn()
        except Exception:
            pass
```

### Notes

- **`async` thay vì sync** — `BackgroundTasks` chạy được cả 2 nhưng async không block thread pool
- **`redis.asyncio`** thay vì sync — match async function context
- **Never raise** — BackgroundTasks không có retry/error handler, exception chỉ log. Failure thể hiện qua `speech_failed` payload
- **DB 0** — bỏ DB 1 broker, gộp về 1 Redis namespace

### Client method (xác minh không cần đổi)

[services/vieneu_tts/client.py](agenticRAG/agentic_rag_gemini/langgraph_agents/services/vieneu_tts/client.py) đã có cả `synthesize()` async và `synthesize_sync()`. BackgroundTasks dùng async → xóa `synthesize_sync()` nếu muốn cleanup (không bắt buộc).

### Acceptance

- [ ] File mới không còn import `celery_app`
- [ ] `synthesize_speech_async(text="hello", task_id="t1")` chạy (mock VieNeu, mock Redis) → Redis key `task_result:t1` populated với `speech_ready`
- [ ] Service down → Redis key có `speech_failed` payload, không raise

---

## 5. Task 3.5.4 — Disable celery_app.py (giữ skeleton)

**File**: [celery_app.py](agenticRAG/agentic_rag_gemini/langgraph_agents/celery_app.py)

### Update

```python
"""Celery app — DISABLED in v2.4.1 (reserved for Phase 7 hybrid cloud).

v2.4 used Celery for TTS background tasks. v2.4.1 switched to FastAPI
BackgroundTasks (in-process async) because:
  - VieNeu-TTS latency < 10s, no need for distributed worker
  - Removes 1 process + 1 Redis DB + 1 dependency
  - Simpler test mocking

This module is kept as a skeleton for Phase 7 (hybrid cloud) when:
  - TTS scale > 100 req/min → need queue + multiple workers
  - Add heavy async jobs (S3 batch upload, doc ingestion, scheduled tasks)
  - Edge worker (HP ProDesk) needs to consume jobs from cloud queue

To reactivate:
  1. Add `celery>=5.4.0` to requirements-langgraph.txt
  2. Add `langgraph.celery` block to config/langgraph.yaml
  3. Convert TTS task in services/vieneu_tts/tasks.py back to @celery_app.task
  4. Wire `synthesize_speech.delay(...)` in api/main.py
"""

from __future__ import annotations

import logging

logger = logging.getLogger("langgraph.celery")

celery_app = None   # disabled

# Phase 7 reactivation snippet (commented):
#
# try:
#     import yaml
#     from celery import Celery
#     from pathlib import Path
#
#     def _load_cfg() -> dict:
#         path = Path(__file__).resolve().parents[3] / "config" / "langgraph.yaml"
#         if not path.exists():
#             return {}
#         with open(path, "r", encoding="utf-8") as f:
#             return yaml.safe_load(f).get("langgraph", {}).get("celery", {})
#
#     cfg = _load_cfg()
#     celery_app = Celery("langgraph_agents")
#     celery_app.conf.update(
#         broker_url=cfg.get("broker_url", "redis://localhost:6379/1"),
#         result_backend=cfg.get("result_backend", "redis://localhost:6379/1"),
#         task_default_queue="langgraph",
#         imports=("langgraph_agents.services.vieneu_tts.tasks",),
#         # ... rest of config
#     )
# except ImportError:
#     pass
```

### Acceptance

- [ ] `from langgraph_agents.celery_app import celery_app` → `celery_app is None`
- [ ] Không import file này trong production path (chỉ test cũ có thể import → fix ở Task 3.5.8)

---

## 6. Task 3.5.5 — Remove celery from requirements

Đã làm trong Task 3.5.1. Verify:

```bash
grep -i "^celery" requirements-langgraph.txt
# Expected: no output
```

---

## 7. Task 3.5.6 — Update `api/main.py` to use BackgroundTasks

**File**: [api/main.py](agenticRAG/agentic_rag_gemini/langgraph_agents/api/main.py)

### Diff chính

```diff
- import json
+ import json
+ import uuid
- from fastapi import FastAPI, HTTPException
+ from fastapi import FastAPI, HTTPException, BackgroundTasks
+ from langgraph_agents.services.vieneu_tts.tasks import synthesize_speech_async

  @application.post("/chat", response_model=ChatResponse)
- async def chat(req: ChatRequest):
+ async def chat(req: ChatRequest, background_tasks: BackgroundTasks):
      request_id = str(uuid.uuid4())
      ...
      result = await _graph.ainvoke(state, config=config)

      speech_task_id: str | None = None
      if req.output_mode in ("speech", "both") and result.get("final_answer"):
          persona = get_persona(req.persona_id)
          voice_path = persona.get("voice_identity", {}).get("voice_path")
-         from langgraph_agents.services.vieneu_tts.tasks import synthesize_speech
-         task = synthesize_speech.delay(
-             result["final_answer"], request_id, req.session_id, voice_path,
-         )
-         speech_task_id = task.id
+         speech_task_id = str(uuid.uuid4())
+         background_tasks.add_task(
+             synthesize_speech_async,
+             text=result["final_answer"],
+             task_id=speech_task_id,
+             voice_path=voice_path,
+         )

      # Empty response fallback (Plan §13.1 step 3)
+     final_answer = result.get("final_answer") or "Xin lỗi, tôi không thể xử lý yêu cầu này."

      return ChatResponse(
          request_id=request_id,
-         final_answer=result.get("final_answer", ""),
+         final_answer=final_answer,
          ...
      )
```

### Acceptance

- [ ] `/chat` không còn import `synthesize_speech.delay`
- [ ] `BackgroundTasks` được inject qua signature
- [ ] Test: POST /chat với `output_mode="speech"` → response có `speech_task_id` non-null
- [ ] Test: empty `final_answer` → response có fallback message thay vì empty

---

## 8. Task 3.5.7 — Harden `tts_result` endpoint

**File**: [api/main.py](agenticRAG/agentic_rag_gemini/langgraph_agents/api/main.py)

### Bug hiện tại

Test mock `redis.get(...)` trả `MagicMock` → `if not raw: 404 else json.loads(raw)` → fall vào else → `json.loads(MagicMock)` → `TypeError`.

Production cũng có rủi ro: Redis trả bytes/str/None bình thường, nhưng nếu key tồn tại với content corrupt (không phải JSON) → 500.

### Fix

```python
@application.get("/tts/{task_id}/result")
async def tts_result(task_id: str):
    raw = _get_redis().get(f"task_result:{task_id}")
    if raw is None or not isinstance(raw, (bytes, str)):
        raise HTTPException(404, "Task not ready or expired")
    try:
        return json.loads(raw)
    except (json.JSONDecodeError, TypeError):
        raise HTTPException(500, "Corrupt task result in cache")
```

### Acceptance

- [ ] `mock_redis.get.return_value = None` → 404
- [ ] `mock_redis.get.return_value = MagicMock()` → 404 (không crash)
- [ ] `mock_redis.get.return_value = b'{"event":"speech_ready","url":"x"}'` → 200 + payload
- [ ] `mock_redis.get.return_value = b'not json'` → 500

---

## 9. Task 3.5.8 — Fix tests

### 9.1 [test_phase3_api.py](tests/langgraph_agents/test_phase3_api.py)

**Bug**: fixture `mock_redis = MagicMock()` không set `get.return_value` → mặc định trả MagicMock truthy.

**Fix fixture**:

```python
@pytest.fixture
def api_client(monkeypatch):
    mock_graph = AsyncMock()
    mock_graph.ainvoke = AsyncMock(return_value={
        "final_answer": "Xin chào! Tôi là ECA.",
        "intent": "conversation",
        "confidence": 0.95,
        "needs_clarification": False,
        "total_tokens": 42,
        "grader_result": "pass",
        "grader_warning": None,
        "errors": [],
    })

    mock_redis = MagicMock()
    mock_redis.get.return_value = None   # ← KEY FIX

    captured_tasks = []

    def fake_add_task(func, **kwargs):
        captured_tasks.append((func.__name__, kwargs))

    # Patch the BackgroundTasks.add_task at the FastAPI level — actually
    # we capture by mocking synthesize_speech_async directly:
    async def fake_synth(**kwargs):
        captured_tasks.append(("synthesize_speech_async", kwargs))

    monkeypatch.setattr(
        "langgraph_agents.api.main.synthesize_speech_async",
        fake_synth,
    )
    monkeypatch.setattr(
        "langgraph_agents.api.main.build_graph_async",
        AsyncMock(return_value=mock_graph),
    )
    monkeypatch.setattr(
        "langgraph_agents.api.main.sync_redis.Redis.from_url",
        lambda *a, **kw: mock_redis,
    )

    from langgraph_agents.api.main import create_app
    from fastapi.testclient import TestClient
    app = create_app()
    client = TestClient(app)

    yield client, mock_graph, captured_tasks, mock_redis


@pytest.mark.unit
def test_chat_text_mode_no_tts_task(api_client):
    client, _, captured, _ = api_client
    resp = client.post("/chat", json={"query": "Xin chào", "output_mode": "text"})
    assert resp.status_code == 200
    data = resp.json()
    assert data["final_answer"] == "Xin chào! Tôi là ECA."
    assert data["speech_task_id"] is None
    assert captured == []


@pytest.mark.unit
def test_chat_speech_mode_fires_background_task(api_client):
    client, _, captured, _ = api_client
    resp = client.post("/chat", json={"query": "Hãy đọc câu này", "output_mode": "speech"})
    assert resp.status_code == 200
    data = resp.json()
    assert data["speech_task_id"] is not None
    # Background task scheduled — wait for TestClient to run it
    assert len(captured) == 1
    assert captured[0][0] == "synthesize_speech_async"


@pytest.mark.unit
def test_tts_result_404_when_missing(api_client):
    client, _, _, mock_redis = api_client
    mock_redis.get.return_value = None
    resp = client.get("/tts/nonexistent/result")
    assert resp.status_code == 404


@pytest.mark.unit
def test_tts_result_200_when_present(api_client):
    client, _, _, mock_redis = api_client
    mock_redis.get.return_value = b'{"event":"speech_ready","url":"http://x.wav"}'
    resp = client.get("/tts/abc/result")
    assert resp.status_code == 200
    assert resp.json()["event"] == "speech_ready"


@pytest.mark.unit
def test_tts_result_500_on_corrupt(api_client):
    client, _, _, mock_redis = api_client
    mock_redis.get.return_value = b'not json'
    resp = client.get("/tts/abc/result")
    assert resp.status_code == 500
```

### 9.2 [test_phase3_tts_task.py](tests/langgraph_agents/test_phase3_tts_task.py) — rewrite hoàn toàn

Test hiện chỉ gọi `_persist()` helper. Phải test `synthesize_speech_async` thật.

```python
"""Tests for async TTS task (v2.4.1: BackgroundTasks pattern)."""

import json
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from langgraph_agents.services.exceptions import ServiceUnavailableError


@pytest.mark.unit
@pytest.mark.asyncio
async def test_synthesize_async_success_persists_speech_ready():
    from langgraph_agents.services.vieneu_tts import tasks

    fake_client = MagicMock()
    fake_client.synthesize = AsyncMock(return_value={"audio_url": "http://x/a.wav"})

    fake_redis = AsyncMock()
    fake_redis.setex = AsyncMock()
    fake_redis.aclose = AsyncMock()

    with patch.object(tasks, "get_vieneu_tts_client", return_value=fake_client):
        with patch.object(tasks.aioredis, "from_url", return_value=fake_redis):
            await tasks.synthesize_speech_async("hello", task_id="t1")

    fake_redis.setex.assert_awaited_once()
    args = fake_redis.setex.await_args[0]
    assert args[0] == "task_result:t1"
    assert args[1] == 3600
    payload = json.loads(args[2])
    assert payload["event"] == "speech_ready"
    assert payload["url"] == "http://x/a.wav"


@pytest.mark.unit
@pytest.mark.asyncio
async def test_synthesize_async_service_down_persists_speech_failed():
    from langgraph_agents.services.vieneu_tts import tasks

    fake_client = MagicMock()
    fake_client.synthesize = AsyncMock(
        side_effect=ServiceUnavailableError("vieneu_tts", "circuit open")
    )

    fake_redis = AsyncMock()
    fake_redis.setex = AsyncMock()
    fake_redis.aclose = AsyncMock()

    with patch.object(tasks, "get_vieneu_tts_client", return_value=fake_client):
        with patch.object(tasks.aioredis, "from_url", return_value=fake_redis):
            await tasks.synthesize_speech_async("hello", task_id="t2")

    payload = json.loads(fake_redis.setex.await_args[0][2])
    assert payload["event"] == "speech_failed"
    assert "circuit open" in payload["error"]


@pytest.mark.unit
@pytest.mark.asyncio
async def test_synthesize_async_unexpected_error_caught():
    from langgraph_agents.services.vieneu_tts import tasks

    fake_client = MagicMock()
    fake_client.synthesize = AsyncMock(side_effect=RuntimeError("boom"))

    fake_redis = AsyncMock()
    fake_redis.setex = AsyncMock()
    fake_redis.aclose = AsyncMock()

    with patch.object(tasks, "get_vieneu_tts_client", return_value=fake_client):
        with patch.object(tasks.aioredis, "from_url", return_value=fake_redis):
            await tasks.synthesize_speech_async("hello", task_id="t3")

    payload = json.loads(fake_redis.setex.await_args[0][2])
    assert payload["event"] == "speech_failed"
    assert "boom" in payload["error"]


@pytest.mark.unit
@pytest.mark.asyncio
async def test_synthesize_async_redis_down_does_not_raise():
    """Persist failure should not propagate to caller (BackgroundTasks has no retry)."""
    from langgraph_agents.services.vieneu_tts import tasks

    fake_client = MagicMock()
    fake_client.synthesize = AsyncMock(return_value={"audio_url": "http://x/a.wav"})

    fake_redis = AsyncMock()
    fake_redis.setex = AsyncMock(side_effect=ConnectionError("redis down"))
    fake_redis.aclose = AsyncMock()

    with patch.object(tasks, "get_vieneu_tts_client", return_value=fake_client):
        with patch.object(tasks.aioredis, "from_url", return_value=fake_redis):
            # Should not raise
            await tasks.synthesize_speech_async("hello", task_id="t4")
```

### Acceptance

- [ ] `test_phase3_api.py`: 6 test pass (đổi tên `test_chat_speech_mode_fires_task` → `_fires_background_task`)
- [ ] `test_phase3_tts_task.py`: 4 test pass (test task body thật, không chỉ helper)
- [ ] Tổng suite: 50+ unit pass, 0 fail

---

## 10. Task 3.5.9 — Verify

```bash
# Unit suite
pytest tests/langgraph_agents/ -m unit -p no:cacheprovider -v

# Expected: ~52 passed (38 from Phase 2.5 + 14 from Phase 3 + new)
# 0 failed, 0 skipped (sau khi env fix + ddg-search installed)

# Integration suite (requires DEEPSEEK_API_KEY + Postgres + Redis up)
pytest tests/langgraph_agents/ -m integration -p no:cacheprovider -v

# Expected: 3 passed (Phase 2.5 integration) + 1-2 passed (Phase 3 MCP retriever)

# Smoke /chat
uvicorn langgraph_agents.api.main:create_app --factory --port 8080 &
curl -X POST localhost:8080/chat -H "Content-Type: application/json" \
  -d '{"query":"Xin chào","output_mode":"text"}'
# Expected: 200 + JSON with final_answer non-empty
```

---

## 11. Task 3.5.10 — Commit

**Option A — gộp 1 commit** (Owner đề xuất ở phiên trước):

```
chore: Phase 3 finalize — TTS BackgroundTasks, simplification, test fixes

[mô tả ngắn]
```

**Option B — 2 commit tách bạch**:

```
1. feat(phase-3): MCP servers + FastAPI /chat + TTS (Celery)
   ← all original Phase 3 work

2. refactor(phase-3.5): Celery → BackgroundTasks (v2.4.1)
   ← refactor + test fix + cleanup
```

K khuyên **Option B** — diff rõ hơn, dễ revert riêng nếu cần.

---

## 12. Files Touched Summary

### Modified

| File | Change |
|------|--------|
| [requirements-langgraph.txt](requirements-langgraph.txt) | Pin versions, bỏ celery + langchain-google-genai + anthropic |
| [config/langgraph.yaml](config/langgraph.yaml) | Bỏ dead keys (manager_model, services.kimodo, celery block) |
| [services/vieneu_tts/tasks.py](agenticRAG/agentic_rag_gemini/langgraph_agents/services/vieneu_tts/tasks.py) | Rewrite: async function thay Celery task |
| [celery_app.py](agenticRAG/agentic_rag_gemini/langgraph_agents/celery_app.py) | Disabled skeleton + Phase 7 reactivation comment |
| [api/main.py](agenticRAG/agentic_rag_gemini/langgraph_agents/api/main.py) | Dùng BackgroundTasks + harden tts_result + empty fallback |
| [tests/langgraph_agents/test_phase3_api.py](tests/langgraph_agents/test_phase3_api.py) | Fix mock pattern + thêm 2 test |
| [tests/langgraph_agents/test_phase3_tts_task.py](tests/langgraph_agents/test_phase3_tts_task.py) | Rewrite hoàn toàn: test async task body |

### Optional (nice-to-have)

| File | Change |
|------|--------|
| [services/vieneu_tts/client.py](agenticRAG/agentic_rag_gemini/langgraph_agents/services/vieneu_tts/client.py) | Xóa `synthesize_sync()` (không ai gọi nữa) |

### Untouched (đã đúng v2.4.1)

- Mọi file trong `langgraph_agents/nodes/`, `mcp/`, `tools/`, `db/`, `personas/`
- `state.py` (`total_tokens` field vẫn dùng cho tracking)
- `graph.py` (`build_graph_async()` không cần thay)
- `routing.py`

---

## 13. Risks & Gotchas

| Risk | Mitigation |
|------|-----------|
| **BackgroundTasks execute SAU response trả về** | Đúng spec FastAPI. Client nhận `speech_task_id` ngay, poll `/tts/{id}/result`. UX: hiển thị "đang tổng hợp giọng nói..." |
| **VieNeu blocking event loop nếu synthesize() không thực sự async** | Verify `client.synthesize()` dùng `httpx.AsyncClient`, không phải sync wrapper |
| **TestClient `BackgroundTasks` chạy sync trong test** | Đúng — `TestClient` await background tasks trước khi return response. Test có thể check captured list ngay |
| **Redis DB migration (DB 1 → DB 0)** | Không có data cũ ở DB 1 (chỉ là broker), wipe an toàn. Nếu lo: `redis-cli -n 1 FLUSHDB` |
| **Pin versions có thể conflict với deps khác trong firstconda** | Test `pip install -r requirements-langgraph.txt` trong env clean trước khi commit |
| **Celery skeleton import có thể fail nếu `celery` package gỡ bỏ** | Đã handle: import wrapped trong try, `celery_app = None` |
| **`@app.on_event` deprecated** | Đã dùng `lifespan` context manager ở Phase 3 — OK |

---

## 14. Reporting Checkpoints

| CP | Sau task | Báo K |
|----|---------|-------|
| **CP1** | 3.5.3 (TTS async function) | Paste test output cho `test_synthesize_async_success_persists_speech_ready` |
| **CP2** | 3.5.6 (api/main.py) | `curl /chat` log với output_mode=text và =speech, paste JSON response |
| **CP3** | 3.5.8 (tests) | Full pytest output, 0 fail |
| **CP4** | 3.5.10 (commit) | `git log --oneline -5` |

---

## 15. Execution time estimate

| Task | Time |
|------|------|
| 3.5.1 Pin versions | 15m |
| 3.5.2 Cleanup config | 15m |
| 3.5.3 Refactor TTS async | 30m |
| 3.5.4 Disable celery_app | 10m |
| 3.5.5 Remove celery dep | (gộp 3.5.1) |
| 3.5.6 api/main.py BackgroundTasks | 30m |
| 3.5.7 Harden tts_result | 15m |
| 3.5.8 Fix tests | 60m |
| 3.5.9 Verify | 15m |
| 3.5.10 Commit | 10m |
| **Total** | **~3h** |

Nửa ngày là xong. Có thể chia: sáng task 3.5.1-3.5.6, chiều task 3.5.7-3.5.10.

---

## 16. After Phase 3.5 — what unblocks

| Phase 5 | Sẵn sàng vì |
|---------|------------|
| SSE streaming | `/chat` endpoint sạch, chỉ cần đổi response từ JSON → SSE event stream qua `astream_events()` |
| Session reopen API | `conversations` table có sẵn, chỉ thêm 2 endpoint `GET /sessions` + `POST /sessions/{id}/resume` |
| Auto-write conversations sau graph | Pattern BackgroundTasks đã quen, dùng `background_tasks.add_task(_write_session_async, ...)` |
| Frontend rework | API contract khóa, không thay đổi nữa |

| Phase 6 | Sẵn sàng vì |
|---------|------------|
| Token tracking | `total_tokens` field active, chỉ cần thêm structured logger |
| Eval framework | Test infra đã có, thêm `tests/eval/golden_set.py` + `pytest tests/eval` |
| Output validators | Insert giữa synthesizer ↔ grader trong graph |
| Runbook | Document failure modes đã encounter |

---

**N**: bắt đầu từ Task 3.5.1. Commit theo Option B (2 commit). Báo K sau mỗi CP.
