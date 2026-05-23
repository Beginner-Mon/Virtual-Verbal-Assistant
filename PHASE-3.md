# Phase 3 — Dispatch + Celery Tasks + Service Clients + Approval Gate

**Architect**: K | **Developer**: N | **Date**: 2026-05-21
**Branch**: `feature/langgraph-rewrite` (continue from Phase 2)
**Estimated time**: ~6-8h

---

## Overview

Implement the async pipeline: Dispatch node fires background tasks, Kimodo/VieNeu-TTS service clients communicate via REST, Celery tasks manage rendering/synthesis lifecycle, approval gate controls motion rendering. This phase connects the LangGraph graph to the outside world.

**Key reality**: Kimodo and VieNeu-TTS services are NOT yet running. All code must handle "service unavailable" gracefully from day 1. Circuit breakers prevent cascade failures.

**Critical rule**: Do NOT modify existing `celery_app.py` or `tasks/`. Create a new Celery configuration under `langgraph_agents/`. Both coexist.

---

## New Files to Create

```
langgraph_agents/
  celery_app.py                    # Celery instance for langgraph tasks
  services/
    kimodo/
      client.py                    # Async httpx client for Kimodo REST API
      tasks.py                     # Celery task: generate_motion
    vieneu_tts/
      client.py                    # Async httpx client for VieNeu-TTS REST API
      tasks.py                     # Celery task: synthesize_speech
  streaming/
    approval.py                    # FastAPI endpoint: POST /render/approve/{request_id}

config/langgraph.yaml              # Update: add services section

tests/langgraph_agents/
  test_phase3_dispatch.py          # Tests for dispatch + approval + graceful degradation
```

## Files to Modify

```
langgraph_agents/nodes/dispatch.py  # Stub → real implementation
config/langgraph.yaml               # Add services + celery section
```

---

## Task 1: Update `config/langgraph.yaml` — add services + celery sections

Append to the existing config:

```yaml
  services:
    kimodo:
      url: "http://localhost:5001"
      endpoint: "/generate"
      timeout: 30                    # seconds — motion render can be slow
      circuit_breaker:
        failure_threshold: 3
        cool_down_seconds: 60
    vieneu_tts:
      url: "http://localhost:5000"
      endpoint: "/synthesize"
      timeout: 15
      circuit_breaker:
        failure_threshold: 3
        cool_down_seconds: 60

  celery:
    broker_url: "redis://localhost:6379/1"   # /1 to separate from existing /0
    result_backend: "redis://localhost:6379/1"
    task_time_limit: 300             # 5 min hard kill
    task_soft_time_limit: 240        # 4 min soft warn
```

---

## Task 2: Create `langgraph_agents/celery_app.py` — new Celery instance

```python
"""Celery app for LangGraph async tasks (Kimodo + VieNeu-TTS).

Separate from the existing celery_app.py to avoid modifying legacy code.
Uses Redis DB 1 (/1) to isolate from existing Celery broker on /0.

Worker command:
  celery -A langgraph_agents.celery_app worker -l info -Q langgraph
"""
```

**Requirements**:
- Import Celery conditionally (same pattern as existing `celery_app.py`)
- Load config from `config/langgraph.yaml` → `langgraph.celery` section
- `include` list: `["langgraph_agents.services.kimodo.tasks", "langgraph_agents.services.vieneu_tts.tasks"]`
- Default queue: `langgraph` (isolate from existing Celery queues)
- `task_serializer="json"`, `accept_content=["json"]`, `result_serializer="json"`
- `task_track_started=True` (for progress monitoring)
- `task_acks_late=True` (prevent task loss on worker crash)
- If Celery not installed: set `celery_app = None`, log warning

---

## Task 3: Create `langgraph_agents/services/kimodo/client.py` — Kimodo REST client

### API Contract (expected from Kimodo service)

```
POST http://localhost:5001/generate
Content-Type: application/json

{
  "prompt": "raise right arm 90 degrees overhead",
  "constraints": [
    {"joint": "right_shoulder", "angle": 90, "axis": "flexion"}
  ],
  "duration_seconds": 3.0
}

→ 200 OK
{
  "status": "completed",
  "motion_file_url": "/output/motion_abc123.mp4",
  "duration_seconds": 3.0,
  "frames": 90,
  "fps": 30
}
```

### Implementation

```python
"""Async HTTP client for Kimodo motion generation service."""

import httpx
from core.circuit_breaker import CircuitBreaker
```

**Class**: `KimodoClient`
- `__init__(self, base_url, endpoint, timeout, circuit_breaker_cfg)`: store params, create `CircuitBreaker` instance
- `async def generate(self, prompt, constraints, duration_seconds) -> dict`: 
  1. Check circuit breaker `.allow()` → if open, raise `ServiceUnavailableError`
  2. `async with httpx.AsyncClient(timeout=self.timeout) as client:`
  3. POST to `{base_url}{endpoint}` with JSON body
  4. On success: `circuit_breaker.record_success()`, return response dict
  5. On failure (timeout, connection error, HTTP error): `circuit_breaker.record_failure()`, raise `ServiceUnavailableError`
- `def is_healthy(self) -> bool`: return `self._circuit_breaker.state != "open"`

**Custom exception**: `ServiceUnavailableError(Exception)` — define in `langgraph_agents/services/exceptions.py` (shared by both clients).

**Module-level singleton** (lazy):

```python
_client = None

def get_kimodo_client() -> KimodoClient:
    global _client
    if _client is None:
        cfg = _load_kimodo_config()
        _client = KimodoClient(
            base_url=cfg.get("url", "http://localhost:5001"),
            endpoint=cfg.get("endpoint", "/generate"),
            timeout=cfg.get("timeout", 30),
            circuit_breaker_cfg=cfg.get("circuit_breaker", {}),
        )
    return _client
```

---

## Task 4: Create `langgraph_agents/services/vieneu_tts/client.py` — VieNeu-TTS REST client

### API Contract (expected from VieNeu-TTS service)

```
POST http://localhost:5000/synthesize
Content-Type: application/json

{
  "text": "Bài tập đầu tiên là Cat-Cow stretch..."
}

→ 200 OK
{
  "status": "completed",
  "audio_url": "/output/speech_abc123.wav",
  "duration_seconds": 4.2
}
```

### Implementation

Same pattern as KimodoClient:

```python
"""Async HTTP client for VieNeu-TTS speech synthesis service."""
```

**Class**: `VieNeuTTSClient`
- `__init__(self, base_url, endpoint, timeout, circuit_breaker_cfg)`
- `async def synthesize(self, text: str) -> dict`: POST → response with audio_url
- `def is_healthy(self) -> bool`

**Module-level singleton**: `get_vieneu_tts_client()`

---

## Task 5: Create `langgraph_agents/services/exceptions.py`

```python
class ServiceUnavailableError(Exception):
    """Raised when a downstream service (Kimodo/VieNeu-TTS) is unreachable or circuit breaker is open."""
    def __init__(self, service_name: str, reason: str = ""):
        self.service_name = service_name
        self.reason = reason
        super().__init__(f"{service_name} unavailable: {reason}")
```

---

## Task 6: Create `langgraph_agents/services/kimodo/tasks.py` — Celery motion task

```python
"""Celery task for Kimodo motion generation."""
```

### Task: `generate_motion`

```python
@celery_app.task(
    name="langgraph.generate_motion",
    bind=True,
    acks_late=True,
    max_retries=1,
    default_retry_delay=5,
)
def generate_motion(self, prompt, constraints, duration_seconds, request_id, session_id):
```

**Steps**:
1. `self.update_state(state="STARTED", meta={"stage": "kimodo_call"})`
2. Call `KimodoClient.generate()` (sync wrapper — Celery tasks are sync)
   - Since `KimodoClient.generate()` is async, use `asyncio.run()` or create a sync `.generate_sync()` method
   - **K's preference**: add a sync `generate_sync()` to KimodoClient that wraps the httpx call synchronously. Celery tasks are sync by design. Don't mix asyncio into Celery.
3. On success:
   - Write result to Redis: `task_result:{self.request.id}` with TTL 3600
   - Publish to Redis pub/sub: `task_events:{session_id}` with `{"event": "motion_ready", "task_id": "...", "url": "..."}`
   - Return result dict
4. On `ServiceUnavailableError`:
   - Do NOT retry (service is genuinely down, not transient)
   - Write failure to Redis: `task_result:{self.request.id}` with error
   - Publish `{"event": "motion_failed", "error": "..."}` to session channel
   - Raise so Celery marks task as FAILURE
5. On other exceptions:
   - Retry once (transient network issue)
   - If retry exhausted: same as ServiceUnavailableError path

### Redis result persistence

Use `redis.Redis` (sync, not async — this runs in Celery worker):

```python
import json
import redis

r = redis.Redis.from_url(REDIS_URL)
r.setex(
    f"task_result:{self.request.id}",
    3600,  # 1h TTL
    json.dumps({"event": "motion_ready", "url": result["motion_file_url"]}),
)
r.publish(
    f"task_events:{session_id}",
    json.dumps({"event": "motion_ready", "task_id": self.request.id, "url": result["motion_file_url"]}),
)
```

---

## Task 7: Create `langgraph_agents/services/vieneu_tts/tasks.py` — Celery speech task

Same pattern as kimodo/tasks.py:

### Task: `synthesize_speech`

```python
@celery_app.task(
    name="langgraph.synthesize_speech",
    bind=True,
    acks_late=True,
    max_retries=1,
    default_retry_delay=3,
)
def synthesize_speech(self, text, request_id, session_id):
```

**Steps**: Same as motion task but:
- Calls `VieNeuTTSClient.synthesize_sync(text)`
- Publishes `speech_ready` event (not `motion_ready`)
- Redis key: `task_result:{self.request.id}`

---

## Task 8: Implement `langgraph_agents/nodes/dispatch.py` — real dispatch

### Architecture

```
dispatch_node(state) →
  1. Read intent, output_mode, final_answer, request_id, session_id
  2. Speech decision:
     if output_mode in ("speech", "both") AND final_answer not empty:
       → try fire Celery synthesize_speech.delay(text, request_id, session_id)
       → set speech_task_id = task.id
       → on failure: RECOVERABLE error, speech_task_id = None
  3. Motion decision:
     if intent in ("exercise_recommendation", "visualize_motion"):
       → check worker health (Celery inspect ping, or circuit breaker state)
       → if healthy:
           store motion_pending:{request_id} in Redis with payload
           set motion_pending = True, motion_payload = {...}
       → if unhealthy:
           RECOVERABLE error: "Hệ thống hình ảnh 3D đang bảo trì"
           set motion_pending = False
  4. Return updated state fields
```

### Worker health check

Don't use Celery inspector (requires round-trip to worker, slow). Instead:
- Check circuit breaker state: `get_kimodo_client().is_healthy()`
- If circuit breaker is open → worker considered down → skip motion

For initial implementation (services don't exist yet): check if Celery app is available, and if Redis is reachable. Both being up = "worker healthy enough to queue tasks".

```python
def _celery_available() -> bool:
    from langgraph_agents.celery_app import celery_app
    return celery_app is not None
```

### Motion pending storage

Store in Redis with 30 minute TTL:

```python
import json
import redis.asyncio as aioredis

r = aioredis.from_url(REDIS_URL)
await r.setex(
    f"motion_pending:{request_id}",
    1800,  # 30 min TTL
    json.dumps({
        "prompt": _build_motion_prompt(state),
        "constraints": _extract_constraints(state),
        "session_id": session_id,
        "request_id": request_id,
    }),
)
```

### Motion prompt building

Build from reasoning_output + expanded_query:
```python
def _build_motion_prompt(state: dict) -> str:
    return state.get("expanded_query") or state.get("query", "")

def _extract_constraints(state: dict) -> list:
    # Phase 3: empty constraints. Phase 6: parse from reasoning_output.
    return []
```

### Return dict

```python
return {
    "speech_task_id": speech_task_id,      # str or None
    "motion_task_id": None,                 # always None here — set by approval gate
    "motion_pending": motion_pending,       # bool
    "motion_payload": motion_payload,       # dict or None
}
```

Add `"errors": [...]` if any RECOVERABLE errors.

### Graceful degradation rules

| Condition | Behavior |
|-----------|----------|
| Celery not installed | Skip all tasks, RECOVERABLE warning |
| Redis unreachable | Skip motion_pending storage, RECOVERABLE warning |
| Kimodo circuit breaker open | Skip motion, RECOVERABLE: "3D đang bảo trì" |
| VieNeu-TTS circuit breaker open | Skip speech, RECOVERABLE warning |
| Task enqueue fails | Log error, return None task_id, RECOVERABLE |

---

## Task 9: Create `langgraph_agents/streaming/approval.py` — approval gate endpoint

**This is a standalone FastAPI router, NOT part of the LangGraph graph.** It will be mounted by the FastAPI app in Phase 5.

```python
"""Approval gate for motion rendering.

Mounted at: POST /render/approve/{request_id}
Called by: frontend "Mô phỏng 3D" button click
Flow: read motion_pending from Redis → fire Celery task → delete pending key
"""

from fastapi import APIRouter, HTTPException

router = APIRouter()

@router.post("/render/approve/{request_id}")
async def approve_motion_render(request_id: str):
```

**Steps**:
1. Read `motion_pending:{request_id}` from Redis
2. If not found → 404: "No pending motion for this request"
3. Parse JSON payload
4. Fire Celery: `generate_motion.delay(prompt, constraints, duration, request_id, session_id)`
5. Delete Redis key `motion_pending:{request_id}`
6. Store `task_id` in Redis: `motion_task:{request_id}` (for SSE to track)
7. Also add to `pending_tasks:{session_id}` set (for reconnect recovery)
8. Return 202 Accepted: `{"task_id": "...", "status": "queued"}`

**Error handling**:
- Celery not available → 503: "Task queue unavailable"
- Redis error → 500
- Payload parse error → 400

---

## Task 10: Write tests — `tests/langgraph_agents/test_phase3_dispatch.py`

### Unit tests (no services needed)

1. **`test_dispatch_no_celery`** — dispatch node with Celery unavailable → all task_ids None, no crash, RECOVERABLE warning
2. **`test_dispatch_conversation_no_tasks`** — conversation intent → no speech/motion tasks fired (output_mode="text")
3. **`test_dispatch_text_mode_no_speech`** — output_mode="text" → speech_task_id = None
4. **`test_dispatch_motion_pending_exercise`** — exercise_recommendation intent → motion_pending=True (mock Redis)
5. **`test_dispatch_motion_pending_visualize`** — visualize_motion intent → motion_pending=True (mock Redis)
6. **`test_dispatch_knowledge_no_motion`** — knowledge_query intent → motion_pending=False
7. **`test_motion_prompt_builder`** — `_build_motion_prompt` returns expanded_query or query
8. **`test_circuit_breaker_integration`** — circuit breaker opens after N failures, blocks subsequent calls

### Service client tests (unit, mock httpx)

9. **`test_kimodo_client_success`** — mock httpx.post returns 200 → result dict
10. **`test_kimodo_client_timeout`** — mock httpx.post raises TimeoutError → ServiceUnavailableError + circuit breaker records failure
11. **`test_vieneu_client_success`** — mock httpx.post returns 200 → result dict
12. **`test_vieneu_client_circuit_open`** — after 3 failures, circuit breaker blocks → ServiceUnavailableError without making HTTP call

### Approval gate tests (unit, mock Redis)

13. **`test_approval_success`** — mock Redis has pending payload → fires task → 202
14. **`test_approval_not_found`** — mock Redis returns None → 404
15. **`test_approval_no_celery`** — Celery unavailable → 503

### Full graph tests

16. **`test_full_graph_dispatch_graceful`** — full graph end-to-end with services down → final_answer still returned, errors contain RECOVERABLE warnings

### Mocking strategy

For Redis in dispatch tests: use `unittest.mock.patch` or `pytest-mock` to mock `redis.asyncio.from_url`.

For httpx in client tests: use `unittest.mock.patch("httpx.AsyncClient.post")` or `respx` library.

For Celery in dispatch tests: mock `generate_motion.delay()` and `synthesize_speech.delay()`.

---

## Acceptance Criteria

1. `dispatch_node` fires speech task when `output_mode` is speech/both
2. `dispatch_node` stores `motion_pending:{request_id}` in Redis for exercise/motion intents
3. `dispatch_node` does NOT fire motion task directly (approval gate does that)
4. `dispatch_node` degrades gracefully when Celery/Redis/services are down
5. `KimodoClient` uses circuit breaker — 3 failures → open → block calls for 60s
6. `VieNeuTTSClient` uses same circuit breaker pattern
7. Celery tasks write `task_result:{task_id}` to Redis + publish to pub/sub
8. `POST /render/approve/{request_id}` reads pending payload → fires Celery task → 202
9. Approval returns 404 when no pending motion exists
10. All unit tests pass without running services: `pytest tests/langgraph_agents/test_phase3_dispatch.py -m unit`
11. Full graph still returns final_answer even with all services down

---

## Execution Order

| Step | Task | Est. |
|------|------|------|
| 1 | Task 1: Update config/langgraph.yaml | 10m |
| 2 | Task 5: Create exceptions.py | 5m |
| 3 | Task 2: Create langgraph_agents/celery_app.py | 30m |
| 4 | Task 3: Create kimodo/client.py | 45m |
| 5 | Task 4: Create vieneu_tts/client.py | 30m (similar pattern) |
| 6 | Task 6: Create kimodo/tasks.py | 45m |
| 7 | Task 7: Create vieneu_tts/tasks.py | 30m (similar pattern) |
| 8 | Task 8: Implement dispatch.py | 60m |
| 9 | Task 9: Create approval.py | 30m |
| 10 | Task 10: Write tests | 60m |
| 11 | Run tests + fix issues | 30m |

---

## What is NOT in Phase 3

- **Actual Kimodo/VieNeu-TTS service installation**: Services don't exist yet. We build the client wrappers against expected API contracts.
- **SSE streaming**: Phase 5. The pub/sub publishing in tasks is the producer side; the SSE consumer is Phase 5.
- **Frontend approval button**: Phase 5. The `POST /render/approve` endpoint is ready but no UI yet.
- **FastAPI app mounting**: Phase 5. The `approval.py` router exists but isn't mounted in any app yet.
- **S3 upload**: Production deployment concern. Local file storage for now.
- **Constraint extraction from reasoning_output**: Phase 6 (hardening). For now, constraints are empty `[]`.
- **Docker Compose for Kimodo/VieNeu-TTS**: Separate setup, not in scope.

---

## Architecture Notes for N

### Why separate Celery app?

Existing `celery_app.py` uses Redis DB 0 (`redis://localhost:6379/0`) and includes `tasks.motion_tasks` (DART-specific). Creating `langgraph_agents/celery_app.py` with Redis DB 1 (`/1`) isolates the queues completely. Both workers can run simultaneously without interference.

### Why sync clients in Celery tasks?

Celery tasks run in a sync worker process. Mixing `asyncio.run()` inside a Celery task is fragile (event loop conflicts). Use synchronous `httpx.Client` (not `AsyncClient`) inside task code. The async `httpx.AsyncClient` in `client.py` is for when dispatch_node needs to check health or future direct calls.

**Solution**: Each client should have both:
- `async def generate(...)` — for use in async LangGraph nodes (future)
- `def generate_sync(...)` — for use in Celery tasks

### Circuit breaker is per-process

The existing `CircuitBreaker` from `core/circuit_breaker.py` is in-memory, per-process. This means:
- The Celery worker has its own breaker state
- The FastAPI process (dispatch node) has its own breaker state
- They don't share state

This is acceptable for Phase 3. If cross-process breaker state is needed later, use Redis-backed breaker (Phase 6 hardening).

### Dispatch node runs in async context

The dispatch node is called by LangGraph which is async. Use `redis.asyncio` for Redis operations inside dispatch. But Celery task enqueueing (`.delay()`) is sync — wrap in `asyncio.to_thread()` or call directly (it's non-blocking, just puts a message on Redis).
