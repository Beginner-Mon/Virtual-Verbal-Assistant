# Phase 5 — SSE Streaming + Session Reopen + Frontend Refactor

**Architect**: K | **Developer**: N | **Date**: 2026-05-24
**Branch**: `feature/langgraph-rewrite` (continue from Phase 3.5 commit `858e8a8`)
**Estimated time**: ~8h
**Reference**: [../plans/v2.4-plan.md](../plans/v2.4-plan.md) §13, §14, §17 (Phase 5)

---

## 0. Why this phase exists

Phase 3.5 đóng sạch backend graph + REST `/chat` JSON. Phase 5 nâng UX:

1. **Token streaming** — user nhìn AI "đang gõ" thay vì chờ 5-15s im lặng (DeepSeek pipeline 4 LLM call sequence)
2. **Stage events** — báo cho user "đang suy luận / đang tìm tài liệu / đang sinh câu trả lời" để biết hệ chưa treo
3. **Session reopen** — UI có sidebar history giống Claude, click vào tiếp tục session cũ
4. **Empty fallback** — backend đã có, verify

Decisions chốt từ Owner (24/05):
- Token-by-token + stage events (full UX)
- POST /chat **trả SSE only** — không có endpoint JSON
- Refactor `ECA_UI/api.js` + thêm history sidebar — KHÔNG rebuild React
- Auto-write session **eager** (trước khi SSE `done`) — đảm bảo reopen có data ngay

---

## 1. Order of execution

```
5.1  Spike: verify astream_events emit token cho conversation node
5.2  Backend helper: SSE EventSourceResponse + event encoder
5.3  Backend: rewrite POST /chat → SSE response
5.4  Backend: TTS speech_ready event pump (Redis poll trong SSE)
5.5  Backend: session list/resume endpoints
5.6  Backend: eager session write helper
5.7  Tests: backend SSE (TestClient + parse stream)
5.8  Frontend: refactor api.js → EventSource client
5.9  Frontend: history sidebar component
5.10 Frontend: streaming text display + stage indicator
5.11 Manual browser smoke + UX polish
5.12 Commit
```

Mỗi task commit riêng nếu N muốn diff sạch.

---

## 2. Task 5.1 — Spike: verify astream_events token streaming

**Effort**: 30m
**Goal**: Trước khi build full SSE infra, confirm DeepSeek + langchain-openai emit `on_chat_model_stream` events qua `graph.astream_events()`. Nếu không emit → cần đổi conversation node sang `llm.astream(...)` thủ công.

### Script kiểm tra

Tạo `scripts/spike_astream.py`:

```python
"""Spike — verify token streaming through astream_events."""

import asyncio
import os
from dotenv import load_dotenv
load_dotenv("agenticRAG/agentic_rag_gemini/.env")

from langgraph_agents.graph import build_graph_async


async def main():
    graph = await build_graph_async()
    state = {"messages": [], "errors": [], "retry_count": 0, "total_tokens": 0}
    config = {"configurable": {
        "user_id": "spike", "session_id": "spike-001",
        "query": "Xin chao",  # conversation intent → ngắn, nhanh
        "persona_id": "eca_default", "output_mode": "text",
        "request_id": "spike", "token_limit": None,
    }}

    seen_events = {}
    async for event in graph.astream_events(state, config=config, version="v2"):
        ev_type = event["event"]
        seen_events[ev_type] = seen_events.get(ev_type, 0) + 1
        if ev_type == "on_chat_model_stream":
            chunk = event["data"]["chunk"]
            print(f"TOKEN: {chunk.content!r}")

    print("\nEvent counts:")
    for k, v in sorted(seen_events.items()):
        print(f"  {k}: {v}")


if __name__ == "__main__":
    asyncio.run(main())
```

### Acceptance

- [ ] Chạy script không crash
- [ ] `on_chat_model_stream` event xuất hiện ≥ 5 lần (token-by-token cho conversation node)
- [ ] Event counts có `on_chain_start`/`on_chain_end` cho mỗi node (planner, memory, conversation, ...)

### Nếu spike fail

DeepSeek **không stream** qua langchain-openai → fallback:
- Đổi `conversation_node` từ `llm.ainvoke()` → `llm.astream()` + tự yield chunks
- Phức tạp hơn: cần thay đổi node signature trả async generator hoặc dùng `langgraph.types.Send`
- Backup plan: skip token streaming, chỉ stage events. UX kém hơn nhưng vẫn có "đang gõ" feedback

K assumption: DeepSeek API hỗ trợ streaming (OpenAI-compatible). 90% pass spike.

---

## 3. Task 5.2 — SSE helper module

**Effort**: 30m
**File mới**: `langgraph_agents/api/sse.py`

### Code

```python
"""SSE event encoding + EventSourceResponse helper.

Wraps sse-starlette for consistent event format across endpoints.
Frontend EventSource parses `event:` + `data:` fields automatically.
"""

import json
from typing import AsyncIterator, Any

from sse_starlette.sse import EventSourceResponse


def encode_event(event_type: str, data: Any) -> dict:
    """Build a dict that sse-starlette serializes to:

        event: {event_type}
        data: {json.dumps(data)}

    """
    return {
        "event": event_type,
        "data": json.dumps(data, ensure_ascii=False),
    }


async def stream_response(generator: AsyncIterator[dict]) -> EventSourceResponse:
    """Wrap an async generator of encoded events into a streaming response."""
    return EventSourceResponse(generator)
```

### Acceptance

- [ ] `encode_event("token", {"content": "Xin"})` returns dict với keys `event`, `data`
- [ ] `data` field là JSON string, không phải dict

---

## 4. Task 5.3 — Rewrite POST /chat to SSE

**Effort**: 90m
**File**: [api/main.py](agenticRAG/agentic_rag_gemini/langgraph_agents/api/main.py)

### Event flow design

```
POST /chat → SSE stream:

  event: stage         data: {"node": "memory", "status": "started"}
  event: stage         data: {"node": "memory", "status": "complete"}
  event: stage         data: {"node": "planner", "status": "started"}
  event: stage         data: {"node": "planner", "status": "complete", "intent": "exercise_recommendation"}
  event: stage         data: {"node": "retriever_agent", "status": "started"}
  event: tool_calling  data: {"tool": "pgvector_search"}
  event: tool_complete data: {"tool": "pgvector_search", "result_count": 5}
  event: stage         data: {"node": "synthesizer", "status": "started"}
  event: stage         data: {"node": "synthesizer", "status": "complete"}
  event: stage         data: {"node": "grader", "status": "complete", "result": "pass"}
  event: stage         data: {"node": "conversation", "status": "started"}
  event: token         data: {"content": "Bài "}
  event: token         data: {"content": "tập "}
  event: token         data: {"content": "đầu "}
  ...
  event: stage         data: {"node": "conversation", "status": "complete"}
  event: session_persisted  data: {"session_id": "..."}
  event: speech_pending     data: {"task_id": "..."}        ← optional (output_mode=speech/both)
  event: speech_ready       data: {"task_id": "...", "url": "..."}  ← optional, từ poll Redis
  event: done               data: {"request_id": "...", "total_tokens": 1234, "intent": "..."}
```

### Rewrite

```python
import uuid
import asyncio
from langgraph_agents.api.sse import encode_event
from sse_starlette.sse import EventSourceResponse


# Map LangGraph astream_events event types → our stage event
_NODE_EVENT_MAP = {
    "on_chain_start": "started",
    "on_chain_end": "complete",
}


@application.post("/chat")
async def chat(req: ChatRequest, background_tasks: BackgroundTasks):
    request_id = str(uuid.uuid4())
    state = {"messages": [], "errors": [], "retry_count": 0, "total_tokens": 0}
    config = {"configurable": {
        "user_id": req.user_id,
        "session_id": req.session_id,
        "query": req.query,
        "persona_id": req.persona_id,
        "output_mode": req.output_mode,
        "request_id": request_id,
        "token_limit": req.token_limit,
    }}

    async def event_generator():
        # Track which node is currently streaming tokens (only conversation)
        in_conversation_stream = False
        final_state = {}

        async for event in _graph.astream_events(state, config=config, version="v2"):
            ev_type = event["event"]
            name = event.get("name", "")
            meta = event.get("metadata", {})

            # 1. Node start/end → stage event (only for our nodes, not LangChain internals)
            if ev_type == "on_chain_start" and name in {
                "memory", "planner", "retriever_agent", "synthesizer",
                "grader", "conversation", "error_handler",
            }:
                yield encode_event("stage", {"node": name, "status": "started"})

            elif ev_type == "on_chain_end" and name in {
                "memory", "planner", "retriever_agent", "synthesizer",
                "grader", "conversation", "error_handler",
            }:
                output = event["data"].get("output", {})
                extra = {}
                if name == "planner" and isinstance(output, dict):
                    extra["intent"] = output.get("intent")
                    extra["needs_clarification"] = output.get("needs_clarification", False)
                if name == "grader" and isinstance(output, dict):
                    extra["result"] = output.get("grader_result")
                yield encode_event("stage", {"node": name, "status": "complete", **extra})

            # 2. Tool call → tool_calling event
            elif ev_type == "on_tool_start":
                yield encode_event("tool_calling", {"tool": name})

            elif ev_type == "on_tool_end":
                output = event["data"].get("output")
                count = len(output) if isinstance(output, list) else 1
                yield encode_event("tool_complete", {"tool": name, "result_count": count})

            # 3. Token stream — only from conversation node (others may also emit but we filter)
            elif ev_type == "on_chat_model_stream":
                # Check if parent run is conversation node
                tags = meta.get("tags") or []
                parent_ids = meta.get("langgraph_node", "")
                if parent_ids == "conversation":
                    chunk = event["data"]["chunk"]
                    content = chunk.content if hasattr(chunk, "content") else str(chunk)
                    if content:
                        yield encode_event("token", {"content": content})

        # 4. After graph completes, get final state from last event
        # (astream_events does not return final state directly — need to extract)
        # Alternative: call ainvoke separately? Wastes 1 run. Instead: parse last on_chain_end
        # of the root graph run. For simplicity, call get_state on checkpoint (if checkpointing on).
        # Phase 5 MVP: re-invoke ainvoke quickly OR track last conversation end.
        # K decision: track conversation output manually from last on_chain_end above.

        # For now use ainvoke to get final state (wasteful — Phase 6 optimize)
        # final_state = await _graph.ainvoke(state, config=config)
        # ↑ DON'T — this double-runs the graph. Instead persist intermediate state above.

        # Simpler MVP: capture state from last `on_chain_end name='LangGraph'` event
        # (LangGraph emits its own final chain_end event with full output)

        # → restructure: collect state inside the loop above (omitted for brevity in this sketch).

        # 5. Eager session write
        if final_state.get("final_answer"):
            await _write_session_async(
                user_id=req.user_id,
                session_id=req.session_id,
                user_query=req.query,
                assistant_answer=final_state["final_answer"],
                intent=final_state.get("intent", ""),
                tokens=final_state.get("total_tokens", 0),
            )
            yield encode_event("session_persisted", {"session_id": req.session_id})

        # 6. Fire TTS BackgroundTask if needed
        speech_task_id = None
        if req.output_mode in ("speech", "both") and final_state.get("final_answer"):
            speech_task_id = str(uuid.uuid4())
            persona = get_persona(req.persona_id)
            voice_path = persona.get("voice_identity", {}).get("voice_path")
            background_tasks.add_task(
                synthesize_speech_async,
                text=final_state["final_answer"],
                task_id=speech_task_id,
                voice_path=voice_path,
            )
            yield encode_event("speech_pending", {"task_id": speech_task_id})

            # Poll Redis for speech_ready (Phase 5 MVP — Phase 7 switch to pub/sub)
            async for sse_event in _poll_speech_result(speech_task_id, timeout=15):
                yield sse_event

        # 7. Done
        yield encode_event("done", {
            "request_id": request_id,
            "total_tokens": final_state.get("total_tokens", 0),
            "intent": final_state.get("intent", ""),
            "speech_task_id": speech_task_id,
        })

    return EventSourceResponse(event_generator())
```

### Complications + plan của K

Vấn đề (a): **lấy final_state từ astream_events** — events stream tách rời, không có "return value" như ainvoke. 2 approaches:

| Option | Trade-off |
|--------|-----------|
| **A**. Track state trong loop bằng cách capture `on_chain_end name="LangGraph"` event | Single graph run, không waste. Code phức tạp 1 chút |
| **B**. Sau khi stream xong, gọi `_graph.aget_state(config)` từ checkpointer | Cần enable checkpointer (PostgresSaver). Phase 7 mới setup |
| **C**. Run `astream_events` + `ainvoke` song song qua `asyncio.gather` | Waste 2x cost, KHÔNG nên |

K chọn **A** — extract state từ root `on_chain_end` event. Code chi tiết trong Task 5.3 thực tế N implement.

Vấn đề (b): **token event lọc đúng conversation node** — `on_chat_model_stream` cũng phát ra từ planner / synthesizer. Cần filter bằng `metadata.langgraph_node == "conversation"`. Verify ở spike (Task 5.1).

Vấn đề (c): **DeepSeek planner (json_mode) có thể không stream** — OK, không ảnh hưởng. Planner không emit `on_chat_model_stream` thì chỉ thấy `started/complete` stage events. Đúng UX mong muốn.

### Acceptance

- [ ] `curl -N -X POST localhost:8080/chat -H "Content-Type: application/json" -d '{"query":"Xin chao","output_mode":"text"}'` trả stream với event: stage + event: token + event: done
- [ ] Token events chỉ xuất hiện trong giai đoạn conversation node
- [ ] Exercise query: thấy `tool_calling` event cho `pgvector_search`
- [ ] Session persisted event xuất hiện trước done

---

## 5. Task 5.4 — Speech result polling

**Effort**: 30m
**File mới**: helper trong [api/main.py](agenticRAG/agentic_rag_gemini/langgraph_agents/api/main.py)

### Code

```python
async def _poll_speech_result(task_id: str, timeout: float = 15.0):
    """Poll Redis task_result:{task_id} every 250ms up to timeout.

    Yields:
        - encode_event("speech_ready", {...}) when payload event="speech_ready"
        - encode_event("speech_failed", {...}) when payload event="speech_failed"
        - Nothing if timeout (caller responsibility to handle)
    """
    deadline = asyncio.get_event_loop().time() + timeout
    key = f"task_result:{task_id}"

    while asyncio.get_event_loop().time() < deadline:
        raw = _get_redis().get(key)
        if raw is not None and isinstance(raw, (bytes, str)):
            try:
                payload = json.loads(raw)
                event_name = payload.get("event", "speech_failed")
                yield encode_event(event_name, payload)
                return
            except json.JSONDecodeError:
                pass
        await asyncio.sleep(0.25)

    # Timeout — emit a failed event so frontend knows to stop waiting
    yield encode_event("speech_failed", {
        "task_id": task_id,
        "error": f"TTS task timeout after {timeout}s",
    })
```

### Acceptance

- [ ] TTS task hoàn thành trong 5s → `_poll_speech_result` yield `speech_ready` event
- [ ] TTS task fail → yield `speech_failed`
- [ ] TTS task hang > 15s → yield `speech_failed` với reason timeout

---

## 6. Task 5.5 — Session list + resume endpoints

**Effort**: 45m
**File**: [api/main.py](agenticRAG/agentic_rag_gemini/langgraph_agents/api/main.py) + [api/schemas.py](agenticRAG/agentic_rag_gemini/langgraph_agents/api/schemas.py)

### Schemas

```python
# schemas.py

class SessionListItem(BaseModel):
    session_id: str
    created_at: str
    updated_at: str
    first_user_message_preview: str
    message_count: int

class SessionListResponse(BaseModel):
    sessions: list[SessionListItem]
    total: int

class SessionResumeResponse(BaseModel):
    session_id: str
    messages: list[dict]            # full conversation history
    stm_populated: bool
    last_updated: str
```

### Endpoints

```python
from langgraph_agents.db.session_store import (
    list_user_sessions, load_session_messages, populate_stm_from_messages,
)


@application.get("/sessions", response_model=SessionListResponse)
async def list_sessions(user_id: str, limit: int = 50):
    """List user's sessions sorted by updated_at DESC."""
    rows = await list_user_sessions(user_id=user_id, limit=limit)
    return SessionListResponse(
        sessions=[SessionListItem(**r) for r in rows],
        total=len(rows),
    )


@application.post("/sessions/{session_id}/resume", response_model=SessionResumeResponse)
async def resume_session(session_id: str, user_id: str):
    """Load session messages from PostgreSQL, populate Redis STM with last 3 Q&A."""
    row = await load_session_messages(user_id=user_id, session_id=session_id)
    if not row:
        raise HTTPException(404, "Session not found")

    messages = row["messages"] or []
    await populate_stm_from_messages(session_id, messages)

    return SessionResumeResponse(
        session_id=session_id,
        messages=messages,
        stm_populated=True,
        last_updated=row["updated_at"].isoformat(),
    )
```

### Helper functions

**File**: [db/session_store.py](agenticRAG/agentic_rag_gemini/langgraph_agents/db/session_store.py) (already exists, extend)

```python
import json
from typing import Optional
import redis.asyncio as aioredis

from langgraph_agents.shared import get_pg_client


_REDIS_URL = "redis://localhost:6379/0"
_STM_MAX = 3


async def list_user_sessions(user_id: str, limit: int = 50) -> list[dict]:
    pg = get_pg_client()
    rows = await pg.fetch(
        """
        SELECT session_id::text AS session_id,
               created_at, updated_at,
               jsonb_array_length(COALESCE(messages, '[]'::jsonb)) AS message_count,
               -- First user message preview (first 80 chars)
               COALESCE(
                 SUBSTRING(
                   (SELECT m->>'content' FROM jsonb_array_elements(messages) AS m
                    WHERE m->>'role' = 'user' LIMIT 1),
                   1, 80
                 ),
                 '(empty)'
               ) AS first_user_message_preview
        FROM conversations
        WHERE user_id = $1::uuid
        ORDER BY updated_at DESC
        LIMIT $2
        """,
        user_id, limit,
    )
    return [
        {
            "session_id": r["session_id"],
            "created_at": r["created_at"].isoformat(),
            "updated_at": r["updated_at"].isoformat(),
            "first_user_message_preview": r["first_user_message_preview"],
            "message_count": r["message_count"],
        }
        for r in rows
    ]


async def load_session_messages(user_id: str, session_id: str) -> Optional[dict]:
    pg = get_pg_client()
    row = await pg.fetchrow(
        """SELECT session_id::text, messages, updated_at
           FROM conversations
           WHERE user_id = $1::uuid AND session_id = $2::uuid""",
        user_id, session_id,
    )
    return dict(row) if row else None


async def populate_stm_from_messages(session_id: str, messages: list[dict]) -> None:
    """Pick last 3 Q&A pairs from full message log → write to Redis STM."""
    pairs = []
    pending_user = None
    for m in messages:
        if m["role"] == "user":
            pending_user = m["content"]
        elif m["role"] == "assistant" and pending_user:
            pairs.append({
                "q": pending_user,
                "a": m["content"],
                "ts": m.get("timestamp", ""),
            })
            pending_user = None

    stm = pairs[-_STM_MAX:]

    r = aioredis.from_url(_REDIS_URL)
    try:
        await r.setex(f"stm:{session_id}", 7200, json.dumps(stm))   # TTL 2h
    finally:
        close_fn = getattr(r, "aclose", None) or r.close
        await close_fn()
```

### Acceptance

- [ ] `GET /sessions?user_id=...` trả list sorted desc by updated_at
- [ ] Mỗi item có `first_user_message_preview` (≤ 80 chars)
- [ ] `POST /sessions/{id}/resume` load messages + populate Redis STM
- [ ] Session không tồn tại → 404

---

## 7. Task 5.6 — Eager session write

**Effort**: 45m
**File**: [db/session_store.py](agenticRAG/agentic_rag_gemini/langgraph_agents/db/session_store.py)

### Function

```python
import json
from datetime import datetime, timezone
from typing import Optional


async def write_session_turn(
    user_id: str,
    session_id: str,
    user_query: str,
    assistant_answer: str,
    intent: str,
    tokens: int,
) -> None:
    """Append 1 user message + 1 assistant message to conversations.

    INSERT if session_id new, UPDATE (append to messages JSONB) if exists.
    Also update Redis STM (FIFO 3 Q&A pairs).
    """
    pg = get_pg_client()
    ts = datetime.now(timezone.utc).isoformat()

    new_turn = [
        {"role": "user", "content": user_query, "timestamp": ts},
        {"role": "assistant", "content": assistant_answer, "timestamp": ts,
         "metadata": {"intent": intent, "tokens": tokens}},
    ]

    # UPSERT pattern using ON CONFLICT
    await pg.execute(
        """
        INSERT INTO conversations (id, user_id, session_id, messages, created_at, updated_at)
        VALUES (gen_random_uuid(), $1::uuid, $2::uuid, $3::jsonb, now(), now())
        ON CONFLICT (session_id) DO UPDATE
        SET messages = conversations.messages || $3::jsonb,
            updated_at = now()
        """,
        user_id, session_id, json.dumps(new_turn),
    )

    # Update Redis STM FIFO
    await _append_stm(session_id, user_query, assistant_answer, ts)


async def _append_stm(session_id: str, q: str, a: str, ts: str) -> None:
    r = aioredis.from_url(_REDIS_URL)
    try:
        raw = await r.get(f"stm:{session_id}")
        stm = json.loads(raw) if raw else []
        stm.append({"q": q, "a": a, "ts": ts})
        stm = stm[-_STM_MAX:]   # FIFO keep last 3
        await r.setex(f"stm:{session_id}", 7200, json.dumps(stm))
    except Exception:
        pass   # graceful — next request will re-read PG via memory_node LTM
    finally:
        close_fn = getattr(r, "aclose", None) or r.close
        await close_fn()
```

### Wire vào /chat endpoint

Trong event_generator (Task 5.3):
```python
# Sau khi graph stream xong, trước done
if final_state.get("final_answer"):
    try:
        await write_session_turn(
            user_id=req.user_id,
            session_id=req.session_id,
            user_query=req.query,
            assistant_answer=final_state["final_answer"],
            intent=final_state.get("intent", ""),
            tokens=final_state.get("total_tokens", 0),
        )
        yield encode_event("session_persisted", {"session_id": req.session_id})
    except Exception as exc:
        logger.warning("Session persist failed: %s", exc)
        # Don't block SSE — session_persisted just missing, graph response still goes through
```

### Schema requirement

[db/init_schema.sql](agenticRAG/agentic_rag_gemini/langgraph_agents/db/init_schema.sql) cần thêm `UNIQUE` constraint:

```sql
ALTER TABLE conversations ADD CONSTRAINT conversations_session_id_unique UNIQUE (session_id);
```

Hoặc nếu wipe được DB: edit init_schema.sql trực tiếp.

### Acceptance

- [ ] POST /chat lần đầu với session_id mới → conversations table có 1 row, messages có 2 entries
- [ ] POST /chat lần 2 cùng session_id → messages JSONB append, không tạo row mới
- [ ] Redis `stm:{session_id}` chứa 3 Q&A pairs gần nhất (FIFO drop oldest)
- [ ] Concurrent writes cùng session → not duplicate rows (UNIQUE constraint)

---

## 8. Task 5.7 — Backend SSE tests

**Effort**: 60m
**File mới**: `tests/langgraph_agents/test_phase5_sse.py`

### Test cases

```python
"""Tests for SSE /chat endpoint + session reopen."""

import json
from unittest.mock import AsyncMock, MagicMock, patch
import pytest


def _parse_sse_stream(raw: bytes) -> list[dict]:
    """Parse SSE text into [{event, data}] list."""
    events = []
    current_event = None
    for line in raw.decode("utf-8").splitlines():
        if line.startswith("event:"):
            current_event = line[len("event:"):].strip()
        elif line.startswith("data:"):
            data_str = line[len("data:"):].strip()
            events.append({"event": current_event, "data": json.loads(data_str)})
            current_event = None
    return events


@pytest.mark.unit
def test_sse_chat_emits_stage_events(api_client):
    """Stage events emitted for each node."""
    client, mock_graph, _, _ = api_client
    # Mock astream_events to yield fake stage events
    async def fake_stream(*args, **kw):
        yield {"event": "on_chain_start", "name": "memory", "data": {}, "metadata": {}}
        yield {"event": "on_chain_end", "name": "memory", "data": {"output": {}}, "metadata": {}}
        yield {"event": "on_chain_start", "name": "planner", "data": {}, "metadata": {}}
        yield {"event": "on_chain_end", "name": "planner",
               "data": {"output": {"intent": "conversation"}}, "metadata": {}}
        # ... etc

    mock_graph.astream_events = fake_stream

    resp = client.post("/chat", json={"query": "Xin chào"})
    assert resp.status_code == 200
    events = _parse_sse_stream(resp.content)
    stage_events = [e for e in events if e["event"] == "stage"]
    assert any(e["data"]["node"] == "memory" for e in stage_events)
    assert any(e["data"]["node"] == "planner" for e in stage_events)


@pytest.mark.unit
def test_sse_chat_emits_token_events_only_from_conversation(api_client):
    """on_chat_model_stream from non-conversation nodes filtered out."""
    # ... mock astream_events with chat_model_stream events tagged different nodes
    ...


@pytest.mark.unit
def test_sse_chat_emits_done_event_last(api_client):
    """Done event always last, contains total_tokens + request_id."""
    ...


@pytest.mark.unit
def test_sse_chat_speech_mode_emits_speech_pending_and_ready(api_client):
    """output_mode=speech → speech_pending then speech_ready (mock TTS)."""
    ...


@pytest.mark.unit
def test_session_persisted_event_before_done(api_client):
    """session_persisted always precedes done."""
    ...


@pytest.mark.unit
def test_session_persist_failure_does_not_break_stream(api_client):
    """If write_session_turn raises, SSE continues to done event."""
    ...


@pytest.mark.unit
@pytest.mark.asyncio
async def test_list_user_sessions_returns_sorted_desc(pg_fixture):
    """Mock 3 sessions with different timestamps → returns by updated_at DESC."""
    ...


@pytest.mark.unit
@pytest.mark.asyncio
async def test_resume_session_populates_stm(pg_fixture, redis_fixture):
    """POST /sessions/{id}/resume loads messages + sets Redis STM key."""
    ...


@pytest.mark.unit
@pytest.mark.asyncio
async def test_write_session_turn_upsert(pg_fixture):
    """First call INSERT, second call APPEND to messages JSONB."""
    ...
```

### Acceptance

- [ ] 9 test pass
- [ ] SSE parsing helper hoạt động đúng
- [ ] Token filter (chỉ conversation) verified

---

## 9. Task 5.8 — Frontend: refactor api.js → EventSource

**Effort**: 60m
**File**: [ECA_UI/api.js](ECA_UI/api.js)

### Diff chính

Xóa axios POST /chat, thay bằng:

```javascript
// New SSE chat handler
async function streamChat({ query, userId, sessionId, personaId, outputMode, onEvent }) {
  const url = `${getApiBaseUrl()}/chat`;
  const resp = await fetch(url, {
    method: "POST",
    headers: {
      "Content-Type": "application/json",
      "Accept": "text/event-stream",
    },
    body: JSON.stringify({
      query, user_id: userId, session_id: sessionId,
      persona_id: personaId, output_mode: outputMode,
    }),
  });

  if (!resp.ok) {
    throw new Error(`HTTP ${resp.status}: ${await resp.text()}`);
  }

  const reader = resp.body.getReader();
  const decoder = new TextDecoder();
  let buffer = "";

  while (true) {
    const { done, value } = await reader.read();
    if (done) break;
    buffer += decoder.decode(value, { stream: true });

    // Parse SSE: event:\n data:\n\n blocks
    const blocks = buffer.split("\n\n");
    buffer = blocks.pop();   // last (possibly incomplete) block stays in buffer

    for (const block of blocks) {
      if (!block.trim()) continue;
      let eventType = null, dataStr = "";
      for (const line of block.split("\n")) {
        if (line.startsWith("event:")) eventType = line.slice(6).trim();
        else if (line.startsWith("data:")) dataStr += line.slice(5).trim();
      }
      if (eventType) {
        try {
          onEvent(eventType, JSON.parse(dataStr));
        } catch (e) {
          console.warn("Failed to parse SSE data:", dataStr, e);
        }
      }
    }
  }
}


async function listSessions(userId) {
  const resp = await fetch(`${getApiBaseUrl()}/sessions?user_id=${encodeURIComponent(userId)}`);
  if (!resp.ok) throw new Error(`HTTP ${resp.status}`);
  return await resp.json();
}


async function resumeSession(sessionId, userId) {
  const resp = await fetch(`${getApiBaseUrl()}/sessions/${sessionId}/resume?user_id=${userId}`, {
    method: "POST",
  });
  if (!resp.ok) throw new Error(`HTTP ${resp.status}`);
  return await resp.json();
}


// Expose to UI
window.ECA_API = { streamChat, listSessions, resumeSession };
```

### Tại sao `fetch + ReadableStream` thay vì `EventSource`?

- `EventSource` chỉ support GET. /chat là POST.
- `fetch` với manual SSE parsing flexible hơn — gửi POST body, parse stream
- Trade-off: không có auto-reconnect built-in (cần code thủ công cho production)

### Acceptance

- [ ] `streamChat({ query: "test", onEvent: console.log })` log từng SSE event
- [ ] Network error → throw exception
- [ ] Token events arrive trong < 1s sau request

---

## 10. Task 5.9 — History sidebar component

**Effort**: 90m
**File**: [ECA_UI/index.html](ECA_UI/index.html) + có thể tạo `ECA_UI/sidebar.js`

### Sidebar HTML

```html
<aside id="history-sidebar">
  <h3>Phiên trò chuyện</h3>
  <button id="new-session-btn">+ Phiên mới</button>
  <ul id="session-list">
    <!-- populated dynamically -->
  </ul>
</aside>
```

### sidebar.js

```javascript
async function refreshSessionList() {
  const userId = getCurrentUserId();
  try {
    const { sessions } = await window.ECA_API.listSessions(userId);
    const list = document.getElementById("session-list");
    list.innerHTML = "";

    for (const s of sessions) {
      const li = document.createElement("li");
      li.className = "session-item";
      li.dataset.sessionId = s.session_id;
      li.innerHTML = `
        <div class="session-preview">${escapeHtml(s.first_user_message_preview)}</div>
        <div class="session-meta">${formatRelativeTime(s.updated_at)} · ${s.message_count} tin</div>
      `;
      li.addEventListener("click", () => resumeSessionInUI(s.session_id));
      list.appendChild(li);
    }
  } catch (e) {
    console.error("Failed to load sessions:", e);
  }
}


async function resumeSessionInUI(sessionId) {
  const userId = getCurrentUserId();
  try {
    const { messages } = await window.ECA_API.resumeSession(sessionId, userId);
    clearChatPanel();
    for (const m of messages) {
      renderMessage(m.role, m.content);
    }
    setActiveSessionId(sessionId);
  } catch (e) {
    alert("Không thể tiếp tục phiên: " + e.message);
  }
}


function startNewSession() {
  const newId = crypto.randomUUID();
  clearChatPanel();
  setActiveSessionId(newId);
}


document.getElementById("new-session-btn").addEventListener("click", startNewSession);

// Refresh list on load + after each successful /chat
window.addEventListener("DOMContentLoaded", refreshSessionList);
```

### CSS (minimal)

```css
#history-sidebar {
  width: 280px;
  background: #f7f7f8;
  border-right: 1px solid #e5e5e5;
  height: 100vh;
  overflow-y: auto;
  padding: 16px;
}

.session-item {
  cursor: pointer;
  padding: 12px;
  margin: 4px 0;
  border-radius: 8px;
}

.session-item:hover { background: #eee; }
.session-item.active { background: #dbeafe; }

.session-preview {
  font-size: 14px;
  line-height: 1.3;
  overflow: hidden;
  text-overflow: ellipsis;
  display: -webkit-box;
  -webkit-line-clamp: 2;
  -webkit-box-orient: vertical;
}

.session-meta {
  font-size: 12px;
  color: #6b7280;
  margin-top: 4px;
}
```

### Acceptance

- [ ] Sidebar hiển thị list sessions sorted by updated_at desc
- [ ] Click session → load messages vào chat panel
- [ ] "+ Phiên mới" tạo session_id mới, clear chat panel
- [ ] Sau /chat thành công, sidebar refresh (preview update)

---

## 11. Task 5.10 — Streaming text display + stage indicator

**Effort**: 60m
**File**: [ECA_UI/index.html](ECA_UI/index.html) + có thể tách `ECA_UI/chat.js`

### Stage indicator

```javascript
const STAGE_LABELS = {
  memory: "Đang nhớ lại ngữ cảnh...",
  planner: "Đang phân tích yêu cầu...",
  retriever_agent: "Đang tìm tài liệu...",
  synthesizer: "Đang suy luận...",
  grader: "Đang kiểm tra chất lượng...",
  conversation: "Đang soạn câu trả lời...",
  error_handler: "Đã có lỗi, đang xử lý...",
};


function showStageIndicator(node, status) {
  const indicator = document.getElementById("stage-indicator");
  if (status === "started") {
    indicator.textContent = STAGE_LABELS[node] || node;
    indicator.style.display = "block";
  } else if (node === "conversation" && status === "complete") {
    indicator.style.display = "none";
  }
}
```

### Token streaming

```javascript
let currentAssistantBubble = null;
let currentAssistantText = "";

function appendToken(content) {
  if (!currentAssistantBubble) {
    currentAssistantBubble = createMessageBubble("assistant", "");
  }
  currentAssistantText += content;
  currentAssistantBubble.querySelector(".bubble-text").textContent = currentAssistantText;
  scrollToBottom();
}


function finalizeAssistantMessage() {
  currentAssistantBubble = null;
  currentAssistantText = "";
}


// Wire to streamChat
async function sendUserMessage(query) {
  renderMessage("user", query);
  currentAssistantBubble = null;

  await window.ECA_API.streamChat({
    query,
    userId: getCurrentUserId(),
    sessionId: getActiveSessionId(),
    personaId: "eca_default",
    outputMode: "text",
    onEvent: (type, data) => {
      switch (type) {
        case "stage":
          showStageIndicator(data.node, data.status);
          break;
        case "token":
          appendToken(data.content);
          break;
        case "speech_ready":
          playAudio(data.url);
          break;
        case "speech_failed":
          console.warn("TTS failed:", data.error);
          break;
        case "done":
          finalizeAssistantMessage();
          refreshSessionList();   // update sidebar preview
          break;
        case "session_persisted":
          // optional: visual confirmation
          break;
      }
    },
  });
}
```

### Acceptance

- [ ] Token-by-token visual ChatGPT-like
- [ ] Stage indicator hiển thị "Đang phân tích..." → "Đang tìm tài liệu..." → "Đang soạn câu trả lời..."
- [ ] Stage indicator ẩn khi conversation complete
- [ ] Speech URL được play (if any)

---

## 12. Task 5.11 — Manual browser smoke

**Effort**: 30m

### Procedure

```powershell
# Terminal 1: backend
uvicorn langgraph_agents.api.main:create_app --factory --port 8080

# Terminal 2: frontend
cd ECA_UI
python -m http.server 3000

# Browser: http://localhost:3000
```

### Smoke checklist

1. ✓ "Xin chào" → stage indicator nháy nhanh → conversation generation streams token
2. ✓ "Bài tập cho đau lưng" → retriever calling pgvector_search (sidebar có spinner cho tool)
3. ✓ Click "+ Phiên mới" → chat panel clear, history vẫn list
4. ✓ Click session cũ → load messages
5. ✓ Output mode = speech → audio plays
6. ✓ Network down → error message hiển thị, không crash

---

## 13. Task 5.12 — Commit

```
feat(phase-5): SSE streaming + session reopen + frontend refactor

Backend:
- POST /chat returns SSE (stage + tool + token + done events)
- Token events filtered to conversation node only
- Session list/resume endpoints (timestamp-sorted, preview)
- Eager session write after graph (UPSERT + Redis STM FIFO update)
- TTS speech_ready polled and pushed via SSE

Frontend:
- api.js: fetch + ReadableStream SSE parser (no EventSource — POST required)
- History sidebar with session preview + click-to-resume
- Token streaming display + stage indicator labels

Tests: 9 new SSE/session tests pass.
Smoke: browser end-to-end verified for text/speech/exercise queries.
```

---

## 14. Files Touched Summary

### Created
| File | Purpose |
|------|---------|
| `langgraph_agents/api/sse.py` | SSE event encoder helper |
| `tests/langgraph_agents/test_phase5_sse.py` | 9 SSE + session tests |
| `ECA_UI/sidebar.js` | History sidebar logic |
| `ECA_UI/chat.js` | Streaming display + stage indicator (optional split) |
| `scripts/spike_astream.py` | One-off spike script |

### Modified
| File | Change |
|------|--------|
| [api/main.py](agenticRAG/agentic_rag_gemini/langgraph_agents/api/main.py) | /chat → SSE; + /sessions endpoints |
| [api/schemas.py](agenticRAG/agentic_rag_gemini/langgraph_agents/api/schemas.py) | + SessionListItem, SessionResumeResponse |
| [db/session_store.py](agenticRAG/agentic_rag_gemini/langgraph_agents/db/session_store.py) | + write_session_turn, list_user_sessions, load_session_messages, populate_stm_from_messages |
| [db/init_schema.sql](agenticRAG/agentic_rag_gemini/langgraph_agents/db/init_schema.sql) | + UNIQUE constraint session_id |
| [ECA_UI/api.js](ECA_UI/api.js) | streamChat, listSessions, resumeSession |
| [ECA_UI/index.html](ECA_UI/index.html) | Sidebar markup + stage indicator div |

### Removed
- `GET /tts/{task_id}/result` polling endpoint — không cần khi SSE push (giữ làm fallback Phase 7 reconnect recovery)

---

## 15. Risks & Gotchas

| Risk | Mitigation |
|------|-----------|
| **DeepSeek không stream tokens qua langchain-openai** | Spike (Task 5.1) verify trước. Fallback: emit conversation_complete event với full text, frontend hiển thị instant (kém UX nhưng working) |
| **astream_events lấy final_state khó** | Track trong loop bằng `on_chain_end name="LangGraph"` event capture. Worst case: dùng PostgresSaver checkpointer (Phase 7) |
| **Token filter sai → leak planner/synthesizer tokens** | Test cẩn thận trong Task 5.7 với mock stream từ 3 nodes |
| **Eager session write block SSE response 50-200ms** | Acceptable cho MVP. Nếu quá chậm: batch nhỏ 2-3 turn sau đó write (Phase 6 optimize) |
| **Multi-worker uvicorn → BackgroundTasks pool isolation** | Phase 5 chỉ chạy 1 worker. Multi-worker → switch sang Redis pub/sub cho speech_ready (Phase 7) |
| **Frontend EventSource không support POST** | Dùng fetch + ReadableStream manual. Code phức tạp hơn nhưng đúng pattern industry (ChatGPT/Claude UI cũng dùng) |
| **Session resume populate STM với pairs SAI** (user msg không có assistant pair) | Helper `populate_stm_from_messages` skip pending_user nếu không có matching assistant. Test edge case |
| **UNIQUE constraint conflict khi `session_id` đã tồn tại từ data cũ** | Migration: `ALTER TABLE ... DROP CONSTRAINT IF EXISTS; ADD CONSTRAINT ...`. Hoặc wipe table nếu dev |
| **TTS poll timeout 15s là cứng** | Configurable. Nếu real Kimodo lâu hơn (motion 10s + TTS 5s = 15s đúng giới hạn), cân nhắc tăng. Phase 6: SSE keep-alive ping mỗi 5s |

---

## 16. Reporting Checkpoints

| CP | Sau task | Báo K |
|----|---------|-------|
| **CP1** | 5.1 (spike) | Paste event counts từ spike script. Confirm `on_chat_model_stream` ≥ 5 |
| **CP2** | 5.3 (/chat SSE) | `curl -N` log 3 query mẫu (xin chao / exercise / motion), paste raw SSE output |
| **CP3** | 5.6 (eager write) | psql query show conversations rows + Redis `KEYS stm:*` |
| **CP4** | 5.7 (tests) | pytest output, 9/9 pass |
| **CP5** | 5.11 (browser smoke) | Screen recording 30s hoặc screenshots: greeting + exercise + reopen flow |

K review từng CP. CP1 fail → cần đổi plan trước khi đi tiếp.

---

## 17. Execution time estimate

| Task | Time |
|------|------|
| 5.1 Spike astream | 30m |
| 5.2 SSE helper | 30m |
| 5.3 /chat → SSE | 90m |
| 5.4 Speech poll | 30m |
| 5.5 Session list/resume | 45m |
| 5.6 Eager write | 45m |
| 5.7 Backend tests | 60m |
| 5.8 api.js refactor | 60m |
| 5.9 Sidebar | 90m |
| 5.10 Token display + stage | 60m |
| 5.11 Browser smoke | 30m |
| 5.12 Commit | 10m |
| **Total** | **~8h** |

Chia 2 ngày:
- Ngày 1 (4-5h): Backend (5.1 → 5.7)
- Ngày 2 (3-4h): Frontend (5.8 → 5.12)

---

## 18. After Phase 5 — what unblocks

| Phase 6 (Hardening) sẵn sàng vì |
|---------------------------------|
| Full request trace (stage events) → wire `request_id` correlation log |
| Token tracking endpoint surface (in `done` event) → aggregate per user/feature |
| Eval framework input: SSE events đầy đủ để compare expected vs actual flow |
| Output validators: insert layer giữa synthesizer ↔ grader, frontend hiển thị warning |

| Phase 7 (Hybrid cloud) sẵn sàng vì |
|------------------------------------|
| Frontend không phụ thuộc backend cụ thể — chỉ cần URL `getApiBaseUrl()` |
| SSE friendly với CloudFront (passthrough cho `/chat`) |
| Polling `task_result:{id}` chuyển sang Redis pub/sub khi multi-worker |
| Session reopen sẵn sàng cho multi-device sync (FastAPI stateless với PG/Redis) |

---

**N**: bắt đầu từ Task 5.1 (spike — 30 phút). Báo K sau CP1 trước khi viết SSE infra. Nếu spike fail, chúng ta sẽ điều chỉnh plan trước khi đi tiếp.
