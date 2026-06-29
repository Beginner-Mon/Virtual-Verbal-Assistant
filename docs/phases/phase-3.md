# Phase 3 — MCP Servers + TTS at FastAPI Layer

**Architect**: K | **Developer**: N | **Date**: 2026-05-23
**Branch**: `feature/langgraph-rewrite` (continue from Phase 2.5 commit `2293430`)
**Estimated time**: ~8h
**Reference**: [../plans/v2.4-plan.md](../plans/v2.4-plan.md) §6, §11, §13

---

## 0. Why this phase exists

Sau Phase 2.5, retriever_agent chỉ có 1 tool (`pgvector_search` in-process). Plan v2.4 §11 chốt: thêm 2 **MCP** tools — `generate_motion` (Kimodo) và `search_medical` (web search) — để retriever execute parallel khi planner yêu cầu.

Đồng thời, TTS được chuyển **ra khỏi graph** (Plan §13): graph trả `final_answer` → FastAPI layer fire Celery TTS task async → response trả về client với optional `speech_task_id`.

### Decisions chốt trước khi code

| Câu hỏi | Lựa chọn |
|--------|---------|
| Kimodo build thật hay mock? | **Mock trước** — placeholder video URL. Plug NVIDIA Kimodo sau |
| Web search provider? | **DuckDuckGo** (`duckduckgo-search` lib, free, no key) |
| FastAPI đặt đâu? | **Mới**: `langgraph_agents/api/` — không đụng `api_server.py` cũ |
| Test scope? | **Unit + integration với mock** — không cần real Kimodo/VieNeu trong CI |
| MCP transport? | **stdio** cho dev simplicity (subprocess). HTTP `streamable_http` để dành cho production deploy |

---

## 1. Order of execution

```
3.1  Audit & cleanup v2.2 services/ leftovers   (refactor)
3.2  Kimodo MCP server (mock)
3.3  Web search MCP server (DuckDuckGo)
3.4  config/mcp_servers.yaml
3.5  MCP client wrapper + wire into RETRIEVER_TOOLS
3.6  Verify grader visualize_motion rule still fires correctly
3.7  VieNeu-TTS Celery task (mostly verify, file đã tồn tại)
3.8  FastAPI /chat endpoint
3.9  Tests (unit + integration with mocks)
```

Commit từng task riêng. Mỗi task xanh test mới sang task kế.

---

## 2. Task 3.1 — Audit & cleanup v2.2 services/ leftovers

### Hiện trạng (carry-over từ v2.2 Phase 3, chưa wire vào graph v2.4)

```
langgraph_agents/
  services/
    exceptions.py                    # ServiceUnavailableError — KEEP
    kimodo/
      __init__.py
      client.py                      # REST httpx client — DELETE (sẽ thay bằng MCP)
      tasks.py                       # Celery motion task — DELETE
    vieneu_tts/
      __init__.py
      client.py                      # REST httpx client — KEEP (Celery task gọi)
      tasks.py                       # Celery synthesize_speech — KEEP/refactor
  streaming/
    approval.py                      # Motion approval endpoint — DELETE (no approval gate in v2.4)
  celery_app.py                      # Celery instance — KEEP, verify config
```

### Action

| File | Action | Note |
|------|--------|------|
| `services/kimodo/` (cả folder) | **DELETE** | Thay bằng `mcp/kimodo_server.py` |
| `services/vieneu_tts/client.py` | KEEP | Sync httpx → VieNeu-TTS REST |
| `services/vieneu_tts/tasks.py` | **REFACTOR** | Bỏ `voice_path` parameter (Plan v2.4 §13: FastAPI đọc persona trực tiếp); bỏ Redis pub/sub (Phase 5 SSE sẽ thêm lại) |
| `services/exceptions.py` | KEEP | Dùng cho TTS errors |
| `streaming/approval.py` | **DELETE** | Plan v2.4 không còn approval gate cho motion |
| `streaming/__init__.py` | DELETE | Folder rỗng sau khi xóa approval.py |
| `celery_app.py` | KEEP | Verify queue name `langgraph`, Redis DB 1 |

### Acceptance

- [ ] `git rm -r langgraph_agents/services/kimodo langgraph_agents/streaming`
- [ ] `langgraph_agents/services/vieneu_tts/tasks.py` không còn reference `voice_path` hoặc Redis pub/sub
- [ ] Unit tests vẫn xanh (38/38)

---

## 3. Task 3.2 — Kimodo MCP server (mock)

**File mới**: `langgraph_agents/mcp/__init__.py`, `langgraph_agents/mcp/kimodo_server.py`

### Tool contract (Plan v2.4 §11.3)

```
Tool: generate_motion
  Input:
    prompt: str                      # natural language motion description
    constraints: list[dict]          # optional, [{joint, angle}]
    duration_seconds: float = 3.0
  Output:
    {
      "video_url": str,              # mock returns "mock://motion/<hash>.mp4"
      "duration_sec": float,
      "format": "mp4",
      "joints_used": list[str],
      "_mock": true                  # flag để N test biết đang mock
    }
```

### Implementation skeleton

```python
"""Kimodo MCP server — mock mode for Phase 3.

Plug NVIDIA Kimodo real inference behind _generate_motion_real() when GPU is ready.

Run standalone (HTTP): python -m langgraph_agents.mcp.kimodo_server
Run via stdio (default in tests): spawned by MultiServerMCPClient
"""

import asyncio
import hashlib
import sys
from typing import Any

from mcp.server import Server
from mcp.server.stdio import stdio_server
from mcp.types import Tool, TextContent


_USE_REAL = False  # flip when NVIDIA Kimodo wrapper is ready


server = Server("kimodo-motion")


@server.list_tools()
async def list_tools() -> list[Tool]:
    return [
        Tool(
            name="generate_motion",
            description=(
                "Generate a 3D motion animation from a natural language description "
                "with optional joint constraints. Use ONLY for visualize_motion intent."
            ),
            inputSchema={
                "type": "object",
                "properties": {
                    "prompt": {"type": "string", "description": "motion description"},
                    "constraints": {
                        "type": "array",
                        "items": {
                            "type": "object",
                            "properties": {
                                "joint": {"type": "string"},
                                "angle": {"type": "number"},
                            },
                        },
                        "default": [],
                    },
                    "duration_seconds": {"type": "number", "default": 3.0},
                },
                "required": ["prompt"],
            },
        )
    ]


@server.call_tool()
async def call_tool(name: str, arguments: dict[str, Any]) -> list[TextContent]:
    if name != "generate_motion":
        raise ValueError(f"Unknown tool: {name}")
    result = await _generate_motion(**arguments)
    import json
    return [TextContent(type="text", text=json.dumps(result))]


async def _generate_motion(prompt: str, constraints: list[dict] | None = None,
                            duration_seconds: float = 3.0) -> dict:
    if _USE_REAL:
        return await _generate_motion_real(prompt, constraints, duration_seconds)
    return _generate_motion_mock(prompt, constraints, duration_seconds)


def _generate_motion_mock(prompt: str, constraints, duration_seconds) -> dict:
    digest = hashlib.sha1(prompt.encode("utf-8")).hexdigest()[:12]
    joints = [c.get("joint") for c in (constraints or []) if c.get("joint")]
    return {
        "video_url": f"mock://motion/{digest}.mp4",
        "duration_sec": float(duration_seconds),
        "format": "mp4",
        "joints_used": joints,
        "_mock": True,
        "_prompt_echo": prompt[:200],
    }


async def _generate_motion_real(prompt, constraints, duration_seconds) -> dict:
    raise NotImplementedError("Set _USE_REAL=True after wiring NVIDIA Kimodo inference here.")


async def main():
    async with stdio_server() as (read_stream, write_stream):
        await server.run(
            read_stream, write_stream,
            server.create_initialization_options(),
        )


if __name__ == "__main__":
    asyncio.run(main())
```

### Acceptance

- [ ] Server start standalone: `python -m langgraph_agents.mcp.kimodo_server` không lỗi
- [ ] Tool list contains `generate_motion`
- [ ] Mock call returns `_mock: True` + valid video_url string

---

## 4. Task 3.3 — Web search MCP server (DuckDuckGo)

**File mới**: `langgraph_agents/mcp/web_search_server.py`

### Tool contract

```
Tool: search_medical
  Input:
    query: str
    max_results: int = 3
    domain_filter: str | None = None  # e.g. "site:pubmed.ncbi.nlm.nih.gov"
  Output:
    [{title, snippet, url, source_domain}, ...]
```

### Dependency

Thêm vào `requirements-langgraph.txt`:
```
duckduckgo-search>=6.0.0
```

### Implementation skeleton

```python
"""Web search MCP server (DuckDuckGo backend).

Free, no API key. Used by retriever_agent as fallback when pgvector returns
low-quality results or when planner explicitly requests web search.

Run standalone: python -m langgraph_agents.mcp.web_search_server
Run via stdio (default in tests): spawned by MultiServerMCPClient
"""

import asyncio
import json
from typing import Any
from urllib.parse import urlparse

from mcp.server import Server
from mcp.server.stdio import stdio_server
from mcp.types import Tool, TextContent

from duckduckgo_search import DDGS


server = Server("web-search")


@server.list_tools()
async def list_tools() -> list[Tool]:
    return [
        Tool(
            name="search_medical",
            description=(
                "Web search via DuckDuckGo. Use as fallback when internal knowledge "
                "base (pgvector) is insufficient. Returns up to max_results items."
            ),
            inputSchema={
                "type": "object",
                "properties": {
                    "query": {"type": "string"},
                    "max_results": {"type": "integer", "default": 3, "minimum": 1, "maximum": 10},
                    "domain_filter": {"type": "string", "description": "optional site: filter, e.g. site:pubmed.ncbi.nlm.nih.gov"},
                },
                "required": ["query"],
            },
        )
    ]


@server.call_tool()
async def call_tool(name: str, arguments: dict[str, Any]) -> list[TextContent]:
    if name != "search_medical":
        raise ValueError(f"Unknown tool: {name}")
    results = await asyncio.to_thread(_search, **arguments)
    return [TextContent(type="text", text=json.dumps(results))]


def _search(query: str, max_results: int = 3, domain_filter: str | None = None) -> list[dict]:
    full_query = f"{query} {domain_filter}" if domain_filter else query
    out = []
    try:
        with DDGS() as ddgs:
            for r in ddgs.text(full_query, max_results=max_results):
                domain = urlparse(r.get("href", "")).netloc
                out.append({
                    "title": r.get("title", ""),
                    "snippet": r.get("body", ""),
                    "url": r.get("href", ""),
                    "source_domain": domain,
                })
    except Exception as exc:
        out.append({"error": f"DuckDuckGo search failed: {exc}"})
    return out


async def main():
    async with stdio_server() as (r, w):
        await server.run(r, w, server.create_initialization_options())


if __name__ == "__main__":
    asyncio.run(main())
```

### Acceptance

- [ ] `pip install duckduckgo-search`
- [ ] Standalone start không lỗi
- [ ] `_search("back pain exercises", max_results=2)` trả list ≥1 dict có `title`, `url`

---

## 5. Task 3.4 — MCP servers config

**File mới**: `config/mcp_servers.yaml`

### Content

```yaml
# MCP server registry — consumed by langgraph_agents/mcp/client.py
#
# Transport options:
#   stdio              → server runs as subprocess (default for dev)
#   streamable_http    → server runs as long-lived HTTP service
#
# To switch a server to HTTP, change transport + add url, remove command/args.

mcp_servers:
  kimodo_motion:
    transport: stdio
    command: python
    args: ["-m", "langgraph_agents.mcp.kimodo_server"]

  web_search:
    transport: stdio
    command: python
    args: ["-m", "langgraph_agents.mcp.web_search_server"]

  # Production HTTP example (commented):
  # kimodo_motion:
  #   transport: streamable_http
  #   url: "http://localhost:5001/mcp"
```

---

## 6. Task 3.5 — MCP client wrapper + wire into RETRIEVER_TOOLS

**File mới**: `langgraph_agents/mcp/__init__.py`, `langgraph_agents/mcp/client.py`
**File sửa**: [retriever_agent.py](agenticRAG/agentic_rag_gemini/langgraph_agents/nodes/retriever_agent.py)

### MCP client wrapper

```python
"""MultiServerMCPClient wrapper — lazy singleton.

Loads config/mcp_servers.yaml on first call, builds the client, returns the
list of LangChain tools discovered from all MCP servers. Cached afterwards.
"""

import asyncio
from pathlib import Path
import yaml

from langchain_mcp_adapters.client import MultiServerMCPClient


_mcp_client = None
_mcp_tools: list = []
_init_lock = asyncio.Lock()


def _load_mcp_config() -> dict:
    config_path = Path(__file__).resolve().parents[3] / "config" / "mcp_servers.yaml"
    if not config_path.exists():
        return {}
    with open(config_path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f).get("mcp_servers", {})


async def get_mcp_tools() -> list:
    """Discover tools from all configured MCP servers (idempotent, cached)."""
    global _mcp_client, _mcp_tools
    if _mcp_tools:
        return _mcp_tools

    async with _init_lock:
        if _mcp_tools:
            return _mcp_tools

        cfg = _load_mcp_config()
        if not cfg:
            return []

        # MultiServerMCPClient consumes the dict shape directly.
        _mcp_client = MultiServerMCPClient(cfg)
        # langchain-mcp-adapters >= 0.2 uses async get_tools()
        _mcp_tools = await _mcp_client.get_tools()
        return _mcp_tools


async def close_mcp_client():
    global _mcp_client, _mcp_tools
    if _mcp_client is not None:
        # Some versions expose .aclose() or are CMs — try both.
        for attr in ("aclose", "close"):
            fn = getattr(_mcp_client, attr, None)
            if fn:
                try:
                    res = fn()
                    if asyncio.iscoroutine(res):
                        await res
                    break
                except Exception:
                    pass
        _mcp_client = None
        _mcp_tools = []
```

### Wire into retriever

Update [retriever_agent.py](agenticRAG/agentic_rag_gemini/langgraph_agents/nodes/retriever_agent.py):

```python
from langgraph_agents.mcp.client import get_mcp_tools


async def _build_tools() -> list:
    """In-process + MCP. Called per node invocation; tool list cached by mcp/client."""
    mcp_tools = await get_mcp_tools()
    return [pgvector_search, *mcp_tools]


async def retriever_agent_node(state: AgentState, config) -> dict:
    tools = await _build_tools()
    # ... existing logic, but use `tools` instead of module-level RETRIEVER_TOOLS
    llm = get_chat_model("retriever").bind_tools(tools)
    ...
```

### Update graph.py

ToolNode cần biết toàn bộ tools để execute. Vì MCP tools chỉ discover được async, graph build phải bất đồng bộ.

**Approach**: tách `build_graph()` thành `async def build_graph_async()`. Sync `build_graph()` gọi `asyncio.run(build_graph_async())` cho convenience.

```python
import asyncio
from langgraph.graph import StateGraph
from langgraph.prebuilt import ToolNode

from langgraph_agents.nodes.retriever_agent import (
    retriever_agent_node, RETRIEVER_TOOLS as _INPROCESS_TOOLS,
)
from langgraph_agents.mcp.client import get_mcp_tools


async def build_graph_async():
    mcp_tools = await get_mcp_tools()
    all_tools = [*_INPROCESS_TOOLS, *mcp_tools]

    g = StateGraph(AgentState)
    # ... existing nodes
    g.add_node("tools", ToolNode(all_tools))
    # ... rest unchanged
    return g.compile().with_config(recursion_limit=_RECURSION_LIMIT)


def build_graph():
    return asyncio.run(build_graph_async())
```

> **Cảnh báo**: `asyncio.run` không hoạt động bên trong event loop đang chạy. Test fixtures và FastAPI startup phải gọi `build_graph_async()` trực tiếp. Document rõ trong docstring.

### Acceptance

- [ ] `await get_mcp_tools()` trả list ≥ 2 tools (`generate_motion`, `search_medical`)
- [ ] `build_graph_async()` compile OK với MCP tools
- [ ] Retriever invoke với query motion → LLM gọi `generate_motion` → ToolMessage có content mock

---

## 7. Task 3.6 — Verify grader visualize_motion rule

**File**: [grader.py](agenticRAG/agentic_rag_gemini/langgraph_agents/nodes/grader.py) — không sửa.

Phase 2.5 grader đã có rule cho `visualize_motion`:

```python
if intent == "visualize_motion":
    has_motion = any(
        isinstance(m, ToolMessage) and m.name == "generate_motion"
        for m in messages
    )
    if not has_motion:
        return False, _MOTION_FAIL_FEEDBACK
```

Sau khi Kimodo MCP wired vào, rule này active. Test:

- [ ] Query "Cho tôi xem động tác bridge" → planner intent=visualize_motion → retriever gọi `generate_motion` → ToolMessage có name=`generate_motion` → grader pass
- [ ] Force retriever skip motion call (mock LLM trả AIMessage không có tool_calls) → grader fail → retry → vẫn không gọi → pass_with_warning

---

## 8. Task 3.7 — VieNeu-TTS Celery task (refactor v2.2 leftovers)

**File**: [services/vieneu_tts/tasks.py](agenticRAG/agentic_rag_gemini/langgraph_agents/services/vieneu_tts/tasks.py)

### Refactor checklist

1. **Bỏ `voice_path` param** — Plan v2.4 §13: FastAPI đọc persona MD trực tiếp, pass voice_path; task ký nhận `voice_path` qua kwargs vẫn OK
2. **Bỏ Redis pub/sub** — Phase 3 minimal: task chạy fire-and-forget, kết quả trong Redis `task_result:{task_id}`. Phase 5 SSE sẽ thêm pub/sub lại
3. **Persist result vào Redis** với TTL 1h: `task_result:{task_id}` → `{event: "speech_ready", url}`

### Skeleton (sửa từ file hiện có)

```python
import json
import redis as sync_redis

from langgraph_agents.celery_app import celery_app
from langgraph_agents.services.vieneu_tts.client import get_vieneu_tts_client
from langgraph_agents.services.exceptions import ServiceUnavailableError


_REDIS_URL = "redis://localhost:6379/1"


@celery_app.task(
    name="langgraph.synthesize_speech",
    bind=True,
    acks_late=True,
    max_retries=1,
    default_retry_delay=3,
)
def synthesize_speech(self, text: str, request_id: str, session_id: str,
                       voice_path: str | None = None) -> dict:
    client = get_vieneu_tts_client()
    try:
        result = client.synthesize_sync(text=text, voice_path=voice_path)
    except ServiceUnavailableError as exc:
        _persist(self.request.id, {"event": "speech_failed", "error": str(exc)})
        raise

    payload = {"event": "speech_ready", "url": result["audio_url"]}
    _persist(self.request.id, payload)
    return payload


def _persist(task_id: str, payload: dict):
    try:
        r = sync_redis.Redis.from_url(_REDIS_URL)
        r.setex(f"task_result:{task_id}", 3600, json.dumps(payload))
    except Exception:
        pass  # graceful — Phase 5 SSE will retry from celery result backend
```

### Acceptance

- [ ] Mock test: `synthesize_speech.delay("hello", "req1", "sess1")` (with mocked httpx) → task_result:* key xuất hiện trong Redis
- [ ] Service down → ServiceUnavailableError → task FAILURE, `speech_failed` payload trong Redis

---

## 9. Task 3.8 — FastAPI `/chat` endpoint

**File mới**: `langgraph_agents/api/__init__.py`, `langgraph_agents/api/main.py`, `langgraph_agents/api/schemas.py`

### Tối thiểu cho Phase 3

Endpoint trả JSON đầy đủ (no streaming). Phase 5 sẽ thay bằng SSE qua `astream_events()`.

### `schemas.py`

```python
from pydantic import BaseModel, Field
from typing import Literal, Optional


class ChatRequest(BaseModel):
    query: str
    user_id: str = "anonymous"
    session_id: str = "default"
    persona_id: str = "eca_default"
    output_mode: Literal["text", "speech", "both"] = "text"
    token_limit: Optional[int] = None


class ChatResponse(BaseModel):
    request_id: str
    final_answer: str
    intent: str
    confidence: float
    needs_clarification: bool = False
    speech_task_id: Optional[str] = None
    total_tokens: int = 0
    grader_result: Optional[str] = None
    grader_warning: Optional[str] = None
    errors: list[dict] = Field(default_factory=list)
```

### `main.py`

```python
"""FastAPI layer for the LangGraph v2.4 pipeline.

POST /chat        → run graph + fire async TTS task
GET  /health      → service health
GET  /tts/{task_id}/result   → poll Redis for TTS result (Phase 3 polling fallback;
                               Phase 5 will replace with SSE push)
"""

import json
import logging
import uuid

import redis as sync_redis
from fastapi import FastAPI, HTTPException

from langgraph_agents.api.schemas import ChatRequest, ChatResponse
from langgraph_agents.graph import build_graph_async
from langgraph_agents.nodes._persona_loader import get_persona


logger = logging.getLogger("langgraph.api")

app = FastAPI(title="VVA LangGraph v2.4")
_graph = None
_redis = sync_redis.Redis.from_url("redis://localhost:6379/1")


@app.on_event("startup")
async def _startup():
    global _graph
    _graph = await build_graph_async()


@app.get("/health")
async def health():
    return {"status": "ok", "graph_loaded": _graph is not None}


@app.post("/chat", response_model=ChatResponse)
async def chat(req: ChatRequest):
    request_id = str(uuid.uuid4())
    state = {
        "messages": [], "errors": [], "retry_count": 0, "total_tokens": 0,
    }
    config = {
        "configurable": {
            "user_id": req.user_id,
            "session_id": req.session_id,
            "query": req.query,
            "persona_id": req.persona_id,
            "output_mode": req.output_mode,
            "request_id": request_id,
            "token_limit": req.token_limit,
        }
    }
    result = await _graph.ainvoke(state, config=config)

    speech_task_id: str | None = None
    if req.output_mode in ("speech", "both") and result.get("final_answer"):
        persona = get_persona(req.persona_id)
        voice_path = persona.get("voice_identity", {}).get("voice_path")
        from langgraph_agents.services.vieneu_tts.tasks import synthesize_speech
        task = synthesize_speech.delay(
            result["final_answer"], request_id, req.session_id, voice_path,
        )
        speech_task_id = task.id

    return ChatResponse(
        request_id=request_id,
        final_answer=result.get("final_answer", ""),
        intent=result.get("intent", ""),
        confidence=result.get("confidence", 0.0),
        needs_clarification=result.get("needs_clarification", False),
        speech_task_id=speech_task_id,
        total_tokens=result.get("total_tokens", 0),
        grader_result=result.get("grader_result"),
        grader_warning=result.get("grader_warning"),
        errors=result.get("errors", []),
    )


@app.get("/tts/{task_id}/result")
async def tts_result(task_id: str):
    raw = _redis.get(f"task_result:{task_id}")
    if not raw:
        raise HTTPException(404, "Task not ready or expired")
    return json.loads(raw)
```

### Run command

```powershell
uvicorn langgraph_agents.api.main:app --reload --port 8080
# Separate terminal:
celery -A langgraph_agents.celery_app worker -l info -Q langgraph
```

### Acceptance

- [ ] `GET /health` trả `{status: "ok", graph_loaded: true}`
- [ ] `POST /chat {query: "Xin chào"}` trả JSON với `final_answer` non-empty, `intent: "conversation"`
- [ ] `POST /chat {query: "...", output_mode: "speech"}` trả `speech_task_id` non-null

---

## 10. Task 3.9 — Tests

### `test_phase3_mcp_kimodo.py` (unit)

- [ ] `test_kimodo_list_tools` — server exposes `generate_motion`
- [ ] `test_kimodo_mock_returns_url` — call returns `_mock: True` + valid URL
- [ ] `test_kimodo_constraints_passthrough` — joints in input appear in `joints_used`

### `test_phase3_mcp_web_search.py` (integration, network)

- [ ] `test_web_search_returns_results` — query "back pain exercises" → list with ≥1 result
- [ ] `test_web_search_handles_failure` — mock DDGS raising → returns `[{error: ...}]`
- Mark with `@pytest.mark.integration` (skip in CI without network)

### `test_phase3_mcp_client.py` (unit, mock MultiServerMCPClient)

- [ ] `test_get_mcp_tools_caches` — second call returns same tools without re-init
- [ ] `test_get_mcp_tools_empty_config` — no config file → returns `[]`

### `test_phase3_retriever_with_mcp.py` (integration, requires DeepSeek)

- [ ] `test_retriever_calls_motion_for_visualize_intent` — full graph with motion query → ToolMessage `generate_motion` appears in messages

### `test_phase3_tts_task.py` (unit, mock httpx + Celery in-memory)

- [ ] `test_synthesize_speech_persists_result` — mock VieNeu returns audio_url → Redis key `task_result:<id>` set
- [ ] `test_synthesize_speech_service_down` — mock raises ServiceUnavailableError → Redis has `speech_failed`

### `test_phase3_api.py` (unit, FastAPI TestClient + mock graph)

- [ ] `test_chat_text_mode_no_tts_task` — output_mode="text" → speech_task_id is None
- [ ] `test_chat_speech_mode_fires_task` — output_mode="speech" → speech_task_id non-null (mock `synthesize_speech.delay`)
- [ ] `test_health_returns_ok`
- [ ] `test_tts_result_404_when_missing`

### Test infrastructure update

Conftest cần fixture cho mock MCP client để retriever tests không cần spawn subprocess.

---

## 11. Files Touched Summary

### Created

| File | Purpose |
|------|---------|
| `langgraph_agents/mcp/__init__.py` | new package |
| `langgraph_agents/mcp/kimodo_server.py` | Kimodo MCP server (mock) |
| `langgraph_agents/mcp/web_search_server.py` | DuckDuckGo MCP server |
| `langgraph_agents/mcp/client.py` | MultiServerMCPClient wrapper |
| `langgraph_agents/api/__init__.py` | new package |
| `langgraph_agents/api/main.py` | FastAPI app |
| `langgraph_agents/api/schemas.py` | Pydantic request/response |
| `config/mcp_servers.yaml` | MCP server registry |
| `tests/langgraph_agents/test_phase3_*.py` | 5 test files |

### Modified

| File | Change |
|------|--------|
| [retriever_agent.py](agenticRAG/agentic_rag_gemini/langgraph_agents/nodes/retriever_agent.py) | `_build_tools()` async, append MCP tools |
| [graph.py](agenticRAG/agentic_rag_gemini/langgraph_agents/graph.py) | Split `build_graph_async()` + sync convenience wrapper |
| [services/vieneu_tts/tasks.py](agenticRAG/agentic_rag_gemini/langgraph_agents/services/vieneu_tts/tasks.py) | Drop pub/sub, keep Redis result persist |
| [requirements-langgraph.txt](requirements-langgraph.txt) | + `duckduckgo-search>=6.0.0`, + `uvicorn[standard]>=0.30` (nếu chưa có) |

### Removed

| File | Reason |
|------|------|
| `langgraph_agents/services/kimodo/` (cả folder) | Replaced by MCP server |
| `langgraph_agents/streaming/` (cả folder) | No approval gate in v2.4 |

---

## 12. Risks & Gotchas

| Risk | Mitigation |
|------|-----------|
| **MCP stdio subprocess** spawn overhead (~100ms first call) | Acceptable for dev. Production: switch to HTTP via `transport: streamable_http` |
| **`build_graph_async` async-only** | `asyncio.run` không chạy trong event loop có sẵn. Test fixture phải gọi `await build_graph_async()` thay vì `build_graph()` sync wrapper |
| **`duckduckgo-search` rate-limit** (~ vài req/giây nếu spam) | Lib tự retry; nếu rate-limit triệt để, swap sang Brave/Tavily |
| **MCP tool naming clash** với pgvector @tool | Đặt prefix nếu cần, hoặc rely on unique tool names (`generate_motion` ≠ `search_medical` ≠ `pgvector_search`) |
| **MultiServerMCPClient lifecycle** | Là async context manager. Singleton wrapper giữ alive. Cleanup ở `app.on_event("shutdown")` (Phase 3 chưa cần — graceful exit OK) |
| **Celery worker bắt buộc chạy riêng** cho TTS | Document trong README. Test mock `synthesize_speech.delay` cho local unit |
| **FastAPI `/tts/{task_id}/result`** là polling — không real-time | Phase 5 SSE thay thế. Đủ cho Phase 3 verification |
| **Mock Kimodo trả URL không tồn tại** | Test chỉ check field shape, không fetch URL. Khi plug real, update test assertion |
| **`@app.on_event` deprecated trong FastAPI 0.100+** | Cân nhắc dùng `lifespan` context manager nếu lint warn. Acceptable cho Phase 3 |

---

## 13. Reporting Checkpoints

| CP | Sau task | Báo K |
|----|---------|-------|
| **CP1** | 3.2 (Kimodo MCP) | Paste output `python -m langgraph_agents.mcp.kimodo_server` không lỗi + 1 test pass |
| **CP2** | 3.4 (config + 3.5 wire) | Paste log retriever_agent gọi `generate_motion` qua MCP — kèm ToolMessage content |
| **CP3** | 3.8 (FastAPI /chat) | `curl localhost:8080/chat` với 3 query mẫu (greeting / exercise / motion), paste JSON response |
| **CP4** | 3.9 (tests) | Full pytest output (unit + integration) |

K review từng CP trước khi N tiếp.

---

## 14. Execution time estimate

| Task | Time |
|------|------|
| 3.1 Cleanup v2.2 leftovers | 30m |
| 3.2 Kimodo MCP (mock) | 60m |
| 3.3 Web search MCP | 60m |
| 3.4 Config file | 15m |
| 3.5 MCP client + retriever wire + graph_async | 90m |
| 3.6 Grader verify | 15m |
| 3.7 VieNeu-TTS Celery refactor | 30m |
| 3.8 FastAPI /chat | 90m |
| 3.9 Tests | 120m |
| **Total** | **~8h** |

Chia 2 ngày: ngày 1 task 3.1–3.5 (3h), ngày 2 task 3.6–3.9 (5h).

---

## 15. After Phase 3 — what unblocks

| Phase tiếp | Ready vì |
|-----------|---------|
| **Phase 5** SSE streaming | `/chat` endpoint đã có. Thay JSON response bằng `astream_events()` SSE. Phase 5 chỉ wrap |
| **Phase 5** Frontend rework | API contract Phase 3 đã chốt — ECA_UI có thể đổi sang fetch JSON ngay |
| **Phase 6** Hardening | MCP circuit breaker, structured logging, tracing |
| **Phase real-Kimodo** | Plug NVIDIA inference vào `_generate_motion_real()` — không động graph |

---

**N**: bắt đầu từ Task 3.1. Commit mỗi task riêng. Báo K sau mỗi CP.
