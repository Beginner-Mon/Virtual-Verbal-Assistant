# Phase 2.5 — Architecture Refactor (v2.2 → v2.4)

**Architect**: K | **Developer**: N | **Date**: 2026-05-23
**Branch**: `feature/langgraph-rewrite` (continue from Phase 2)
**Estimated time**: ~8–10h
**Reference**: [PLAN-v2.4-DRAFT.md](PLAN-v2.4-DRAFT.md), worklog [23-05-2026.md](docs/worklogs/23-05-2026.md)

---

## 0. Why this phase exists

Phase 0–4 đã code xong theo **Plan v2.2**. Sau grill session 22–23/05, Plan v2.4 chốt 9 thay đổi kiến trúc. Phase 2.5 = **sửa lại Phase 0–4 hiện tại** cho khớp v2.4. **Không phá** code đã chạy được — refactor có kiểm soát, mỗi bước có test xanh trước khi sang bước kế.

### Lệch chính giữa code hiện tại và Plan v2.4

| Khu vực | Code hiện tại (v2.2) | Plan v2.4 |
|---------|---------------------|-----------|
| **Flow** | manager → memory → retrieval → reasoning → validator → conversation → dispatch | memory → planner → retriever_agent → synthesizer → grader → conversation |
| **LLM** | Custom `llm_gateway.py` | LangChain ChatModel (`langchain-google-genai`) |
| **Manager** | Trả `intent + confidence + expanded_query` | Planner trả `PlanOutput` Pydantic (structured plan) |
| **Memory** | Luôn chạy LTM | Conditional LTM (keyword trigger), STM 3 Q&A FIFO |
| **Retrieval** | Direct pgvector + web | LLM + ToolNode (pgvector @tool + MCP tools) |
| **Reasoning** | Đọc `retrieval_results` từ state | Synthesizer đọc `messages` (ToolMessages) |
| **Validator** | Build `raw_answer` + fallback | **XÓA** — thay bằng Grader rule-based + retry max 1 |
| **Conversation** | Chỉ styling | **Dual mode**: styling + generation |
| **Dispatch** | Approval gate + Celery tasks | **XÓA** — TTS chuyển sang FastAPI layer |
| **State** | `raw_answer`, `voice_path`, `motion_*` | `plan`, `needs_clarification`, `grader_*`, `total_tokens`, `messages` |
| **error_handler** | Ghi `raw_answer` | Ghi `reasoning_output` |
| **Kimodo** | REST client + Celery task | **MCP server** (Phase 3) |

---

## 1. Order of execution (strict)

Bám đúng thứ tự — mỗi bước phụ thuộc bước trước:

```
1.  state.py rewrite              (foundation)
2.  Dependencies + LangChain ChatModel adapter
3.  RENAME manager.py → planner.py + PlanOutput Pydantic
4.  UPDATE memory.py — STM 3 Q&A + conditional LTM
5.  CREATE tools/pgvector_tool.py — @tool wrapper
6.  REPLACE retrieval.py → retriever_agent.py (ToolNode + pgvector @tool only; MCP wiring stub)
7.  RENAME reasoning.py → synthesizer.py — read messages
8.  REPLACE validator.py → grader.py — rule-based + retry
9.  UPDATE conversation.py — dual mode
10. UPDATE error_handler.py — reasoning_output
11. UPDATE routing.py — planner routing + grader retry
12. UPDATE graph.py — new flow
13. REMOVE dispatch.py + llm_gateway.py + (Phase-3 dispatch dependencies)
14. UPDATE tests — adapt to new nodes
15. Smoke + integration test pass
```

**Quy tắc**: mỗi bước **commit riêng**. Nếu break, revert chính xác bước đó.

---

## 2. Task 2.5.1 — Rewrite `state.py`

**File**: [agenticRAG/agentic_rag_gemini/langgraph_agents/state.py](agenticRAG/agentic_rag_gemini/langgraph_agents/state.py)

### Mục tiêu

Thay schema cũ (v2.2) bằng schema mới (v2.4). Loại bỏ các field theo intent cũ. Thêm field cho planner + grader + token budget.

### Code mới

```python
from __future__ import annotations
import operator
from enum import Enum
from typing import Annotated, Optional, TypedDict

from langchain_core.messages import add_messages


class ErrorSeverity(str, Enum):
    CRITICAL = "critical"
    RECOVERABLE = "recoverable"
    IGNORABLE = "ignorable"


class AgentState(TypedDict):
    # ── LangGraph message passing (retriever_agent ToolNode) ─────────
    messages: Annotated[list, add_messages]

    # ── Planner output ────────────────────────────────────────────────
    intent: str                         # conversation | knowledge_query | exercise_recommendation | visualize_motion | clarify
    confidence: float
    expanded_query: str
    plan: dict                          # PlanOutput.model_dump()
    needs_clarification: bool

    # ── Memory output ────────────────────────────────────────────────
    memory_context: dict                # {short_term: [...], long_term: {...}, user_profile: {...}}

    # ── Synthesizer output ───────────────────────────────────────────
    reasoning_output: str               # clinical response OR error msg (from error_handler)

    # ── Grader output ────────────────────────────────────────────────
    grader_result: str                  # "pass" | "retry" | "pass_with_warning"
    grader_warning: Optional[str]
    grader_feedback: Optional[str]      # injected into retriever on retry
    retry_count: int                    # 0 → max 1

    # ── Conversation output ──────────────────────────────────────────
    final_answer: str

    # ── Token tracking (reducer auto-accumulates) ────────────────────
    total_tokens: Annotated[int, operator.add]

    # ── Error tracking (append-only) ─────────────────────────────────
    errors: Annotated[list[dict], operator.add]
```

### Fields ĐÃ XÓA (không tồn tại nữa)

| Field cũ | Lý do xóa |
|----------|-----------|
| `user_id`, `session_id`, `query`, `persona_id`, `output_mode`, `request_id` | Chuyển sang `RunnableConfig.configurable` (immutable) |
| `conversation_history` | Đã có trong `memory_context.short_term` |
| `raw_answer` | Validator removed; conversation đọc `reasoning_output` |
| `voice_path` | TTS chuyển sang FastAPI layer, đọc persona trực tiếp |
| `motion_task_id`, `speech_task_id`, `motion_pending`, `motion_payload` | Dispatch removed |
| `retrieval_results`, `retrieval_metadata` | Nằm trong `messages` (ToolMessage) |

### Truy cập configurable từ node

```python
async def some_node(state: AgentState, config) -> dict:
    user_id = config["configurable"]["user_id"]
    query   = config["configurable"]["query"]
    ...
```

### Acceptance

- [ ] `from langgraph_agents.state import AgentState, ErrorSeverity` chạy được
- [ ] `messages` field có `add_messages` reducer
- [ ] `total_tokens` field có `operator.add` reducer
- [ ] Không còn import `raw_answer` hay `motion_*` ở bất kỳ file nào → `grep -r "raw_answer\|motion_pending\|motion_payload\|voice_path" agenticRAG/agentic_rag_gemini/langgraph_agents/` rỗng (trừ file sắp xóa)

---

## 3. Task 2.5.2 — Dependencies + LangChain ChatModel adapter

### 3.1 Update `requirements-langgraph.txt`

Thêm:
```
langchain-google-genai>=2.0.0
langchain-mcp-adapters>=0.1.0    # cho Phase 3, install sẵn
mcp>=1.0.0
pydantic>=2.0
```

Có thể xóa `anthropic>=0.34.0` (chưa dùng, để lại cũng được — không phá).

### 3.2 Tạo helper `langgraph_agents/llm.py`

**Không thay `llm_gateway.py` ngay** — sẽ xóa ở Task 2.5.13 sau khi mọi node migrate xong.

**File mới**: `langgraph_agents/llm.py`

```python
"""Thin factory for LangChain ChatModels.

Centralizes model selection per node role. Replaces llm_gateway.py.
Tất cả node import từ đây thay vì khởi tạo ChatModel rải rác.
"""

import os
from functools import lru_cache
from langchain_google_genai import ChatGoogleGenerativeAI


_DEFAULT_MODELS = {
    "planner":      "gemini-2.5-flash",
    "synthesizer":  "gemini-2.5-flash",   # đổi sang Claude khi sẵn sàng
    "conversation": "gemini-2.5-flash",
    "retriever":    "gemini-2.5-flash",
}

_DEFAULT_TEMPS = {
    "planner": 0.0,
    "synthesizer": 0.7,
    "conversation": 0.7,
    "retriever": 0.0,
}


@lru_cache(maxsize=8)
def get_chat_model(role: str, *, temperature: float | None = None):
    """Return a LangChain ChatModel for the given node role.

    Cached so each role shares one client instance.
    """
    model_name = _DEFAULT_MODELS.get(role, "gemini-2.5-flash")
    temp = _DEFAULT_TEMPS.get(role, 0.7) if temperature is None else temperature
    api_key = os.getenv("GEMINI_API_KEY") or os.getenv("GOOGLE_API_KEY")
    return ChatGoogleGenerativeAI(
        model=model_name,
        temperature=temp,
        google_api_key=api_key,
    )
```

### Acceptance

- [ ] `pip install -r requirements-langgraph.txt` thành công
- [ ] `from langgraph_agents.llm import get_chat_model; m = get_chat_model("planner")` không lỗi
- [ ] Gọi thử `m.invoke([("user", "ping")])` trả về AIMessage

---

## 4. Task 2.5.3 — RENAME `manager.py` → `planner.py`

**File cũ**: [agenticRAG/agentic_rag_gemini/langgraph_agents/nodes/manager.py](agenticRAG/agentic_rag_gemini/langgraph_agents/nodes/manager.py)
**File mới**: `langgraph_agents/nodes/planner.py`

### Bước 1: Tạo `PlanOutput` Pydantic

Trong `planner.py`:

```python
from typing import Optional
from pydantic import BaseModel, Field


class PlanOutput(BaseModel):
    intent: str = Field(description="conversation | knowledge_query | exercise_recommendation | visualize_motion | clarify")
    confidence: float = Field(ge=0.0, le=1.0)
    expanded_query: str
    needs_clarification: bool = False
    clarification_question: Optional[str] = None
    required_outputs: list[str] = Field(default_factory=list)
    search_strategy: list[str] = Field(default_factory=list)
    constraints_detected: list[str] = Field(default_factory=list)
    notes: Optional[str] = None
```

### Bước 2: Planner node dùng `with_structured_output()`

```python
import time
from datetime import datetime, timezone

from langgraph_agents.state import AgentState, ErrorSeverity
from langgraph_agents.llm import get_chat_model


_PLANNER_SYSTEM_PROMPT = """You are the planning brain for a physical therapy AI assistant.

Analyze the user query + memory context and produce a structured plan.

Intents:
- conversation            : greetings, follow-ups, no exercise content
- knowledge_query         : explanation, facts, non-motion advice
- exercise_recommendation : exercises, stretches, workouts
- visualize_motion        : asks to SEE / animate a specific movement
- clarify                 : query too vague to route

Required outputs per intent:
- knowledge_query         : ["answer", "sources"]
- exercise_recommendation : ["exercise_name", "description", "sets_reps", "safety_warnings"]
- visualize_motion        : ["motion_description", "joint_constraints"]
- conversation            : ["greeting_response"]
- clarify                 : ["clarification_question"]

Search strategy (suggest tools for retriever):
- "pgvector_search"           : internal knowledge base
- "web_search_if_low_quality" : fallback to web
- "generate_motion"           : Kimodo motion synthesis (visualize_motion only)

Rules:
- confidence < 0.5 → needs_clarification = true + provide clarification_question
- Detect missing critical info (e.g. exercise without body region) → needs_clarification
- expanded_query: add anatomical/physiotherapy synonyms
- For greetings, set intent=conversation, required_outputs=["greeting_response"], search_strategy=[]
"""

_VALID_INTENTS = {"conversation", "knowledge_query", "exercise_recommendation", "visualize_motion", "clarify"}


async def planner_node(state: AgentState, config) -> dict:
    """Intent classification + query expansion + structured plan output."""
    llm = get_chat_model("planner")
    structured_llm = llm.with_structured_output(PlanOutput)

    query = config["configurable"]["query"]
    memory = state.get("memory_context", {})

    # Build context snippet from memory
    stm = memory.get("short_term") or []
    history_snippet = ""
    if stm:
        history_snippet = "\n\nRecent Q&A:\n" + "\n".join(
            f"Q: {p['q']}\nA: {p['a']}" for p in stm[-3:]
        )

    profile = memory.get("user_profile") or {}
    profile_snippet = f"\n\nUser profile: {profile}" if profile else ""

    ltm = memory.get("long_term") or {}
    ltm_snippet = ""
    if ltm.get("ambiguous"):
        ltm_snippet = "\n\nNote: Multiple past sessions matched recall — ask user for clarification."
    elif ltm.get("results"):
        ltm_snippet = "\n\nRelevant past context found in memory."

    user_msg = query + history_snippet + profile_snippet + ltm_snippet

    t0 = time.perf_counter()
    try:
        plan: PlanOutput = await structured_llm.ainvoke([
            ("system", _PLANNER_SYSTEM_PROMPT),
            ("user", user_msg),
        ])
        elapsed_ms = (time.perf_counter() - t0) * 1000
    except Exception as exc:
        elapsed_ms = (time.perf_counter() - t0) * 1000
        return {
            "intent": "clarify",
            "confidence": 0.3,
            "expanded_query": query,
            "plan": {},
            "needs_clarification": True,
            "errors": [{
                "node": "planner",
                "severity": ErrorSeverity.RECOVERABLE,
                "message": f"LLM call failed ({elapsed_ms:.0f}ms): {exc}",
                "timestamp": datetime.now(timezone.utc).isoformat(),
            }],
        }

    # Validate intent
    intent = plan.intent if plan.intent in _VALID_INTENTS else "clarify"

    return {
        "intent": intent,
        "confidence": plan.confidence,
        "expanded_query": plan.expanded_query or query,
        "plan": plan.model_dump(),
        "needs_clarification": plan.needs_clarification,
    }
```

### Bước 3: Xóa `manager.py` sau khi `planner.py` chạy được

### Acceptance

- [ ] `planner.py` tồn tại, `manager.py` đã xóa
- [ ] Test: planner trả `plan.required_outputs` không rỗng cho intent `exercise_recommendation`
- [ ] Test: query mơ hồ → `needs_clarification=True` + `clarification_question` không rỗng
- [ ] LLM fail → fallback `intent="clarify"` + `needs_clarification=True` + RECOVERABLE error

---

## 5. Task 2.5.4 — UPDATE `memory.py` — STM 3 Q&A + Conditional LTM

**File**: [agenticRAG/agentic_rag_gemini/langgraph_agents/nodes/memory.py](agenticRAG/agentic_rag_gemini/langgraph_agents/nodes/memory.py)

### Thay đổi chính

| Cũ | Mới |
|----|-----|
| LTM luôn chạy (pgvector search mọi query) | LTM chỉ chạy khi detect recall keyword |
| `short_term` = session buffer (định dạng tự do) | `short_term` = list 3 Q&A pairs, FIFO |
| Trả `{long_term: [...], short_term: {...}, user_profile: {...}}` | Cùng schema nhưng `long_term` có cấu trúc: `{found, results?, ambiguous?, sessions?}` |

### Bộ keyword detection (module-level)

```python
import re

_RECALL_PATTERNS = [
    re.compile(r"\bcon\s+nho\b", re.IGNORECASE),
    re.compile(r"\bnho(\s+lai)?\b", re.IGNORECASE),
    re.compile(r"\blan\s+truoc\b", re.IGNORECASE),
    re.compile(r"\btruoc\s+do\b", re.IGNORECASE),
    re.compile(r"\b(tuan|hom\s+qua|thang)\s+truoc\b", re.IGNORECASE),
    re.compile(r"\bhom\s+qua\b", re.IGNORECASE),
    re.compile(r"\bda\s+(noi|hoi|trao\s+doi|lam)\b", re.IGNORECASE),
    re.compile(r"\b(remember|last\s+time|previously)\b", re.IGNORECASE),
]


def _needs_recall(query: str) -> bool:
    return any(p.search(query) for p in _RECALL_PATTERNS)
```

### STM read (Redis, max 3 Q&A)

```python
import json
import redis.asyncio as aioredis

_STM_KEY = "stm:{session_id}"
_STM_MAX = 3


async def _read_stm(session_id: str) -> list[dict]:
    """Returns list of up to 3 Q&A pairs, oldest first."""
    r = aioredis.from_url("redis://localhost:6379")
    try:
        raw = await r.get(_STM_KEY.format(session_id=session_id))
        if not raw:
            return []
        return json.loads(raw)[-_STM_MAX:]
    except Exception:
        return []
    finally:
        await r.aclose()
```

> **Lưu ý**: STM **ghi** xảy ra ở FastAPI layer sau khi graph hoàn thành (xem Plan v2.4 §4.4). Node `memory` **chỉ đọc**.

### LTM lookup (PostgreSQL + pgvector, conditional)

Logic 3 nhánh theo Plan §4.2:

```python
async def _lookup_ltm(user_id: str, query: str, expanded_query: str) -> dict:
    """Run only if _needs_recall(query). Returns one of:
      {found: False}
      {found: True, results: [...]}        # 1 session matched
      {ambiguous: True, sessions: [...]}    # 2+ sessions
    """
    pg = get_pg_client()
    # 1. Find candidate sessions by timestamp (calendar week, ±3 days expansion)
    rows = await pg.fetch(
        """
        SELECT session_id, summary, created_at
        FROM conversations
        WHERE user_id = $1
          AND created_at >= now() - interval '14 days'
        ORDER BY created_at DESC
        LIMIT 5
        """,
        user_id,
    )

    if not rows:
        return {"found": False}

    if len(rows) >= 2:
        # 2+ sessions matched → ambiguous
        return {
            "ambiguous": True,
            "sessions": [
                {"session_id": str(r["session_id"]), "summary": r["summary"], "created_at": r["created_at"].isoformat()}
                for r in rows
            ],
        }

    # Exactly 1 session — pgvector search within that session
    target_session = rows[0]["session_id"]
    svc = get_embedding_service()
    import asyncio
    embedding = await asyncio.to_thread(svc.embed_texts, expanded_query)

    vb = VectorBackend(pg)
    results = await vb.search(
        query_embedding=embedding[0] if isinstance(embedding[0], list) else embedding,
        top_k=5,
        source_type="conversation",
        # TODO: filter by source_id = target_session (add filter to VectorBackend.search)
    )

    return {"found": True, "results": results}
```

### Memory node — orchestrator

```python
async def memory_node(state: AgentState, config) -> dict:
    user_id = config["configurable"]["user_id"]
    session_id = config["configurable"]["session_id"]
    query = config["configurable"]["query"]
    expanded = state.get("expanded_query") or query   # planner chạy SAU memory → expanded chưa có lần đầu

    # 1. STM (always)
    short_term = await _read_stm(session_id)

    # 2. LTM (conditional)
    if _needs_recall(query):
        long_term = await _lookup_ltm(user_id, query, query)
    else:
        long_term = {"found": False, "skipped": True}

    # 3. User profile
    user_profile = await _get_user_profile(user_id)

    return {
        "memory_context": {
            "short_term": short_term,
            "long_term": long_term,
            "user_profile": user_profile,
        }
    }
```

> **Note quan trọng**: Memory chạy **TRƯỚC** planner → `expanded_query` chưa tồn tại. LTM dùng `query` thô. Acceptable trade-off (Plan §4.2).

### VectorBackend cần update (thêm `source_id` filter)

**File**: [agenticRAG/agentic_rag_gemini/langgraph_agents/db/vector_backend.py](agenticRAG/agentic_rag_gemini/langgraph_agents/db/vector_backend.py)

Thêm tham số `source_id: Optional[UUID]` cho `search()`. SQL: `AND ($4::uuid IS NULL OR source_id = $4)`.

### Acceptance

- [ ] `_needs_recall("Xin chào")` → False, `_needs_recall("ban con nho bai tap tuan truoc khong")` → True
- [ ] Memory với query thường → `long_term = {"found": False, "skipped": True}`, không query PostgreSQL
- [ ] Memory với recall keyword + 0 session → `long_term = {"found": False}`
- [ ] Memory với recall + 2+ session → `long_term = {"ambiguous": True, "sessions": [...]}`
- [ ] STM trả tối đa 3 Q&A, định dạng `[{"q": ..., "a": ..., "ts": ...}]`

---

## 6. Task 2.5.5 — CREATE `tools/pgvector_tool.py` — @tool wrapper

**File mới**: `langgraph_agents/tools/pgvector_tool.py` (tạo folder `tools/` + `__init__.py`)

### Code

```python
"""In-process @tool wrappers cho retriever agent.

pgvector KHÔNG phải MCP server — chạy trực tiếp cùng process, 0ms network.
Chỉ Kimodo + web_search là MCP (Phase 3).
"""

import asyncio
from langchain_core.tools import tool

from langgraph_agents.shared import get_pg_client, get_embedding_service
from langgraph_agents.db.vector_backend import VectorBackend


@tool
async def pgvector_search(query: str, top_k: int = 5, source_type: str = "document") -> list[dict]:
    """Search internal medical knowledge base for exercises, treatments, and PT theory.

    Use for knowledge_query and exercise_recommendation intents.
    Returns documents ranked by cosine similarity (highest first).

    Args:
        query: Semantic search query (use expanded_query from planner)
        top_k: Number of results to return (default 5)
        source_type: One of "document", "humanml3d" (default "document")
    """
    pg = get_pg_client()
    svc = get_embedding_service()

    # Embedding service is sync → wrap
    embedding = await asyncio.to_thread(svc.embed_texts, query)
    if isinstance(embedding[0], list):
        embedding = embedding[0]

    vb = VectorBackend(pg)
    results = await vb.search(
        query_embedding=embedding,
        top_k=top_k,
        source_type=source_type,
    )
    # Compact result for LLM consumption
    return [
        {
            "content": r["content"],
            "similarity": round(r["similarity"], 3),
            "source_type": r.get("metadata", {}).get("source_type", source_type),
        }
        for r in results
    ]
```

### Acceptance

- [ ] `from langgraph_agents.tools.pgvector_tool import pgvector_search` chạy
- [ ] `pgvector_search.name == "pgvector_search"`, `pgvector_search.description` non-empty
- [ ] Test: gọi tool với query thật → trả list dict (có thể rỗng nếu DB rỗng, miễn không crash)

---

## 7. Task 2.5.6 — REPLACE `retrieval.py` → `retriever_agent.py`

**File cũ**: [agenticRAG/agentic_rag_gemini/langgraph_agents/nodes/retrieval.py](agenticRAG/agentic_rag_gemini/langgraph_agents/nodes/retrieval.py)
**File mới**: `langgraph_agents/nodes/retriever_agent.py`

### Pattern: LLM + ToolNode

Phase 2.5 **chỉ wire `pgvector_search`** (in-process). MCP tools (Kimodo, web_search) **stub ra** — sẽ thêm ở Phase 3.

### Code

```python
"""Retriever Agent — executes planner's plan via tool calls.

Pattern: ChatModel.bind_tools() → invoke → if tool_calls → ToolNode → loop
Phase 2.5: only pgvector_search. MCP tools added in Phase 3.
"""

from datetime import datetime, timezone
from langchain_core.messages import SystemMessage, HumanMessage, ToolMessage

from langgraph_agents.state import AgentState, ErrorSeverity
from langgraph_agents.llm import get_chat_model
from langgraph_agents.tools.pgvector_tool import pgvector_search


_RETRIEVER_TOOLS = [pgvector_search]   # Phase 3: extend with MCP tools


_RETRIEVER_SYSTEM_PROMPT = """You are a tool execution agent. Execute the plan from the planner.

Rules:
- For knowledge_query and exercise_recommendation: ALWAYS call pgvector_search first
- Use the planner's expanded_query as the search query
- You may call tools in parallel via multiple tool_calls in one response
- Do NOT generate the final answer — only retrieve evidence and let synthesizer compose it
- If a tool fails, note in your response and continue

{retry_note}

Plan from planner:
{plan}
"""


async def retriever_agent_node(state: AgentState, config) -> dict:
    """Run one tool-calling step. ToolNode handles execution."""
    plan = state.get("plan", {})
    expanded = state.get("expanded_query") or config["configurable"]["query"]

    retry_note = ""
    feedback = state.get("grader_feedback")
    if feedback:
        retry_note = f"## Retry context\nPrevious attempt was rejected. Grader feedback:\n{feedback}\nTry a different query or call additional tools."

    system = _RETRIEVER_SYSTEM_PROMPT.format(
        retry_note=retry_note,
        plan=plan,
    )

    llm = get_chat_model("retriever").bind_tools(_RETRIEVER_TOOLS)

    try:
        ai_msg = await llm.ainvoke([
            SystemMessage(content=system),
            HumanMessage(content=f"Execute plan for query: {expanded}"),
        ])
    except Exception as exc:
        return {
            "errors": [{
                "node": "retriever_agent",
                "severity": ErrorSeverity.CRITICAL,
                "message": f"Retriever LLM failed: {exc}",
                "timestamp": datetime.now(timezone.utc).isoformat(),
            }],
        }

    return {
        "messages": [ai_msg],
        "total_tokens": (ai_msg.usage_metadata or {}).get("total_tokens", 0) if hasattr(ai_msg, "usage_metadata") else 0,
    }
```

### ToolNode wiring trong `graph.py` (preview)

```python
from langgraph.prebuilt import ToolNode, tools_condition

tool_node = ToolNode(_RETRIEVER_TOOLS)
graph.add_node("retriever_agent", retriever_agent_node)
graph.add_node("tools", tool_node)

# Conditional: nếu LLM yêu cầu tool call → "tools", else → synthesizer
graph.add_conditional_edges("retriever_agent", tools_condition, {
    "tools": "tools",
    "__end__": "synthesizer",   # no more tool calls → synthesize
})
graph.add_edge("tools", "retriever_agent")   # loop back for follow-up tool calls
```

### Acceptance

- [ ] `retriever_agent_node` trả `{"messages": [AIMessage]}` (có thể có hoặc không tool_calls)
- [ ] Khi state có `grader_feedback` → retry_note xuất hiện trong system prompt (verify bằng mock)
- [ ] Integration test: query exercise → LLM gọi `pgvector_search` → ToolMessage append vào `messages`

---

## 8. Task 2.5.7 — RENAME `reasoning.py` → `synthesizer.py`

**File cũ**: [agenticRAG/agentic_rag_gemini/langgraph_agents/nodes/reasoning.py](agenticRAG/agentic_rag_gemini/langgraph_agents/nodes/reasoning.py)
**File mới**: `langgraph_agents/nodes/synthesizer.py`

### Thay đổi chính

| Cũ | Mới |
|----|-----|
| Đọc `retrieval_results` từ state | Đọc `messages` (tìm `ToolMessage`) |
| Đọc `conversation_history` | Đọc `memory_context.short_term` |
| Dùng `LLMGateway` | Dùng `get_chat_model("synthesizer")` |
| Không có `plan` | Đọc `plan` từ state để biết `required_outputs` |
| Không track token | Trả `total_tokens` từ `usage_metadata` |

### Code skeleton

```python
import time
from datetime import datetime, timezone
from langchain_core.messages import SystemMessage, HumanMessage, ToolMessage

from langgraph_agents.state import AgentState, ErrorSeverity
from langgraph_agents.llm import get_chat_model


def _extract_tool_results(messages: list) -> str:
    """Format ToolMessage content from retriever_agent's tool calls."""
    parts = []
    for i, m in enumerate(messages, 1):
        if isinstance(m, ToolMessage):
            parts.append(f"[Tool {i}: {m.name}]\n{m.content}")
    return "\n\n".join(parts) if parts else "No tool results."


def _format_memory(memory: dict) -> str:
    parts = []
    stm = memory.get("short_term") or []
    if stm:
        parts.append("Recent Q&A:\n" + "\n".join(f"- Q: {p['q']}\n  A: {p['a']}" for p in stm))
    profile = memory.get("user_profile") or {}
    if profile:
        parts.append(f"User profile: {profile}")
    return "\n\n".join(parts) if parts else "No memory."


_SYNTH_SYSTEM_PROMPT = """You are an expert physical therapist AI assistant.

## Plan requirements
{required_outputs}
{constraints}
{notes}

## Retrieved evidence
{tool_results}

## Patient memory
{memory}

Instructions:
- Cover ALL required_outputs from the plan
- Use Vietnamese if user query is in Vietnamese
- Include safety warnings for exercise recommendations
- Cite sources when available
- Keep under 500 words unless topic requires detail
"""


async def synthesizer_node(state: AgentState, config) -> dict:
    plan = state.get("plan", {})
    messages = state.get("messages", [])
    memory = state.get("memory_context", {})
    query = config["configurable"]["query"]

    tool_results = _extract_tool_results(messages)
    memory_str = _format_memory(memory)

    system = _SYNTH_SYSTEM_PROMPT.format(
        required_outputs="Required: " + ", ".join(plan.get("required_outputs", [])),
        constraints=("Constraints: " + ", ".join(plan.get("constraints_detected", []))) if plan.get("constraints_detected") else "",
        notes=("Notes: " + plan["notes"]) if plan.get("notes") else "",
        tool_results=tool_results,
        memory=memory_str,
    )

    llm = get_chat_model("synthesizer")

    t0 = time.perf_counter()
    try:
        ai_msg = await llm.ainvoke([
            SystemMessage(content=system),
            HumanMessage(content=query),
        ])
    except Exception as exc:
        elapsed_ms = (time.perf_counter() - t0) * 1000
        return {
            "reasoning_output": "",
            "errors": [{
                "node": "synthesizer",
                "severity": ErrorSeverity.CRITICAL,
                "message": f"Synthesizer LLM failed ({elapsed_ms:.0f}ms): {exc}",
                "timestamp": datetime.now(timezone.utc).isoformat(),
            }],
        }

    tokens = (ai_msg.usage_metadata or {}).get("total_tokens", 0) if hasattr(ai_msg, "usage_metadata") else 0

    return {
        "reasoning_output": ai_msg.content,
        "total_tokens": tokens,
    }
```

### Acceptance

- [ ] `synthesizer.py` tồn tại, `reasoning.py` đã xóa
- [ ] Test: state có 2 ToolMessage → `_extract_tool_results` trả 2 block "[Tool N: name]"
- [ ] Test: LLM fail → reasoning_output rỗng + CRITICAL error
- [ ] Test: state có plan với required_outputs → system prompt chứa các string đó

---

## 9. Task 2.5.8 — REPLACE `validator.py` → `grader.py`

**File cũ**: [agenticRAG/agentic_rag_gemini/langgraph_agents/nodes/validator.py](agenticRAG/agentic_rag_gemini/langgraph_agents/nodes/validator.py)
**File mới**: `langgraph_agents/nodes/grader.py`

### Logic

Rule-based, không gọi LLM. Retry max 1 lần (Plan §8).

### Code

```python
from langgraph_agents.state import AgentState


_GENERIC_FAIL_FEEDBACK = "Câu trả lời quá ngắn hoặc thiếu nội dung. Hãy tìm thêm evidence và viết chi tiết hơn."
_EXERCISE_FAIL_FEEDBACK = "Câu trả lời thiếu các bước thực hiện cụ thể (sets, reps, hoặc steps). Hãy bổ sung hướng dẫn động tác."
_MOTION_FAIL_FEEDBACK = "Chưa thấy kết quả từ generate_motion tool. Hãy gọi tool đó."

_WARNING_MSG = "Câu trả lời có thể chưa đầy đủ. Vui lòng tham khảo bác sĩ vật lý trị liệu."


def _check_rules(state: dict) -> tuple[bool, str | None]:
    """Run rule checks. Returns (pass: bool, fail_feedback: str|None)."""
    reasoning = (state.get("reasoning_output") or "").strip()
    intent = state.get("intent", "conversation")
    messages = state.get("messages", [])

    # Rule 1: non-empty
    if not reasoning:
        return False, _GENERIC_FAIL_FEEDBACK

    # Rule 2: under word limit
    if len(reasoning.split()) > 500:
        # Not a hard failure — just truncate later. For now, pass.
        pass

    # Rule 3: intent-specific
    if intent == "knowledge_query":
        if len(reasoning) < 50:
            return False, _GENERIC_FAIL_FEEDBACK

    if intent == "exercise_recommendation":
        markers = ["1.", "2.", "-", "bước", "lần", "hiệp", "sets", "reps"]
        if not any(m in reasoning for m in markers):
            return False, _EXERCISE_FAIL_FEEDBACK

    if intent == "visualize_motion":
        # Must have called generate_motion tool
        from langchain_core.messages import ToolMessage
        has_motion = any(
            isinstance(m, ToolMessage) and m.name == "generate_motion"
            for m in messages
        )
        if not has_motion:
            return False, _MOTION_FAIL_FEEDBACK

    return True, None


async def grader_node(state: AgentState) -> dict:
    """Rule-based quality check. Retry max 1 time."""
    passed, feedback = _check_rules(state)
    retry_count = state.get("retry_count", 0)

    if passed:
        return {"grader_result": "pass"}

    if retry_count == 0:
        return {
            "grader_result": "retry",
            "retry_count": 1,
            "grader_feedback": feedback,
        }

    # retry_count >= 1 — fail-safe pass with warning
    return {
        "grader_result": "pass_with_warning",
        "grader_warning": _WARNING_MSG,
    }
```

### Acceptance

- [ ] Exercise rec có "Bridge: 3 hiệp 10 lần" → `pass`
- [ ] Exercise rec rỗng/không có markers → `retry` lần đầu, `pass_with_warning` lần 2
- [ ] visualize_motion mà `messages` không có ToolMessage `generate_motion` → `retry`
- [ ] `retry_count` được tăng đúng (state reducer hoặc node tự ghi)

---

## 10. Task 2.5.9 — UPDATE `conversation.py` — Dual mode

**File**: [agenticRAG/agentic_rag_gemini/langgraph_agents/nodes/conversation.py](agenticRAG/agentic_rag_gemini/langgraph_agents/nodes/conversation.py)

### Logic chính

```
if reasoning_output non-empty AND not needs_clarification:
    STYLING MODE — restyle reasoning_output theo persona
else:
    GENERATION MODE — generate từ persona + query + plan + memory
```

### Code skeleton

```python
from datetime import datetime, timezone
from langchain_core.messages import SystemMessage, HumanMessage

from langgraph_agents.state import AgentState, ErrorSeverity
from langgraph_agents.llm import get_chat_model


# Persona loader giữ nguyên từ Phase 4 — chỉ thay LLM call.
from langgraph_agents.nodes._persona_loader import get_persona, build_persona_prompt


_STYLING_INSTRUCTION = """Restyle the following clinical response to match your personality and formatting rules.
- Do NOT add new medical information — only restyle.
- Preserve all safety warnings (rephrase in your tone).
- Respond in the same language as the input."""

_GENERATION_INSTRUCTION_CONVERSATION = """Respond naturally to the user's greeting or casual message in your persona.
- Stay in character.
- Be concise (under 50 words for greetings)."""

_GENERATION_INSTRUCTION_CLARIFY = """The planner detected that the user's query needs clarification.
Style the clarification question naturally in your persona.

Clarification question: {question}"""

_GENERATION_INSTRUCTION_FALLBACK = """The system could not produce a clinical response. Respond to the user politely in character,
explaining you need more information or that you cannot help with this specific query right now."""


async def conversation_node(state: AgentState, config) -> dict:
    persona_id = config["configurable"].get("persona_id", "eca_default")
    persona = get_persona(persona_id)

    reasoning = (state.get("reasoning_output") or "").strip()
    intent = state.get("intent", "conversation")
    needs_clarification = state.get("needs_clarification", False)
    grader_warning = state.get("grader_warning")
    query = config["configurable"].get("query", "")
    plan = state.get("plan", {})

    persona_system = build_persona_prompt(persona, intent)

    # Mode selection
    if reasoning and not needs_clarification:
        # STYLING MODE
        user_msg = f"{_STYLING_INSTRUCTION}\n\n---\n{reasoning}"
    elif needs_clarification and plan.get("clarification_question"):
        # GENERATION MODE — clarify
        user_msg = _GENERATION_INSTRUCTION_CLARIFY.format(question=plan["clarification_question"])
    elif intent == "conversation":
        # GENERATION MODE — greeting/chat
        user_msg = f"{_GENERATION_INSTRUCTION_CONVERSATION}\n\nUser said: {query}"
    else:
        # GENERATION MODE — fallback (empty reasoning, no clarification question)
        user_msg = _GENERATION_INSTRUCTION_FALLBACK

    llm = get_chat_model("conversation")

    try:
        ai_msg = await llm.ainvoke([
            SystemMessage(content=persona_system),
            HumanMessage(content=user_msg),
        ])
        final = ai_msg.content
    except Exception as exc:
        # Fallback: return reasoning as-is if styling failed
        return {
            "final_answer": reasoning or "Xin lỗi, tôi không thể trả lời lúc này.",
            "errors": [{
                "node": "conversation",
                "severity": ErrorSeverity.RECOVERABLE,
                "message": f"Persona styling failed: {exc}",
                "timestamp": datetime.now(timezone.utc).isoformat(),
            }],
        }

    # Append warning if grader flagged it
    if grader_warning:
        final = f"{final}\n\n_{grader_warning}_"

    tokens = (ai_msg.usage_metadata or {}).get("total_tokens", 0) if hasattr(ai_msg, "usage_metadata") else 0

    return {
        "final_answer": final,
        "total_tokens": tokens,
    }
```

### Persona loader — di chuyển ra file riêng

Phase 4 đã có persona loader trong `conversation.py`. Tách ra `nodes/_persona_loader.py` để conversation_node sạch hơn. Giữ nguyên logic parse MD + cache.

### Voice path — XÓA HOÀN TOÀN

Field `voice_path` không còn trong state. Persona MD vẫn có `## Voice Identity`, nhưng FastAPI layer (Phase 5) đọc trực tiếp file persona để truyền cho TTS — node không pass voice_path nữa.

### Acceptance

- [ ] reasoning_output non-empty + not needs_clarification → styling mode
- [ ] needs_clarification + plan.clarification_question có → clarify generation
- [ ] intent="conversation" + reasoning rỗng → greeting generation
- [ ] grader_warning có → append vào final_answer
- [ ] LLM fail → fallback to reasoning (hoặc generic message nếu reasoning cũng rỗng), RECOVERABLE error
- [ ] Không còn return `voice_path` trong dict

---

## 11. Task 2.5.10 — UPDATE `error_handler.py`

**File**: [agenticRAG/agentic_rag_gemini/langgraph_agents/nodes/error_handler.py](agenticRAG/agentic_rag_gemini/langgraph_agents/nodes/error_handler.py)

### Thay đổi duy nhất

Ghi `reasoning_output` thay vì `raw_answer`. Conversation node sau đó sẽ style nó.

```python
from langgraph_agents.state import AgentState, ErrorSeverity


async def error_handler_node(state: AgentState) -> dict:
    errors = state.get("errors", [])
    critical = [e for e in errors if e.get("severity") == ErrorSeverity.CRITICAL]

    if critical:
        msg = "Xin lỗi, hệ thống đang gặp sự cố. Vui lòng thử lại sau."
    else:
        msg = "Đã có lỗi nhỏ, nhưng tôi vẫn cố gắng trả lời."

    return {"reasoning_output": msg}
```

### Acceptance

- [ ] Không còn return `raw_answer`
- [ ] Test: CRITICAL error → reasoning_output = "Xin lỗi..."
- [ ] Test: chỉ RECOVERABLE → reasoning_output = "Đã có lỗi nhỏ..."

---

## 12. Task 2.5.11 — UPDATE `routing.py`

**File**: [agenticRAG/agentic_rag_gemini/langgraph_agents/routing.py](agenticRAG/agentic_rag_gemini/langgraph_agents/routing.py)

### Thay routing functions cũ bằng các function mới

```python
from langgraph_agents.state import AgentState, ErrorSeverity


def check_errors(state: AgentState) -> str:
    for err in state.get("errors", []):
        if err.get("severity") == ErrorSeverity.CRITICAL:
            return "error_handler"
    return "continue"


def route_after_memory(state: AgentState) -> str:
    """Memory always goes to planner unless CRITICAL error."""
    if check_errors(state) == "error_handler":
        return "error_handler"
    return "planner"


def route_after_planner(state: AgentState) -> str:
    """Decide path based on intent and needs_clarification."""
    if check_errors(state) == "error_handler":
        return "error_handler"

    if state.get("needs_clarification"):
        return "conversation"

    intent = state.get("intent", "conversation")
    if intent == "conversation":
        return "conversation"      # generation mode
    if intent in ("knowledge_query", "exercise_recommendation", "visualize_motion"):
        return "retriever_agent"
    # clarify or unknown
    return "conversation"


def route_after_grader(state: AgentState) -> str:
    result = state.get("grader_result", "pass")
    if result == "retry":
        return "retriever_agent"
    # pass or pass_with_warning → conversation
    return "conversation"
```

### Xóa các function cũ

- `route_by_intent` (cũ) — replaced by `route_after_planner`
- `route_after_conversation` — không cần, conversation luôn đi END

### Acceptance

- [ ] 4 function trên import OK
- [ ] Unit test cho từng routing function với state giả

---

## 13. Task 2.5.12 — UPDATE `graph.py` — New flow

**File**: [agenticRAG/agentic_rag_gemini/langgraph_agents/graph.py](agenticRAG/agentic_rag_gemini/langgraph_agents/graph.py)

### Flow mới (Plan §2.1)

```
START → memory → planner → [route_after_planner]:
  ├─ conversation  → END
  ├─ retriever_agent → [tools_condition]:
  │     ├─ tools → retriever_agent (loop)
  │     └─ __end__ → synthesizer → grader → [route_after_grader]:
  │           ├─ retriever_agent (retry)
  │           └─ conversation → END
  └─ error_handler → conversation → END
```

### Code

```python
from langgraph.graph import StateGraph, END, START
from langgraph.prebuilt import ToolNode, tools_condition

from langgraph_agents.state import AgentState
from langgraph_agents.nodes.memory import memory_node
from langgraph_agents.nodes.planner import planner_node
from langgraph_agents.nodes.retriever_agent import retriever_agent_node, _RETRIEVER_TOOLS
from langgraph_agents.nodes.synthesizer import synthesizer_node
from langgraph_agents.nodes.grader import grader_node
from langgraph_agents.nodes.conversation import conversation_node
from langgraph_agents.nodes.error_handler import error_handler_node
from langgraph_agents.routing import (
    route_after_memory,
    route_after_planner,
    route_after_grader,
    check_errors,
)


def build_graph():
    g = StateGraph(AgentState)

    # Nodes
    g.add_node("memory", memory_node)
    g.add_node("planner", planner_node)
    g.add_node("retriever_agent", retriever_agent_node)
    g.add_node("tools", ToolNode(_RETRIEVER_TOOLS))
    g.add_node("synthesizer", synthesizer_node)
    g.add_node("grader", grader_node)
    g.add_node("conversation", conversation_node)
    g.add_node("error_handler", error_handler_node)

    # Entry
    g.add_edge(START, "memory")

    # memory → planner OR error_handler
    g.add_conditional_edges("memory", route_after_memory, {
        "planner": "planner",
        "error_handler": "error_handler",
    })

    # planner → conversation | retriever_agent | error_handler
    g.add_conditional_edges("planner", route_after_planner, {
        "conversation": "conversation",
        "retriever_agent": "retriever_agent",
        "error_handler": "error_handler",
    })

    # retriever_agent → tools (loop) OR synthesizer (no more tool calls)
    g.add_conditional_edges("retriever_agent", tools_condition, {
        "tools": "tools",
        "__end__": "synthesizer",
    })
    g.add_edge("tools", "retriever_agent")

    # synthesizer → grader (or error_handler if CRITICAL)
    g.add_conditional_edges("synthesizer", check_errors, {
        "continue": "grader",
        "error_handler": "error_handler",
    })

    # grader → retriever_agent (retry) OR conversation
    g.add_conditional_edges("grader", route_after_grader, {
        "retriever_agent": "retriever_agent",
        "conversation": "conversation",
    })

    # conversation → END
    g.add_edge("conversation", END)

    # error_handler → conversation → END
    g.add_edge("error_handler", "conversation")

    return g.compile()
```

### Acceptance

- [ ] `build_graph()` không raise
- [ ] Smoke: `await graph.ainvoke({...}, config={...})` cho query "Xin chào" → final_answer non-empty (đi qua memory → planner → conversation generation)
- [ ] Smoke: query exercise → đi qua đầy đủ memory → planner → retriever_agent → tools → synthesizer → grader → conversation

---

## 14. Task 2.5.13 — REMOVE deprecated files

Chạy **sau khi** Task 2.5.12 pass smoke test:

```bash
# Trong langgraph_agents/
rm nodes/manager.py
rm nodes/reasoning.py
rm nodes/retrieval.py
rm nodes/validator.py
rm nodes/dispatch.py
rm llm_gateway.py

# Dispatch-related (Phase 3 sẽ rebuild khi cần)
# Giữ lại services/ vì sẽ tái sử dụng cho MCP server wrappers
# rm -r services/   ← KHÔNG xóa; sẽ refactor ở Phase 3
# Giữ lại celery_app.py vì FastAPI TTS task sẽ cần
# Giữ lại streaming/approval.py — có thể dùng cho token budget approval (Plan §11.4)
```

### Kiểm tra không còn reference

```bash
grep -rn "from langgraph_agents.nodes.manager\|from langgraph_agents.nodes.reasoning\|from langgraph_agents.nodes.retrieval\|from langgraph_agents.nodes.validator\|from langgraph_agents.nodes.dispatch\|from langgraph_agents.llm_gateway" .
```

Kết quả phải **rỗng** (trừ test file đã update).

### Acceptance

- [ ] 6 file đã xóa
- [ ] `pytest tests/langgraph_agents/ -x --co -q` collect không lỗi import

---

## 15. Task 2.5.14 — UPDATE Tests

**File**: [tests/langgraph_agents/test_phase2_integration.py](tests/langgraph_agents/test_phase2_integration.py) — refactor để dùng node names mới

### Tests cần thay đổi

| Test cũ | Thay bằng |
|---------|-----------|
| `test_format_retrieval_context` | `test_extract_tool_results` (từ synthesizer) |
| `test_validator_fallback_*` (4 tests) | XÓA — replaced bằng grader tests |
| `test_validator_passes_reasoning` | XÓA |
| `test_quality_assessment` | XÓA (logic moved to grader rules, không còn quality scoring) |
| `test_retrieval_graceful_degradation` | `test_retriever_agent_llm_fail` |
| `test_reasoning_*` | `test_synthesizer_*` |
| `test_reasoning_critical_error_on_bad_model` | `test_synthesizer_critical_error_on_bad_model` (dùng `langgraph_agents.llm` thay vì `llm_gateway`) |
| `test_full_graph_*` | Giữ nhưng update state shape (config + state mới) |

### Test mới cần viết

| Test | File |
|------|------|
| `test_planner_structured_output_exercise` | `test_phase2_5_planner.py` |
| `test_planner_clarification_low_confidence` | `test_phase2_5_planner.py` |
| `test_planner_llm_fail_returns_clarify` | `test_phase2_5_planner.py` |
| `test_memory_no_recall_skips_ltm` | `test_phase2_5_memory.py` |
| `test_memory_recall_keyword_triggers_ltm` | `test_phase2_5_memory.py` |
| `test_memory_ambiguous_multi_session` | `test_phase2_5_memory.py` |
| `test_pgvector_tool_invocation` | `test_phase2_5_tools.py` |
| `test_retriever_agent_no_grader_feedback` | `test_phase2_5_retriever.py` |
| `test_retriever_agent_with_grader_feedback` | `test_phase2_5_retriever.py` |
| `test_grader_exercise_pass` | `test_phase2_5_grader.py` |
| `test_grader_exercise_retry_then_warning` | `test_phase2_5_grader.py` |
| `test_grader_motion_missing_tool_call` | `test_phase2_5_grader.py` |
| `test_conversation_styling_mode` | `test_phase2_5_conversation.py` |
| `test_conversation_generation_greeting` | `test_phase2_5_conversation.py` |
| `test_conversation_generation_clarification` | `test_phase2_5_conversation.py` |
| `test_conversation_warning_appended` | `test_phase2_5_conversation.py` |
| `test_error_handler_writes_reasoning_output` | `test_phase2_5_error.py` |
| `test_full_graph_v24_conversation_path` | `test_phase2_5_integration.py` |
| `test_full_graph_v24_exercise_full_pipeline` | `test_phase2_5_integration.py` |
| `test_full_graph_v24_grader_retry_loop` | `test_phase2_5_integration.py` |

### Config + state shape mới cho tests

Helper update:

```python
def _base_invoke_args(**overrides):
    """Returns (initial_state, config) for graph.ainvoke."""
    state = {
        "messages": [],
        "errors": [],
        "retry_count": 0,
        "total_tokens": 0,
    }
    config = {
        "configurable": {
            "user_id": "test-user",
            "session_id": "test-session",
            "query": overrides.pop("query", "Xin chào"),
            "persona_id": "eca_default",
            "output_mode": "text",
            "request_id": "test-001",
            "token_limit": None,
        }
    }
    state.update(overrides)
    return state, config


# Usage:
state, config = _base_invoke_args(query="bài tập cho đau lưng")
result = await graph.ainvoke(state, config=config)
```

### Acceptance

- [ ] Old test file giữ nguyên hoặc xóa hẳn (tùy N), không có test nào còn reference `raw_answer`, `validator_node`, `retrieval_node`, `reasoning_node`, `dispatch_node`
- [ ] Unit suite (no API key): `pytest tests/langgraph_agents/ -m unit -v` xanh hết
- [ ] Integration suite (cần GEMINI_API_KEY): `pytest tests/langgraph_agents/ -m integration -v` xanh hết
- [ ] Full graph test cho 3 path: conversation / knowledge / exercise — tất cả trả `final_answer` non-empty

---

## 16. Files Touched Summary

### Created (mới)

| File | Mục đích |
|------|---------|
| `langgraph_agents/llm.py` | LangChain ChatModel factory |
| `langgraph_agents/nodes/planner.py` | (rename từ manager.py) |
| `langgraph_agents/nodes/synthesizer.py` | (rename từ reasoning.py) |
| `langgraph_agents/nodes/retriever_agent.py` | LLM + ToolNode |
| `langgraph_agents/nodes/grader.py` | Rule-based quality check |
| `langgraph_agents/nodes/_persona_loader.py` | Tách persona loader |
| `langgraph_agents/tools/__init__.py` | New package |
| `langgraph_agents/tools/pgvector_tool.py` | @tool wrapper |
| `tests/langgraph_agents/test_phase2_5_*.py` | New test suite |

### Modified

| File | Thay đổi |
|------|---------|
| `langgraph_agents/state.py` | Schema rewrite |
| `langgraph_agents/graph.py` | New flow |
| `langgraph_agents/routing.py` | New routing functions |
| `langgraph_agents/nodes/memory.py` | STM 3 Q&A + conditional LTM |
| `langgraph_agents/nodes/conversation.py` | Dual mode + remove voice_path |
| `langgraph_agents/nodes/error_handler.py` | Write reasoning_output |
| `langgraph_agents/db/vector_backend.py` | Thêm `source_id` filter |
| `requirements-langgraph.txt` | langchain-google-genai, mcp, pydantic |

### Removed

| File | Lý do |
|------|------|
| `langgraph_agents/nodes/manager.py` | Renamed to planner.py |
| `langgraph_agents/nodes/reasoning.py` | Renamed to synthesizer.py |
| `langgraph_agents/nodes/retrieval.py` | Replaced by retriever_agent.py |
| `langgraph_agents/nodes/validator.py` | Replaced by grader.py |
| `langgraph_agents/nodes/dispatch.py` | TTS chuyển sang FastAPI layer |
| `langgraph_agents/llm_gateway.py` | Thay bằng LangChain ChatModel |

### NOT touched (giữ nguyên cho Phase 3+)

- `services/` — sẽ refactor thành MCP server wrappers ở Phase 3
- `celery_app.py` — sẽ tái dùng cho TTS task (FastAPI layer)
- `streaming/approval.py` — sẽ dùng cho token budget interrupt (Plan §11.4)
- `db/postgres.py`, `db/init_schema.py`, `db/session_store.py` — không đổi
- `shared.py` — không đổi
- `personas/eca_default.md` — không đổi

---

## 17. Risks & Gotchas

| Risk | Mitigation |
|------|-----------|
| **`add_messages` reducer + retry loop** có thể chèn duplicate messages | Khi grader retry, retriever_agent append AIMessage mới — `add_messages` xử lý theo ID. Nếu có vấn đề, manual filter trong synthesizer. |
| **`with_structured_output` Gemini chưa stable cho mọi schema** | Test sớm với `PlanOutput`. Nếu fail, fallback sang `chat_json` style (system prompt yêu cầu JSON). |
| **pgvector @tool gọi async trong sync context** (LangChain wrapper) | Dùng `@tool` async — LangChain ToolNode support async tools tự nhiên. |
| **Memory chạy trước Planner ⇒ không có expanded_query** | Đã accept trong Plan §4.2 — Memory dùng raw query cho LTM lookup. |
| **`retry_count` reducer** | TypedDict không có reducer mặc định cho int. Phải ghi explicit `{"retry_count": 1}` trong grader, hoặc dùng `Annotated[int, lambda a, b: b]` (last-write-wins). Plan v2.4 dùng explicit write. |
| **Token tracking double-count** | `total_tokens: Annotated[int, operator.add]` — mỗi node tự return `total_tokens` của riêng lần invoke đó, không cộng dồn state cũ. |
| **`tools_condition` không tìm tool_calls** | Đảm bảo `bind_tools()` được gọi đúng cách. Mock test trước khi integration test. |
| **Persona prompt cho generation mode khác styling mode** | Build prompt khác nhau theo mode. Đừng dùng cùng template. |
| **STM ghi sau graph (FastAPI layer)** | Phase 2.5 không động đến FastAPI. Mock STM write trong test. Phase 5 sẽ implement. |

---

## 18. Reporting checkpoints

N báo K sau mỗi mốc:

| Checkpoint | Sau task | Báo gì |
|------------|----------|--------|
| **CP1** | 2.5.3 (planner.py done) | Show PlanOutput cho 3 query mẫu (greeting / knowledge / exercise) |
| **CP2** | 2.5.6 (retriever_agent done) | Show messages sequence khi gọi 1 query exercise (AIMessage tool_calls + ToolMessage) |
| **CP3** | 2.5.8 (grader done) | Show grader result + retry feedback cho 1 case fail |
| **CP4** | 2.5.12 (graph done) | Smoke test 3 path đầy đủ — paste logs |
| **CP5** | 2.5.14 (tests done) | Full pytest output |

K review từng CP trước khi N tiếp tục bước kế.

---

## 19. Sau Phase 2.5

Phase 3 sẽ:
- Implement Kimodo MCP server thật (`mcp/kimodo_server.py`)
- Implement web_search MCP server thật (`mcp/web_search_server.py`)
- Thêm 2 MCP tools vào `_RETRIEVER_TOOLS` (qua `MultiServerMCPClient`)
- Refactor `services/vieneu_tts/` thành Celery task gọi từ FastAPI layer (không qua graph)
- Test retriever_agent gọi tool song song (pgvector + MCP)

Phase 5 sẽ thêm SSE qua `astream_events()` + session reopen + token budget interrupt.

---

**N**: bắt đầu từ Task 2.5.1. Commit mỗi task riêng. Báo K sau mỗi CP. Tổng thời gian ước ~8–10h chia 2 ngày.
