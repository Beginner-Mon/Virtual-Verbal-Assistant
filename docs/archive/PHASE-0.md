---
title: "Phase 0 — Foundation"
plan: v2.2
assignee: N (Developer)
reviewer: K (Architect)
status: NOT STARTED
date: 2026-05-20
tags: [phase-0, foundation, langgraph, scaffold]
---

# Phase 0 — Foundation

> K's implementation orders for N. Don't delete existing agenticRAG code — only add new files/folders.

---

## Task 0.1 — Branch + Directory Structure

### Branch

```bash
git checkout develop
git checkout -b feature/langgraph-rewrite
```

### Directory

Create inside `agenticRAG/agentic_rag_gemini/`:

```
langgraph_agents/
  __init__.py
  state.py
  graph.py
  routing.py
  llm_gateway.py
  nodes/
    __init__.py
    manager.py
    memory.py
    retrieval.py
    reasoning.py
    validator.py
    conversation.py
    dispatch.py
    error_handler.py
  personas/
    eca_default.md
streaming/
  __init__.py
  sse_handler.py
  approval.py
services/
  __init__.py
  kimodo/
    __init__.py
    api_server.py
    celery_task.py
  vieneu_tts/
    __init__.py
    api_server.py
    celery_task.py
db/
  __init__.py
  postgres.py
  vector_backend.py
  migrations/
```

Also create at project root:

```
tests/langgraph_agents/
  __init__.py
  test_phase0_smoke.py
```

---

## Task 0.2 — `state.py` (AgentState + ErrorSeverity)

**File**: `langgraph_agents/state.py`

```python
from __future__ import annotations
import operator
from enum import Enum
from typing import Annotated, Optional, TypedDict


class ErrorSeverity(str, Enum):
    CRITICAL = "critical"       # LLM/DB down -> stop, apologize
    RECOVERABLE = "recoverable" # Retrieval timeout, worker down -> continue with available data
    IGNORABLE = "ignorable"     # Non-essential fail -> log only


class AgentState(TypedDict):
    # Input
    user_id: str
    session_id: str
    query: str
    conversation_history: list[dict]
    output_mode: str            # "text" | "speech" | "both"
    persona_id: str

    # Manager output
    intent: str                 # conversation | knowledge_query | exercise_recommendation | visualize_motion | clarify
    confidence: float
    expanded_query: str

    # Memory output
    memory_context: dict        # {long_term: [...], short_term: {...}, user_profile: {...}}

    # Retrieval output
    retrieval_results: list[dict]
    retrieval_metadata: dict

    # Reasoning output
    reasoning_output: str

    # Validator output (pre-persona)
    raw_answer: str

    # Conversation output (post-persona)
    final_answer: str

    # Dispatch / Async tasks
    motion_task_id: Optional[str]       # Celery task ID (set after approval)
    speech_task_id: Optional[str]       # Celery task ID (fired immediately)
    motion_pending: bool                # True if motion payload stored, awaiting approval
    motion_payload: Optional[dict]      # {prompt, constraints} for approval gate

    # Operational
    errors: Annotated[list[dict], operator.add]  # [{node, severity, message, timestamp}]
    request_id: str
```

**Rules**:
- Each node writes ONLY to its own fields. No cross-writes.
- Error entry format: `{"node": "reasoning", "severity": "critical", "message": "Claude API timeout", "timestamp": "..."}`

---

## Task 0.3 — Stub Nodes (8 files in `nodes/`)

Every node is a **pure Python async function**. Zero imports from `langgraph` or `langchain` inside node files. Phase 0 = stubs that return correct fields.

### `nodes/manager.py`

```python
from langgraph_agents.state import AgentState


async def manager_node(state: AgentState) -> dict:
    """Intent classification + routing. Phase 0: stub returns 'conversation'."""
    return {
        "intent": "conversation",
        "confidence": 0.9,
        "expanded_query": state["query"],
    }
```

### `nodes/memory.py`

```python
from langgraph_agents.state import AgentState


async def memory_node(state: AgentState) -> dict:
    """Redis STM + pgvector LTM lookup. Phase 0: stub returns empty context."""
    return {
        "memory_context": {
            "long_term": [],
            "short_term": {},
            "user_profile": {},
        }
    }
```

### `nodes/retrieval.py`

```python
from langgraph_agents.state import AgentState


async def retrieval_node(state: AgentState) -> dict:
    """pgvector search + web fallback. Phase 0: stub returns empty results."""
    return {
        "retrieval_results": [],
        "retrieval_metadata": {"source": "stub", "count": 0},
    }
```

### `nodes/reasoning.py`

```python
from langgraph_agents.state import AgentState


async def reasoning_node(state: AgentState) -> dict:
    """Clinical analysis + constraint extraction. Phase 0: stub echo."""
    return {
        "reasoning_output": f"[Stub reasoning] Query: {state['query']}",
    }
```

### `nodes/validator.py`

This is the **one node with real logic in Phase 0** — validates fields, builds `raw_answer`, applies fallback.

```python
from langgraph_agents.state import AgentState


async def validator_node(state: AgentState) -> dict:
    """Validate sub-agent outputs + build raw_answer + fallback if data missing."""
    parts = []

    reasoning = state.get("reasoning_output")
    if reasoning:
        parts.append(reasoning)

    retrieval = state.get("retrieval_results")
    if retrieval:
        parts.append(f"Found {len(retrieval)} relevant results.")

    if parts:
        raw = "\n\n".join(parts)
    else:
        raw = "Xin loi, toi khong tim duoc thong tin phu hop."

    return {"raw_answer": raw}
```

### `nodes/conversation.py`

```python
from langgraph_agents.state import AgentState


async def conversation_node(state: AgentState) -> dict:
    """Apply persona styling to raw_answer. Phase 0: passthrough."""
    return {
        "final_answer": state.get("raw_answer", ""),
    }
```

### `nodes/dispatch.py`

```python
from langgraph_agents.state import AgentState


async def dispatch_node(state: AgentState) -> dict:
    """Approval gate + fire Celery tasks. Phase 0: no-op."""
    return {
        "motion_pending": False,
        "motion_payload": None,
        "motion_task_id": None,
        "speech_task_id": None,
    }
```

### `nodes/error_handler.py`

```python
from langgraph_agents.state import AgentState, ErrorSeverity


async def error_handler_node(state: AgentState) -> dict:
    """Generate graceful error message from errors list."""
    errors = state.get("errors", [])
    critical = [e for e in errors if e.get("severity") == ErrorSeverity.CRITICAL]

    if critical:
        msg = "Xin loi, he thong dang gap su co. Vui long thu lai sau."
    else:
        msg = "Da co loi nho, nhung toi van co gang tra loi."

    return {"raw_answer": msg}
```

---

## Task 0.4 — `routing.py` (Conditional edges + error routing)

**File**: `langgraph_agents/routing.py`

```python
from langgraph_agents.state import AgentState, ErrorSeverity


def check_errors(state: AgentState) -> str:
    """After each node: route to error_handler if CRITICAL error exists."""
    for err in state.get("errors", []):
        if err.get("severity") == ErrorSeverity.CRITICAL:
            return "error_handler"
    return "continue"


def route_by_intent(state: AgentState) -> str:
    """After memory: decide next node based on intent."""
    intent = state.get("intent", "conversation")

    if intent in ("knowledge_query", "exercise_recommendation"):
        return "retrieval"
    if intent == "visualize_motion":
        return "validator"
    if intent == "clarify":
        return "validator"
    # default: simple conversation
    return "validator"


def route_after_conversation(state: AgentState) -> str:
    """After conversation: skip dispatch for clarify intent."""
    if state.get("intent") == "clarify":
        return "end"
    return "dispatch"
```

---

## Task 0.5 — `graph.py` (StateGraph construction)

**File**: `langgraph_agents/graph.py`

Build the graph following this flow:

```
START -> manager -> [check_errors] -> memory -> [check_errors] -> [route_by_intent]:

  Route "retrieval":
    retrieval -> [check_errors] -> reasoning -> [check_errors] -> validator -> conversation -> [route_after_conversation]:
      -> dispatch -> END
      -> END (clarify)

  Route "validator" (simple/clarify/visualize_motion):
    validator -> conversation -> [route_after_conversation]:
      -> dispatch -> END
      -> END (clarify)

  Error routing (any node):
    error_handler -> conversation -> END
```

```python
from langgraph.graph import StateGraph, END
from langgraph_agents.state import AgentState
from langgraph_agents.nodes.manager import manager_node
from langgraph_agents.nodes.memory import memory_node
from langgraph_agents.nodes.retrieval import retrieval_node
from langgraph_agents.nodes.reasoning import reasoning_node
from langgraph_agents.nodes.validator import validator_node
from langgraph_agents.nodes.conversation import conversation_node
from langgraph_agents.nodes.dispatch import dispatch_node
from langgraph_agents.nodes.error_handler import error_handler_node
from langgraph_agents.routing import check_errors, route_by_intent, route_after_conversation


def build_graph():
    graph = StateGraph(AgentState)

    # Add all nodes
    graph.add_node("manager", manager_node)
    graph.add_node("memory", memory_node)
    graph.add_node("retrieval", retrieval_node)
    graph.add_node("reasoning", reasoning_node)
    graph.add_node("validator", validator_node)
    graph.add_node("conversation", conversation_node)
    graph.add_node("dispatch", dispatch_node)
    graph.add_node("error_handler", error_handler_node)

    # Entry
    graph.set_entry_point("manager")

    # manager -> error check
    graph.add_conditional_edges("manager", check_errors, {
        "error_handler": "error_handler",
        "continue": "memory",
    })

    # memory -> error check -> route by intent
    graph.add_conditional_edges("memory", check_errors, {
        "error_handler": "error_handler",
        "continue": "route_intent",  # placeholder — see below
    })
    # NOTE: LangGraph doesn't chain conditionals directly.
    # Use a pass-through or combine check_errors + route_by_intent.
    # N: figure out the cleanest way to do "check_errors THEN route_by_intent"
    # Option A: single function that checks errors first, then routes by intent
    # Option B: intermediate node

    # retrieval -> error check -> reasoning
    graph.add_conditional_edges("retrieval", check_errors, {
        "error_handler": "error_handler",
        "continue": "reasoning",
    })

    # reasoning -> error check -> validator
    graph.add_conditional_edges("reasoning", check_errors, {
        "error_handler": "error_handler",
        "continue": "validator",
    })

    # validator -> conversation (always)
    graph.add_edge("validator", "conversation")

    # conversation -> dispatch or END
    graph.add_conditional_edges("conversation", route_after_conversation, {
        "dispatch": "dispatch",
        "end": END,
    })

    # dispatch -> END
    graph.add_edge("dispatch", END)

    # error_handler -> conversation -> END
    graph.add_edge("error_handler", "conversation")

    return graph.compile()
```

**Known issue for N to solve**: The `memory -> route_by_intent` transition needs to combine error checking AND intent routing. Recommended approach — write a single routing function:

```python
def route_after_memory(state: AgentState) -> str:
    """Combined: check errors first, then route by intent."""
    if check_errors(state) == "error_handler":
        return "error_handler"
    return route_by_intent(state)
```

Then use: `graph.add_conditional_edges("memory", route_after_memory, {...})`

---

## Task 0.6 — Docker Compose (PostgreSQL + pgvector + Redis)

**File**: `docker-compose.langgraph.yml` (separate file, don't modify existing docker-compose if any)

```yaml
version: "3.8"

services:
  postgres:
    image: pgvector/pgvector:pg16
    container_name: vva-postgres
    environment:
      POSTGRES_DB: vva
      POSTGRES_USER: vva
      POSTGRES_PASSWORD: vva_dev
    ports:
      - "5432:5432"
    deploy:
      resources:
        limits:
          memory: 2G
    volumes:
      - pgdata:/var/lib/postgresql/data
    command: >
      postgres
      -c shared_buffers=512MB
      -c work_mem=16MB
      -c effective_cache_size=1G

  redis:
    image: redis:7-alpine
    container_name: vva-redis
    ports:
      - "6379:6379"
    command: redis-server --maxmemory 512mb --maxmemory-policy allkeys-lru
    deploy:
      resources:
        limits:
          memory: 512M

volumes:
  pgdata:
```

---

## Task 0.7 — Dependencies

**File**: `requirements-langgraph.txt` (project root, separate from existing requirements)

```
# Core framework
langgraph>=0.2.0
langchain-core>=0.3.0

# LLM
anthropic>=0.34.0

# Database
asyncpg>=0.29.0
pgvector>=0.3.0
alembic>=1.13.0

# SSE streaming
sse-starlette>=2.0.0

# HTTP client (for Kimodo/VieNeu-TTS REST calls)
httpx>=0.27.0

# Testing
pytest>=8.0.0
pytest-asyncio>=0.23.0
```

---

## Task 0.8 — Smoke Tests

**File**: `tests/langgraph_agents/test_phase0_smoke.py`

```python
import pytest
from langgraph_agents.state import AgentState, ErrorSeverity
from langgraph_agents.graph import build_graph


def _base_state(**overrides) -> dict:
    """Create minimal valid AgentState for testing."""
    state = {
        "user_id": "test-user",
        "session_id": "test-session",
        "query": "Xin chao",
        "conversation_history": [],
        "output_mode": "text",
        "persona_id": "eca_default",
        "errors": [],
        "request_id": "test-001",
    }
    state.update(overrides)
    return state


@pytest.mark.asyncio
@pytest.mark.unit
async def test_happy_path_returns_final_answer():
    """Query traverses all stub nodes and returns a non-empty final_answer."""
    graph = build_graph()
    result = await graph.ainvoke(_base_state())

    assert result["final_answer"], "final_answer should not be empty"
    assert result["intent"] == "conversation"
    assert result["confidence"] > 0


@pytest.mark.asyncio
@pytest.mark.unit
async def test_critical_error_routes_to_error_handler():
    """Injecting a CRITICAL error should route through error_handler."""
    graph = build_graph()
    result = await graph.ainvoke(_base_state(
        errors=[{
            "node": "external",
            "severity": ErrorSeverity.CRITICAL,
            "message": "Simulated DB failure",
            "timestamp": "2026-05-20T00:00:00Z",
        }],
    ))

    assert "su co" in result["final_answer"].lower() or "loi" in result["final_answer"].lower()


@pytest.mark.asyncio
@pytest.mark.unit
async def test_knowledge_query_traverses_retrieval_and_reasoning():
    """Intent 'knowledge_query' should go through retrieval -> reasoning path."""
    # This test requires manager to return knowledge_query intent.
    # For Phase 0 stubs, manager always returns "conversation".
    # Override by testing the routing function directly.
    from langgraph_agents.routing import route_by_intent
    assert route_by_intent({"intent": "knowledge_query"}) == "retrieval"
    assert route_by_intent({"intent": "exercise_recommendation"}) == "retrieval"
    assert route_by_intent({"intent": "conversation"}) == "validator"
    assert route_by_intent({"intent": "clarify"}) == "validator"


@pytest.mark.asyncio
@pytest.mark.unit
async def test_graph_compiles():
    """Graph should compile without errors."""
    graph = build_graph()
    assert graph is not None
```

---

## Task 0.9 — Persona Stub

**File**: `langgraph_agents/personas/eca_default.md`

```markdown
## Identity
Name: ECA | Role: Physical therapy AI assistant | Avatar: eca_default.png

## Voice
Tone: Warm, professional, encouraging | Formality: Semi-formal

## Behavioral Rules
- Acknowledge pain before suggesting exercises
- Use anatomical terms with plain-language explanations
- End exercise recs with safety reminders
- Refer to medical professionals for anything beyond wellness

## Response Formatting
- Bullet points for exercise lists, include rep/set counts
- Bold safety warnings
- Keep under 300 words
```

---

## Acceptance Criteria

| # | Check | How to verify |
|---|-------|---------------|
| 1 | Branch `feature/langgraph-rewrite` exists | `git branch` |
| 2 | All files in directory structure created | `ls -R langgraph_agents/ streaming/ services/ db/` |
| 3 | `state.py` has `ErrorSeverity` + full `AgentState` | K code review |
| 4 | 8 stub nodes import successfully, return correct fields | Unit test |
| 5 | `graph.py` compiles — `build_graph()` returns runnable | `test_graph_compiles` |
| 6 | Happy path: query -> all nodes -> `final_answer` returned | `test_happy_path` |
| 7 | Error path: CRITICAL -> `error_handler` -> graceful message | `test_critical_error` |
| 8 | Docker compose up -> PostgreSQL + Redis running | `docker compose -f docker-compose.langgraph.yml up -d` |
| 9 | No existing files modified or deleted | `git diff develop --name-only` shows only new files |

---

## Order of Execution

```
0.1  Branch + directories     (5 min)
0.2  state.py                 (10 min)
0.3  Stub nodes (8 files)     (20 min)
0.4  routing.py               (10 min)
0.5  graph.py                 (20 min)  <-- likely needs debugging
0.6  Docker compose           (5 min)
0.7  requirements             (2 min)
0.8  Smoke tests              (15 min)
0.9  Persona stub             (2 min)
     ---
     Total: ~1.5 hours
```

N: start from 0.1, work sequentially. Report to K after 0.5 (graph compile) for mid-review, and again after 0.8 (tests pass) for final review.
