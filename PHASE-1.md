---
title: "Phase 1 — Manager + Memory"
plan: v2.2
assignee: N (Developer)
reviewer: K (Architect)
status: NOT STARTED
date: 2026-05-20
depends_on: Phase 0 (COMPLETE)
tags: [phase-1, manager, memory, intent-classification, redis, pgvector]
---

# Phase 1 — Manager + Memory

> K's implementation orders for N. Replace stub `manager.py` and `memory.py` with real logic. Add `llm_gateway.py`, `db/postgres.py`, `db/vector_backend.py`. Migrate session store from Firebase to PostgreSQL.

**Reference code** (read before coding):
- `agents/api_orchestrator.py:78-111` — `_UNIFIED_ANALYSIS_PROMPT` (current intent classification prompt)
- `agents/api_orchestrator.py:275-344` — `classify_intent_and_analyze()` (current intent+routing logic)
- `agents/intents.py` — canonical `IntentType`, `ActionType`, aliases, parsers
- `memory/memory_manager.py` — current memory logic (ChromaDB + VectorStore)
- `memory/embedding_service.py` — SentenceTransformer embeddings (keep as-is)
- `memory/session_store.py` — Firebase Firestore session persistence (to be replaced)
- `config/config.yaml` — current config structure

---

## Task 1.1 — `llm_gateway.py` (Unified LLM abstraction)

**File**: `langgraph_agents/llm_gateway.py`

This is the single point of contact for ALL LLM calls across nodes. Nodes never import `anthropic`, `openai`, or `google.generativeai` directly.

```python
"""Unified LLM gateway — single abstraction for all LLM providers.

Nodes call: `response = await llm_gateway.chat(messages, **kwargs)`
Provider is selected by config. Nodes don't know or care which LLM is behind it.
"""

import os
from typing import Optional


class LLMGateway:
    """Async LLM client that routes to the configured provider."""

    def __init__(self, provider: str = None, model: str = None):
        """
        Args:
            provider: "anthropic" | "gemini" | "ollama". Defaults to config/env.
            model: Model name override. Defaults to config.
        """
        self.provider = provider or os.getenv("LLM_PROVIDER", "gemini")
        self.model = model
        self._client = None

    async def chat(
        self,
        messages: list[dict],
        temperature: float = 0.1,
        max_tokens: int = 1024,
        response_format: Optional[dict] = None,
    ) -> str:
        """Send chat completion request. Returns content string."""
        ...

    async def chat_json(
        self,
        messages: list[dict],
        temperature: float = 0.0,
        max_tokens: int = 1024,
    ) -> dict:
        """Chat + parse JSON response. Raises on invalid JSON."""
        ...
```

### Implementation requirements:

1. **Gemini provider** (primary, since current stack uses Gemini):
   - Reuse the OpenAI-compatible endpoint that `GeminiClientWrapper` already uses
   - OR use `google-generativeai` SDK directly
   - Model default: `gemini-2.5-flash` (from `config.yaml`)

2. **Anthropic provider** (future primary):
   - Use `anthropic.AsyncAnthropic` client
   - Model: `claude-sonnet-4-20250514` or configurable

3. **Ollama provider** (local fallback):
   - Use `httpx.AsyncClient` → `POST http://localhost:11434/api/chat`
   - Model: `qwen2.5:3b` (from `config.yaml`)

4. **`chat_json()`**: Wraps `chat()` + strips markdown code fences + `json.loads()`. Reuse `clean_json_response()` logic from `api_orchestrator.py:51-70`.

**Phase 1 minimum**: Implement Gemini provider only (it's what we have). Anthropic + Ollama can be stubs that raise `NotImplementedError`. Mark them with `# TODO: Phase 2+`.

**Config**: Add a `langgraph` section to `config/config.yaml` (or create a separate `config/langgraph.yaml`):

```yaml
langgraph:
  llm:
    provider: "gemini"              # "gemini" | "anthropic" | "ollama"
    manager_model: "gemini-2.5-flash"
    reasoning_model: "gemini-2.5-flash"   # will change to Claude later
    conversation_model: "gemini-2.5-flash"
    temperature:
      manager: 0.0        # deterministic routing
      reasoning: 0.7
      conversation: 0.7
    max_tokens:
      manager: 1024
      reasoning: 4096
      conversation: 4096
```

---

## Task 1.2 — `manager.py` (Real intent classification)

**File**: `langgraph_agents/nodes/manager.py`

Replace the stub. Port logic from `api_orchestrator.py:classify_intent_and_analyze()`.

### Input (from AgentState):
- `query` — user's message
- `conversation_history` — last N turns

### Output (write to AgentState):
```python
{
    "intent": str,           # "conversation" | "knowledge_query" | "exercise_recommendation" | "visualize_motion" | "clarify"
    "confidence": float,
    "expanded_query": str,   # enriched query with synonyms for better retrieval
}
```

### Implementation:

```python
from langgraph_agents.state import AgentState
from langgraph_agents.llm_gateway import LLMGateway

_MANAGER_SYSTEM_PROMPT = """You are the routing brain for a physical therapy AI assistant.
Analyze the user query and return a single JSON object.

Intents:
- conversation            : general chat, greetings, or follow-ups with no exercise content
- knowledge_query         : asks for explanation, facts, or non-motion advice
- exercise_recommendation : asks for one or more exercises, stretches, or workouts
- visualize_motion        : asks to see / animate / show a specific single movement
- clarify                 : query is too vague to route confidently

Respond with valid JSON ONLY:
{
  "intent": "<intent_value>",
  "confidence": 0.0-1.0,
  "expanded_query": "<rephrase the query with 2-3 extra keywords for retrieval>"
}

Rules:
- expanded_query should add anatomical/physiotherapy synonyms when relevant
- confidence < 0.5 → use "clarify"
- For simple greetings, confidence should be high (>0.9)"""


async def manager_node(state: AgentState) -> dict:
    gateway = LLMGateway()  # or inject via config

    history_snippet = ""
    if state.get("conversation_history"):
        turns = state["conversation_history"][-3:]
        history_snippet = "\n\nRecent conversation:\n" + "\n".join(
            f"{t['role']}: {t['content']}" for t in turns
        )

    result = await gateway.chat_json(
        messages=[
            {"role": "system", "content": _MANAGER_SYSTEM_PROMPT},
            {"role": "user", "content": state["query"] + history_snippet},
        ],
        temperature=0.0,
        max_tokens=1024,
    )

    intent = result.get("intent", "conversation")
    # Validate intent is known
    valid_intents = {"conversation", "knowledge_query", "exercise_recommendation", "visualize_motion", "clarify"}
    if intent not in valid_intents:
        intent = "knowledge_query"

    return {
        "intent": intent,
        "confidence": float(result.get("confidence", 0.8)),
        "expanded_query": result.get("expanded_query") or state["query"],
    }
```

### Key differences from current system:

| Current (`api_orchestrator.py`) | New (`manager.py`) |
|---|---|
| Returns `IntentType` enum | Returns plain string (LangGraph state is TypedDict, not enum) |
| Returns `action`, `needs_rag`, `generate_motion`, `motion_type` | Returns only `intent`, `confidence`, `expanded_query` — routing logic moved to `routing.py` |
| Uses `GeminiClientWrapper` directly | Uses `LLMGateway` (provider-agnostic) |
| Sync | Async |

**Why simpler output**: Manager's job is ONLY classification + query expansion. Action selection (which tools to use, whether to generate motion) is now handled by graph routing edges in `routing.py`. This is the separation-of-concerns win from LangGraph.

---

## Task 1.3 — `db/postgres.py` (Async PostgreSQL client)

**File**: `langgraph_agents/db/postgres.py`

Thin async wrapper around `asyncpg`. Used by MemoryAgent and future nodes.

```python
"""Async PostgreSQL client for the LangGraph agent system.

Uses asyncpg connection pool. All queries go through this module.
"""

import asyncpg
from typing import Optional


class PostgresClient:
    """Async PostgreSQL connection pool manager."""

    def __init__(self, dsn: str = None):
        """
        Args:
            dsn: PostgreSQL connection string.
                 Default: "postgresql://vva:vva_dev@localhost:5432/vva"
        """
        self.dsn = dsn or "postgresql://vva:vva_dev@localhost:5432/vva"
        self._pool: Optional[asyncpg.Pool] = None

    async def connect(self):
        """Create connection pool."""
        if self._pool is None:
            self._pool = await asyncpg.create_pool(self.dsn, min_size=2, max_size=10)

    async def close(self):
        """Close connection pool."""
        if self._pool:
            await self._pool.close()
            self._pool = None

    async def execute(self, query: str, *args):
        """Execute a query (INSERT, UPDATE, DELETE)."""
        await self.connect()
        async with self._pool.acquire() as conn:
            return await conn.execute(query, *args)

    async def fetch(self, query: str, *args) -> list:
        """Fetch multiple rows."""
        await self.connect()
        async with self._pool.acquire() as conn:
            return await conn.fetch(query, *args)

    async def fetchrow(self, query: str, *args):
        """Fetch single row."""
        await self.connect()
        async with self._pool.acquire() as conn:
            return await conn.fetchrow(query, *args)

    async def fetchval(self, query: str, *args):
        """Fetch single value."""
        await self.connect()
        async with self._pool.acquire() as conn:
            return await conn.fetchval(query, *args)
```

---

## Task 1.4 — `db/vector_backend.py` (pgvector operations)

**File**: `langgraph_agents/db/vector_backend.py`

Replaces ChromaDB backend for vector similarity search. Uses the `embeddings` table from the plan's PostgreSQL schema.

```python
"""pgvector operations — vector similarity search via PostgreSQL.

Replaces ChromaDB backend. Uses the 'embeddings' table with
IVFFlat index for cosine similarity search.
"""

from typing import Optional
from langgraph_agents.db.postgres import PostgresClient


class VectorBackend:
    """pgvector-based vector similarity search."""

    def __init__(self, pg: PostgresClient):
        self.pg = pg

    async def search(
        self,
        query_embedding: list[float],
        top_k: int = 5,
        source_type: Optional[str] = None,
        user_id: Optional[str] = None,
    ) -> list[dict]:
        """Cosine similarity search against embeddings table.

        Args:
            query_embedding: Query vector (384-dim for MiniLM-L6-v2)
            top_k: Number of results
            source_type: Filter by 'conversation' | 'document' | 'humanml3d'
            user_id: Filter by user (via source_id -> users/conversations)

        Returns:
            List of {id, content, metadata, similarity}
        """
        ...

    async def insert(
        self,
        content: str,
        embedding: list[float],
        source_type: str,
        source_id: str,
        metadata: Optional[dict] = None,
    ) -> str:
        """Insert a new embedding record. Returns the new record's UUID."""
        ...

    async def delete_by_source(self, source_type: str, source_id: str) -> int:
        """Delete all embeddings for a given source. Returns count deleted."""
        ...
```

### SQL for search:

```sql
SELECT id, content, metadata,
       1 - (embedding <=> $1::vector) AS similarity
FROM embeddings
WHERE ($2::text IS NULL OR source_type = $2)
ORDER BY embedding <=> $1::vector
LIMIT $3
```

**Note**: `asyncpg` doesn't natively understand the `vector` type. N must register the pgvector type with asyncpg. Use the `pgvector` Python package:

```python
from pgvector.asyncpg import register_vector

async def connect(self):
    self._pool = await asyncpg.create_pool(self.dsn, ...)
    async with self._pool.acquire() as conn:
        await register_vector(conn)
```

---

## Task 1.5 — Alembic Migrations (PostgreSQL schema)

**File**: `langgraph_agents/db/migrations/`

Set up Alembic and create the initial migration with the schema from Plan Section 3.2:

```sql
-- Enable pgvector extension
CREATE EXTENSION IF NOT EXISTS vector;

-- Core tables
CREATE TABLE users (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    profile JSONB DEFAULT '{}',
    created_at TIMESTAMPTZ DEFAULT now(),
    updated_at TIMESTAMPTZ DEFAULT now()
);

CREATE TABLE conversations (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    user_id UUID REFERENCES users(id),
    session_id UUID,
    messages JSONB DEFAULT '[]',
    summary TEXT,
    created_at TIMESTAMPTZ DEFAULT now()
);

CREATE TABLE documents (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    user_id UUID REFERENCES users(id),
    filename TEXT,
    metadata JSONB DEFAULT '{}',
    created_at TIMESTAMPTZ DEFAULT now()
);

-- Vector search
CREATE TABLE embeddings (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    source_type TEXT NOT NULL,      -- 'conversation' | 'document' | 'humanml3d'
    source_id UUID,
    content TEXT,
    embedding vector(384),          -- MiniLM-L6-v2 dimension
    metadata JSONB DEFAULT '{}',
    created_at TIMESTAMPTZ DEFAULT now()
);

CREATE INDEX ON embeddings USING ivfflat (embedding vector_cosine_ops) WITH (lists = 100);
CREATE INDEX idx_embeddings_source ON embeddings (source_type, source_id);
CREATE INDEX idx_conversations_user ON conversations (user_id);
CREATE INDEX idx_conversations_session ON conversations (session_id);
```

### Alembic setup:

```bash
cd agenticRAG/agentic_rag_gemini/langgraph_agents/db
alembic init migrations
# Edit alembic.ini: sqlalchemy.url = postgresql://vva:vva_dev@localhost:5432/vva
# Create migration: alembic revision --autogenerate -m "initial schema"
# Run: alembic upgrade head
```

**Alternative**: If Alembic feels heavy for Phase 1, a simple `init_schema.sql` file + a Python script that runs it via `asyncpg` is acceptable. Alembic can be added in Phase 6 (production hardening).

---

## Task 1.6 — `memory.py` (Real Memory Agent)

**File**: `langgraph_agents/nodes/memory.py`

Replace the stub. This node runs on EVERY query (unconditionally). It reads both:
1. **Redis STM** (short-term memory) — session buffer, user snapshot, behavioral patterns
2. **PostgreSQL/pgvector LTM** (long-term memory) — conversation history, user profile

### Input (from AgentState):
- `user_id`, `session_id`, `query`, `expanded_query` (from Manager)

### Output (write to AgentState):
```python
{
    "memory_context": {
        "long_term": [...],       # relevant past interactions from pgvector
        "short_term": {...},      # current session buffer from Redis
        "user_profile": {...},    # user preferences, conditions, constraints
    }
}
```

### Implementation approach:

```python
import json
from typing import Optional
import redis.asyncio as aioredis
from langgraph_agents.state import AgentState
from langgraph_agents.db.postgres import PostgresClient
from langgraph_agents.db.vector_backend import VectorBackend


# Redis STM key patterns (from Plan Section 3.3)
_STM_PROFILE_KEY = "stm:{user_id}:profile"           # TTL 24h
_STM_SESSION_KEY = "stm:{user_id}:session:{session_id}"  # TTL 2h
_STM_PATTERNS_KEY = "stm:{user_id}:patterns"          # TTL 7d


async def memory_node(state: AgentState) -> dict:
    user_id = state["user_id"]
    session_id = state["session_id"]
    query = state.get("expanded_query") or state["query"]

    # 1. Redis STM reads (fast, in-memory)
    short_term = await _read_stm(user_id, session_id)

    # 2. pgvector LTM search (semantic similarity)
    long_term = await _search_ltm(query, user_id)

    # 3. User profile from PostgreSQL
    user_profile = await _get_user_profile(user_id)

    return {
        "memory_context": {
            "long_term": long_term,
            "short_term": short_term,
            "user_profile": user_profile,
        }
    }


async def _read_stm(user_id: str, session_id: str) -> dict:
    """Read short-term memory from Redis."""
    r = aioredis.from_url("redis://localhost:6379")
    try:
        session_data = await r.get(f"stm:{user_id}:session:{session_id}")
        patterns = await r.get(f"stm:{user_id}:patterns")
        return {
            "session_buffer": json.loads(session_data) if session_data else [],
            "patterns": json.loads(patterns) if patterns else {},
        }
    finally:
        await r.close()


async def _search_ltm(query: str, user_id: str) -> list[dict]:
    """Search long-term memory via pgvector semantic similarity."""
    # Use EmbeddingService to get query vector
    # Then use VectorBackend.search()
    # Return top 5 relevant past interactions
    ...


async def _get_user_profile(user_id: str) -> dict:
    """Get user profile from PostgreSQL users table."""
    pg = PostgresClient()
    row = await pg.fetchrow("SELECT profile FROM users WHERE id = $1", user_id)
    if row:
        return row["profile"] or {}
    return {}
```

### Key design decisions:

1. **Redis + pgvector in parallel**: Both reads are independent — use `asyncio.gather()` to run them concurrently.

2. **Embedding for LTM search**: Reuse `memory/embedding_service.py` (SentenceTransformer, MiniLM-L6-v2). Don't create a new embedding service. Import and use the existing one. If it's sync-only, wrap with `asyncio.to_thread()`.

3. **STM write** (session buffer update): The Memory node READS on every query. WRITING to STM (storing the current interaction) happens AFTER the full graph completes — either in a post-graph hook or a separate Celery task. Don't write inside the Memory node.

4. **Graceful degradation**: If Redis is down → return empty `short_term`, log RECOVERABLE error. If PostgreSQL is down → return empty `long_term` + `user_profile`, log RECOVERABLE error. Graph continues either way.

---

## Task 1.7 — Session Store Migration (Firebase → PostgreSQL)

**File**: This is NOT a new file. It's about making the `conversations` table work.

### What to implement:

1. **Write a helper** in `db/postgres.py` (or a new `db/session_store.py`) that provides:
   - `save_session(user_id, session_id, messages)` → INSERT/UPDATE `conversations` table
   - `load_session(user_id, session_id)` → SELECT from `conversations`
   - `list_sessions(user_id, limit=10)` → list recent sessions

2. **Don't touch** `memory/session_store.py` (Firebase). It stays for the old code path. New LangGraph code uses the PostgreSQL path. Both coexist.

3. **Data migration script** (optional for Phase 1): A one-time script to export Firebase sessions → INSERT into PostgreSQL. Can be deferred to Phase 6 if there's no critical data.

---

## Task 1.8 — Update `routing.py` (use `expanded_query`)

**File**: `langgraph_agents/routing.py`

No change needed to routing functions — they already read `state["intent"]`. But verify that downstream nodes (retrieval, reasoning) use `expanded_query` from Manager instead of raw `query` when available.

Add to `routing.py` if needed:

```python
def get_search_query(state: AgentState) -> str:
    """Utility: return expanded_query if available, else raw query."""
    return state.get("expanded_query") or state.get("query", "")
```

---

## Task 1.9 — Integration Test

**File**: `tests/langgraph_agents/test_phase1_integration.py`

### Test cases:

1. **Manager intent accuracy** — test against the same queries used in the current system:
   ```python
   # Greetings → conversation
   assert (await run_manager("Xin chào"))["intent"] == "conversation"
   assert (await run_manager("Hello"))["intent"] == "conversation"

   # Knowledge → knowledge_query
   assert (await run_manager("What exercises help lower back pain?"))["intent"] == "knowledge_query"

   # Exercise → exercise_recommendation
   assert (await run_manager("Cho tôi bài tập giãn cơ lưng"))["intent"] == "exercise_recommendation"

   # Motion → visualize_motion
   assert (await run_manager("Show me a squat animation"))["intent"] == "visualize_motion"
   ```

2. **Manager expanded_query** — verify it adds relevant synonyms:
   ```python
   result = await run_manager("back pain exercises")
   assert "lumbar" in result["expanded_query"].lower() or "spine" in result["expanded_query"].lower()
   ```

3. **Memory node with empty DB** — verify it returns empty context gracefully (no crash):
   ```python
   result = await graph.ainvoke(base_state)
   assert result["memory_context"]["long_term"] == []
   assert result["memory_context"]["user_profile"] == {}
   ```

4. **Memory node with Redis down** — verify RECOVERABLE error, not crash:
   ```python
   # Kill Redis before test, or use wrong port
   result = await graph.ainvoke(base_state)
   assert result["memory_context"]["short_term"] == {"session_buffer": [], "patterns": {}}
   # Check RECOVERABLE error logged (not CRITICAL)
   ```

5. **Full graph end-to-end** — real Manager + real Memory + stub downstream:
   ```python
   result = await graph.ainvoke(base_state(query="Bài tập cho đau lưng"))
   assert result["intent"] == "exercise_recommendation"
   assert result["final_answer"]  # even though downstream is stub
   ```

6. **Manager latency profile** — measure and log:
   ```python
   import time
   start = time.time()
   await run_manager("What exercises help lower back pain?")
   elapsed_ms = (time.time() - start) * 1000
   print(f"Manager latency: {elapsed_ms:.0f}ms")
   # Flag if >500ms (plan says merge with Reasoning if too slow)
   assert elapsed_ms < 2000, "Manager should respond within 2s"
   ```

---

## Task 1.10 — Config Extension

Either extend `config/config.yaml` with a `langgraph:` section, or create `config/langgraph.yaml`. K recommends a separate file to avoid merge conflicts with existing config.

**File**: `config/langgraph.yaml`

```yaml
langgraph:
  llm:
    provider: "gemini"
    manager_model: "gemini-2.5-flash"
    reasoning_model: "gemini-2.5-flash"
    conversation_model: "gemini-2.5-flash"
    temperature:
      manager: 0.0
      reasoning: 0.7
      conversation: 0.7
    max_tokens:
      manager: 1024
      reasoning: 4096
      conversation: 4096

  memory:
    redis_url: "redis://localhost:6379"
    stm_profile_ttl: 86400     # 24h
    stm_session_ttl: 7200      # 2h
    stm_patterns_ttl: 604800   # 7d
    ltm_top_k: 5
    ltm_similarity_threshold: 0.3

  postgres:
    dsn: "postgresql://vva:vva_dev@localhost:5432/vva"
    pool_min: 2
    pool_max: 10
```

---

## Dependency Map (what imports what)

```
manager.py ──→ llm_gateway.py ──→ GeminiClientWrapper (or anthropic SDK)
memory.py  ──→ db/postgres.py ──→ asyncpg
           ──→ db/vector_backend.py ──→ db/postgres.py + pgvector
           ──→ redis.asyncio
           ──→ memory/embedding_service.py (existing, for LTM query vector)
graph.py   ──→ (unchanged, imports nodes)
routing.py ──→ (unchanged)
```

---

## Acceptance Criteria

| # | Check | How to verify |
|---|-------|---------------|
| 1 | `llm_gateway.py` exists, Gemini provider works | Unit test: call with simple prompt, get response |
| 2 | `manager.py` returns correct intent for 5+ test queries | Integration test |
| 3 | Manager latency < 2s (flag if > 500ms) | Latency test with timing |
| 4 | `expanded_query` adds relevant synonyms | Integration test |
| 5 | `db/postgres.py` connects to Docker PostgreSQL | Connection test |
| 6 | `db/vector_backend.py` can insert + search embeddings | Insert 3 docs, search, verify top result |
| 7 | Alembic migration (or init SQL) creates all 4 tables + indexes | `\dt` in psql |
| 8 | `memory.py` returns `memory_context` with all 3 keys | Integration test |
| 9 | Memory graceful degradation (Redis/PG down → empty, not crash) | Kill service → test |
| 10 | Full graph: real Manager + real Memory + stub downstream → `final_answer` | End-to-end test |
| 11 | Config in `config/langgraph.yaml` loads correctly | Unit test |
| 12 | No existing files modified (except `config/config.yaml` if extending) | `git diff develop --name-only` |

---

## Order of Execution

```
1.1   llm_gateway.py            (30 min)  ← foundation, everything depends on this
1.10  config/langgraph.yaml     (10 min)  ← needed by llm_gateway
1.2   manager.py                (30 min)  ← depends on 1.1
1.3   db/postgres.py            (20 min)
1.5   Alembic / init SQL        (20 min)  ← depends on 1.3
1.4   db/vector_backend.py      (30 min)  ← depends on 1.3
1.6   memory.py                 (45 min)  ← depends on 1.3, 1.4
1.7   Session store helper      (20 min)  ← depends on 1.3
1.8   routing.py update         (5 min)
1.9   Integration tests         (30 min)
      ---
      Total: ~4 hours
```

**Mid-review checkpoint**: After 1.2 (Manager works with real LLM) — report to K with:
1. Manager latency numbers
2. Intent accuracy on 5+ test queries
3. Any LLM provider issues

**Final review**: After 1.9 (all tests pass).

---

## Risk: Manager Latency

Plan says: if Manager latency > 500ms, consider merging Manager + Reasoning into one node.

**What to measure**: Time from `manager_node()` entry to return (including LLM call).

**How to measure**: Add `time.perf_counter()` before/after the `gateway.chat_json()` call. Log the result.

**If > 500ms**: Don't merge yet. Report the number to K. We'll decide based on:
- Is it the LLM cold start? (First call is always slow)
- Is it the model? (Gemini Flash should be ~200-400ms)
- Is it the prompt size? (History snippet may be large)

This is a data-driven decision. Collect the data first.
