# Phase 2 — Retrieval + Reasoning + Validator

**Architect**: K | **Developer**: N | **Date**: 2026-05-21
**Branch**: `feature/langgraph-rewrite` (continue from Phase 1)
**Estimated time**: ~5-6h

---

## Overview

Replace the 3 stub nodes (`retrieval.py`, `reasoning.py`, `validator.py`) with real implementations that absorb logic from:

- `agents/knowledge_librarian.py` — source routing, fact extraction
- `agents/double_rag.py` — clinical dispatch + constraint extraction
- `agents/query_transform.py` — HyDE generation (deferred; Manager already provides `expanded_query`)
- `retrieval/rag_pipeline.py` — context retrieval, web search fallback, prompt building, LLM generation, validation

**Critical rule**: Do NOT modify or import from the old files. Port the logic into new files. Both architectures coexist.

---

## Dependencies on Phase 1

These are ready and tested:

| Component | How Phase 2 uses it |
|-----------|-------------------|
| `LLMGateway` | Reasoning node calls `chat()` for response generation |
| `VectorBackend` | Retrieval node calls `search()` for pgvector similarity |
| `PostgresClient` | Shared via `_get_pg_client()` singleton pattern |
| `EmbeddingService` | `memory/embedding_service.py` — embed query for vector search |
| `memory_node` | Populates `memory_context` in state before retrieval |
| `routing.py` | `route_by_intent()` sends `knowledge_query`/`exercise_recommendation` to retrieval |

---

## Task 1: Update `config/langgraph.yaml` — add retrieval section

Add a `retrieval` section under `langgraph:`:

```yaml
  retrieval:
    top_k: 5
    min_context_threshold: 2          # fewer results triggers web fallback
    quality_threshold: 0.4            # avg similarity below this triggers web fallback
    web_search_enabled: true
    web_search_max_results: 3
    web_search_timeout: 8             # seconds
    source_weights:                   # for quality scoring
      document: 1.0
      conversation: 0.9
      web_search: 0.75
```

---

## Task 2: Implement `retrieval.py` — pgvector search + web fallback

**File**: `langgraph_agents/nodes/retrieval.py`

### Architecture

```
retrieval_node(state) →
  1. Get expanded_query from state (Manager already did expansion)
  2. Embed query via EmbeddingService
  3. pgvector search: source_type filtering based on intent
     - knowledge_query → search "document" + "conversation" types
     - exercise_recommendation → search "document" + "conversation" types
  4. Assess context quality (weighted avg similarity)
  5. Web search fallback if quality too low or too few results
  6. Return retrieval_results + retrieval_metadata
```

### Inputs from state

- `expanded_query` (or fallback to `query`)
- `intent` — determines which source_types to search
- `user_id` — for future user-scoped doc filtering

### Implementation details

**Singleton pattern**: Use the same `_get_embedding_service()` and `_get_pg_client()` pattern from `memory.py`. Import them:

```python
from langgraph_agents.nodes.memory import _get_embedding_service, _get_pg_client
```

Or better: create a shared `langgraph_agents/shared.py` with these singletons so both memory and retrieval can import from the same place. K's preference: shared module, avoids circular deps and cross-node imports.

**pgvector search**: Use `VectorBackend.search()`:

```python
from langgraph_agents.db.vector_backend import VectorBackend

pg = _get_pg_client()
vb = VectorBackend(pg)
embedding = await asyncio.to_thread(svc.embed_texts, query)
results = await vb.search(query_embedding=embedding, top_k=top_k, source_type="document")
```

Search multiple source_types and merge:
- `source_type="document"` — exercise KB, uploaded PDFs
- `source_type="conversation"` — relevant past conversations

**Quality assessment** (ported from `rag_pipeline.py` `_assess_context_quality`):

```python
def _assess_quality(results: list[dict], source_weights: dict) -> float:
    if not results:
        return 0.0
    total = sum(
        r["similarity"] * source_weights.get(r.get("metadata", {}).get("source_type", "document"), 0.8)
        for r in results
    )
    return total / len(results)
```

**Web search fallback**: Use existing `utils/web_search.py`:

```python
from utils.web_search import get_web_search_service
web_svc = get_web_search_service(max_results=cfg_max, timeout=cfg_timeout)
if web_svc.is_available():
    web_text = await asyncio.to_thread(web_svc.search_and_summarize, query)
```

Only trigger web search when:
- `len(results) < min_context_threshold` OR
- `quality < quality_threshold`

Web results get appended to `retrieval_results` with `source_type="web_search"`, `similarity=0.7`.

**Error handling**: Wrap everything in try/except. On failure, return empty results with RECOVERABLE error. Retrieval failure should NOT kill the pipeline — Reasoning can still generate a response from memory_context alone.

### Return dict

```python
return {
    "retrieval_results": results,  # list[dict] with id, content, metadata, similarity
    "retrieval_metadata": {
        "source_types_searched": ["document", "conversation"],
        "total_results": len(results),
        "web_search_used": bool,
        "quality_score": float,
        "elapsed_ms": float,
    },
}
```

If errors occurred, add `"errors": [...]` (same RECOVERABLE pattern as memory.py).

---

## Task 3: Create `langgraph_agents/shared.py` — shared singletons

Move the singleton pattern out of `memory.py` into a shared module:

```python
"""Shared singletons for LangGraph nodes — avoids re-creating heavy objects."""

_embedding_service = None
_pg_client = None

def get_embedding_service():
    global _embedding_service
    if _embedding_service is None:
        from memory.embedding_service import EmbeddingService
        _embedding_service = EmbeddingService()
    return _embedding_service

def get_pg_client():
    global _pg_client
    if _pg_client is None:
        from langgraph_agents.db.postgres import PostgresClient
        _pg_client = PostgresClient()
    return _pg_client
```

Then update `memory.py` to import from `shared.py` instead of defining its own singletons:

```python
from langgraph_agents.shared import get_embedding_service, get_pg_client
# remove _embedding_service, _pg_client, _get_embedding_service(), _get_pg_client()
```

And `retrieval.py` also imports from `shared.py`.

---

## Task 4: Implement `reasoning.py` — clinical analysis + response generation

**File**: `langgraph_agents/nodes/reasoning.py`

### Architecture

```
reasoning_node(state) →
  1. Build context from retrieval_results + memory_context
  2. Select prompt template based on intent
  3. LLM call via LLMGateway (reasoning_model, higher temp + max_tokens)
  4. Return reasoning_output
```

### Inputs from state

- `query`, `expanded_query`
- `intent`
- `retrieval_results` — from retrieval node
- `memory_context` — from memory node
- `conversation_history` — for conversational continuity

### Prompt design

Two prompt templates (not hardcoded strings — define at module level):

**Knowledge query prompt** (absorbs `_build_prompt` from rag_pipeline.py):

```
You are a knowledgeable health and wellness assistant.

## Retrieved Context
{formatted_context}

## Memory Context
{formatted_memory}

## Conversation History
{formatted_history}

## User Question
{query}

Instructions:
- Answer based on the retrieved context. Cite specific facts when possible.
- If context is insufficient, say so honestly — do not hallucinate.
- Use Vietnamese if the user's query is in Vietnamese.
- Keep response under 300 words unless the topic requires detail.
```

**Exercise recommendation prompt** (absorbs constraint extraction from double_rag.py + generation from rag_pipeline.py):

```
You are an expert physical therapist AI assistant.

## Clinical Context
{formatted_context}

## Patient Profile
{formatted_profile}

## Memory Context
{formatted_memory}

## User Question
{query}

Instructions:
- Recommend specific exercises with sets, reps, and safety cues.
- Extract any biomechanical constraints from the clinical context.
- Include safety warnings in bold.
- Mention when to stop or consult a professional.
- If context is insufficient for safe recommendations, recommend consulting a PT.
- Use Vietnamese if the user's query is in Vietnamese.

Respond in plain text (not JSON).
```

### Context formatting helpers

```python
def _format_retrieval_context(results: list[dict]) -> str:
    if not results:
        return "No relevant documents found."
    parts = []
    for i, r in enumerate(results, 1):
        source = r.get("metadata", {}).get("source_type", "unknown")
        parts.append(f"[{i}] ({source}, similarity={r['similarity']:.2f})\n{r['content']}")
    return "\n\n".join(parts)

def _format_memory_context(memory: dict) -> str:
    parts = []
    lt = memory.get("long_term", [])
    if lt:
        parts.append("Long-term memories:\n" + "\n".join(f"- {m['content']}" for m in lt[:3]))
    profile = memory.get("user_profile", {})
    if profile:
        parts.append(f"User profile: {profile}")
    return "\n".join(parts) if parts else "No relevant memory."

def _format_history(history: list[dict]) -> str:
    if not history:
        return "No prior conversation."
    recent = history[-5:]  # last 5 turns
    return "\n".join(f"{m['role']}: {m['content']}" for m in recent)
```

### LLM call

```python
from langgraph_agents.llm_gateway import LLMGateway

_CFG = _load_config()  # load reasoning model + temp + max_tokens from config
gateway = LLMGateway(model=_CFG.get("llm", {}).get("reasoning_model"))

response = await gateway.chat(
    messages=[
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_message},
    ],
    temperature=_CFG.get("llm", {}).get("temperature", {}).get("reasoning", 0.7),
    max_tokens=_CFG.get("llm", {}).get("max_tokens", {}).get("reasoning", 4096),
)
```

### Error handling

On LLM failure (timeout, API error): return CRITICAL error + empty `reasoning_output`. The error routing will send to `error_handler`.

On partial failure (e.g., context formatting issue): return RECOVERABLE error + best-effort response.

### Return dict

```python
return {
    "reasoning_output": response,  # str — the generated answer text
}
```

If errors: add `"errors": [...]`.

### Timing

Log elapsed time via `time.perf_counter()`. If Reasoning takes >3s, log as warning (helps us decide on Manager+Reasoning merge later).

---

## Task 5: Implement `validator.py` — validation + raw_answer assembly

**File**: `langgraph_agents/nodes/validator.py`

### Architecture

```
validator_node(state) →
  1. Check which fields are populated based on intent
  2. Build raw_answer from reasoning_output (or fallback)
  3. Enrich with metadata hints for conversation node
  4. Return raw_answer
```

### Validation rules by intent

| Intent | Required | Optional | Fallback if missing |
|--------|----------|----------|-------------------|
| `knowledge_query` | `reasoning_output` | `retrieval_results` | "Xin loi, toi khong tim duoc thong tin phu hop. Vui long thu cau hoi khac." |
| `exercise_recommendation` | `reasoning_output` | `retrieval_results` | "Xin loi, toi khong the dua ra bai tap luc nay. Vui long tham khao bac si." |
| `conversation` | (none — direct pass) | `memory_context` | State's `query` echoed with polite fallback |
| `visualize_motion` | (none — Phase 3) | — | "Motion visualization will be available soon." |
| `clarify` | (none — direct pass) | — | "Vui long cung cap them thong tin." |

### raw_answer assembly

```python
intent = state.get("intent", "conversation")
reasoning = state.get("reasoning_output", "")

if reasoning and reasoning.strip():
    raw_answer = reasoning.strip()
elif intent in ("knowledge_query", "exercise_recommendation"):
    raw_answer = _FALLBACK_MESSAGES[intent]
else:
    raw_answer = _FALLBACK_MESSAGES.get(intent, _FALLBACK_MESSAGES["conversation"])
```

### Metadata logging (debug only, not in state)

Log to `logger.info`:
- Which fields were populated vs empty
- Intent
- Whether fallback was used
- retrieval_results count (if any)
- quality_score from retrieval_metadata

### Return dict

```python
return {"raw_answer": raw_answer}
```

No errors expected from Validator itself — it's a pure data assembly node. If something is truly wrong, it was already caught by upstream nodes.

---

## Task 6: Update `routing.py` — add error check after retrieval/reasoning

Already wired in `graph.py` via `check_errors`. No changes needed to routing.py.

**Verify**: the existing graph edges handle the flow correctly:
- `retrieval → check_errors → reasoning` (or error_handler)
- `reasoning → check_errors → validator` (or error_handler)

This is already correct in `graph.py`. No changes needed.

---

## Task 7: Write tests — `tests/langgraph_agents/test_phase2_integration.py`

### Unit tests (no services needed)

1. **`test_format_retrieval_context`** — formatting helper produces expected output from mock results
2. **`test_format_memory_context`** — formats long_term + profile correctly, handles empty
3. **`test_format_history`** — truncates to last 5 turns
4. **`test_validator_fallback_knowledge`** — empty reasoning_output → Vietnamese fallback message
5. **`test_validator_fallback_exercise`** — empty reasoning_output → clinical safety fallback
6. **`test_validator_passes_reasoning`** — populated reasoning_output → passes through as raw_answer
7. **`test_quality_assessment`** — weighted quality scoring with source_weights
8. **`test_retrieval_graceful_degradation`** — retrieval node with PG/embedding down → empty results + RECOVERABLE error

### Integration tests (require GEMINI_API_KEY)

9. **`test_reasoning_knowledge_query`** — real LLM call with mock retrieval context → non-empty reasoning_output
10. **`test_reasoning_exercise_recommendation`** — real LLM call → response contains exercise-related content
11. **`test_reasoning_latency`** — reasoning should complete within 10s (generous for Gemini)

### Full graph tests (require GEMINI_API_KEY)

12. **`test_full_graph_knowledge_query`** — "What exercises help lower back pain?" → traverses retrieval + reasoning → final_answer non-empty
13. **`test_full_graph_exercise_recommendation`** — "Bai tap cho dau lung" → intent=exercise_recommendation → final_answer non-empty
14. **`test_full_graph_conversation_skips_retrieval`** — "Xin chao" → conversation intent → skips retrieval/reasoning → final_answer non-empty

### Test structure

```python
@pytest.mark.unit  # or @pytest.mark.integration
@pytest.mark.asyncio
async def test_xxx():
    ...
```

Use `_base_state()` helper from Phase 1 tests — copy or import it.

---

## Task 8: Update `config/langgraph.yaml` retrieval section (if not done in Task 1)

Ensure the full config is:

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
    stm_profile_ttl: 86400
    stm_session_ttl: 7200
    stm_patterns_ttl: 604800
    ltm_top_k: 5
    ltm_similarity_threshold: 0.3

  retrieval:
    top_k: 5
    min_context_threshold: 2
    quality_threshold: 0.4
    web_search_enabled: true
    web_search_max_results: 3
    web_search_timeout: 8
    source_weights:
      document: 1.0
      conversation: 0.9
      web_search: 0.75

  postgres:
    dsn: "postgresql://vva:vva_dev@localhost:5432/vva"
    pool_min: 2
    pool_max: 10
```

---

## Acceptance Criteria

1. `retrieval_node` returns real results from pgvector (or empty + RECOVERABLE when DB is down)
2. `retrieval_node` falls back to web search when pgvector results are insufficient
3. `reasoning_node` generates contextual responses via LLM for both knowledge + exercise intents
4. `reasoning_node` returns CRITICAL error when LLM API fails
5. `validator_node` builds raw_answer from reasoning_output, or uses fallback
6. Shared singletons in `shared.py` — no duplicate EmbeddingService/PostgresClient instances
7. `memory.py` updated to use `shared.py` (no behavior change, just import refactor)
8. Config updated with retrieval section
9. All unit tests pass: `pytest tests/langgraph_agents/test_phase2_integration.py -m unit`
10. Integration tests pass with GEMINI_API_KEY: `pytest tests/langgraph_agents/test_phase2_integration.py -m integration`
11. Full graph test: knowledge query → retrieval → reasoning → validator → conversation → dispatch → non-empty final_answer

---

## Execution Order

| Step | Task | Est. |
|------|------|------|
| 1 | Task 1 + Task 8: Update config/langgraph.yaml | 10m |
| 2 | Task 3: Create shared.py + update memory.py imports | 20m |
| 3 | Task 2: Implement retrieval.py | 90m |
| 4 | Task 4: Implement reasoning.py | 90m |
| 5 | Task 5: Implement validator.py | 30m |
| 6 | Task 7: Write tests | 60m |
| 7 | Run tests + fix issues | 30m |

---

## What is NOT in Phase 2

- **HyDE document generation**: Manager already provides `expanded_query`. HyDE is an optimization for Phase 3 (motion search with Kimodo).
- **Double-RAG motion search**: Phase 3. The conditioned motion search using constraints + HyDE is Kimodo's job.
- **Reflection/self-correction loop**: The RAG pipeline's iterative reflection is heavyweight. Omit for now. If response quality is low, add in Phase 6 (hardening).
- **Query reformulation loop**: Same — omit for now. The expanded_query from Manager is sufficient.
- **Structured JSON responses (exercises list)**: Phase 4/5 (Conversation node handles output formatting).
- **Streaming**: Phase 5 (SSE).
- **Source routing via SLM**: KnowledgeLibrarian uses qwen:0.5b for ambiguous routing. Our Manager already classifies intent. Not needed.
