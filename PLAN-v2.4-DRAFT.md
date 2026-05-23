# Plan: VVA (KineticChat) Re-Architecture — LangGraph Multi-Agent (v2.4)

> Architect: K | Reviewer: T | Developer: N | Date: 2026-05-23
> Status: **v2.4 — Plan-Execute Pattern, Dual-mode Conversation, Conditional LTM, Rule-based Grader**
> Base: v2.2 (K) + v2.3 debate (T) + architecture session (22-23/05/2026)

---

## Changelog v2.2 → v2.4

| # | Thay doi | Ly do |
|---|---------|-------|
| 1 | **Bo Validator Node** | Grader check quality. Empty output fallback tai FastAPI layer |
| 2 | **Bo Dispatch Node** | TTS = async tai FastAPI layer. Kimodo = MCP tool trong retriever |
| 3 | **Them Grader Node** (rule-based) | Thay validator. Check theo intent + retry max 1 lan |
| 4 | **Manager → Planner** (renamed + upgraded) | Intent classification + query expansion + structured plan output |
| 5 | **Memory chay TRUOC Planner** | Planner can context (STM + LTM + profile) de len plan chinh xac |
| 6 | **Retriever Agent** (LLM + ToolNode) | Execute plan cua planner. Goi pgvector @tool + MCP tools |
| 7 | **Reasoning → Synthesizer** (renamed) | Tong hop tool results + generate clinical response |
| 8 | **Conversation Dual Mode** | Styling mode (co reasoning_output) + Generation mode (conversation/clarify) |
| 9 | **pgvector KHONG phai MCP** | In-process @tool wrapper cho retriever, direct call cho memory LTM |
| 10 | **Kimodo + web_search = MCP servers** | External services, uniform MCP protocol |
| 11 | **LLMGateway → LangChain ChatModel** | Xoa custom LLM abstraction, dung LangChain native everywhere |
| 12 | **Accept LangChain/LangGraph coupling** | Bo yeu cau "pure Python nodes". Framework coupling accepted |
| 13 | **SSE via astream_events()** | LangGraph native streaming, emit stage events per node |
| 14 | **Token budget** | User-set limit, Annotated reducer, interrupt() pause |
| 15 | **Memory architecture moi** | STM: Redis 3 Q&A FIFO. LTM: conditional keyword trigger. Session: PostgreSQL |
| 16 | **TTS co dinh async** | Sau graph, fire tai FastAPI layer (Celery), khong trong graph |
| 17 | **SSE stage events** | Adopt tu v2.3 — progress updates cho tung node |

### Giu nguyen tu v2.2

- Error routing: CRITICAL / RECOVERABLE / IGNORABLE + error_handler node
- Persona system (MD files) + conversation node
- PostgreSQL (pgvector) + Redis architecture
- SSE + REST POST protocol
- Celery cho TTS background task
- Hybrid Edge-Cloud deployment target
- Phase structure (co update)

---

## 1. Architecture Overview

### 1.1 Graph Nodes (7 nodes)

| # | Node | Vai tro | LLM calls | Implementation |
|---|------|---------|-----------|----------------|
| 1 | `memory` | Redis STM + PostgreSQL LTM (conditional) | 0 | LangChain compatible (asyncpg + redis) |
| 2 | `planner` | Intent classification + query expansion + structured plan | 1 (fast model) | LangChain ChatModel + `with_structured_output()` |
| 3 | `retriever_agent` | Execute plan: goi pgvector @tool + MCP tools song song | 1+ | LangGraph ToolNode + MCP adapters |
| 4 | `synthesizer` | Tong hop tool results → generate clinical response | 1 (heavy model) | LangChain ChatModel |
| 5 | `grader` | Rule-based quality check, intent-based validation | 0 | Rule engine |
| 6 | `conversation` | Dual mode: styling (co content) hoac generation (conversation/clarify) | 1 | LangChain ChatModel |
| 7 | `error_handler` | Graceful Vietnamese error message tu errors list | 0 | Template-based |

### 1.2 Infrastructure (ngoai graph)

| Component | Vai tro | Nam o dau |
|-----------|---------|-----------|
| TTS (VieNeu) | Speech synthesis | Async function tai FastAPI layer, fire sau graph xong |
| MCP Servers | Tool providers (Kimodo, web search) | Standalone services |
| Celery | TTS background task + future heavy tasks | Worker on Edge machine |
| FastAPI | REST endpoints + SSE streaming (astream_events) + fallback | Gateway layer |

---

## 2. Graph Flow

### 2.1 Flow Diagram

```
START
  └─► memory (Redis STM + conditional PostgreSQL LTM)
        │
        ├─ [check_errors → CRITICAL?] ──► error_handler ──► conversation ──► END
        │
        └─► planner (intent + query expansion + structured plan)
              │
              ├─ [check_errors → CRITICAL?] ──► error_handler ──► conversation ──► END
              │
              ├─ needs_clarification ──► conversation (generation mode: style question) ──► END
              │
              ├─ intent = conversation ──► conversation (generation mode: generate response) ──► END
              │
              └─ intent = knowledge / exercise / motion
                    │
                    └─► retriever_agent (execute plan: pgvector @tool + MCP tools)
                          │
                          ├─ [check_errors → CRITICAL?] ──► error_handler ──► conversation ──► END
                          │
                          └─► synthesizer (LLM generate clinical response)
                                │
                                ├─ [check_errors → CRITICAL?] ──► error_handler ──► conversation ──► END
                                │
                                └─► grader (rule-based check)
                                      ├─ FAIL + retry_count < 1 ──► retriever_agent (retry with feedback)
                                      ├─ FAIL + retry_count >= 1 ──► conversation (pass with warning)
                                      └─ PASS ──► conversation (styling mode) ──► END

END ──► [FastAPI layer: async TTS if output_mode = speech/both]
```

### 2.2 Routing Rules

| Sau node | Condition | Di dau |
|----------|-----------|--------|
| memory | CRITICAL error | error_handler |
| memory | OK | planner |
| planner | CRITICAL error | error_handler |
| planner | needs_clarification = true | conversation (generation mode) |
| planner | intent = conversation | conversation (generation mode) |
| planner | intent = knowledge / exercise / motion | retriever_agent |
| retriever_agent | CRITICAL error | error_handler |
| retriever_agent | OK | synthesizer |
| synthesizer | CRITICAL error | error_handler |
| synthesizer | OK | grader |
| grader | FAIL + retry_count < 1 | retriever_agent |
| grader | FAIL + retry_count >= 1 | conversation (+ warning) |
| grader | PASS | conversation (styling mode) |
| conversation | always | END |
| error_handler | always | conversation |

### 2.3 Special Paths

**Conversation / Clarify → Skip retriever + synthesizer + grader**

Intent `conversation` ("Xin chao") va `needs_clarification` di thang conversation node. Khong qua retriever/synthesizer/grader vi:
- Khong co retrieval → khong co gi de synthesize → khong co gi de grade
- Conversation node o generation mode tu generate response tu persona + query + memory_context
- Hoac style clarification question tu planner's plan

**Error path → conversation styles error message**

error_handler ghi error message vao `reasoning_output`. Conversation node styles error message theo persona truoc khi tra ve user.

---

## 3. State Schema

### 3.1 AgentState (mutable, thay doi qua nodes)

```python
from langchain_core.messages import add_messages

class AgentState(TypedDict):
    # LangGraph message passing (retriever_agent ToolNode)
    messages: Annotated[list, add_messages]

    # Planner output
    intent: str                     # conversation | knowledge_query | exercise_recommendation | visualize_motion | clarify
    confidence: float
    expanded_query: str
    plan: dict                      # structured plan (required_outputs, search_strategy, etc.)
    needs_clarification: bool       # True → route to conversation generation mode

    # Memory output
    memory_context: dict            # {short_term: [...], long_term: {...}, user_profile: {...}}

    # Synthesizer output
    reasoning_output: str           # clinical response OR error message (from error_handler)

    # Grader output
    grader_result: str              # "pass" | "retry" | "pass_with_warning"
    grader_warning: Optional[str]   # warning message appended to response
    grader_feedback: Optional[str]  # feedback for retry (sent to retriever_agent)
    retry_count: int                # 0 initially, max 1

    # Conversation output
    final_answer: str

    # Token tracking
    total_tokens: Annotated[int, operator.add]  # auto-accumulated via reducer

    # Error tracking (append-only)
    errors: Annotated[list[dict], operator.add]  # [{node, severity, message, timestamp}]
```

### 3.2 RunnableConfig (immutable, set 1 lan boi FastAPI layer)

```python
config = {
    "configurable": {
        "user_id": "user-123",
        "session_id": "sess-456",
        "query": "bai tap cho dau lung",
        "persona_id": "eca_default",
        "output_mode": "text",           # "text" | "speech" | "both"
        "request_id": "req-789",
        "token_limit": None,             # user-set, None = disabled
    }
}
# Invoke: await graph.astream_events(initial_state, config=config)
# Nodes access: config["configurable"]["user_id"]
```

### 3.3 Removed from v2.2

| Field | Ly do |
|-------|-------|
| `user_id`, `session_id`, `query`, `persona_id`, `output_mode`, `request_id` | Moved to RunnableConfig (immutable) |
| `conversation_history` | Replaced by memory_context.short_term (3 Q&A pairs from Redis) |
| `raw_answer` | Validator removed. Conversation reads reasoning_output |
| `voice_path` | TTS tai FastAPI layer, doc persona config truc tiep |
| `motion_task_id`, `speech_task_id`, `motion_pending`, `motion_payload` | Dispatch removed |
| `retrieval_results`, `retrieval_metadata` | Nam trong `messages` (ToolMessage tu retriever_agent) |

---

## 4. Memory Architecture

### 4.1 Short-Term Memory (STM) — Redis

| Aspect | Detail |
|--------|--------|
| **Storage** | Redis key `stm:{session_id}` |
| **Format** | JSON array, max 3 Q&A pairs |
| **Eviction** | FIFO: new pair drops oldest when > 3 |
| **TTL** | 2 hours (session duration) |

```json
[
  {"q": "bai tap cho dau lung", "a": "Co 3 bai tap tot cho lung...", "ts": "2026-05-23T10:00:00Z"},
  {"q": "bai tap nao de nhat?", "a": "Bird-dog la bai tap nhe...", "ts": "2026-05-23T10:01:00Z"},
  {"q": "lam bao nhieu lan?", "a": "3 hiep, moi hiep 10 lan...", "ts": "2026-05-23T10:02:00Z"}
]
```

### 4.2 Long-Term Memory (LTM) — PostgreSQL + pgvector (conditional)

LTM **KHONG chay mac dinh**. Chi trigger khi user query co recall keywords.

**Keyword detection (regex, khong dung LLM):**

```python
_RECALL_PATTERNS = [
    r"(con )?nho",                          # "ban con nho...", "nho lai..."
    r"lan truoc|truoc do",                  # "lan truoc toi hoi..."
    r"tuan truoc|hom qua|thang truoc",      # temporal markers
    r"da (noi|hoi|trao doi|lam)",
    r"remember|last time|previously",
]
```

**LTM flow — 3 nhanh:**

```
Memory detects recall keywords?
  │
  ├─ NO → skip LTM, return STM only
  │
  └─ YES → PostgreSQL query sessions by timestamp + user_id
        │
        ├─ 0 sessions match
        │   → memory_context.long_term = {found: false}
        │   → planner decides: proceed as knowledge query or inform user
        │
        ├─ 1 session match
        │   → pgvector search WITHIN that session's messages
        │   → return matched content in memory_context.long_term.results
        │   → planner proceeds with context
        │
        └─ 2+ sessions match
            → memory_context.long_term = {ambiguous: true, sessions: [...summaries]}
            → planner sees ambiguity → sets needs_clarification = true
            → conversation asks: "Toi tim thay nhieu phien truoc do. Ban nho them chi tiet gi?"
            → user provides keywords → new request → memory pgvector search
```

**Temporal keyword handling:**

"Tuan truoc" = calendar week truoc (Monday-Sunday). Neu khong match → mo rong range +-3 ngay.

### 4.3 Session Persistence — PostgreSQL

Sessions luu tren PostgreSQL. Support reopen session cu.

```sql
-- Reuse existing conversations table
CREATE TABLE conversations (
    id UUID PRIMARY KEY,
    user_id UUID REFERENCES users(id),
    session_id UUID UNIQUE,
    messages JSONB,           -- full conversation history (all messages)
    summary TEXT,             -- auto-generated summary for session listing
    created_at TIMESTAMPTZ,
    updated_at TIMESTAMPTZ
);
```

**Session reopen** (Phase 5+ API feature):

```
GET  /sessions?user_id=...         → list sessions (id, summary, created_at)
POST /sessions/{id}/resume         → load messages from PostgreSQL
                                   → populate Redis STM with last 3 Q&A
                                   → return session_id for SSE connection
```

### 4.4 STM Read/Write Lifecycle

```
Request lifecycle:
1. FastAPI receives POST /chat
2. Read STM from Redis (last 3 Q&A) → pass to graph as initial memory hint
3. Invoke graph via astream_events()
4. Memory node reads STM from Redis internally
5. Graph runs → returns final_answer
6. FastAPI writes complete Q+A pair to Redis STM (FIFO, drop oldest if > 3)
7. FastAPI writes to PostgreSQL conversations table (full history, for LTM + session reopen)
```

Graph chi DOC memory. FastAPI layer DOC + GHI.

### 4.5 pgvector Usage Distinction

| | Memory LTM (Section 4.2) | Retriever Agent (Section 6) |
|---|---|---|
| **Tim gi** | User's past conversations | Medical knowledge base |
| **source_type filter** | `"conversation"` | `"document"`, `"humanml3d"` |
| **Khi nao** | Chi khi recall keywords detected | Moi knowledge/exercise/motion query |
| **Ai goi** | Memory node — direct call (in-process) | Retriever agent — via @tool wrapper |
| **Can LLM quyet dinh?** | Khong (keyword detection) | Co (LLM chon tool) |

Cung bang `embeddings`, khac `source_type` filter. Khong conflict.

---

## 5. Planner Node (was Manager)

### 5.1 Responsibilities

Planner la **brain** cua pipeline. Nhan query + memory_context → output:
1. **Intent classification**: conversation / knowledge_query / exercise_recommendation / visualize_motion / clarify
2. **Query expansion**: them anatomical/physiotherapy synonyms
3. **Structured plan**: required outputs, search strategy, detected constraints
4. **Clarification detection**: neu thieu thong tin → set needs_clarification + cau hoi

Planner **chi len ke hoach**, KHONG goi tool. Retriever Agent execute plan.

### 5.2 Plan Output Schema

Planner dung `with_structured_output()` tra Pydantic model:

```python
class PlanOutput(BaseModel):
    intent: str                         # classified intent
    confidence: float                   # 0.0-1.0
    expanded_query: str                 # query with added synonyms
    needs_clarification: bool           # True → route to conversation
    clarification_question: Optional[str]  # question for user (if needs_clarification)
    required_outputs: list[str]         # what the response MUST contain
    search_strategy: list[str]          # suggested tools for retriever
    constraints_detected: list[str]     # detected physical constraints
    notes: Optional[str]                # additional context for retriever/synthesizer
```

**Vi du per intent:**

Exercise recommendation:
```json
{
  "intent": "exercise_recommendation",
  "confidence": 0.92,
  "expanded_query": "bai tap vat ly tri lieu cho dau cot song that lung man tinh",
  "needs_clarification": false,
  "clarification_question": null,
  "required_outputs": ["exercise_name", "description", "sets_reps", "safety_warnings"],
  "search_strategy": ["pgvector_search", "web_search_if_low_quality"],
  "constraints_detected": ["lower_back", "chronic_pain"],
  "notes": "User mentioned chronic pain, prioritize gentle exercises"
}
```

Clarification needed:
```json
{
  "intent": "exercise_recommendation",
  "confidence": 0.6,
  "expanded_query": "bai tap cho dau lung",
  "needs_clarification": true,
  "clarification_question": "Ban co the mo ta ro hon vung dau lung khong? Phia tren hay phia duoi? Dau cap tinh hay man tinh?",
  "required_outputs": [],
  "search_strategy": [],
  "constraints_detected": [],
  "notes": "Insufficient detail for safe exercise recommendation"
}
```

### 5.3 Plan Templates per Intent

Planner system prompt chua huong dan cho tung intent type. Template dinh nghia `required_outputs` toi thieu:

| Intent | Required outputs | Notes |
|--------|-----------------|-------|
| knowledge_query | `["answer", "sources"]` | Phai co evidence-based answer |
| exercise_recommendation | `["exercise_name", "description", "sets_reps", "safety_warnings"]` | Phai co safety |
| visualize_motion | `["motion_description", "joint_constraints"]` | Can constraints cho Kimodo |
| conversation | `["greeting_response"]` | Don gian |
| clarify | `["clarification_question"]` | Hoi ro rang |

### 5.4 Clarification Flow

Planner set `needs_clarification = true` khi:
- confidence < 0.5
- Thieu thong tin critical (vd: exercise rec nhung khong biet vung dau)
- Memory LTM ambiguous (2+ sessions match)

Flow: planner → conversation (generation mode) → conversation doc `plan.clarification_question` + style theo persona → END.

---

## 6. Retriever Agent

Retriever Agent **execute plan** tu planner. Nhan plan → chon tools → goi song song → tra ket qua.

### 6.1 pgvector @tool (in-process)

pgvector search la LangChain `@tool` function, chay in-process (0ms network overhead):

```python
from langchain_core.tools import tool

@tool
def pgvector_search(query: str, top_k: int = 5, source_type: str = "document") -> list[dict]:
    """Search internal medical knowledge base for exercises, treatments, and PT theory.
    
    Use for knowledge_query and exercise_recommendation intents.
    Returns documents ranked by cosine similarity.
    
    Args:
        query: semantic search query (expanded by planner)
        top_k: number of results (default 5)
        source_type: filter by "document" or "humanml3d"
    """
    # Calls VectorBackend directly — no network hop
    ...
```

### 6.2 MCP Tools (Kimodo, web_search)

Kimodo va web_search la MCP servers. `langchain-mcp-adapters` auto-discover tools.

Retriever Agent's ToolNode co ca 3 loai tool:
- `pgvector_search` — @tool (in-process)
- `generate_motion` — MCP tool (Kimodo, port 5001)
- `search_medical` — MCP tool (web_search, port 5020)

LLM quyet dinh goi tool nao dua tren planner's `search_strategy` va `required_outputs`.

### 6.3 Implementation Sketch

```python
# langgraph_agents/nodes/retriever_agent.py

async def build_retriever_tools():
    """Load pgvector @tool + MCP tools. Called at startup."""
    # In-process tools
    tools = [pgvector_search]
    
    # MCP tools from config
    config = load_mcp_config()  # config/mcp_servers.yaml
    if config:
        client = MultiServerMCPClient(config)
        await client.__aenter__()
        tools.extend(client.get_tools())
    
    return tools

# In graph construction:
# retriever_llm = ChatModel.bind_tools(all_tools)
# tool_node = ToolNode(all_tools)  # parallel execution
```

### 6.4 Retriever System Prompt

```
You are a tool execution agent. Follow the plan strictly.

Plan from planner:
{plan}

Rules:
- Execute the search_strategy from the plan
- For knowledge/exercise queries, ALWAYS search pgvector first
- Call multiple tools in parallel when possible
- If a tool fails, note the error and continue with available results
- Do NOT generate answers — only retrieve and return tool results
```

---

## 7. Synthesizer

### 7.1 Role

Synthesizer la **heavy thinking LLM**. Nhan tool results + memory context + plan → **generate** clinical response.

Day KHONG chi la aggregation — synthesizer **suy nghi**, **phan tich**, va **viet** response hoan chinh (huong dan dong tac, giai thich y khoa, safety warnings).

### 7.2 Input

| Source | Field | Content |
|--------|-------|---------|
| Retriever results | `messages` (ToolMessages) | Raw tool outputs |
| Memory | `memory_context` | STM (last 3 Q&A) + LTM (if any) + user profile |
| Planner | `plan` | Required outputs, constraints, notes |
| Config | `configurable.query` | Original user query |

### 7.3 Output

```python
return {
    "reasoning_output": "...",  # complete clinical response
    "total_tokens": response.usage_metadata["total_tokens"],
}
```

### 7.4 Prompt Structure

```
System: You are an expert physical therapist AI assistant.

## Plan Requirements
{plan.required_outputs}
{plan.constraints_detected}
{plan.notes}

## Retrieved Context
{formatted tool results from messages}

## Patient Memory
{memory_context.short_term}
{memory_context.user_profile}

## Instructions
- Generate response that covers ALL required_outputs from the plan
- Use Vietnamese if user query is in Vietnamese
- Include safety warnings for exercise recommendations
- Cite sources when available
- Keep under 500 words unless topic requires detail

User: {query}
```

---

## 8. Grader Node — Rule-based Quality Check

### 8.1 Grader Rules (by intent label)

```python
GRADER_RULES = {
    # Universal rules
    "non_empty": lambda s: bool(s.get("reasoning_output", "").strip()),
    "under_word_limit": lambda s: len(s.get("reasoning_output", "").split()) <= 500,
    
    # Intent-specific rules
    "knowledge_has_content": {
        "intents": ["knowledge_query"],
        "check": lambda s: len(s.get("reasoning_output", "")) > 50,
    },
    "exercise_has_steps": {
        "intents": ["exercise_recommendation"],
        "check": lambda s: any(marker in s.get("reasoning_output", "")
                              for marker in ["1.", "2.", "-", "buoc", "lan", "hiep"]),
    },
    "motion_has_output": {
        "intents": ["visualize_motion"],
        "check": lambda s: any(
            "generate_motion" in str(m) for m in s.get("messages", [])
        ),
    },
}
```

### 8.2 Retry Logic

```
grader receives synthesizer output
    │
    ├─ Run all applicable rules (by intent from state)
    │
    ├─ ALL PASS
    │   → return {"grader_result": "pass"}
    │   → route to conversation (styling mode)
    │
    ├─ ANY FAIL + retry_count == 0
    │   → return {
    │       "grader_result": "retry",
    │       "retry_count": 1,
    │       "grader_feedback": "Thieu buoc thuc hien bai tap. Can them sets/reps."
    │   }
    │   → route to retriever_agent (retry)
    │   → retriever_agent reads grader_feedback from state
    │   → re-calls tools → synthesizer re-generates → grader checks again
    │
    └─ ANY FAIL + retry_count >= 1
        → return {
            "grader_result": "pass_with_warning",
            "grader_warning": "Cau tra loi co the chua day du. Vui long tham khao bac si."
        }
        → route to conversation (warning appended to reasoning_output)
```

**Retry path**: grader → retriever_agent → synthesizer → grader (with retry_count = 1). Worst case = 2x pipeline cost. Acceptable for quality assurance.

**grader_feedback injection**: Retriever agent reads `state["grader_feedback"]` va include trong system prompt de LLM biet tai sao phai retry va can tim them gi.

---

## 9. Conversation Node (Dual Mode)

### 9.1 Styling Mode

**Trigger**: intent = knowledge/exercise/motion AND reasoning_output co noi dung.

```
Input: reasoning_output (clinical response from synthesizer)
Process: LLM restyle theo persona MD file
Output: final_answer (styled response)
```

- Load persona MD → build system prompt
- LLM nhan reasoning_output → restyle tone, formatting, language
- KHONG them medical info moi — chi restyle

### 9.2 Generation Mode

**Trigger**: intent = conversation HOAC needs_clarification = true HOAC reasoning_output rong.

```
Input: query (from RunnableConfig) + plan (from state) + memory_context (from state)
Process: LLM generate response truc tiep tu persona + inputs
Output: final_answer (generated response)
```

3 sub-cases:

| Case | Input | Response |
|------|-------|----------|
| Conversation ("Xin chao") | query + persona | Greeting in persona style |
| Clarification | plan.clarification_question + persona | Styled clarification question |
| Error (from error_handler) | reasoning_output (error msg) + persona | Styled error message |

### 9.3 Detection Logic

```python
async def conversation_node(state: AgentState, config: RunnableConfig) -> dict:
    persona = load_persona(config["configurable"]["persona_id"])
    reasoning = state.get("reasoning_output", "")
    intent = state.get("intent", "conversation")
    needs_clarification = state.get("needs_clarification", False)
    
    if reasoning and reasoning.strip() and not needs_clarification:
        # STYLING MODE: restyle existing content
        final = await style_with_persona(persona, reasoning, intent)
    else:
        # GENERATION MODE: generate from scratch
        query = config["configurable"]["query"]
        plan = state.get("plan", {})
        memory = state.get("memory_context", {})
        final = await generate_with_persona(persona, query, plan, memory, intent)
    
    return {"final_answer": final}
```

---

## 10. Error Handler

error_handler ghi error message vao `reasoning_output` (KHONG phai `raw_answer`).

```python
async def error_handler_node(state: AgentState) -> dict:
    errors = state.get("errors", [])
    critical = [e for e in errors if e.get("severity") == ErrorSeverity.CRITICAL]

    if critical:
        msg = "Xin loi, he thong dang gap su co. Vui long thu lai sau."
    else:
        msg = "Da co loi nho, nhung toi van co gang tra loi."

    return {"reasoning_output": msg}
```

Flow: error_handler → conversation (styling mode, vi reasoning_output co noi dung) → END.

---

## 11. MCP Architecture

### 11.1 MCP Servers (Kimodo + web_search ONLY)

pgvector **KHONG phai MCP server**. Chi Kimodo va web_search dung MCP protocol.

```
┌─────────────────────────────────────────────────────────┐
│  Retriever Agent (LangGraph ToolNode)                   │
│                                                         │
│  Tools available:                                       │
│  ┌──────────────────┐  ┌──────────────┐ ┌────────────┐ │
│  │ pgvector_search   │  │ Kimodo MCP   │ │ Web Search │ │
│  │ @tool (in-process)│  │ Port 5001    │ │ MCP 5020   │ │
│  └──────────────────┘  └──────────────┘ └────────────┘ │
│         │                     ▲               ▲        │
│    direct call           MCP Protocol    MCP Protocol   │
└─────────────────────────────────────────────────────────┘
```

### 11.2 MCP Server Config

```yaml
# config/mcp_servers.yaml
mcp_servers:
  kimodo_motion:
    url: "http://localhost:5001/mcp"
    transport: "streamable_http"
  web_search:
    url: "http://localhost:5020/mcp"
    transport: "streamable_http"
  # External (user-added, can approval)
  # google_calendar:
  #   url: "https://mcp.example.com/gcal"
  #   transport: "streamable_http"
  #   api_token: "${GCAL_TOKEN}"
  #   requires_approval: true
```

**Them MCP server moi = them entry trong config. Khong sua code retriever.**

### 11.3 MCP Tool Specifications

#### kimodo_motion MCP Server

| Tool | `generate_motion` |
|------|-------------------|
| **Khi nao goi** | Khi user yeu cau xem/hien thi/mo phong mot chuyen dong cu the. CHI goi khi intent = `visualize_motion` hoac khi user noi ro "cho toi xem", "mo phong", "animate". KHONG goi cho exercise recommendation thong thuong. |
| **Input** | `prompt: str` — mo ta chuyen dong bang ngon ngu tu nhien. `constraints: list[{joint: str, angle: float}]` (optional) — rang buoc goc khop. |
| **Output** | `{video_url: str, duration_sec: float, format: "mp4", joints_used: list[str]}` |
| **Latency** | 5-10s (GPU inference) |

#### web_search MCP Server

| Tool | `search_medical` |
|------|-------------------|
| **Khi nao goi** | Khi knowledge base noi bo (pgvector) khong du thong tin, hoac user hoi ve chu de moi/cap nhat ma tai lieu local chua co. Dung lam fallback khi pgvector tra ket qua similarity thap (< 0.5). |
| **Input** | `query: str` — search query. `max_results: int` (default 3). `domain_filter: str` (optional). |
| **Output** | `list[{title, snippet, url, source_domain}]` |
| **Latency** | 1-3s |

### 11.4 Approval Rules

| Trigger | Khi nao | Cach xu ly |
|---------|---------|-------------|
| **External MCP voi API token** | Config co `requires_approval: true` | SSE `approval_required` → user approve → tiep tuc |
| **Token budget > threshold** | total_tokens > token_limit * 0.8 | interrupt() → SSE event → user approve/stop |

Internal tools (pgvector @tool, Kimodo, web_search) **KHONG can approval**.

---

## 12. Token Budget

### 12.1 Tracking via State Reducer

```python
# In AgentState:
total_tokens: Annotated[int, operator.add]  # auto-accumulated

# Each LLM-calling node returns token count:
return {
    "reasoning_output": response.content,
    "total_tokens": response.usage_metadata["total_tokens"],
}
```

LangChain standardized `usage_metadata` — works across Gemini, Claude, OpenAI. No provider-specific parsing needed.

### 12.2 Interrupt Mechanism

```python
from langgraph.types import interrupt

async def synthesizer_node(state, config):
    token_limit = config["configurable"].get("token_limit")
    if token_limit and state.get("total_tokens", 0) > token_limit * 0.8:
        interrupt("Token budget approaching limit")
    # ... rest of node logic
```

**Khi interrupt() duoc goi:**
1. Graph "dong bang" state → luu vao checkpointer (PostgresSaver)
2. FastAPI gui SSE event: `approval_required` + reason
3. User bam "Dong y" → client gui REST POST
4. Server resume graph voi `Command(resume=True)` + same thread_id
5. Graph "ra dong" va chay tiep

**Neu user khong set token_limit** → toan bo mechanism bi skip. Khong co check, khong co interrupt.

---

## 13. TTS — Async tai FastAPI Layer

TTS **KHONG nam trong LangGraph graph**. Graph ket thuc sau `conversation` node.

```python
# FastAPI layer (sketch)
@app.post("/chat")
async def chat(request: ChatRequest):
    # 1. Stream graph via astream_events
    async for event in graph.astream_events(state, config=config):
        if event["event"] == "on_chat_model_stream":
            await sse_send(session_id, "token", event["data"]["chunk"].content)
        # ... other event handling
    
    # 2. Get final result
    final_answer = result["final_answer"]
    
    # 3. Fire TTS async (if needed)
    if request.output_mode in ("speech", "both") and final_answer:
        task = celery_app.send_task(
            "langgraph.synthesize_speech",
            args=(final_answer, request.request_id, request.session_id),
        )
        # SSE event when done: speech_ready + audio URL
    
    # 4. Empty response fallback
    if not final_answer:
        await sse_send(session_id, "token", "Xin loi, toi khong the xu ly yeu cau nay.")
    
    await sse_send(session_id, "done", {})
```

---

## 14. SSE Event Schema

SSE su dung `astream_events()` tu LangGraph. FastAPI forward events to client.

```typescript
type SSEEvent =
  // Progress tracking (moi node bat dau/ket thuc)
  | { event: "stage";             data: { node: string, status: "started" | "complete" } }
  
  // Tool execution progress (tu retriever_agent)
  | { event: "tool_executing";    data: { tool: string, status: "started" | "complete" } }
  | { event: "tool_output";       data: { tool: string, summary: string } }
  
  // Approval (external MCP hoac token budget)
  | { event: "approval_required"; data: { reason: "external_tool" | "token_budget", details?: string } }
  
  // Streaming response (tu conversation node)
  | { event: "token";             data: { content: string } }
  
  // Async results (tu Celery)
  | { event: "speech_ready";      data: { url: string } }
  
  // System
  | { event: "error";             data: { code: string, message: string } }
  | { event: "done";              data: {} }
```

---

## 15. File Structure

```
agenticRAG/agentic_rag_gemini/
  langgraph_agents/
    __init__.py
    state.py                        # AgentState TypedDict (updated v2.4)
    graph.py                        # StateGraph construction (updated flow)
    nodes/
      __init__.py
      planner.py                    # RENAME from manager.py — intent + plan
      memory.py                     # UPDATE — STM/LTM architecture
      retriever_agent.py            # NEW — LLM + ToolNode (pgvector @tool + MCP)
      synthesizer.py                # RENAME from reasoning.py — generate clinical response
      grader.py                     # NEW — rule-based quality check
      conversation.py               # UPDATE — dual mode (styling + generation)
      error_handler.py              # UPDATE — writes reasoning_output
    tools/                          # NEW — in-process tool wrappers
      pgvector_tool.py              # @tool wrapper for pgvector search
    routing.py                      # UPDATE — planner routing + grader retry
    mcp/                            # NEW — MCP server implementations
      kimodo_server.py              # Kimodo wrapped as MCP server
      web_search_server.py          # Web search MCP server
    personas/
      eca_default.md                # Default persona — KEEP

config/
  mcp_servers.yaml                  # NEW — MCP server registry (Kimodo + web_search)
  langgraph.yaml                    # Existing config (extended)
```

### Changed Files

| File | Action | Detail |
|------|--------|--------|
| `manager.py` | RENAME → `planner.py` | Intent + query expansion + structured plan |
| `reasoning.py` | RENAME → `synthesizer.py` | Generate clinical response from tool results |
| `memory.py` | UPDATE | STM (3 Q&A) + conditional LTM + keyword detection |
| `conversation.py` | UPDATE | Dual mode: styling + generation |
| `error_handler.py` | UPDATE | Write `reasoning_output` instead of `raw_answer` |
| `state.py` | UPDATE | New fields: plan, needs_clarification, grader_*, total_tokens |
| `graph.py` | UPDATE | New flow: memory → planner → routing |
| `routing.py` | UPDATE | Planner routing + grader retry + clarification |

### Removed Files

| File | Reason |
|------|--------|
| `nodes/validator.py` | Replaced by grader.py |
| `nodes/dispatch.py` | TTS at FastAPI layer, Kimodo is MCP tool |
| `nodes/retrieval.py` | Replaced by retriever_agent.py |
| `llm_gateway.py` | Replaced by LangChain ChatModel |

### New Files

| File | Purpose |
|------|---------|
| `nodes/planner.py` | (renamed from manager.py) |
| `nodes/retriever_agent.py` | LLM + ToolNode executor |
| `nodes/grader.py` | Rule-based quality check |
| `tools/pgvector_tool.py` | @tool wrapper for pgvector |
| `mcp/kimodo_server.py` | Kimodo MCP server |
| `mcp/web_search_server.py` | Web search MCP server |
| `config/mcp_servers.yaml` | MCP config |

---

## 16. Dependencies

```
# Core
langgraph>=0.2.0
langchain-core>=0.3.0

# LangChain LLM providers
langchain-google-genai>=2.0.0      # Gemini via LangChain ChatModel
# langchain-anthropic>=0.3.0       # Claude (future option)

# MCP Integration
langchain-mcp-adapters>=0.1.0      # MCP → LangChain tool conversion
mcp>=1.0.0                         # MCP protocol SDK (for building MCP servers)

# Database
asyncpg>=0.29.0
pgvector>=0.3.0
alembic>=1.13.0

# SSE
sse-starlette>=2.0.0

# Existing
redis>=5.0.0
celery>=5.4.0
httpx
```

### Removed in v2.4

| Package | Reason |
|---------|--------|
| (custom `llm_gateway.py`) | Replaced by `langchain-google-genai` ChatModel |

---

## 17. Phased Implementation

### Phase 0: Foundation (DONE)
- Directory structure, state.py, graph.py with stub nodes
- Error routing + error_handler
- PostgreSQL + pgvector Docker setup
- Smoke tests + error tests

### Phase 1: Manager + Memory (DONE)
- manager.py — intent classification
- memory.py — Redis STM + pgvector LTM
- db/postgres.py + db/vector_backend.py

### Phase 2: Retrieval + Reasoning (DONE — needs refactor in Phase 2.5)
- retrieval.py + reasoning.py implemented
- validator.py implemented (to be replaced)
- Tests passing (43/43)

### Phase 2.5: Architecture Refactor (implement v2.4 changes)

| Task | Detail |
|------|--------|
| **RENAME** manager.py → planner.py | Add structured plan output, clarification detection |
| **RENAME** reasoning.py → synthesizer.py | Update to read from messages + plan |
| **REPLACE** retrieval.py → retriever_agent.py | LLM + ToolNode, pgvector @tool + MCP |
| **REPLACE** validator.py → grader.py | Rule-based quality check |
| **REMOVE** dispatch.py | TTS at FastAPI layer |
| **REMOVE** llm_gateway.py | Replace with LangChain ChatModel |
| **CREATE** tools/pgvector_tool.py | @tool wrapper for pgvector |
| **CREATE** config/mcp_servers.yaml | MCP server registry |
| **UPDATE** state.py | New fields, remove old fields |
| **UPDATE** graph.py | memory → planner → routing, grader retry loop |
| **UPDATE** routing.py | Planner routing + clarification + grader retry |
| **UPDATE** memory.py | STM 3 Q&A FIFO + conditional LTM + keyword detection |
| **UPDATE** conversation.py | Dual mode (styling + generation) |
| **UPDATE** error_handler.py | Write reasoning_output |
| **UPDATE** tests | Adapt to new node names + grader + dual mode tests |

### Phase 3: MCP Servers + Kimodo + VieNeu-TTS
- Implement Kimodo MCP server (GPU motion generation)
- Implement web search MCP server
- VieNeu-TTS as Celery task (called from FastAPI layer, not graph)
- Test: retriever_agent discovers + calls MCP tools in parallel
- Test: grader retry → re-call tools → pass with warning if fail

### Phase 4: Conversation + Personas
- Persona MD files (eca_default + 2-3 additional)
- conversation.py — dual mode fully wired
- Test: same reasoning_output → different styled outputs per persona
- Test: conversation intent → generation mode → persona greeting
- Test: clarification → generation mode → persona-styled question

### Phase 5: SSE Streaming + Frontend + Session Management
- SSE endpoint via astream_events() with stage events
- REST POST /chat endpoint
- Token streaming from conversation node
- Async TTS via Celery → SSE speech_ready event
- Session reopen API (GET /sessions, POST /sessions/{id}/resume)
- Empty response fallback at FastAPI layer
- Token budget interrupt + approval flow
- Frontend: EventSource + REST POST
- Test: end-to-end streaming in browser

### Phase 6: Production Hardening
- Structured logging, tracing, circuit breakers
- Comprehensive test suite
- Health check endpoints

### Phase 7: Hybrid Edge-Cloud Deployment
- VPS + Supabase + local worker

---

## 18. Risks & Mitigations

| Risk | Mitigation |
|------|-----------|
| **LangGraph lock-in at retriever_agent** | 1/7 nodes coupled. ToolNode replaceable with custom ReAct loop (~50 lines) |
| **LLM skips pgvector @tool** | Retriever system prompt: "ALWAYS search pgvector first". Grader catches missing content → retry |
| **Kimodo latency (5-10s) blocks pipeline** | ToolNode runs tools in parallel. Only called for visualize_motion intent |
| **langchain-mcp-adapters maturity** | Fallback: custom MCP client (~100 lines) |
| **Grader false positive retry** | Max 1 retry. Deterministic rules. Worst case = 2x pipeline |
| **Empty response reaches user** | Defense in depth: grader + conversation fallback + FastAPI fallback |
| **Planner wrong intent** | Low confidence → clarify. Grader catches downstream issues |
| **Memory LTM keyword false positive** | Conservative patterns. False positive = unnecessary PostgreSQL query (low cost) |
| **Memory LTM temporal ambiguity** | Calendar week + +-3 day expansion. Multiple matches → ask user |
| **LLMGateway removal** | LangChain ChatModel is stable, actively maintained. Gemini provider well-supported |
| **Token budget tracking accuracy** | LangChain usage_metadata standardized. Threshold at 80% gives margin |

---

## 19. Verification Checklist

### Core Flow
1. `pytest tests/langgraph_agents/` — unit tests per node
2. **Memory → Planner → Retriever → Synthesizer → Grader → Conversation**: full pipeline
3. **Conversation intent**: memory → planner → conversation (generation mode) → END
4. **Clarification**: planner detects missing info → conversation asks user → END
5. **Error routing**: CRITICAL error at each node → error_handler → conversation → END

### Memory
6. **STM**: 3 Q&A pairs stored, FIFO eviction works
7. **LTM trigger**: recall keywords → PostgreSQL query → results in memory_context
8. **LTM ambiguous**: 2+ sessions → clarification flow
9. **Session reopen**: load from PostgreSQL → populate STM → continue

### Retriever + Tools
10. **pgvector @tool**: in-process, returns relevant documents
11. **MCP discovery**: retriever_agent auto-discovers Kimodo + web_search
12. **Parallel execution**: multiple tools called in parallel, latency = max(tool times)
13. **Dynamic MCP**: add new MCP server to config → retriever discovers → can call

### Quality
14. **Grader pass**: valid response → grader passes → conversation styles → END
15. **Grader retry**: invalid response → retriever re-calls → synthesizer re-generates → pass
16. **Grader fail-safe**: retry fails → pass with warning → user sees warning
17. **Grader feedback**: grader_feedback visible to retriever on retry

### Conversation
18. **Styling mode**: reasoning_output → persona-styled final_answer
19. **Generation mode**: conversation/clarify → persona-generated response
20. **Persona variety**: same reasoning_output → different outputs per persona

### Infrastructure
21. **Async TTS**: text streams first → TTS fires background → SSE speech_ready
22. **SSE stage events**: each node sends stage started/complete via astream_events
23. **Empty fallback**: graph returns empty → FastAPI sends fallback message
24. **Token budget**: total_tokens tracked → interrupt at threshold → resume after approval
