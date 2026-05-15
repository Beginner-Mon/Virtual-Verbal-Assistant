---
title: "Architecture Review"
description: "Senior-architect critique of the multi-agent Physiotherapy ECA system."
tags:
  - architecture
  - review
  - critique
  - p0
  - p1
  - p2
  - agentic-rag
  - dart
  - speechllm
  - refactoring
  - roadmap
---

# Architecture Review — Virtual Verbal Assistant

> Senior-architect critique của hệ thống multi-agent Physiotherapy ECA, đánh giá theo
> chuẩn "Senior engineer + đồ án tốt nghiệp" với ràng buộc CPU-only 16 GB và yêu cầu
> rành mạch để vẽ DFD/Sequence cho luận văn.

**Phạm vi:** AgenticRAG (port 8000), Orchestrator gateway (port 8080), DART (port 5001),
SpeechLLm (port 5000), ECA UI / test-ui.

**Phương pháp:** static review của các module trọng yếu — `agents/api_orchestrator.py`,
`agents/local_orchestrator.py`, `orchestration/pipeline_orchestrator.py`,
`agents/knowledge_librarian.py`, `agents/semantic_bridge.py`, `agents/tools/*`,
`api_server_pkg/app.py`, `services/*`, `routers/*`.

---

## 1. Strengths (đã làm đúng)

| # | Điểm mạnh | Trích dẫn |
|---|-----------|-----------|
| S1 | Đã modularise `api_server.py` thành shim → `api_server_pkg/app.py`, `routers/`, `services/` | `agenticRAG/agentic_rag_gemini/api_server.py:6` |
| S2 | `OrchestratorAgent._run_tools` chạy tools song song bằng `ThreadPoolExecutor` đúng pattern Agent + Tools | `agents/api_orchestrator.py:688-747` |
| S3 | `LocalOrchestrator` có **regex pre-router** cho greeting/knowledge/visualize/clear-context — quyết định đúng cho CPU-only | `agents/local_orchestrator.py:84-160` |
| S4 | `RAGPipeline` có `RateLimiter` thread-safe + parallel web search; loại bỏ query-expansion để cắt LLM calls | `retrieval/rag_pipeline.py` (theo README §3.5) |
| S5 | Async motion qua Celery + `MotionJobManager` với in-flight stages, timeline tracking | `motion_jobs.py:37-149` |
| S6 | DART có contract `duration_seconds` first-class, sạch hơn legacy `action*N` | `agents/tools/motion_generation_tool.py:100-105` |
| S7 | `SessionStore` Firestore với JSON-on-disk fallback — graceful degradation | `memory/session_store.py:1-92` |
| S8 | `MotionCandidateRetriever` đã có cả Pinecone + ChromaDB + JSONL fallback — đúng định hướng hybrid | `agents/tools/motion_candidate_retriever.py:73-155` |

---

## 2. Issues — P0 Architecture (CRITICAL)

### A1. Ba "orchestrator" trùng tên, vai trò mơ hồ

```
agents/api_orchestrator.py     → LLM routing (Gemini)
agents/local_orchestrator.py   → LLM routing (Ollama Qwen)
orchestration/pipeline_orchestrator.py → Multi-service HTTP fanout
```

Cả ba đều có method `process_query` / `analyze_query`. Trong luận văn không thể vẽ
DFD vì không phân biệt "Orchestrator" là cái nào. Mỗi lần đọc code mất 5 phút để xác
định scope.

**Đề xuất:** đặt tên theo trách nhiệm:
- `Router` (cũ: `*_orchestrator`) — chỉ ra quyết định routing, không retrieve, không HTTP.
- `Coordinator` (cũ: `pipeline_orchestrator`) — phối hợp multi-service.
- Trong code, dùng đúng 1 từ "Orchestrator" để chỉ tổng thể (gateway 8080).

### A2. Intent vocabulary phân kỳ giữa 2 router

| File | Intent định nghĩa |
|------|-------------------|
| `agents/api_orchestrator.py:72-92` (IntentType enum) | `conversation`, `knowledge_query`, `exercise_recommendation`, `visualize_motion` (4 giá trị) |
| `agents/local_orchestrator.py:299-303` (`valid_intents` list) | `ask_exercise_info`, `visualize_motion`, `greeting`, `followup_question`, `resume_conversation`, `general_fitness_question`, `exercise_recommendation`, `conversation`, `knowledge_query`, `unknown` (10 giá trị) |

Hệ quả:
- UI nhận intent shape khác nhau tuỳ router nào chạy.
- `IntentType.CONVERSATION` ≠ `local_orchestrator`'s `"greeting"` mặc dù cùng nghĩa.
- Test contract impossible — không có single source of truth.
- Báo cáo phải giải thích 2 lần.

**Fix (đã implement ở D2):** `agents/intents.py` định nghĩa duy nhất `IntentType` +
`ActionType` + `from_local_intent_str()` mapping. Cả hai router import từ đây.

### A3. Double-RAG inlined trong `OrchestratorAgent.process_query`

```@d:\Project_A\Virtual-Verbal-Assistant\agenticRAG\agentic_rag_gemini\agents\api_orchestrator.py:520-542
        # ── Step 2: Multi-Stage Orchestration (Double-RAG) ───────────────────
        # If we need clinical knowledge and motion execution, execute Double-RAG
        double_rag_results = {}
        if self._document_tool and (action == ActionType.GENERATE_MOTION or analysis["needs_rag"]):
            # 1. Clinical Dispatch
            clinical_docs = self._document_tool.search_documents(expanded_query)
            
            # 2. Constraint Extraction
            constraints = self._extract_constraints(clinical_docs)
            logger.info(f"Extracted Constraints: {constraints}")
            
            # 3. Conditioned Motion Search
            conditioned_query = f"{hyde_document} Constraints: {constraints}"
            motion_candidates = self._motion_retriever.retrieve_top_k(conditioned_query, k=1)
            
            if motion_candidates:
                hyde_document = motion_candidates[0].text_description
            
            double_rag_results = {
                "clinical_docs": clinical_docs,
                "constraints": constraints,
                "motion_candidates": motion_candidates
            }
```

Vấn đề:
- Orchestrator vừa **route**, vừa **retrieve clinical**, vừa **gọi LLM extract constraint**, vừa **retrieve motion candidate**. Vi phạm SRP nghiêm trọng.
- `_extract_constraints` (line 601-624) là LLM call **ẩn** không tính trong claim "2 LLM calls max".
- Khi viết luận văn, không tách được Hub & Spoke architecture vì code không phản ánh nó.

**Fix (D4):** tách thành class `DoubleRAGAgent` riêng, orchestrator chỉ gọi
`self._double_rag.run(...)`.

### A4. Tool registry implicit qua magic strings

```@d:\Project_A\Virtual-Verbal-Assistant\agenticRAG\agentic_rag_gemini\agents\api_orchestrator.py:145-151
_ACTION_TOOL_MAP: Dict[ActionType, List[str]] = {
    ActionType.RETRIEVE_MEMORY: ["memory"],
    ActionType.CALL_LLM:        ["memory", "documents", "web_search"],
    ActionType.GENERATE_MOTION: [],
    ActionType.HYBRID:          ["memory", "documents", "web_search"],
    ActionType.CLARIFY:         [],
}
```

Thêm 1 tool mới (ví dụ `ExerciseRanker`) phải sửa **4 chỗ**:
1. `_ACTION_TOOL_MAP` thêm string `"exercise_ranker"`.
2. `_select_tools` thêm filter logic.
3. `_run_tools` thêm if branch + executor.submit.
4. Constructor inject parameter mới.

**Fix (D3):** `Tool` ABC + `ToolRegistry`. Tool tự khai báo nó áp dụng cho intent
nào. Orchestrator chỉ gọi `registry.tools_for(intent)` và iterate.

### A5. `AgenticRAGAPI` God Class

`api_server_pkg/app.py` = 2874 dòng, ~15 module imported ở top, `class AgenticRAGAPI`
chứa cả FastAPI route handler, business logic, agent wiring, lifecycle.

Hậu quả:
- Cold-start chậm (xem P8).
- Không unit-test được nếu không stub 15 module.
- Refactor 1 method dễ vỡ 5 cái khác.

**Fix (out of scope step này, nhưng D2-D6 sẽ giảm dần phụ thuộc):** các step refactor
sẽ rút bớt logic ra khỏi class này. Không big-bang rewrite.

### A6. Schema drift `motion_prompt` ↔ `exercise_motion_prompt`

```@d:\Project_A\Virtual-Verbal-Assistant\agenticRAG\agentic_rag_gemini\orchestration\pipeline_orchestrator.py:117-119
                result.text_answer = rag_response.get("text_answer", "")
                motion_prompt = rag_response.get("motion_prompt")
                voice_prompt = rag_response.get("voice_prompt")
```

`api_orchestrator` mới đã đổi sang `exercise_motion_prompt` (theo README §2 contract +
service `main_api_downstream.py:build_motion_from_agenticrag`). Pipeline orchestrator
vẫn dùng key cũ → silently skip motion khi gọi qua đường này.

**Fix (D8):** cập nhật pipeline_orchestrator hoặc đánh dấu legacy + chuyển hết qua
`services/main_api_downstream.py`.

### A7. Không có agent decision trace có cấu trúc

Toàn bộ decision được log dưới dạng f-string. Không thể:
- Replay 1 request để debug.
- Show cho UI biết "tại sao agent quyết định thế này".
- Đo timing per-decision-stage một cách có hệ thống.

**Fix (D5):** `core/tracing.py` cung cấp `AgentTrace` TypedDict, propagate qua
`request_id`, attach vào response khi `AGENTIC_TRACE=1`.

---

## 3. Issues — P0 Performance (CPU-only 16 GB constraint)

### P1. `process_query_sync` deadlock-prone

```@d:\Project_A\Virtual-Verbal-Assistant\agenticRAG\agentic_rag_gemini\orchestration\pipeline_orchestrator.py:352-367
        try:
            loop = asyncio.get_event_loop()
            if loop.is_running():
                # Already in async context, need to create new loop or use run_in_executor
                import concurrent.futures
                with concurrent.futures.ThreadPoolExecutor() as executor:
                    future = executor.submit(
                        asyncio.run,
                        self.process_query(query, user_id, conversation_history),
                    )
                    return future.result()
```

Spawn `ThreadPoolExecutor` → submit `asyncio.run` bên trong → blocking `future.result()`.
Trên CPU yếu/bận, dễ treo. Đây là anti-pattern phổ biến.

**Fix (D8):** xoá luôn `process_query_sync`, ép caller dùng async-only.

### P2. Celery worker chạy `-P solo`

Tất cả async motion job chạy serial. Trên 16 GB chạy được 2-3 task song song nếu
prefork. Hiện tại defeat purpose của queue.

**Fix:** README ghi khuyến nghị `-P prefork --concurrency=2` (Linux/WSL). Windows
giữ `solo` do Celery prefork bug đã biết — đó là trade-off có chủ đích.

### P3. Per-user ChromaDB collection

```@d:\Project_A\Virtual-Verbal-Assistant\agenticRAG\agentic_rag_gemini\agents\knowledge_librarian.py:8-11
    1. HumanML3D    → humanml3d_library (kinematic motion descriptions)
    2. Documents    → user_{id}_documents (uploaded PDFs, DOCX, exercise KB)
    3. User Context → user_{id}_collection (conversation memory)
```

N user → 2N collections. ChromaDB metadata grow tuyến tính. Trên 16 GB, demo nhiều
user dễ swap.

**Fix (D6):** `VectorBackend` ABC. HumanML3D dùng Pinecone (cố định, ~23k rows). User
docs/memory dùng Chroma (giữ nguyên). Khi scale, chuyển user collections sang Pinecone
multi-tenant namespace = đổi 1 dòng config.

### P4. Hybrid vector backend half-baked

`VECTOR_DB_TYPE` env biến chỉ được tôn trọng trong `motion_candidate_retriever.py` và
`semantic_bridge.py`. Còn `vector_store.py`, `document_store.py` hard-code Chroma.

**Fix (D6):** centralize qua `get_vector_backend(collection)` — adapter chọn backend
theo collection name. Khi Pinecone hết quota, fallback Chroma.

### P5. Không cap `ThreadPoolExecutor` workers

```@d:\Project_A\Virtual-Verbal-Assistant\agenticRAG\agentic_rag_gemini\agents\api_orchestrator.py:715
        with ThreadPoolExecutor(max_workers=len(selected_tools)) as executor:
```

OK với 3 tool. Khi thêm motion retrieval / exercise ranker / safety filter song song,
1 request có thể fork 5-6 thread, mỗi thread call HTTP/LLM. 5 request đồng thời = 30
thread = trên 16 GB dễ contention.

**Fix (D7):** `ResourceGuard.tool_executor()` — global `ThreadPoolExecutor` shared,
`max_workers = min(4, cpu_count // 2)`.

### P6. 2 sentence-transformer instance song song

`MotionCandidateRetriever` load `all-MiniLM-L6-v2` (~80MB), `EmbeddingService` cũng
load `all-MiniLM-L6-v2`. Hai instance → 160MB không cần thiết.

**Fix (D7):** `ResourceGuard.shared_embedder()` singleton.

### P7. Không có circuit breaker

Pipeline orchestrator có `timeout` đơn lẻ. Nếu DART crash liên tục, mỗi request vẫn
chờ 10s rồi mới fail. 5 request liên tiếp = 50s wasted.

**Fix (D8):** trivial in-memory circuit breaker — fail >= 3 consecutive trong 60s →
skip DART luôn trong 60s tiếp theo.

### P8. Cold-start nặng

```@d:\Project_A\Virtual-Verbal-Assistant\agenticRAG\agentic_rag_gemini\api_server_pkg\app.py:47-72
from agents.api_orchestrator import OrchestratorAgent
from agents.local_orchestrator import LocalOrchestrator
from agents.safety_filter import SafetyFilter
from agents.query_transform import QueryTransformer
... (15+ heavy imports at top-level)
```

Top-level import trigger:
- `sentence-transformers` load (~3s).
- `chromadb` PersistentClient (~1s).
- `firebase-admin` SDK init (~0.5s).
- `OllamaClient.check_connection` (~0.3s + timeout).

Cold-start hiện tại có thể 8-12s. Demo bấm "Start" phải chờ.

**Fix (out of scope, ghi nhận):** lazy-load các agent ở constructor, chỉ
`gemini_client`/`logger` ở top. Đề xuất viết trong `docs/ARCHITECTURE_REVIEW.md` để
thực hiện sau.

---

## 4. Issues — P1 Code quality

| # | Issue | Trích dẫn |
|---|-------|-----------|
| Q1 | Magic strings tool name | `agents/api_orchestrator.py:145-151, 715-734` |
| Q2 | `clean_json_response` duplicated logic | `agents/api_orchestrator.py:41-60` (cũng có biến thể trong `local_orchestrator._parse_response`) |
| Q3 | `_INTENT_TOKEN_BUDGETS` rời rạc trong code thay vì config | `agents/local_orchestrator.py:12-23` |
| Q4 | Không có `Tool` ABC, không có `Agent` base | toàn module |
| Q5 | Test rời rạc: `test_auditor.py`, `test_sbr.py` ở root | repo root |
| Q6 | Threshold qua `os.getenv` không validate | `knowledge_librarian.py:48-68`, `semantic_bridge.py:62-68` |
| Q7 | `ActionType.CLARIFY` được khai báo nhưng không có flow xử lý | `api_orchestrator.py:69` (chỉ map, không có logic clarify dialog) |
| Q8 | `MotionGenerationTool` docstring nói "not wired into pipeline yet" — comment đã lỗi thời | `agents/tools/motion_generation_tool.py:42-44` |

---

## 5. Issues — P2 Ops & Observability

- Logs unstructured strings → khó grep theo `request_id`.
- Không có `request_id` propagate xuyên 3 services. Khi DART log error, không biết thuộc request nào của user.
- Health check shallow: chỉ HTTP 200, không probe Gemini/Chroma/Redis thực sự.
- `firebase-service-account.json` nằm trong repo — nguy cơ leak nếu push public.
- README có cảnh báo về Linux/WSL split deployment nhưng không có script kiểm tra
  trạng thái stack tổng thể.

---

## 6. Architecture đề xuất

```
┌──────────────── ECA UI ─────────────────┐
│              POST /answer               │
└────────────────────┬────────────────────┘
                     ▼
        ┌─────────── Coordinator (8080) ──────┐
        │  (was: pipeline_orchestrator)       │
        │  - routers/answer.py                │
        │  - services/main_api_*.py           │
        │  - CircuitBreaker (D8)              │
        └────────┬───────────────────┬────────┘
                 ▼                   ▼
       ┌─────────────────┐  ┌─────────────────┐
       │  AgenticRAG API │  │  DART API (5001)│
       │  (api_server_pkg│  │  (WSL)          │
       │   /app.py 8000) │  └─────────────────┘
       └────────┬────────┘
                ▼
       ┌─────────────────────────────────┐
       │  Router (single)                │  ← merge api_orchestrator
       │   - Regex pre-router            │     + local_orchestrator
       │   - LLM router (Gemini|Ollama)  │
       │   - Returns RouterDecision      │
       │   - emits AgentTrace.decision   │  (D5)
       └────────┬────────────────────────┘
                │
                ▼
       ┌─────────────────────────────────┐
       │  Plan Executor                  │
       │   - ToolRegistry (D3)           │
       │   - ResourceGuard.executor (D7) │
       │   - emits AgentTrace.tools_run  │
       └────────┬────────────────────────┘
        ┌───────┼─────────────┬────────────┐
        ▼       ▼             ▼            ▼
    Librarian DoubleRAG   Memory      WebSearch
   (existing) (D4 split)  Tool        Tool
        │         │
        └─────────┴──────────────┐
                                  ▼
                       VectorBackend (D6)
                       ├── ChromaBackend  (user_*, local)
                       └── PineconeBackend (humanml3d, clinical)
```

**Boundary cho luận văn:**

| Component | Trách nhiệm | KHÔNG làm |
|-----------|-------------|-----------|
| **Coordinator** | HTTP fanout, circuit breaker, timeout | LLM, retrieval |
| **Router** | Intent + Action plan | Retrieval, motion gen |
| **Plan Executor** | Chạy tool song song, ráp result | Quyết định plan |
| **Librarian** | Multi-collection retrieval, fact summarisation | LLM generation |
| **DoubleRAG** | Hub & Spoke: clinical → constraint → motion candidate | Routing |
| **Translator** (`semantic_bridge`) | Natural language → DART-ready prompt | Retrieval, generation |
| **VectorBackend** | Hybrid vector store với fallback | Domain logic |
| **ResourceGuard** | Concurrency cap, shared executor, shared embedder | Business logic |

---

## 7. Roadmap thực thi

| Step | Module mới | Patch | Risk |
|------|-----------|-------|------|
| D2 | `agents/intents.py` | re-export trong `api_orchestrator` (backward compat) | LOW |
| D3 | `agents/tools/base.py` + `registry.py` | tools opt-in, không break existing call site | LOW |
| D4 | `agents/double_rag.py` | xoá block 520-542 + `_extract_constraints` khỏi orchestrator | MED |
| D5 | `core/tracing.py` | thêm field optional `agent_trace` vào response | LOW |
| D6 | `memory/vector_backend.py` | adapter, các module hiện tại không bắt buộc dùng ngay | LOW |
| D7 | `core/resource_guard.py` | replace ThreadPoolExecutor spawn | LOW |
| D8a | patch `api_orchestrator.py` | dùng D2-D7 | MED |
| D8b | patch `local_orchestrator.py` | dùng D2 | LOW |
| D8c | patch `pipeline_orchestrator.py` | drop sync, add CircuitBreaker, fix contract | MED |
| D8d | wire `request_id` qua `routers/answer.py` | additive | LOW |
| D9 | update `README_DEV.md` | docs only | NONE |

---

## 8. Acceptance gates

1. `python api_server.py` cold-start < 8s trên Windows firstconda (mục tiêu).
2. Chạy 5 query song song qua Coordinator: P95 latency text-only < 12s, không OOM.
3. `agent_trace` JSON xuất hiện trong response khi `AGENTIC_TRACE=1`, có ≥ 5 decision points.
4. `grep -rn "class IntentType\|class ActionType" agents/` chỉ thấy 1 file (`agents/intents.py`).
5. Thêm tool dummy `EchoTool` chỉ qua `registry.register(EchoTool())` — KHÔNG sửa orchestrator.
6. README_DEV chứa Mermaid sequence diagram tái hiện flow Coordinator → Router → Plan Executor → DART/AgenticRAG.
7. `VECTOR_DB_TYPE=hybrid` chạy: humanml3d hits Pinecone, user docs hits Chroma, log xác nhận adapter routing.

---

## 9. Out of scope (cố ý bỏ qua)

- Kubernetes, Helm, GitHub Actions CI/CD phức tạp.
- Multi-tenant SaaS (auth/billing/quota).
- Distributed tracing (OpenTelemetry, Jaeger) — quá nặng cho 16 GB demo.
- Refactor full DART training stack — chỉ chạm `api_server.py` của DART nếu cần fix contract.
- Big-bang rewrite `api_server_pkg/app.py` — sẽ bóc dần qua D2-D6.

---

## 10. Key takeaways cho luận văn

1. **Multi-agent boundary phải code-enforced**, không chỉ doc-enforced. Tools, Router, Plan Executor là 3 lớp phải có ABC riêng để readers hiểu được flow.
2. **Hybrid vector backend** không thể làm half-baked qua env flag rải rác. Phải có adapter pattern, mỗi collection biết backend nào của nó.
3. **CPU-only 16 GB** ép phải có ResourceGuard từ ngày 1: shared executor, shared embedder, bound caches. Không tin vào "Python sẽ tự lo".
4. **AgentTrace** không phải optional — đây là cách duy nhất để bảo vệ luận văn khi giảng viên hỏi "tại sao agent quyết định thế này?".
5. **Single Source of Truth** cho Intent + Action enum là quy tắc cứng: 1 module duy nhất, mọi chỗ khác `from agents.intents import …`.

---

## Related Notes

- [[agentic_rag_refactor]] — Implementation of D1-D9 fixes described in this review
- [[system_overview]] — Service topology and query flow
- [[api_contract]] — Gateway request/response schemas
- [[setup_guide]] — Running the stack locally
- [[troubleshooting]] — Debugging common issues
- [[dart_architecture]] — DART motion synthesis internals
