# VVA — Status & Roadmap

> Last update: 2026-06-11 | Branch: `feature/langgraph-rewrite`
> Audience: new team members, manager takeover

---

## 1. Tổng quan

Healthcare/wellness AI assistant — bài tập physical therapy, tư vấn chống chỉ định, 3D motion synthesis, voice I/O. Kiến trúc: **LangGraph 1.2.4 + DeepSeek v4 + PostgreSQL/pgvector + Redis**.

```
user query → memory (STM+facts) → planner (3-axis intent) → retriever (tool selection)
   ├─ needs_retrieval=false → synthesizer (persona-styled) → grader (tag-driven) → response
   └─ needs_retrieval=true  → kb_search ∥ memory_search ∥ web_search → synthesizer → grader
```

8 nodes, 2 cổng routing độc lập (D15): retriever ⟸ `needs_retrieval`, grader ⟸ `required_outputs != []`

---

## 2. Đã hoàn thành ✅

### Core architecture (REUPDATE_PLAN §M, 33 decisions D1-D33)
- [x] **M.4 Schema**: 7 tables (users, conversations, messages, summaries, user_memory, documents, kb_embeddings). HNSW indexes. e5-small vector(384). No user_id trên messages/summaries (D19).
- [x] **8 nodes**: memory → planner → retriever_agent ⇄ tools → kimodo → synthesizer → grader → error_handler
- [x] **3-axis intent**: `required_outputs` / `resolved_query` / routing bits. Bỏ 6-enum cũ.
- [x] **Grader rule-based**: TAG_RULES 8 tag (3 safety + 5 quality), deterministic regex, no LLM.
- [x] **Prompt caching layout**: tĩnh đầu/động cuối (M.7), DeepSeek auto-cache prefix.
- [x] **GDPR from day 1**: gdpr.py — mark-dirty, hard-delete, empty-chunk cleanup (M.8).

### Infrastructure
- [x] **Docker**: PostgreSQL 16 (pgvector), Redis 7, SearXNG (docker-compose.langgraph.yml)
- [x] **Alembic migration**: 002_m4_fresh chạy thành công, 7 tables created
- [x] **Embedding**: `intfloat/multilingual-e5-small` (384 dims) + prefix query:/passage:
- [x] **Config**: langgraph.yaml (DSN, breaker, memory, persona)

### Testing
- [x] **187/187 tests passing** (60+ unit, 5 integration LLM, SSE, circuit breaker, health, logging)
- [x] Session store sync M.4 schema
- [x] API schemas sync 3-axis model
- [x] Persona loader sync 4 modes (chat/clarify/refuse/synthesize)

### Cleanup
- [x] Gỡ 4 test files cũ (planner, grader, memory, retriever)
- [x] Gỡ nút 📎 upload khỏi UI (tàn dư kiến trúc orchestrator cũ)

---

## 3. Đang làm 🟡

- [ ] **STATUS.md** (file này — chưa có phiên bản trước 11/06)

---

## 4. Còn thiếu (theo thứ tự ưu tiên)

### 🔴 Trước khi có user thật

| # | Task | Effort | Where |
|---|---|---|---|
| 1 | ~~LTM write path~~ → ✅ XONG 12/06: background summarizer M.5 (`nodes/summarizer.py`) + bind memory tools + fix tenant scope. Xem worklog 12/06 + `PREDEPLOY-AUDIT.md` | — | done |
| 2 | **users.profile write path**: 0 chỗ ghi → luôn `{}`. Cần `PATCH /users/{id}/profile` | 1h | `api/main.py` |
| 3 | **Verify general_query**: test "giá vàng?" qua SearXNG thật, xác nhận web search hoạt động | 30m | manual |
| 4 | **YouTube paste link**: detect link → get transcript → context cho synthesizer | 3h | `nodes/planner.py`, `tools/youtube_ingest.py` |

### 🟠 Trước demo

| # | Task | Effort | Where |
|---|---|---|---|
| 5 | **CI pipeline**: GitHub Actions chạy pytest toàn bộ suite | 1h | `.github/workflows/` |
| 6 | **Eval dataset**: 50 case golden test (5 chat, 10 safety, 15 exercise, 10 clarify, 10 refuse) | 3h | `tests/eval/` |
| 7 | **LLM fallback**: DeepSeek → Gemini qua circuit breaker | 2h | `llm.py` |
| 8 | **Auth JWT middleware**: bỏ `user_id="anonymous"` cho production | 2h | `api/main.py` |
| 9 | **Rate limiting**: `slowapi` hoặc Redis-based, 20 req/min/user | 1h | `api/main.py` |
| 10 | **Persona prompt versioning**: snapshot prompt version mỗi lần đổi | 1h | `nodes/_persona_loader.py` |

### 🟡 Nice to have

| # | Task | Effort |
|---|---|---|
| 11 | `ai_understanding` — AI tự đúc kết facts về user từ hội thoại (background, throttled) | 3h |
| 12 | LangSmith tracing — full graph trace cho debug production | 1h |
| 13 | User upload document — bảng `user_documents` + `user_doc_embeddings` (defer per M.10) | 4h |
| 14 | Multi-turn clarification loop | 3h |
| 15 | A/B testing framework cho prompt | 5h |

---

## 5. Phase 7 — Hybrid Cloud Deployment (chưa bắt đầu)

[Xem REUPDATE_PLAN.md lines 716-883 để biết chi tiết.]

| # | Task |
|---|---|
| 1 | Init `infra/` CDK project (TypeScript): VPC, RDS, ElastiCache, ECS Fargate, Lambda, CloudFront |
| 2 | SQS + TTS worker (reactivate `celery_app.py` với SQS broker) |
| 3 | Encryption at rest + RLS (khi có nhiều đường vào DB) |
| 4 | DNS + SSL |
| 5 | Supabase vs self-hosted RDS decision |
| 6 | Push-to-talk vs continuous streaming voice decision |

---

## 6. Cách chạy

```bash
# Docker services
docker compose -f docker-compose.langgraph.yml up -d postgres redis

# Migration (port 5433 — host may conflict with local PostgreSQL)
cd agenticRAG/langgraph_agents
alembic upgrade head

# Tests
pytest ../../tests/langgraph_agents/ -v

# API (dev)
uvicorn langgraph_agents.api.main:create_app --port 8080
```

## 7. Key files — đọc nếu muốn đào sâu

| File | Nội dung | Khi nào đọc |
|---|---|---|
| `REUPDATE_PLAN.md §M` | 33 decisions + spec chi tiết | Cần hiểu "vì sao" |
| `TECH_DEBT.md` | Danh sách việc tồn (có thể hơi cũ) | Check còn gì chưa làm |
| `docs/worklogs/05-06-2026.md` | Phiên grill chốt kiến trúc | Hiểu 6 lần đảo ngược quyết định |
| `docs/worklogs/06-06-2026.md` | N implement 15 bước M.9 | Biết ai code gì |
| `docs/worklogs/11-06-2026.md` | Test suite rebuild + bug fixes | Biết đã sửa những gì |
| `.claude/CLAUDE.md` | Roles, conventions, skills | Role của K, N, Owner |

## 8. Conventions

- **Worklog**: ghi `docs/worklogs/DD-MM-YYYY.md` mỗi phiên làm việc
- **Plan**: `REUPDATE_PLAN.md §M` là nguồn chân lý kiến trúc
- **Test**: `pytest -m unit|integration -v`. 187 tests, tất cả phải xanh trước merge.
- **Branch**: `feature/langgraph-rewrite`
- **Roles**: K = Architect (AI, không code, viết plan, review). N = Developer (Human, code, test).
- **Ngôn ngữ**: Code = English, docs = Vietnamese + English mix
