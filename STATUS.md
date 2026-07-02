# VVA — Status & Roadmap

> Last update: 2026-07-02 (K) | Branch: `feature/langgraph-rewrite`
> Audience: new team members, manager/K takeover after context compaction

---

## 0. TRẠNG THÁI ĐANG TREO (đọc trước — dễ mất khi compact)

- **Git**: local đã **sync với origin** — merge `1b734b1` (kéo về 3 commit auth-FE của Tri) trên
  commit của tôi `b857c36` (auth backend + 3 fix memory + FE wire + markdown + web-search + tests).
  Local đang **ahead 2, CHƯA PUSH**. (Owner đã nói "ok push" nhưng bị ngắt → push còn treo.)
- **TEMP đang bật để test local** (uncommitted, PHẢI khôi phục trước deploy):
  - `ECA_UI/frontend/src/components/AuthGuard.tsx`: đã **comment khối redirect `/login`** (có marker
    `⚠️ TEMP`) để vào thẳng chat không cần Cognito/Google login. Khôi phục = bỏ comment 3 dòng.
  - `agenticRAG/agentic_rag_gemini/.env`: `ALLOWED_ORIGINS` đã thêm `:5173` (local, gitignored).
- **Backend chạy trên :8000** (conda `firstconda`; DeepSeek key đã nạp balance). ⚠️ **8080 dành
  cho service Spring của Owner — KHÔNG bind backend vào 8080.** Frontend trỏ :8000 qua
  `ECA_UI/frontend/.env.local` (gitignored) + default trong `src/lib/api.ts`.
- **Không nhắc chuyện commit** với Owner trừ khi Owner chủ động (feedback 19/06). Không commit/push
  khi chưa được lệnh. Gọi Owner là **Mr. Senryuu**.
- **Demo chạy UI React** `:5173` (Vite dev), backend `:8000`, Docker PG(5433)/Redis/SearXNG.

---

## 1. Tổng quan

Healthcare/wellness AI assistant — bài tập physical therapy, chống chỉ định, 3D motion, voice.
Kiến trúc: **LangGraph 1.2.4 + DeepSeek + PostgreSQL/pgvector + Redis + React/Vite FE + Cognito**.

```
user query → memory (STM+facts+history) → planner (3-axis) → retriever (tool select)
   ├─ needs_retrieval=false → synthesizer (persona + markdown, đọc history) → grader → SSE
   └─ needs_retrieval=true  → kb_search ∥ memory_search ∥ web_search ∥ youtube_transcript → synthesizer → grader
```
8 nodes, 2 cổng routing độc lập (D15). Package: `agenticRAG/langgraph_agents/` (đã relevel từ
`agentic_rag_gemini/`). **247 tests** (unit + integration LLM/PG).

---

## 2. Đã hoàn thành ✅

### Core (REUPDATE_PLAN §M, D1-D33) — nền tảng, xong từ 06-11/06
- M.4 schema 7 bảng, HNSW, e5-small 384, no user_id trên messages/summaries (D19).
- 8 nodes, 3-axis intent, grader rule-based (TAG_RULES 8 tag), prompt caching M.7, GDPR M.8.
- Alembic 002, Docker (PG16/Redis7/SearXNG), embedding e5-small.

### Cụm A closeout + R1/R2 (12-13/06)
- **Background summarizer M.5** (`nodes/summarizer.py`): trigger 10k token, CAS, retry.
- **Memory tools bind + tenant scope fix** (memory_search `session_id=ANY`, inject scope qua config).
- **Clarify động M.2b**: memory_search/resume_last_session emit `{ambiguous, candidates}`.
- **GDPR wiring**: endpoint delete message/user → mark-dirty → `rebuild_dirty_chunk` (fix R1).
- **user_memory write path** (A3): `POST/GET/DELETE /users/{id}/memory`.
- **Path traversal `persona_id`** (A0) vá; **iterative_scan** bật ở pool; persona cache không cache fallback.
- **YouTube paste-link**: `youtube_transcript` tool (retrieval tool, không ghi KB).

### Auth + Frontend + Memory fixes (18/06 → nay)
- **Auth integration** (`api/auth.py`): verify Cognito ID token (JWKS RS256 + aud + iss + token_use),
  `user_id=sub`, gated cờ `REQUIRE_AUTH` (default false = demo). Wired mọi endpoint. CORS +:5173.
- **Frontend nối backend thật**: `src/lib/api.ts` streamChat SSE; ChatPanel bỏ mock; **web-search toggle**
  (chip xám "Web"); **ChatMessage render markdown** (react-markdown + remark-gfm); `amplify.ts`
  `import.meta.glob` → dev chạy không cần amplify_outputs.json (demo mode).
- **3 bug memory "câm" đã fix** (bug sau che bug trước): (1) `PostgresClient.executemany` thiếu;
  (2) `write_session_turn` truyền chuỗi ISO cho timestamptz dưới executemany; (3) synthesizer không
  đưa history vào LLM call. → memory đa lượt CHẠY (verify: "Tên bạn là Nguyễn"). +2 regression test.
- **Merge feature/frontend của Tri** (React UI + Cognito login/create-account/set-password pages +
  Amplify functions). Sync sạch, git tự 3-way merge (amplify.ts gộp cả 2, package.json cả 2 bộ dep).

### FE debug 02/07 (verify qua playwright-cli)
- **VRM không hiện**: `<Environment preset>` (PMREM HDR 256px) làm D3D11 device-removal
  (`DXGI 0x887A0020`) → WebGL context lost → canvas trắng. Fix: `resolution={64}` trong
  `CharacterViewer.tsx` (giữ IBL, texture nhỏ). VRM render OK.
- **Bold chữ trắng ẩn**: `ChatMessage.tsx` hardcode `prose-invert` (light theme → bold trắng).
  Fix: `dark:prose-invert` + `prose-strong/headings:text-foreground`.
- **Chat "something went wrong" tức thì**: FE gọi `/chat` ở :8080 (Spring Owner) → 404.
  Fix: default API base 8080→8000 (`api.ts`) + `.env.local` (gitignored). Verify: `POST /chat 200`,
  trợ lý Seele trả lời thật.

---

## 3. Còn thiếu (theo ưu tiên)

### 🔴 Trước khi có user / mở ra mạng
| # | Task | Ghi chú |
|---|---|---|
| 1 | **Bật auth thật**: `REQUIRE_AUTH=true` + 3 biến Cognito + deploy Cognito (`ampx sandbox`). Khôi phục AuthGuard redirect (đang comment TEMP). Đóng IDOR. | chặn deploy mạng |
| 2 | **Fix `MobileNavBar.tsx` TS6133** (`onOpenModal` unused) — chặn `npm run build` production. Lỗi pre-existing của Tri. | chặn build FE |
| 3 | **Persist sessionId FE** (localStorage) — đóng/mở panel Chat hiện tạo session mới → mất memory. | UX memory |
| 4 | **Verify E2E thủ công**: summarizer 10k token → row summaries; general_query "giá vàng?" SearXNG. | cần services |

### 🟠 Trước demo rộng
CI pipeline (GitHub Actions pytest) · Eval dataset 50 golden case · LLM fallback DeepSeek→Gemini
(hiện hết balance = cả hệ câm) · Rate limiting · Persona prompt versioning.

### 🟡 Nice to have
`ai_understanding` (AI tự trích facts) · LLM gợi-ý profile · LangSmith tracing · user upload doc.

---

## 4. Phase 7 — Hybrid Cloud (ON HOLD, chờ Owner bàn)
- Tri's `infra/` CDK (Python) đã merge: VPC isolated + RDS Proxy + Lambda CRUD + API Gateway.
- **Đã chốt**: Alembic = nguồn migration duy nhất (xóa `infra/sql/init_schema.sql`); `/chat` KHÔNG
  qua API Gateway (timeout 29s) → ECS Fargate; Kimodo host = edge RTX 3060 + SQS pull.
- **Còn treo (cần Owner)**: Supabase vs RDS (chi phí AWS-full ~$80/mo vs lean ~$30); chốt trước khi K
  viết PHASE-7.x specs. Voice = push-to-talk (đã chốt, giữ SSE).
- **Sync với Tri**: cần branch-protection/PR để hết va (đã va merge 2 lần).

---

## 5. Cách chạy (local demo)

```bash
# 1. Docker (postgres/redis không có restart policy — Docker restart là phải bật lại)
docker compose -f docker-compose.langgraph.yml up -d postgres redis searxng

# 2. Migration
cd agenticRAG/langgraph_agents && alembic upgrade head

# 3. Backend (:8000 — 8080 dành cho Spring của Owner) — conda env firstconda
cd agenticRAG
python -m uvicorn langgraph_agents.api.main:create_app --factory --port 8000

# 4. Frontend React (:5173) — demo mode, không cần login (AuthGuard đang TEMP-bypass)
cd ECA_UI/frontend && npm install && npm run dev

# Tests
python -m pytest tests/langgraph_agents/ -q       # 247 passed (cần PG/Redis + DeepSeek key)
```
Chi tiết + troubleshooting: `docs/RUNBOOK.md` (đã cập nhật: 2 UI, REQUIRE_AUTH, CORS :5173, Docker-down).

## 6. Key files
| File | Nội dung |
|---|---|
| `REUPDATE_PLAN.md §M` | 33 decisions D1-D33 — nguồn chân lý kiến trúc |
| `TECH_DEBT.md` | Việc tồn (cập nhật 13/06) |
| `PREDEPLOY-AUDIT.md` | Security review + lộ trình pre-deploy (A/B/C) |
| `FIX-*.md` (root) | Spec/handoff các cụm (memory, R1, auth, chatpanel, youtube) |
| `docs/worklogs/*` | Nhật ký từng phiên |
| `.claude/CLAUDE.md` | Roles K/N/Owner, conventions |

## 7. Conventions
- Worklog `docs/worklogs/DD-MM-YYYY.md` mỗi phiên. Test phải xanh trước merge.
- K = Architect (plan/review, thường không code — phiên gần đây K spawn subagent implement rồi tự
  verify code + playwright trước khi report). N = Developer. Owner = "Mr. Senryuu", chốt vision.
- Code = English, docs = Việt + Anh. UI verify bằng skill `playwright-cli` (không npx). Không dùng port 8080 cho smoke-test.
