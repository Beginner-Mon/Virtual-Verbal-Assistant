# VVA — Status & Roadmap

> Last update: 2026-07-21 (K) | Branch: `feature/langgraph-rewrite`
> Audience: K/N/Owner takeover after context compaction — đọc mục 0 trước tiên.

---

## 0. TRẠNG THÁI ĐANG TREO (đọc trước — dễ mất khi compact)

- **Backend + Docker hiện ĐANG TẮT** (verify lúc viết file này: `curl :8000/health` → down,
  `docker ps` → không có postgres/redis/searxng). Muốn chạy tiếp:
  ```bash
  docker compose -f docker-compose.langgraph.yml up -d postgres redis searxng
  cd agenticRAG && conda activate firstconda
  python -m uvicorn langgraph_agents.api.main:create_app --factory --port 8000 --host 0.0.0.0
  ```
  ⚠️ **Port 8080 = service Spring của Owner, KHÔNG bind backend vào đó.** Luôn dùng **8000**.
- **Git**: toàn bộ thay đổi phiên này đang ở trạng thái **STAGED** (`git status` ra cột index
  `M`/`A`/`R` — không rõ do lệnh nào add, có thể Owner tự add ở terminal khác). **21 file, chưa
  commit.** Không tự commit/push khi chưa được lệnh (quy tắc cũ vẫn giữ). Danh sách file ở §8.
- **Docs đã bị reorg** (không phải do K chủ động, phát hiện giữa phiên qua git log
  "re structure docs"): root `FIX-*.md`, `TECH_DEBT.md`, `REUPDATE_PLAN.md`, `RUNBOOK.md` đã dời
  vào `docs/{architecture,ops,plans,fixes,tracking,archive}/`. File này giờ nằm ở
  **`docs/tracking/status.md`** (không phải root `STATUS.md` nữa). Xem §6 cho path đầy đủ.
- **Auth demo bypass** giờ là **env-gate sạch** (không còn TEMP-comment-hack cũ):
  `ECA_UI/frontend/.env.local` có `VITE_AUTH_DISABLED=true` → vào thẳng chat không cần Cognito.
  Production: bỏ/`=false` biến này + set `REQUIRE_AUTH=true` + 3 biến Cognito backend.
- **DeepSeek + Gemini**: cả hai đã từng hết tiền/quota cùng lúc trong phiên này (đã verify sống,
  không phải bug) → Owner đã nạp lại DeepSeek. Nếu gặp lỗi `402`/`429` lại, đó là vấn đề tài
  khoản, không phải code.
- **Test hiện tại: 296 collected (259 unit pass + 37 integration cần Docker/DeepSeek)**.
- Gọi Owner là **Mr. Senryuu**. Không nhắc chuyện commit trừ khi Owner chủ động.

---

## 1. Tổng quan

Healthcare/wellness AI assistant — physical therapy exercise recommendation, clinical safety
grading, 3D motion avatar (Kimodo), voice I/O (VieNeu, optional/chưa có server). Kiến trúc:
**LangGraph 8-node multi-agent + DeepSeek/Gemini + PostgreSQL/pgvector + Redis + React/Vite FE +
Cognito auth.**

```
memory (STM+LTM) → planner (3-axis intent) → retriever_agent ⇄ tools (cap 2 rounds)
   → kimodo (nếu needs_motion) → synthesizer (persona) → grader (nếu có safety tag) → SSE
```

8 node, 2 cổng routing độc lập, MCP tool servers (web search qua SearXNG, motion qua Kimodo).
Package: `agenticRAG/langgraph_agents/`. **296 test** (259 unit + 37 integration).
README.md (root) đã viết lại đầy đủ — kiến trúc, sơ đồ mermaid, benchmark thật, dùng để
tham khảo CV/portfolio.

---

## 2. Đã hoàn thành ✅ (từ đầu tới nay, gộp các phiên)

### Core + Cụm A/B + Auth + Frontend (trước 02/07 — xem worklog cũ nếu cần chi tiết)
- LangGraph 8-node, 3-axis intent, grader rule-based, GDPR delete+re-summarize, memory tools,
  YouTube transcript tool, Cognito JWT verify (JWKS RS256), frontend React nối backend thật,
  3 bug memory "câm" đã fix, merge frontend của Tri.

### FE debug 02/07
- VRM WebGL context-lost fix (`Environment resolution={64}`), bold-text-ẩn fix (dark:prose-invert),
  port 8080→8000 fix cho chat "something went wrong".

### Health-test dashboard + log setup (phiên này, đầu)
- `ECA_UI/test-ui/health-test/`: port 8000, CORS `null` cho phép mở qua `file://`, sửa dashboard
  đọc trạng thái TTS/SearXNG qua `/health/detailed` (server-side probe, tránh false-negative do
  CORS) thay vì fetch thẳng client-side. Bỏ hẳn card VieNeu TTS (không có server, chỉ có client
  code — báo "down" gây hiểu lầm).
- `vva.log`: backend ghi log ra `agenticRAG/vva.log` (biến `LOG_FILE` trong `.env`, path
  relative theo CWD lúc chạy uvicorn). Chạy qua `cmd /c "... > vva.log 2>&1"` để log sạch (tránh
  PowerShell bọc ErrorRecord khi dùng `*>`).

### Retrieval perf P1/P2/P3 — `docs/fixes/retrieval-perf-p123.md`
- **P1**: embedding load offline thật sự 0-HF-call (set `HF_HUB_OFFLINE=1` ở đầu `api/main.py`
  trước import — set trong hàm load là quá trễ vì huggingface_hub cache cờ lúc import).
- **P2**: hard-cap retriever⇄tools ở 2 vòng trong `routing.py` (state `retriever_rounds`,
  không tin prompt "max 2 rounds").
- **P3**: web_search toggle enforce ở MỌI tầng — prompt có điều kiện (không mời gọi tool khi
  tắt) + guard node bọc `ToolNode` (chặn thật nếu LLM lỡ gọi). Verify live cả 2 trạng thái.

### README.md rewrite
- Viết lại toàn bộ: kiến trúc 8-node đúng thực tế, 2 sơ đồ mermaid (request flow + system stack),
  số liệu đo thật (296 test, ~7.800 LOC backend, ~4.200 LOC FE, 33 quyết định D1-D33), mục
  "Engineering highlights" phục vụ CV/portfolio.

### Benchmark thật + Latency fix #1-4 — `docs/fixes/latency-optimization-1234.md`
- Benchmark 29 request thật → phát hiện 100% chi phí là LLM call, đuôi p90 khủng (planner 24s,
  synth 30s), retriever lãng phí (tool_calls trùng, vòng rỗng).
- **#1** Log cache-hit/miss token DeepSeek (`extract_cache_tokens` trong `llm.py`) → verify live
  **~91% cache-hit rate** trên planner system prompt (M.7 vốn đã hoạt động, giờ đo được).
- **#2** Timeout (fast 20s/heavy 35s) + `max_retries=1` DeepSeek + fallback Gemini một-lần khi
  primary lỗi/timeout (`get_fallback_chat_model`, dùng `GEMINI_API_KEYS` có sẵn trong `.env`).
  K tự phát hiện + sửa 2 bug khi review: (a) streaming dở + fallback ghi đè → chữ trùng lộn xộn
  (guard `already_streamed` trong `synthesizer.py`); (b) Gemini SDK tự retry 36s khi 429 → thêm
  `max_retries=0` cho fallback client (one-shot thật). Verify live dưới double-failure thật
  (DeepSeek 402 + Gemini 429 cùng lúc): fail nhanh 40.8s→4.5s.
- **#3** `max_tokens` theo role (fast 512/heavy 1024) + rút prompt synthesizer "500 từ"→"350 từ".
  Verify live: response dài nhất 243-252 từ, kết câu trọn vẹn không cụt.
- **#4** Dedupe tool_calls trùng hệt (name+args) trong 1 vòng retriever trước khi ToolNode chạy.
  Verify live: không có false-positive (query khác nhau không bị xoá nhầm); true-positive đã
  unit-test riêng.
- **Bonus — bug tìm thấy khi live-test, không nằm trong spec ban đầu**: race-condition trong
  `E5EmbeddingService.model` (singleton lazy-load KHÔNG có lock) — 3 kb_search song song load
  model 3 lần cùng lúc → tràn RAM → **crash backend 2 lần khi benchmark**. Fix: thêm
  `threading.Lock` double-checked locking (`shared/embedding.py`) — giữ nguyên 100% hành vi lazy
  (Owner dặn không bỏ), chỉ serialize lần load đầu tiên khi có nhiều thread đua nhau. Verify:
  test đồng thời 5 thread → còn 1 lần construct; live retry đúng câu từng crash → chạy được.

### Cleanup (dọn dẹp)
- `youtube-transcript-api` pin `<1.0` (1.x bỏ API code đang gọi `get_transcript` → sẽ crash
  runtime nếu không pin).
- `npm run build` sửa 9 lỗi TS/6 file: AuthGuard (unused import + null-safety `tokens?.`),
  MobileNavBar (bỏ prop `onOpenModal` thừa), 4 trang auth (`'select_account'`→`'SELECT_ACCOUNT'`,
  Amplify đổi enum casing). AuthGuard bypass chuyển từ comment-hack → env-gate
  `VITE_AUTH_DISABLED` (xem §0). **Build production giờ chạy được** (`✓ built in ~12s`).
- `.env` (`agenticRAG/agentic_rag_gemini/.env`): xoá config chết kiến trúc cũ (Qdrant, ChromaDB,
  **secret Pinecone đang phơi**, Firebase/Firestore) — giữ đúng những gì code hiện tại đọc.
- 3 file `FIX-*.md` root dời vào `docs/fixes/` (đồng bộ với reorg).

### axios migration (frontend)
- Cài `axios`, tạo `http` instance (`ECA_UI/frontend/src/lib/api.ts`) với request interceptor
  gắn Cognito idToken 1 chỗ. Migrate 6 hàm REST (`listSessions`, `getSession`, `deleteSession`,
  `listUserMemory`, `createUserMemory`, `deleteUserMemory`) sang axios.
  **`streamChat` (SSE `/chat`) CỐ Ý giữ `fetch`** — axios/XHR không stream token tiến dần được.
  ⚠️ Các hàm REST axios **chưa có UI caller** (`ChatSessionsPanel` vẫn là mock tĩnh) — verify
  bằng tsc (0 lỗi) + chat end-to-end vẫn chạy, không verify được qua UI thật vì chưa ai gọi.

---

## 3. Còn thiếu / pending (theo mức ưu tiên)

### 🔴 Chặn trước khi ra mạng thật
| # | Task | Ghi chú |
|---|---|---|
| 1 | Bật auth thật: `REQUIRE_AUTH=true` + config Cognito thật + `VITE_AUTH_DISABLED=false` | cơ chế đã code xong, chỉ chưa bật |
| 2 | Rate limiting cho `/chat` | chưa có gì chặn spam → cháy quota LLM |
| 3 | Secret management (chuyển `.env` key sang secret manager) | trước khi deploy cloud |
| 4 | Gỡ `null` khỏi CORS allow-list | tôi thêm để test `file://`, phải bỏ trước production |

### 🟠 Độ tin cậy / vận hành
- Docker không có restart policy — tắt máy là tắt container, phải tay động `docker compose up -d`.
- `/health/detailed` trả 503 khi thiếu TTS (optional dep kéo cả health status) — nên tách
  critical vs optional trước khi có LB/orchestrator thật.

### 🟡 Việc code cụ thể
- **21 file đang staged, chưa commit** (xem §8) — rủi ro nếu máy có sự cố.
- Memory FE reset khi đóng sidebar (Owner đã chủ động bỏ persistence — biết và chấp nhận).
- Animation nhân vật "cúi đầu" lúc load trang — Owner nói để sau khi logic xong.
- Bundle FE nặng (JS ~1.9MB gzip 549KB + VRM asset 9-29MB bundle thẳng) — chưa lazy-load/CDN.

### 🟢 Tối ưu có dư địa, chưa làm
- Eval dataset ~50 golden case (đo recall/latency trước-sau khi đổi prompt/model).
- CI chạy test mỗi PR + branch protection (có `release-tests.yml`, chưa xác nhận chặn merge).
- Đổi embedding model (đã bàn kỹ — khuyến nghị `gte-multilingual-base` — nhưng cần eval dataset
  trước để đo ROI, chưa làm).

---

## 4. Phase 7 — Hybrid Cloud (ON HOLD, chờ Owner bàn)
- Tri's `infra/` CDK (Python) đã merge: VPC isolated + RDS Proxy + Lambda CRUD + API Gateway.
- Đã chốt: Alembic = nguồn migration duy nhất; `/chat` KHÔNG qua API Gateway (timeout 29s) →
  ECS Fargate; Kimodo host = edge RTX 3060 + SQS pull; voice = push-to-talk (giữ SSE).
- Còn treo (cần Owner): Supabase vs RDS (chi phí ~$80/mo vs ~$30 lean) — chốt trước khi K viết
  spec Phase 7 chi tiết.

---

## 5. Cách chạy (local demo)

```bash
# 1. Docker
docker compose -f docker-compose.langgraph.yml up -d postgres redis searxng

# 2. Migration (nếu schema đổi)
cd agenticRAG/langgraph_agents && alembic upgrade head

# 3. Backend (:8000 — KHÔNG dùng 8080) — conda env firstconda
cd agenticRAG
python -m uvicorn langgraph_agents.api.main:create_app --factory --port 8000 --host 0.0.0.0

# 4. Frontend React (:5173) — demo mode, VITE_AUTH_DISABLED=true trong .env.local
cd ECA_UI/frontend && npm install && npm run dev

# Tests
python -m pytest tests/langgraph_agents/ -m unit -q   # 259 passed, không cần service sống
python -m pytest tests/langgraph_agents/ -q            # 296 (cần Docker + DeepSeek key thật)
```

## 6. Key files (path MỚI sau reorg — đừng tìm ở root)
| File | Nội dung |
|---|---|
| `docs/plans/reupdate-plan.md` | 33 decisions D1-D33 — nguồn chân lý kiến trúc |
| `docs/tracking/tech-debt.md` | Việc tồn (nhánh riêng khỏi status.md này) |
| `docs/fixes/*.md` | Spec/handoff từng cụm fix (memory, auth, chatpanel, retrieval-perf, latency) |
| `docs/ops/runbook.md`, `docs/ops/troubleshooting.md` | Chạy + debug chi tiết |
| `docs/architecture/*.md` | Kiến trúc chi tiết theo chủ đề |
| `docs/worklogs/DD-MM-YYYY.md` | Nhật ký từng phiên |
| `README.md` (root) | Đã viết lại đầy đủ — kiến trúc + sơ đồ + benchmark, dùng cho CV/portfolio |
| `.claude/CLAUDE.md` | Roles K/N/Owner, conventions |

## 7. Conventions
- Worklog `docs/worklogs/DD-MM-YYYY.md` mỗi phiên đáng kể. Test phải xanh trước khi coi là xong.
- K = Architect (spec + review; khi việc lớn thì viết spec → spawn subagent Sonnet implement →
  **K tự đọc diff + tự chạy test + tự verify live trước khi báo cáo** — không tin lời subagent).
  N = Developer. Owner = "Mr. Senryuu", chốt vision, không tự commit/push khi chưa được lệnh.
- Code = English, docs = Việt + Anh. UI verify bằng skill `playwright-cli` (không npx). Backend
  port cố định **8000** (8080 = Spring của Owner, không đụng).

## 8. Danh sách file đang staged (git status lúc viết file này)
```
M  ECA_UI/frontend/package-lock.json, package.json
M  ECA_UI/frontend/src/components/{AuthGuard,MobileNavBar,ProfileContent}.tsx
M  ECA_UI/frontend/src/lib/api.ts
M  ECA_UI/frontend/src/pages/{CreateAccountPage,EnterPasswordPage,LoginPage}.tsx
M  README.md
M  agenticRAG/langgraph_agents/{llm.py, nodes/planner.py, nodes/retriever_agent.py,
   nodes/synthesizer.py, shared/embedding.py}
R  FIX-AUTH-INTEGRATION.md → docs/fixes/auth-integration.md
R  FIX-CHATPANEL-WIRE.md → docs/fixes/chatpanel-wire.md
R  FIX-RETRIEVAL-PERF-P123.md → docs/fixes/retrieval-perf-p123.md
A  docs/fixes/latency-optimization-1234.md
A  docs/worklogs/06-07-2026.md
A  tests/langgraph_agents/test_fix_latency_1234.py
M  requirements-langgraph.txt
M  tests/langgraph_agents/{test_phase2_5_planner.py, test_phase6_circuit_breaker.py}
```
Chưa commit — chờ lệnh Owner.
