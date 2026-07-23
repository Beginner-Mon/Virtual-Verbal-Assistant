# VVA — Status & Roadmap

> Last update: 2026-07-22 (K) | Branch: `feature/langgraph-rewrite`
> Audience: K/N/Owner takeover after context compaction — đọc mục 0 trước tiên.

---

## 0. TRẠNG THÁI ĐANG TREO (đọc trước — dễ mất khi compact)

- **Backend + Docker hiện ĐANG CHẠY** (verify lúc viết file này: `curl :8000/health` →
  `{"status":"ok"}`, `docker ps` → postgres/redis/searxng healthy, up 31h). Không cần khởi động
  lại — nếu tắt, lệnh chạy vẫn ở §5.
- **Git**: batch phiên 21-22/07 (Gemini caching + D34 web fallback + stage indicator) **đã
  commit** bởi N: `35985d5 fix bugs in UI streaming` (đè lên `621fcd5` + `878c6ed`). Working
  tree hiện còn phiên 23/07 (2 fix streaming) **CHƯA commit**:
  `M agenticRAG/langgraph_agents/nodes/synthesizer.py` (sleep(0) drain fix),
  `M ECA_UI/frontend/src/lib/api.ts` (CRLF boundary fix),
  `?? docs/worklogs/23-07-2026.md`. Không tự commit/push khi chưa được lệnh.
- **Docs đã reorg** (đã ổn định từ phiên trước): root `FIX-*.md` cũ giờ ở
  `docs/{architecture,ops,plans,fixes,tracking,archive}/`. File này ở **`docs/tracking/status.md`**.
- **Auth demo bypass**: env-gate `VITE_AUTH_DISABLED=true` trong `ECA_UI/frontend/.env.local` →
  vào thẳng chat không cần Cognito. Production: bỏ/`=false` + `REQUIRE_AUTH=true` + 3 biến Cognito.
- **DeepSeek + Gemini**: từng hết tiền/quota cùng lúc phiên 21/07 (đã verify sống, không phải
  bug) → Owner đã nạp lại DeepSeek. Lỗi `402`/`429` lại = vấn đề tài khoản, không phải code.
- **Gemini context caching** (phiên 22/07, `docs/worklogs/22-07-2026.md`): hạ tầng đã code + test
  xong, nhưng **luôn inert** — free-tier API key có cache-storage quota = 0 (verify live, 429).
  Chỉ áp dụng cho `planner` fallback (prompt tĩnh). Không tối ưu latency hiện tại — chuẩn bị cho
  tính năng BYO-key tương lai. Chưa commit (xem mục Git ở trên).
  Xem `docs/architecture/streaming-vs-validated.md` nếu tồn tại — **chưa có**, quyết định
  "stream trước hay sau grader" đang treo, xem mục dưới.
- **Design treo, chưa code**: Owner hỏi liệu guardrail/grader có làm chậm response không (do lo
  ngại pattern "stream thẳng rồi ghi đè khi grader reject"). K đọc code thật, phát hiện **bug thật
  chưa fix**: khi grader trigger retry, `ChatPanel.tsx` không có xử lý `stage` event nên 2 lần
  sinh của synthesizer bị nối chữ liền nhau trên UI (không tách/xoá buffer). Owner sau đó đề xuất
  hướng triệt để hơn: **bỏ hẳn live-stream, chỉ gửi câu trả lời sau khi qua grader** — K đã phân
  tích trade-off (tổng thời gian xử lý không đổi, nhưng UI sẽ im lặng hoàn toàn 7-21s thay vì thấy
  chữ chạy dần từ ~1-3s) và đề xuất giữ typing-indicator suốt thời gian chờ để giảm nhẹ. **Owner
  chưa chốt hướng nào** — chưa động tới `synthesizer.py`/`main.py`/`ChatPanel.tsx`.
- **Test hiện tại: 296 collected (270 unit pass + 37 integration cần Docker/DeepSeek), 0 fail**
  (verify lại lúc viết file này, dùng đúng `firstconda` env — `python` trên PATH mặc định của
  Bash tool KHÔNG có `langchain_google_genai`, phải gọi thẳng
  `/c/Users/Nguyen/miniconda3/envs/firstconda/python` hoặc `conda activate firstconda` trước).
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

### Synthesizer model tier: deepseek-v4-pro → deepseek-v4-flash — `docs/worklogs/21-07-2026.md`
- Live A/B test thật (5 kịch bản × 7 lần/model, không mock): flash thắng **mọi** lần đo (1.5-4x
  nhanh hơn), kể cả worst-case flash < best-case pro. Đọc tay chất lượng output (kể cả kịch bản
  an toàn cao — đau ngực): không thấy khoảng cách, flash còn cụ thể hơn (có số cấp cứu 115).
- `_HEAVY_ROLES` rỗng, tách riêng `_LONG_OUTPUT_ROLES = {"synthesizer"}` để giữ `max_tokens=1024`/
  `timeout=35s` dù đổi model nhẹ (tránh cắt cụt response dài). Fallback Gemini synthesizer đồng bộ
  theo sang flash tier.
- Verify live qua Docker + backend thật: request đau lưng thật, grader reject lần đầu → retry,
  2 lần synthesizer flash (8.57s+7.93s=16.5s) **vẫn nhanh hơn** 1 lần heavy cũ (21.05s benchmark
  trước) dù chạy gấp đôi.
- Follow-up cùng ngày: đổi Gemini fallback model `gemini-2.0-flash` → `gemini-2.5-flash`. Verify
  trực tiếp bằng API thật trước khi chọn — phát hiện `gemini-3.1-flash-lite` trả `.content` dạng
  list cấu trúc (không phải string) qua `langchain_google_genai==4.2.3` hiện cài, sẽ **crash**
  code production đúng lúc fallback cần chạy nhất → loại, chọn `2.5-flash` (verify sạch).

### Gemini explicit context caching (hạ tầng, inert trên free tier) — `docs/worklogs/22-07-2026.md`
- Owner: "cứ tạo đi" — chuẩn bị hạ tầng cho tính năng tương lai (user tự upload API key + chọn
  provider). Research trước khi code: Gemini caching không tự động (khác DeepSeek) — phải tạo
  tường minh `CachedContent` + TTL, ràng buộc `cached_content` không đi kèm `system_instruction`
  riêng. Thử tạo cache thật với đúng prompt planner → **429 quota=0** (giới hạn free-tier, không
  phải lỗi code) — đã honest-disclose, không overclaim đã verify được cache thật giảm latency.
- Scope: chỉ `planner` fallback (prompt tĩnh 100%, giống lý do DeepSeek cache ăn ~91% ở đó).
  Tách `get_warm_gemini_cache()` (tra cứu, không gọi mạng, an toàn dùng trong fallback hot path)
  khỏi `warm_gemini_cache()` (gọi API thật, cố ý KHÔNG auto-invoke từ fallback — tránh lặp lại
  bug "fallback chậm hơn không-fallback" đã sửa trước đó).
  `llm.py` +130 dòng, `planner.py` thêm nhánh thử cached model trước khi cache đã ấm.
- Test: 10 test mới, full suite 270 passed 0 regression. Live thật (không mock): bắt đúng lỗi 429,
  trả `None`, không exception lọt lên trên — xác nhận degrade an toàn trên điều kiện lỗi thật.
- **Chưa commit** (xem §0).

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
- **5 file phiên 22/07 chưa commit** (Gemini caching — `llm.py`, `planner.py`,
  `test_fix_latency_1234.py`, 2 worklog mới) — xem §0. Batch 21 file phiên 21/07 đã commit.
- **Bug thật chưa fix, đang chờ Owner chốt hướng**: grader-retry làm `ChatPanel.tsx` nối chữ 2
  lần sinh của synthesizer liền nhau (không tách buffer). Owner đề xuất hướng triệt để hơn (bỏ
  live-stream, chỉ gửi sau grader) thay vì vá buffer — xem §0, chưa code.
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

## 8. Danh sách file chưa commit (git status lúc viết file này, 22/07)
```
M  agenticRAG/langgraph_agents/llm.py
M  agenticRAG/langgraph_agents/nodes/planner.py
M  tests/langgraph_agents/test_fix_latency_1234.py
?? docs/worklogs/21-07-2026.md
?? docs/worklogs/22-07-2026.md
```
Batch 21 file phiên 21/07 (liệt kê ở bản trước của file này) **đã commit** —
`621fcd5 local_version_finalize` + `878c6ed status_update`. Chưa commit đợt mới — chờ lệnh Owner.
