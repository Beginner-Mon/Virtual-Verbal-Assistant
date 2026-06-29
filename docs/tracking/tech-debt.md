# Tech Debt & Pending Tasks

> Checklist các việc đã biết nhưng CHƯA làm. Cập nhật khi đóng/ mở item.
> Last update: 2026-06-13 (K — review cụm A; A0/A3/A5 đóng, R1 GDPR bug mở). Nguồn: worklogs 05→12/06-cont.

Mức: 🔴 critical (phải làm trước Phase 7 deploy) · 🟠 quan trọng · 🟡 nên làm · ⚪ optional

---

## 🔴 Critical — chặn Phase 7

- [ ] **No-auth IDOR toàn bộ session endpoints** — `user_id` client tự khai + uuid5 đoán được
      → đọc/xóa session người khác, mạo danh `/chat` rút LTM. Localhost dev OK; **chặn mọi
      deploy có network**. Fix = JWT/Cognito middleware (Cognito đã có ở nhánh
      `feature/frontend` — tích hợp). (Security review 12/06 — Vuln 1, chi tiết `tracking/predeploy-audit.md`)

## 🟠 Quan trọng

- [ ] **Summarizer E2E với LLM thật** — PR 2 mới có unit test (mock LLM/PG). Cần chạy thủ công:
      hội thoại vượt 10k token → row `summaries` xuất hiện, turn sau memory node load chunk,
      `memory_search` từ session khác cùng user tìm thấy. (worklog 12/06 defer)
- [ ] **`users.auth` cho Phase 7** — bỏ uuid5 coercion ở production, thêm
      `auth_provider/auth_subject` (cột đã có trong schema M.4, chưa có flow).
      uuid5 giữ cho dev/anonymous. (F3 / plan §3.1)
- [ ] **Verify general_query thủ công** — cần LLM + SearXNG chạy thật: hỏi "giá vàng?",
      confirm needs_retrieval=true (3-axis, không còn intent enum), web search fire,
      trả lời hữu ích (không refuse). (worklog 28/05 §5, cập nhật theo 3-axis)

## 🟡 Nên làm

- [ ] **`ai_understanding` (AI tự đúc kết về user)** — AI-auto trích facts vào `user_memory`
      (background, throttled mỗi 5 turn). Advisory; `valid` flag sẵn cho conflict. (D14 phase sau)
## ⚪ Optional / Phase sau

- [ ] **User upload tài liệu riêng** — ĐÃ QUYẾT (29/05): **Option 1 — KB chỉ của hệ thống.**
      Bỏ `documents.user_id` (luôn = system KB), `search_kb` giữ "all public" an toàn, không có
      private doc nên không leak. Khi thật sự cần upload riêng → thêm bảng riêng (`user_documents`
      + `user_doc_embeddings`) lúc đó, design đúng kịch bản thật. Không schema cho feature chưa có.
- [ ] **LLM gợi-ý profile** — LLM phát hiện fact (age/injury) → đề xuất → user confirm → ghi
      `user_memory`. Không tự ghi thẳng. (plan §4.4 optional) — FEATURE, cần Owner quyết build.
- [ ] **Profile trigger nâng cao** — ngoài endpoint, cân nhắc trích từ hội thoại. = gộp vào
      `ai_understanding` (🟡). FEATURE, cần Owner quyết.

> **`vector(384)` hardcode — ĐÃ THỎA, không cần làm**: `E5_DIM=384` (shared/embedding.py) đã là
> single constant bên Python; chỗ `vector(384)` còn lại nằm trong Alembic migration ĐÃ CHẠY —
> đổi model = viết migration mới dù sao. Refactor thêm = cosmetic (karpathy #3). K 13/06.

---

## Đã xong (tham chiếu — không phải pending)

- ✅ **HNSW iterative_scan** — pgvector 0.8.2 trên DB; bật `SET hnsw.iterative_scan='relaxed_order'`
  ở pool `_init_conn` (postgres.py, best-effort guarded) → mọi connection có, không cần wrap
  transaction. Recall đúng khi memory_search filter `session_id=ANY`. K 13/06.
- ✅ **Persona cache không cache fallback** (R3 nit #2) — `_fallback_persona` gắn cờ `_fallback`;
  `get_persona` chỉ cache persona thật → flood id xấu không phình cache. +1 assert test A0. K 13/06.
- ✅ **YouTube paste-link Q&A (cụm B)** — `youtube_transcript(url)` tool (KHÔNG ghi KB/LTM,
  reuse `_extract_video_id`, truncate 12k chars, empty≠error D23), trong `RETRIEVER_BASE_TOOLS`
  + prompt retriever. Hướng TOOL thay "planner detect" (D2b/D16 — tái dùng đường evidence sẵn,
  0 đổi graph/state/synthesizer). +12 test. Subagent (Sonnet), K verify 237/237. worklog 13/06.
  Spec `FIX-YOUTUBE-PASTE.md`.
- ✅ **`test-ui/app.js` done-label stale field** — `payload.intent` → `required_outputs` (SSE
  done event không còn `intent`). Phần resume rendering đã đúng shape M.4 (không tham chiếu
  `metadata/intent`). + assert thật cho test cache persona (R3 nit #1). K 13/06.
- ✅ **GDPR re-summarize bug (R1) + acceptance tests (R2)** — `rebuild_dirty_chunk` mới ở
  `nodes/summarizer.py` (tái dùng `_summarize_messages` tách từ `_run_summarize`); api/main.py
  fire đúng signature. +17 test (dirty-window, re-summarize round-trip, R1 regression, A1/A3).
  Subagent code (Sonnet, karpathy-guidelines), K verify code + tự chạy 225/225 pass (PG thật,
  không skip). worklog 13/06.
- ✅ **Path traversal `persona_id` (A0)** — 2 tầng: Pydantic `pattern` ở `ChatRequest` +
  validate regex + `relative_to` containment ở `_persona_loader`. 4 test pass. K verify 13/06.
  worklog 12/06-cont (cụm A).
- ✅ **`user_memory` write path (A3)** — 3 endpoints POST/GET/DELETE `/users/{id}/memory`,
  ownership check ở DELETE. K verify 13/06. worklog 12/06-cont.
- ✅ **Task registry `_pending_summarizer_tasks` (A5)** — chuyển về `nodes/summarizer.py`
  module-level, bỏ lazy-import ngược. worklog 12/06-cont.
- ✅ **Clarify động M.2b — tool emit ambiguity (A1 cụm A)** — `memory_search` (gap sim<0.05) +
  `resume_last_session` (gap thời gian<24h) trả `{ambiguous, candidates}`; synthesizer clarify
  nhận tool_results. K verify 13/06. worklog 12/06-cont. (⚠️ còn thiếu test — R2 ở trên.)
- ✅ **`memory_search` tenant leak (A1 tái xuất)** — SQL thêm `session_id = ANY($ids)` cả 2
  nhánh; test tenant-isolation 2-user chạy PG thật PASS. K verify 12/06. worklog 12/06 (PR 1).
- ✅ **`memory_search` + `resume_last_session` bind vào graph** — vào `RETRIEVER_BASE_TOOLS`,
  scope inject qua `config: RunnableConfig` (LLM không thấy `user_id` — test schema verify).
  worklog 12/06 (PR 1).
- ✅ **Background summarizer M.5** — `nodes/summarizer.py` mới: trigger 10k (D13), nền
  (create_task + strong-ref), CAS `ON CONFLICT uq_chunk`, retry 1×. Hook sau
  `write_session_turn`. 10 unit tests. worklog 12/06 (PR 2). (E2E LLM thật → item 🟠 trên.)
- ✅ **Redis STM key/format lệch** — reader đổi sang `stm:{session_id}` +
  `_normalize_redis_format` chấp nhận cả 2 format. worklog 12/06 (PR 1).
- ✅ **Memory & Intent rebuild (M.9 15 bước)** — schema M.4, 3-axis intent, 8 nodes, TAG_RULES,
  GDPR cascade, e5-small. worklogs 06/06 + 11/06. (Còn sót: summarizer M.5 + bind memory tools —
  xem 🔴 phía trên.)
- ✅ **Migration tool (Alembic)** — `002_m4_fresh` chạy thành công, 7 tables. worklog 11/06.
- ✅ **Integration test với PostgreSQL thật** — 187/187 passed (PG 5433 + Redis + embedding).
  worklog 11/06.
- ✅ **HNSW thay IVFFlat** — schema 002 dùng HNSW mọi bảng vector (A5 đóng). worklog 06/06.
- ✅ **DROP cột chết `conversations`** — fresh schema 002 drop toàn bộ bảng cũ, không còn
  JSONB messages. worklog 06/06.
- ✅ **3 test `test_phase3_api` fail** — Redis mock `AsyncMock` fix. worklog 11/06 (bug #9).
- ✅ **Nút 📎 + `/documents/upload` chết** — đã gỡ nút khỏi UI (quyết: Option 1 KB-hệ-thống,
  user upload không thuộc MVP). tracking/status.md 11/06.
- ✅ **Tenant isolation (item cũ 29/05)** — superseded: cụ thể hóa thành 🔴 "`memory_search`
  tenant leak" phía trên (12/06).
- ✅ Phase 6.10 — 8 tasks (CORS, log rotation, stop-gen, health checks, TTS cleanup, STM
  token budget, messages table, youtube ingest). worklog 27/05.
- ✅ Corrections 28/05 — messages dùng cột riêng (bỏ JSONB), resume POST→GET, STM lazy populate,
  `_to_uuid` import, breaker→degraded. worklog 28/05.
- ✅ `reasoning_output`/`final_answer` dedup. worklog 28/05.
- ✅ general_query support (off-domain). worklog 28/05 §5. (còn cần verify thủ công — xem trên)
