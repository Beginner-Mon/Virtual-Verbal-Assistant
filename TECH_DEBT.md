# Tech Debt & Pending Tasks

> Checklist các việc đã biết nhưng CHƯA làm. Cập nhật khi đóng/ mở item.
> Last update: 2026-06-13 (K — review cụm A; A0/A3/A5 đóng, R1 GDPR bug mở). Nguồn: worklogs 05→12/06-cont.

Mức: 🔴 critical (phải làm trước Phase 7 deploy) · 🟠 quan trọng · 🟡 nên làm · ⚪ optional

---

## 🔴 Critical — chặn Phase 7

- [ ] **No-auth IDOR toàn bộ session endpoints** — `user_id` client tự khai + uuid5 đoán được
      → đọc/xóa session người khác, mạo danh `/chat` rút LTM. Localhost dev OK; **chặn mọi
      deploy có network**. Fix = JWT/Cognito middleware (Cognito đã có ở nhánh
      `feature/frontend` — tích hợp). (Security review 12/06 — Vuln 1, chi tiết `PREDEPLOY-AUDIT.md`)

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
- [ ] **HNSW iterative_scan** — khi memory_search có filter `session_id = ANY(...)`, HNSW
      over-fetch cần `iterative_scan` (pgvector 0.8) để recall đúng. Verify sau khi fix
      tenant-scope. (REUPDATE §M.4 note)
- [ ] **User paste YouTube link — Q&A feature (MỚI)** — ĐÃ QUYẾT (29/05). User dán link YouTube
      vào ô chat, AI lấy transcript của video đó → đưa vào context → trả lời ngay lượt đó.
      **3 đường YouTube tách bạch, đừng lẫn:**
      | Đường | Ai | Lưu | Nơi |
      |---|---|---|---|
      | `youtube_ingest.py` CLI | Admin (paste list link) | KB chung, mọi user thấy | `kb_embeddings` |
      | Paste link khi chat | End-user | KHÔNG lưu transcript | context tức thời |
      | LTM | Hệ thống tự | "user đưa video về ABC" | `summaries` (M.4 — không còn `memory_embeddings`) |
      **Implement**: planner detect link YouTube trong query → tái dùng `_extract_video_id` +
      `get_transcript` (từ `youtube_ingest.py`) → **truncate/tóm tắt transcript** (video dài =
      hàng chục nghìn từ, sẽ vượt context — lấy ~2-3k token hoặc chunk theo câu hỏi) → nhét context
      cho synthesizer. KHÔNG ghi KB.
      **LTM nhớ**: dùng cơ chế LTM sẵn có (embed Q&A pair mỗi turn — plan §4.3), KHÔNG code riêng.
      Câu trả lời của AI đã chứa "video nói về ABC" → embed tự nhiên → LTM nhớ. *Option thay thế
      (A): tóm tắt transcript thành 1 dòng rồi embed riêng — tốn 1 LLM call, chỉ làm nếu thấy cần.*
      **Giới hạn**: chỉ đọc lời nói (transcript), KHÔNG hiểu động tác hình ảnh. Video không phụ đề
      → rỗng (trừ khi thêm Whisper STT, mà Whisper cũng chỉ ra lời nói, không phải động tác).
## ⚪ Optional / Phase sau

- [ ] **User upload tài liệu riêng** — ĐÃ QUYẾT (29/05): **Option 1 — KB chỉ của hệ thống.**
      Bỏ `documents.user_id` (luôn = system KB), `search_kb` giữ "all public" an toàn, không có
      private doc nên không leak. Khi thật sự cần upload riêng → thêm bảng riêng (`user_documents`
      + `user_doc_embeddings`) lúc đó, design đúng kịch bản thật. Không schema cho feature chưa có.
- [ ] **`vector(384)` hardcode** — gắn với MiniLM-L6-v2. Đổi embedding model → re-migrate cả 2
      bảng vector. Giữ dim ở 1 config constant để đổi 1 chỗ. (plan §3.5 note)
- [ ] **LLM gợi-ý profile** — LLM phát hiện fact (age/injury) → đề xuất → user confirm → ghi
      `profile`. Không tự ghi thẳng. (plan §4.4 optional)
- [ ] **Profile trigger nâng cao** — ngoài settings endpoint, cân nhắc trích từ hội thoại.

---

## Đã xong (tham chiếu — không phải pending)

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
  user upload không thuộc MVP). STATUS.md 11/06.
- ✅ **Tenant isolation (item cũ 29/05)** — superseded: cụ thể hóa thành 🔴 "`memory_search`
  tenant leak" phía trên (12/06).
- ✅ Phase 6.10 — 8 tasks (CORS, log rotation, stop-gen, health checks, TTS cleanup, STM
  token budget, messages table, youtube ingest). worklog 27/05.
- ✅ Corrections 28/05 — messages dùng cột riêng (bỏ JSONB), resume POST→GET, STM lazy populate,
  `_to_uuid` import, breaker→degraded. worklog 28/05.
- ✅ `reasoning_output`/`final_answer` dedup. worklog 28/05.
- ✅ general_query support (off-domain). worklog 28/05 §5. (còn cần verify thủ công — xem trên)
