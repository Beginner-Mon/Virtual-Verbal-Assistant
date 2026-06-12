# Tech Debt & Pending Tasks

> Checklist các việc đã biết nhưng CHƯA làm. Cập nhật khi đóng/ mở item.
> Last update: 2026-06-12 (K — PR 1+2 review, 3 item 🔴 đóng). Nguồn: worklogs 05→12/06.

Mức: 🔴 critical (phải làm trước Phase 7 deploy) · 🟠 quan trọng · 🟡 nên làm · ⚪ optional

---

## 🔴 Critical — chặn Phase 7

- [ ] **Path traversal qua `persona_id`** — `nodes/_persona_loader.py:67` join thẳng
      request-controlled string vào path, không validate → đọc file `.md` bất kỳ trên host
      qua `persona_id="../../..."`, nội dung đổ vào system prompt → LLM echo = exfiltrate;
      `_persona_cache` còn bị đầu độc tới khi restart. Fix: regex `^[A-Za-z0-9_-]+$` +
      `resolved.relative_to(personas_dir)`. 30 phút. (Security review 12/06 — Vuln 2,
      chi tiết `PREDEPLOY-AUDIT.md`)
- [ ] **No-auth IDOR toàn bộ session endpoints** — `user_id` client tự khai + uuid5 đoán được
      → đọc/xóa session người khác, mạo danh `/chat` rút LTM. Localhost dev OK; **chặn mọi
      deploy có network**. Fix = JWT middleware (nâng từ 🟠 "trước demo" lên 🔴).
      (Security review 12/06 — Vuln 1, chi tiết `PREDEPLOY-AUDIT.md`)

## 🟠 Quan trọng

- [ ] **Summarizer E2E với LLM thật** — PR 2 mới có unit test (mock LLM/PG). Cần chạy thủ công:
      hội thoại vượt 10k token → row `summaries` xuất hiện, turn sau memory node load chunk,
      `memory_search` từ session khác cùng user tìm thấy. (worklog 12/06 defer)
- [ ] **Task registry `_pending_summarizer_tasks` đặt nhầm chỗ** — set sống ở `api/main.py`,
      `summarizer.py` lazy-import ngược lại api.main để lấy nó (circular smell; unit test
      maybe_summarize kéo cả FastAPI module). Chuyển set về `summarizer.py`, api.main chỉ gọi
      `maybe_summarize`. Refactor 15', không khẩn. (K review PR 12/06)
- [ ] **`user_memory` write path** — bảng `user_memory` (M.4 thay `users.profile`) chỉ có code
      ĐỌC (`memory.py::_load_user_facts`), không chỗ nào ghi → facts luôn rỗng. Cần endpoint
      user tự nhập facts (MVP per D14). (D1 / plan §4.4, cập nhật theo schema M.4)
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
- [ ] **`test-ui/app.js` resume** — đã đổi sang GET; verify lại sau rebuild (response shape đổi:
      `required_outputs/needs_retrieval/needs_motion` thay `intent/confidence`).
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
