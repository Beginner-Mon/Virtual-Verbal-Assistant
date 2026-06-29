# FIX — §M.9 Closeout + Security Patch (Cụm A)

> Author: K | Date: 2026-06-12 | Audience: N
> Nguồn: `PREDEPLOY-AUDIT.md` Phần 2-A (Owner approve 12/06: A→B trước demo nội bộ).
> Path mới sau relevel: package ở `agenticRAG/langgraph_agents/`.
> Effort tổng: ~2 ngày. Thứ tự bắt buộc: A0 trước (security), còn lại tùy N.

---

## A0 — Vá path traversal `persona_id` (30m — LÀM TRƯỚC TIÊN)

**Lỗ hổng** (security review 12/06, Vuln 2, confidence 8/10): `_persona_loader.py`
join thẳng `persona_id` từ request vào path → `persona_id="../../docs/worklogs/x"`
đọc file `.md` bất kỳ vào system prompt → LLM echo = exfiltrate. `_persona_cache`
giữ persona độc tới khi restart.

**Fix 2 tầng (defense in depth):**

1. `api/schemas.py` — chặn từ cửa:
```python
persona_id: str = Field(default="eca_default", pattern=r"^[A-Za-z0-9_-]{1,64}$")
```
2. `nodes/_persona_loader.py::_load_persona` — chặn tại nơi dùng (vì loader còn được
   gọi từ chỗ khác ngoài request):
```python
import re
if not re.fullmatch(r"[A-Za-z0-9_-]{1,64}", persona_id):
    logger.warning("invalid_persona_id", extra={"persona_id": persona_id[:80]})
    return _fallback_persona(...)
resolved = (personas_dir / f"{persona_id}.md").resolve()
try:
    resolved.relative_to(personas_dir.resolve())   # containment check
except ValueError:
    return _fallback_persona(...)
```

**Nghiệm thu:**
- [ ] Test: `persona_id="../../../README"` → fallback persona, KHÔNG đọc file ngoài
- [ ] Test: `persona_id="..%2F..%2Fx"`, `"a/b"`, `"a\\b"` → fallback
- [ ] Test: `"eca_default"`, `"eca_clinical"`, `"eca_friendly"` vẫn load đúng
- [ ] Test: POST /chat với persona_id bẩn → 422 (Pydantic chặn)
- [ ] `_persona_cache` không bao giờ chứa key chưa qua validate

---

## A1 — Clarify động M.2b: tool emit ambiguity metadata (2h)

**Gap**: synthesizer đã có `_check_tool_ambiguous()` đọc `{ambiguous: true}` nhưng
không tool nào phát. Spec M.2b/D22: tool trả `{found N, ambiguous, candidates}`.

**Tiêu chí ambiguous (K chốt — giữ đơn giản, deterministic):**

1. `resume_last_session`: nếu ≥2 session khớp filter VÀ `updated_at` của top-2
   cách nhau < 24h → trả:
```python
{"found": True, "ambiguous": True,
 "candidates": [{"session_id", "preview" (first user msg 80 chars), "updated_at"}, ...]}  # max 3
```
   Nếu top-1 cách top-2 ≥ 24h → không mơ hồ, trả như hiện tại.
2. `memory_search`: nếu top-2 kết quả thuộc ≥2 session KHÁC nhau và
   `similarity[0] - similarity[1] < 0.05` → thêm `"ambiguous": True` +
   `"candidates"` (summary_text 80 chars + session_id) vào response hiện có.
   Kết quả vẫn trả đủ — synthesizer tự quyết hỏi hay dùng.

**Synthesizer**: `_CLARIFY_TASK` thêm 1 dòng: khi có candidates trong tool results,
liệt kê 2-3 lựa chọn cho user chọn (đã có khung "list them briefly" — chỉ cần đảm
bảo tool results được đưa vào prompt ở mode clarify; HIỆN clarify mode KHÔNG nhận
`tool_results` — sửa `_CLARIFY_TASK.format` để nhận thêm `tool_results` khi có).

**Nghiệm thu:**
- [ ] Unit: 2 session updated_at cách 1h → ambiguous=true + ≤3 candidates
- [ ] Unit: cách 3 ngày → ambiguous absent/false
- [ ] Unit: memory_search 2 kết quả cùng session, gap nhỏ → KHÔNG ambiguous
- [ ] Unit: `_derive_mode` → "clarify" khi ToolMessage chứa ambiguous=true (test có sẵn, mở rộng)
- [ ] Integration: synthesizer mode clarify nhận được candidates trong prompt

---

## A2 — GDPR wiring M.8 (3h)

**Gap**: `db/gdpr.py` đủ logic (mark-dirty, hard-delete, empty-chunk, re_summarize_chunk)
nhưng zero call sites.

**Endpoints mới (`api/main.py`)** — tạm nhận `user_id` query param như các endpoint
khác cho tới khi JWT (C3); MỌI endpoint phải verify ownership 2-step trước khi xóa
(message → session → user khớp `user_id` đã `_to_uuid`):

| Endpoint | Hành vi |
|---|---|
| `DELETE /sessions/{session_id}/messages/{message_id}?user_id=` | verify ownership → `gdpr.delete_message` (xóa row + mark-dirty chunks chứa seq đó + empty-chunk cleanup) → fire background `re_summarize_chunk` cho từng chunk dirty (pattern create_task + strong-ref như summarizer) |
| `DELETE /users/{user_id}` | `gdpr.delete_user` (FK cascade conversations→messages/summaries + user_memory) → xóa Redis `stm:{sid}` của mọi session user đó |

(`DELETE /sessions/{user_id}/{session_id}` đã có — giữ nguyên, cascade lo phần còn lại.)

**Nghiệm thu:**
- [ ] Integration (PG thật): xóa 1 message nằm giữa chunk → chunk `status='dirty'`
      → memory node KHÔNG load chunk đó (`_load_summary_chunks` lọc `status='active'`
      — đã đúng, viết test chứng minh) VÀ `memory_search` không trả nó
- [ ] Integration: sau re_summarize nền → chunk `active` lại, summary_text KHÔNG còn
      nội dung message đã xóa, embedding đã regen (so vector khác trước)
- [ ] Integration: xóa hết message trong range chunk → chunk bị XÓA hẳn (empty-chunk)
- [ ] Test ownership: user B gọi delete message của user A → 404, không xóa
- [ ] DELETE /users → 0 rows còn lại ở conversations/messages/summaries/user_memory + STM Redis sạch

---

## A3 — `user_memory` write path (1h)

**Gap**: bảng Tier-1 chỉ có code đọc (`memory.py::_load_user_facts`) → facts vĩnh viễn rỗng.

**Endpoints (`api/main.py`)** — D14 MVP user tự ghi, AI-auto là phase sau:

| Endpoint | Hành vi |
|---|---|
| `POST /users/{user_id}/memory` body `{fact_text, category?}` | ensure user row (INSERT ON CONFLICT) → INSERT fact, trả `{id}`. Validate: fact_text 1-500 chars |
| `GET /users/{user_id}/memory` | list facts `valid=true`, mới nhất trước |
| `DELETE /users/{user_id}/memory/{fact_id}` | hard delete (user tự xóa fact của mình ≠ conflict flag) — verify fact thuộc user |

Conflict resolution (`valid=false` cho fact cũ cùng chủ đề) **KHÔNG làm đợt này** —
để cùng `ai_understanding` (STATUS 🟡 #11).

**Nghiệm thu:**
- [ ] Integration: POST fact → memory node turn sau inject `[USER FACTS]` vào SystemMessage
- [ ] Test ownership: user B xóa fact user A → 404
- [ ] Validate: fact_text rỗng/quá dài → 422

---

## A4 — Verify E2E thủ công (1h, cần services chạy thật)

Checklist ghi kết quả vào worklog:
- [ ] Hội thoại >10k token (viết script POST /chat lặp hoặc paste dài) → row `summaries`
      xuất hiện, `embedding NOT NULL` → turn sau memory node load chunk (check log
      `summary_chunks_count > 0`) → từ session MỚI cùng user: "tôi đã hỏi gì về X?"
      → retriever gọi `memory_search` → tìm thấy
- [ ] "giá vàng hôm nay?" → planner `needs_retrieval=true` (log) → `search_medical`
      fired qua SearXNG :6666 → trả lời có thông tin web + không refuse
- [ ] Web toggle OFF + câu real-time → no-source path, giải thích "web đang tắt" (D27)

---

## A5 — Chuyển task registry về đúng module (15m)

`_pending_summarizer_tasks` đang ở `api/main.py`, `summarizer.py` lazy-import ngược
→ chuyển set vào `nodes/summarizer.py` (module-level), api.main chỉ gọi `maybe_summarize`.
Re-summarize background của A2 dùng CHUNG registry này (hoặc set riêng trong gdpr.py
— cùng pattern). Test PR2 hiện patch `asyncio.create_task` — không cần đổi.

## A6 — Test nits (15m)

- `test_pr1_memory_fix.py::test_resume_last_session_schema_hides_user_id`: bỏ dòng
  `assert "since_days" in fields or True` (no-op) → assert thật
- `test_pr1_memory_fix.py::test_memory_search_tenant_isolation`: bọc cleanup trong
  `try/finally` để fail giữa chừng không để rác trong DB

---

## Định nghĩa XONG cụm A

1. Toàn bộ checklist nghiệm thu A0-A3 + A5-A6 pass; A4 có kết quả ghi worklog
2. Full suite xanh (204 + tests mới), không test cũ nào bị sửa để "cho qua"
3. Worklog `docs/worklogs/DD-MM-YYYY.md` + K review trước khi merge
4. Sau cụm A: REUPDATE §M.9 = 15/15 trọn vẹn; TECH_DEBT 🔴 path-traversal đóng
