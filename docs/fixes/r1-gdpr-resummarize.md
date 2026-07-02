# FIX — R1 GDPR re-summarize + R2 tests (cụm A re-work)

> Author: K | Date: 2026-06-13 | Audience: N
> Nguồn: K review cụm A (worklog 12-06-2026-cont.md §K REVIEW).
> Trạng thái: cụm A bị REQUEST CHANGES. R1 + R2 là điều kiện merge. R3 optional.
> Path: package ở `agenticRAG/langgraph_agents/`.

---

## Bối cảnh 1 đoạn

A2 (GDPR wiring) nối endpoint xóa message → mark-dirty → fire re-summarize nền.
Khâu mark-dirty chạy đúng. Khâu re-summarize **gọi sai hàm** + **thiếu hẳn logic tóm
tắt lại** → chunk dirty không bao giờ về active → đoạn hội thoại đó mất khỏi memory
vĩnh viễn. Test không bắt vì A2 chưa có test nào.

---

## R1 — BLOCKING: re-summarize không bao giờ chạy

### Lỗi (chính xác)

`api/main.py:317`:
```python
task = asyncio.create_task(re_summarize_chunk(chunk["id"], session_id))   # 2 args — SAI
```
Signature thật (`db/gdpr.py:185`):
```python
async def re_summarize_chunk(session_id, chunk_id, new_summary_text, new_embedding) -> bool
```
→ thiếu 2 args bắt buộc → `TypeError` → vì là `create_task` fire-and-forget, exception
bị nuốt → chunk ở `status='dirty'` mãi mãi → `_load_summary_chunks` + `memory_search`
lọc `status='active'` nên đoạn đó **biến mất khỏi context + search vĩnh viễn**.

### Gốc rễ

`gdpr.re_summarize_chunk` chỉ làm 1 việc: UPDATE summary_text + embedding + set active.
Nó **kỳ vọng caller đưa SẴN** `new_summary_text` + `new_embedding`. Việc "đọc messages
còn lại trong chunk → nhờ LLM tóm tắt lại → tạo embedding mới" **chưa ai viết**. Đó là
phần phải bổ sung.

### Cách sửa

**Bước 1 — tách helper tóm tắt dùng chung** (refactor nhẹ `nodes/summarizer.py`):

Logic LLM-summarize-từ-list-messages đang nằm inline trong `_run_summarize` (dòng
105-149). Tách thành 1 hàm tái dùng được:
```python
async def _summarize_messages(rows: list[dict], session_id: str) -> str | None:
    """rows = [{role, content, token_count}]. Trả summary_text hoặc None nếu fail.
    Dùng prompt + retry hiện có. (tách từ _run_summarize để rebuild_dirty_chunk dùng lại.)"""
```
`_run_summarize` gọi lại helper này — hành vi M.5 không đổi (test PR2 vẫn xanh).

**Bước 2 — viết `rebuild_dirty_chunk` trong `nodes/summarizer.py`:**
```python
async def rebuild_dirty_chunk(session_id: str, chunk_id: str) -> bool:
    """Re-summarize 1 dirty chunk từ messages CÒN LẠI trong range của nó (M.8 #5,#6).
    Trả True nếu chunk về active; False nếu bỏ qua (empty/LLM fail)."""
    pg = get_pg_client(); await pg.connect()

    # 1. Lấy range của chunk dirty
    chunk = await pg.fetchrow(
        "SELECT covers_from_seq, covers_up_to_seq FROM summaries "
        "WHERE id = $1 AND session_id = $2 AND status = 'dirty'",
        chunk_id, session_id)
    if not chunk:
        return False   # đã bị xử lý / không còn dirty (idempotent)

    # 2. Load messages CÒN LẠI trong range (đã xóa thì không còn ở đây)
    rows = await pg.fetch(
        "SELECT role, content, token_count FROM messages "
        "WHERE session_id = $1 AND seq_id BETWEEN $2 AND $3 ORDER BY seq_id",
        session_id, chunk["covers_from_seq"], chunk["covers_up_to_seq"])
    if not rows:
        return False   # empty-chunk: delete_message đã xóa rồi (M.8 #3) — không tới đây

    # 3. Tóm tắt lại + embed + UPDATE (gọi đúng signature gdpr 4-args)
    summary_text = await _summarize_messages(rows, session_id)
    if not summary_text:
        return False   # LLM fail → để dirty, lần sau thử lại (KHÔNG để mất an toàn)
    embedding = await get_embedding_service().aembed_passage(summary_text)
    from langgraph_agents.db.gdpr import re_summarize_chunk
    return await re_summarize_chunk(session_id, chunk_id, summary_text, embedding)
```

**Bước 3 — sửa `api/main.py:316-319`:**
```python
from langgraph_agents.nodes.summarizer import rebuild_dirty_chunk, _pending_summarizer_tasks
for chunk in await get_dirty_chunks(session_id):
    task = asyncio.create_task(rebuild_dirty_chunk(session_id, chunk["id"]))   # đúng args
    _pending_summarizer_tasks.add(task)
    task.add_done_callback(_pending_summarizer_tasks.discard)
```

> Lưu ý thứ tự: `delete_message` đã xóa empty-chunk TRƯỚC khi endpoint gọi
> `get_dirty_chunks` → list trả về chỉ còn chunk dirty CÓ messages → an toàn. Vẫn giữ
> guard `if not rows: return False` cho chắc.

---

## R2 — Acceptance tests (BLOCKING, thiếu thì R1 lại lọt lần nữa)

Tạo `tests/langgraph_agents/test_a1_a2_a3.py` (hoặc tách 3 file). Tối thiểu:

### A2 — GDPR (integration, PG thật — quan trọng nhất)
- [ ] **dirty-window**: tạo session + 1 summary chunk cover seq 1-10 + messages; xóa 1
      message giữa range → chunk `status='dirty'` → `_load_summary_chunks` KHÔNG trả nó
      → `memory_search` KHÔNG trả nó
- [ ] **re-summarize round-trip** (sau R1 fix): sau `rebuild_dirty_chunk` → chunk về
      `active`, `summary_text` KHÔNG còn chứa nội dung message đã xóa, embedding đổi
- [ ] **empty-chunk**: xóa hết message trong range → chunk bị DELETE (không còn row)
- [ ] **ownership**: user B `DELETE /sessions/{sid}/messages/{mid}` của user A → 404, message còn nguyên
- [ ] **delete_user**: sau `DELETE /users/{id}` → 0 row ở conversations/messages/summaries/user_memory + STM Redis sạch
- [ ] **regression R1**: gọi endpoint delete_message với chunk dirty → assert
      `rebuild_dirty_chunk` chạy KHÔNG raise (mock LLM/embed), chunk về active

### A1 — clarify động (unit)
- [ ] resume: 2 session gap 1h → `ambiguous=true` + ≤3 candidates; gap 3 ngày → không có key ambiguous
- [ ] memory_search: 2 kết quả khác session, sim gap < 0.05 → ambiguous; cùng session hoặc gap lớn → không
- [ ] `synthesizer._derive_mode` → "clarify" khi ToolMessage chứa `ambiguous: true`

### A3 — user_memory (integration)
- [ ] POST fact → memory node turn sau inject `[USER FACTS]` vào SystemMessage
- [ ] ownership: user B DELETE fact user A → 404
- [ ] validate: fact_text rỗng / >500 chars → 422

---

## R3 — nit (KHÔNG chặn, làm nếu rảnh)

1. `test_a0_persona_security.py::test_cache_not_polluted_by_invalid` kết thúc bằng
   comment, không có `assert` về cache → thêm assert thật (vd id invalid không tạo
   entry "valid", hoặc fallback có `title=="ECA Default"`).
2. `_persona_cache` cache cả fallback cho id hợp-pattern-nhưng-không-tồn-tại → 1 user
   spam nhiều id lạ = cache phình. Cân nhắc: KHÔNG cache fallback (chỉ cache khi đọc
   file thành công). 5 phút.

---

## Định nghĩa XONG (re-review)

1. R1 fix theo 3 bước trên; `rebuild_dirty_chunk` chạy không raise
2. R2: đủ test list trên, đặc biệt dirty-window + re-summarize round-trip + regression R1
3. Full suite xanh (208 + test mới), test PR2 (M.5) vẫn xanh sau khi tách `_summarize_messages`
4. Worklog cập nhật → K re-review (vòng này chỉ soi R1 + test, nhanh)
