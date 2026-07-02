# FIX — Memory Layer Gaps (post M-rebuild)

> Author: K | Date: 2026-06-12 | Audience: N
> Nguồn gốc: K takeover review 12/06 — 3 defect 🔴 + 1 🟠 trong `tracking/tech-debt.md`.
> Design KHÔNG đổi — mọi quyết định đã có trong `plans/reupdate-plan.md §M.5/M.6` (D13, D16, D19).
> Đây là spec NỐI DÂY phần đã thiết kế nhưng chưa nối / nối sai.

---

## Bối cảnh 1 đoạn

M.9 rebuild (06/06) viết xong tools + schema nhưng: (1) `memory_search`/`resume_last_session`
không được bind vào graph; (2) SQL của `memory_search` quên scope tenant; (3) không ai viết
summarizer nên bảng `summaries` vĩnh viễn rỗng; (4) Redis cache key lệch giữa writer/reader.
Hệ quả chuỗi: dù có fix (1)+(2), memory_search vẫn trả rỗng cho tới khi có (3).
Thứ tự PR dưới đây là thứ tự dependency thật.

---

## PR 1 — Bind memory tools + tenant scope + STM key (1 PR, sửa cùng nhau)

### Task 1.1 — Inject scope từ config, KHÔNG để LLM sinh

`tools/pgvector_tool.py` — đổi signature 2 tool:

```python
@tool
async def memory_search(
    query: str,
    since_days: Optional[int] = None,
    top_k: int = 3,
    config: RunnableConfig = None,   # ToolNode tự inject, LLM KHÔNG thấy
) -> dict:
    user_id = _to_uuid(config["configurable"]["user_id"])
    current_session_id = config["configurable"]["session_id"]
    ...
```

- LangChain tool nhận param `config: RunnableConfig` → ToolNode tự truyền runtime config,
  param này KHÔNG xuất hiện trong tool schema đưa cho LLM. Đúng M.6: "LLM sinh ARGS
  (query + since), backend bơm scope".
- `resume_last_session` tương tự (bỏ `user_id`, `current_session_id` khỏi args LLM-visible).
- Nhớ `_to_uuid()` (UI gửi "anonymous") — import từ `db/session_store.py`.

### Task 1.2 — Fix tenant scope SQL (leak A1)

`memory_search` Step 2: thêm điều kiện dùng kết quả Step 1 (hiện fetch xong vứt đi):

```sql
WHERE s.session_id = ANY($ids)        -- ← THÊM (đang thiếu — leak chéo user)
  AND s.session_id != $current
  AND s.status = 'active'
```

Áp cho CẢ 2 nhánh (có/không `since_days`).

### Task 1.3 — Bind tools vào graph

- `nodes/retriever_agent.py::_build_tools()`: thêm `memory_search`, `resume_last_session`
  vào danh sách (comment "Step 6" đã hứa nhưng code chưa làm).
- `graph.py::build_graph_async()`: thêm 2 tool vào `all_tools` của `ToolNode`.
- Cập nhật mô tả tool trong `_RETRIEVER_SYSTEM_PROMPT` nếu signature hiển-thị-LLM đổi.

### Task 1.4 — Hợp nhất STM key/format

Quyết: **reader thích nghi writer** (writer `stm:{session_id}` có 3 call sites + test sẵn).

- `nodes/memory.py`: `_RECENT_RAW_KEY = "stm:{session_id}"` + parser chấp nhận format
  `[{q, a, ts}]` → convert thành cặp `{role: "user"/"assistant", content}`.
- Xóa nhánh format `{role, content}` chết nếu không còn ai ghi.

### Nghiệm thu PR 1

- [ ] **Test tenant-isolation (BẮT BUỘC)**: 2 user, mỗi user 1 session + 1 summary;
      `memory_search` của user A không trả summary của user B. Integration test với PG thật.
- [ ] Test: LLM tool schema của `memory_search` KHÔNG chứa `user_id`/`config`
      (assert trên `tool.tool_call_schema`).
- [ ] Test: memory node đọc được STM do `write_session_turn` ghi (round-trip Redis).
- [ ] E2E: query "tôi đã hỏi gì lần trước?" → retriever gọi memory_search không lỗi
      (kết quả rỗng chấp nhận được cho tới PR 2).
- [ ] 187 test cũ vẫn xanh.

---

## PR 2 — Background summarizer (M.5 — phần còn thiếu)

> Spec đầy đủ: `plans/reupdate-plan.md §M.5` + D13. Tóm tắt vận hành:

### Task 2.1 — Trigger sau mỗi turn

Trong `write_session_turn` (hoặc hook sau nó ở `api/main.py`):

1. Tính token cộng dồn của messages có `seq_id > covers_up_to_seq` cuối cùng
   (`token_count` null → estimate `len//4`).
2. Nếu ≥ 10_000 (D13) → fire background task (pattern `asyncio.create_task` +
   strong-ref set, y như `_pending_tts_tasks` trong `api/main.py` — KHÔNG block stream).

### Task 2.2 — Summarize chunk (đóng băng 1 lần)

1. Load messages trong khoảng `(last_covers_up_to_seq, mốc mới]`.
2. LLM summarize (model rẻ — dùng `get_chat_model("planner")` tier).
3. `embed_passage(summary_text)` — e5 prefix `passage:` (D10).
4. INSERT cùng transaction:
   ```sql
   INSERT INTO summaries (session_id, summary_text, covers_from_seq,
                          covers_up_to_seq, embedding, status)
   VALUES (...,'active')
   ON CONFLICT ON CONSTRAINT uq_chunk DO NOTHING   -- CAS idempotent (M.5)
   ```
5. Chunk ĐÓNG BĂNG — không bao giờ UPDATE summary_text (trừ luồng GDPR đã có ở `db/gdpr.py`).

### Task 2.3 — Edge A (summarize fail)

- Retry 1 lần. Vẫn fail → log WARNING, để nguyên (turn sau trigger lại).
- Hard cap: recent raw vượt 2× ngưỡng (20k) mà chưa nén → memory node cắt bớt raw
  theo budget hiện có (`_select_recent_raw` đã làm) — không phình vô hạn, không crash.

### Nghiệm thu PR 2

- [ ] Unit: trigger đúng ngưỡng (9.9k không fire, 10.1k fire), CAS không tạo chunk trùng
      (gọi 2 lần song song → 1 row).
- [ ] Integration (PG thật): hội thoại vượt 10k token → row `summaries` xuất hiện,
      `embedding` NOT NULL, memory node turn sau load được chunk vào context.
- [ ] E2E nối PR 1: sau khi có summary, `memory_search` từ session KHÁC của CÙNG user
      tìm thấy nó; user khác KHÔNG thấy.
- [ ] Đo: latency `/chat` không tăng (task chạy nền).

---

## Ngoài phạm vi (đừng kéo vào)

- `user_memory` write path / endpoint facts — item 🟠 riêng trong tracking/tech-debt.
- Grounding check, query-rewrite — M.11 defer.
- `iterative_scan` pgvector — verify sau PR 1, item 🟡.
- Nâng cấp grader — Mr. N để bàn lại sau khi chạy thực tế.

## Worklog

N log vào `docs/worklogs/DD-MM-YYYY.md` như mọi khi; K review trước khi merge từng PR.
