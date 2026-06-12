# Schema Redesign Plan — VVA PostgreSQL

> Architect: K | Date: 2026-05-29 | Cập nhật: 2026-06-02
> Audience: N (Developer), Owner, T (Reviewer)
> Branch target: `feature/langgraph-rewrite`

---

> ## ⚠️ TRẠNG THÁI (02/06): SCHEMA SECTIONS SUPERSEDED
> Phần **§3-6 (schema + code changes)** của tài liệu này đã bị **THAY** bởi
> `REUPDATE_PLAN.md` mục **🔧 MEMORY & INTENT REBUILD (§M.4 schema, §M.5-M.8)** — bản đó
> hoàn thiện hơn: có `seq_id`, `summaries` (chunk đóng băng), `user_memory`, cột `status`
> (GDPR), e5-small embedding. **N implement theo REUPDATE_PLAN, KHÔNG theo §3-6 dưới đây.**
>
> Phần **§1 (audit 15 vấn đề)** vẫn GIÁ TRỊ — đây là phân tích gốc giải thích *tại sao* phải
> redesign (mapping A1-F4 → bảng). Giữ để tham chiếu lý do. REUPDATE §M.4 là *giải pháp* cho
> chính các vấn đề liệt kê ở §1 này.
>
> | Phần | Trạng thái |
> |------|-----------|
> | §0 Tại sao redesign | ✅ Còn đúng (bối cảnh) |
> | §1 Audit 15 vấn đề | ✅ Còn đúng (lý do gốc) |
> | §2 Nguyên tắc thiết kế | ✅ Còn đúng |
> | §3 Schema (6 bảng) | ❌ Superseded → REUPDATE §M.4 |
> | §4 Code changes | ❌ Superseded → REUPDATE §M.5-M.6 |
> | §5 Alembic | ✅ Còn đúng (vẫn dùng Alembic) |
> | §6 Thứ tự | ⚠️ Tham chiếu → REUPDATE §M.9 mới hơn |
> | §8 Quyết định | ✅ Còn đúng (đã merge vào REUPDATE §M.0) |

---

## 0. Tại sao redesign bây giờ

Audit schema (cho Phase 7 deploy) lộ ra 15 vấn đề, trong đó 3 critical. Gốc rễ: bảng
`embeddings` được **bê thẳng từ ChromaDB** ("1 collection + metadata dict") sang PostgreSQL
mà không redesign cho relational paradigm → mất FK, mất tenant column, mất typed constraint.
Bảng nhạy cảm nhất (chứa hội thoại riêng tư + tài liệu user) lại là bảng ẩu nhất.

**Thời điểm đúng**: chưa có data production, chưa multi-user. Sửa structural bây giờ rẻ;
sửa sau khi có data thật + nhiều user = migration đau đớn + nguy cơ leak đã xảy ra.

**Quyết định scope (Owner, 29/05)**:
1. Redesign **toàn bộ** schema + thêm migration tool (Alembic)
2. LTM implement **đầy đủ** — bảng memory riêng có `user_id`, viết write path
3. Phase 7 **có auth thật** — `users` thiết kế cho auth, bỏ uuid5 coercion

---

## 1. Vấn đề tổng hợp (từ audit)

| # | Vấn đề | Mức | Bảng |
|---|--------|------|------|
| A1 | `embeddings` không có `user_id` → leak chéo user (PHI) | 🔴 | embeddings |
| A2 | `source_id UUID` vs YouTube string → INSERT/search vỡ | 🔴 | embeddings |
| F1 | Không migration tool → schema drift thủ công | 🔴 | all |
| A3 | Filtered-ANN recall: lọc `source_type` SAU top-k → thiếu kết quả | 🟠 | embeddings |
| A4 | `source_id` không FK → orphan vector khi xóa nguồn | 🟠 | embeddings |
| A6 | Trộn public KB + private user data 1 bảng | 🟠 | embeddings |
| B1 | `conversations.id` thừa (mọi thứ dùng session_id) | 🟠 | conversations |
| F2 | LTM write path không tồn tại (chỉ có code đọc) | 🟠 | embeddings/memory |
| F3 | uuid5 coercion không lưu mapping → không truy ngược | 🟠 | users |
| B2 | `conversations.summary` cột chết | 🟡 | conversations |
| B3 | `conversations.messages JSONB` thừa sau Correction 1 | 🟡 | conversations |
| C2 | `messages` không có `user_id` (phải JOIN) | 🟡 | messages |
| D1 | `users.profile` không có write path → luôn rỗng | 🟡 | users |
| E1 | `documents` ↔ embeddings rời rạc, không liên kết | 🟡 | documents |
| A5 | IVFFlat `lists=100` tạo trên bảng rỗng → vô dụng | 🟡 | embeddings |
| F4 | `embeddings.metadata` lưu lại `source_type` (duplicate cột) | 🟡 | embeddings |

---

## 2. Nguyên tắc thiết kế mới

1. **Tách public vs private.** Knowledge base (document, youtube, humanml3d — đọc chung)
   tách hẳn khỏi user memory (conversation — private per-user). Hai vòng đời, hai quyền truy
   cập, hai index.
2. **Tenant isolation bằng structure, không bằng convention.** Mọi bảng chứa dữ liệu user có
   `user_id NOT NULL`. Search private LUÔN filter `user_id` ở SQL, không dựa vào code nhớ.
3. **FK ở đâu có thể.** Bỏ polymorphic `source_id`. Mỗi loại embedding FK về bảng nguồn của
   nó (hoặc NULL có chủ đích cho external source không có bảng).
4. **Field cố định → cột; tùy biến → JSONB.** (Đã áp dụng ở `messages` Correction 1.)
5. **Migration versioned.** Alembic. Không `CREATE TABLE IF NOT EXISTS` + ALTER thủ công nữa.
6. **`source_id` là TEXT, không UUID** — chứa được cả UUID document lẫn youtube video_id.

---

## 3. Schema mới (target)

### 3.1 `users` — thiết kế cho auth thật

```sql
CREATE TABLE users (
    id            UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    -- Auth identity. Phase 7: Firebase uid / OAuth sub. Nullable cho dev/anon.
    auth_provider TEXT,                          -- 'firebase' | 'google' | 'anonymous' | NULL
    auth_subject  TEXT,                          -- provider's user id (uid/sub/email)
    display_name  TEXT,
    -- FACTS user khẳng định (settings, hoặc LLM gợi-ý-rồi-user-confirm). High trust.
    profile          JSONB NOT NULL DEFAULT '{}',   -- age, injury_history, fitness_level...
    -- IMPRESSION AI tự đúc kết từ hội thoại. Advisory, AI-owned, user xem/xóa được.
    ai_understanding TEXT,                           -- "thích bài nhẹ, hay đau buổi sáng..."
    created_at    TIMESTAMPTZ NOT NULL DEFAULT now(),
    updated_at    TIMESTAMPTZ NOT NULL DEFAULT now(),
    UNIQUE (auth_provider, auth_subject)
);
CREATE INDEX idx_users_auth ON users (auth_provider, auth_subject);
```

**Giải quyết**: F3 (auth identity rõ ràng, không cần uuid5 coercion ở production — dev/anon
vẫn dùng uuid5 nhưng lưu `auth_provider='anonymous'`), D1 (profile có write path — xem §4.4).

**`profile` vs `ai_understanding` — vì sao tách 2 cột (provenance)**:
- `profile` = điều **user khẳng định** (facts). Ghi qua settings, hoặc LLM gợi ý → user
  confirm → ghi. High trust, dùng cho gợi ý quan trọng.
- `ai_understanding` = điều **AI suy luận** từ hội thoại (impression). AI tự ghi (advisory),
  không cần confirm vì tự gắn nhãn "tôi nghĩ". User xem/xóa được.
- Tách để AI suy luận sai KHÔNG đè lên fact user khai. Hệ thống ở mức **advisory wellness**,
  không phải healthcare chuyên nghiệp → rủi ro thấp, nhưng giữ ranh giới provenance vẫn đúng.

**Migration uuid5**: giữ `_to_uuid()` cho dev/anonymous (backward-compat), nhưng khi Phase 7
auth bật, user thật được tạo với `auth_provider/auth_subject` và `id` mới. Không trộn.

### 3.2 `conversations` — session header gọn

```sql
CREATE TABLE conversations (
    session_id  UUID PRIMARY KEY,                -- bỏ cột id thừa (B1)
    user_id     UUID NOT NULL REFERENCES users(id) ON DELETE CASCADE,
    created_at  TIMESTAMPTZ NOT NULL DEFAULT now(),
    updated_at  TIMESTAMPTZ NOT NULL DEFAULT now()
);
CREATE INDEX idx_conversations_user_updated ON conversations (user_id, updated_at DESC);
```

**Giải quyết**: B1 (session_id PK), B2 (summary bỏ hẳn), B3 (messages JSONB biến mất).
**Không có cột title** — UI session label dùng `first_user_message_preview` (đã có trong
`list_user_sessions`, ~50 ký tự tin nhắn user đầu). Không tốn LLM, không cột thừa. Đổi tên
session thủ công (rename) là feature tương lai, thêm cột lúc đó.
Index `(user_id, updated_at DESC)` phục vụ `list_user_sessions` trực tiếp (không cần sort sau).

### 3.3 `messages` — giữ từ Correction 1, thêm user_id

```sql
CREATE TABLE messages (
    id             UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    session_id     UUID NOT NULL REFERENCES conversations(session_id) ON DELETE CASCADE,
    user_id        UUID NOT NULL REFERENCES users(id) ON DELETE CASCADE,  -- C2: denormalized
    role           TEXT NOT NULL CHECK (role IN ('user', 'assistant')),
    content        TEXT NOT NULL,
    intent         TEXT,
    tokens         INT,
    grader_result  TEXT,
    created_at     TIMESTAMPTZ NOT NULL DEFAULT now()
);
CREATE INDEX idx_messages_session_created ON messages (session_id, created_at);  -- ASC
CREATE INDEX idx_messages_user_created    ON messages (user_id, created_at);     -- per-user query
```

**Giải quyết**: C2. `user_id` denormalized (đã có qua session) nhưng cho phép "tất cả message
của user X" không cần JOIN — hữu ích cho LTM write + analytics + GDPR delete. Index ASC
(sớm→trễ), backward-scan phục vụ DESC query.

### 3.4 `documents` — nguồn của KB embeddings

```sql
CREATE TABLE documents (
    id          UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    source_type TEXT NOT NULL,                   -- 'document' | 'youtube' | 'humanml3d'
    external_id TEXT,                            -- youtube video_id, dataset id... (TEXT! A2)
    title       TEXT,
    metadata    JSONB NOT NULL DEFAULT '{}',     -- doi, author, channel_url... (heterogeneous)
    created_at  TIMESTAMPTZ NOT NULL DEFAULT now()
);
CREATE INDEX idx_documents_source ON documents (source_type, external_id);
```

**Giải quyết**: E1 (documents giờ là parent thật của KB chunks), A2 (`external_id TEXT` chứa
youtube string).

**Quyết định Option 1 (29/05)** — `documents` KHÔNG có `user_id`: toàn bộ KB là **của hệ
thống** (admin/CLI nạp). Không có user upload riêng → không leak qua `search_kb`. Khớp thực
tế: codebase hiện không có endpoint upload nào. Khi thật sự cần upload riêng → thêm bảng riêng
`user_documents` + `user_doc_embeddings` lúc đó (xem TECH_DEBT.md). Không schema sẵn cho
feature chưa tồn tại.

### 3.5 `kb_embeddings` — KB CÔNG CỘNG (tách khỏi private)

```sql
CREATE TABLE kb_embeddings (
    id           UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    document_id  UUID NOT NULL REFERENCES documents(id) ON DELETE CASCADE,  -- FK thật! (A4)
    chunk_index  INT NOT NULL DEFAULT 0,
    content      TEXT NOT NULL,
    embedding    vector(384) NOT NULL,
    created_at   TIMESTAMPTZ NOT NULL DEFAULT now()
);
CREATE INDEX idx_kb_emb_document ON kb_embeddings (document_id);
-- IVFFlat tạo SAU khi có data (xem §5 bước cuối) — A5
```

**Giải quyết**: A4 (FK CASCADE — xóa document, chunks tự xóa, không orphan), E1, A6 (public
tách riêng). `source_type` đọc qua JOIN `documents` — không duplicate (F4). Search KB: filter
qua `documents.source_type` nếu cần, hoặc search toàn KB (đều public, không leak).

### 3.6 `memory_embeddings` — LTM PRIVATE per-user

```sql
CREATE TABLE memory_embeddings (
    id           UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    user_id      UUID NOT NULL REFERENCES users(id) ON DELETE CASCADE,      -- A1: tenant!
    session_id   UUID NOT NULL REFERENCES conversations(session_id) ON DELETE CASCADE,
    content      TEXT NOT NULL,                  -- Q+A pair hoặc turn summary
    embedding    vector(384) NOT NULL,
    created_at   TIMESTAMPTZ NOT NULL DEFAULT now()
);
CREATE INDEX idx_mem_emb_user ON memory_embeddings (user_id);
-- IVFFlat tạo SAU khi có data
```

**Giải quyết**: A1 (`user_id NOT NULL` — leak chéo bất khả về mặt structure), A3 (private tách
riêng → ANN chỉ chạy trên memory vector của 1 user, recall đúng), F2 (đây là đích cho LTM
write path — xem §4.3), A6.

---

## 4. Code changes

### 4.1 `vector_backend.py` — tách 2 backend

Thay 1 class `VectorBackend` mơ hồ bằng 2 method rõ ràng (hoặc 2 class):

```python
async def search_kb(query_embedding, top_k=5, source_type=None) -> list[dict]:
    """Search PUBLIC knowledge base. No user filter (all public)."""
    # JOIN documents để lấy source_type/title; filter source_type optional
    # ORDER BY embedding <=> $1 LIMIT top_k

async def search_memory(query_embedding, user_id, top_k=5, session_id=None) -> list[dict]:
    """Search PRIVATE user memory. user_id MANDATORY — enforced in SQL."""
    # WHERE user_id = $2  ← NOT optional, NOT NULL-bypassed
    # AND ($3::uuid IS NULL OR session_id = $3)
    # ORDER BY embedding <=> $1 LIMIT top_k
```

**Then chốt**: `search_memory` không có nhánh `user_id IS NULL` — thiếu user_id = lỗi, không
phải "search tất cả". Đảo ngược lỗi A1.

### 4.2 `pgvector_tool.py` — retriever dùng `search_kb`

`pgvector_search` tool → gọi `search_kb` (KB công cộng). Không đụng memory. `source_type`
arg giữ để filter document/youtube/humanml3d trong KB.

### 4.3 LTM write path (F2) — MỚI

Trong `write_session_turn` (hoặc hàm async riêng fire sau response):
```python
# Sau khi ghi messages, embed Q+A pair và lưu memory_embeddings
emb = await embed(f"Q: {user_query}\nA: {assistant_answer}")
await pg.execute(
    """INSERT INTO memory_embeddings (user_id, session_id, content, embedding)
       VALUES ($1, $2, $3, $4)""",
    user_id, session_id, qa_text, emb,
)
```
`memory.py::_lookup_ltm` → đổi sang `search_memory(user_id=..., session_id=...)`. LTM giờ
chạy thật.

### 4.4 Profile write path (D1) — MỚI, qua settings endpoint

`update_user_profile(user_id, patch: dict)` + endpoint `PATCH /users/{id}/profile`:
```python
await pg.execute(
    "UPDATE users SET profile = profile || $2::jsonb, updated_at = now() WHERE id = $1",
    user_id, json.dumps(patch),
)
```
JSONB merge (`||`) cập nhật từng field. **Trigger = settings endpoint, KHÔNG để LLM ghi thẳng.**
*Optional Phase sau*: LLM phát hiện fact (age/injury) → ĐỀ XUẤT "Tôi ghi nhận bạn 65 tuổi,
đúng không?" → user confirm → mới gọi `update_user_profile`. Không tự ghi.

### 4.6 AI understanding write path — MỚI (background, throttled)

`ai_understanding` = ấn tượng AI tự đúc kết. Khác `profile` (facts), đây là advisory, AI ghi
tự do không cần confirm.

```python
# Background task sau response (cùng lane TTS, KHÔNG block /chat). Throttle: mỗi K turn.
async def refresh_ai_understanding(user_id, session_id, k=5):
    turn_count = await count_user_turns(session_id)
    if turn_count % k != 0:
        return                                    # chỉ update mỗi 5 turn
    recent  = await load_recent_messages(session_id, limit=20)   # FULL TEXT, không phải vector
    current = await get_ai_understanding(user_id)
    new_text = await llm_summarize_understanding(current, recent)  # 1 LLM call / 5 turn
    await pg.execute(
        "UPDATE users SET ai_understanding = $2, updated_at = now() WHERE id = $1",
        user_id, new_text,
    )
```

**Lưu ý nguồn**: đúc kết từ **recent message history (full text)**, KHÔNG đọc vector
`memory_embeddings` — vector chỉ để retrieve khi search, không để đọc tuần tự. Memory node
load `ai_understanding` kèm `profile` → planner/synthesizer biết "user này thường...".
Throttle K=5 giữ chi phí thấp (1 LLM call / 5 turn, chạy background).

### 4.5 `youtube_ingest.py` — ghi qua documents + kb_embeddings

```python
# 1. Tạo document row (external_id = video_id, TEXT)
doc_id = await create_document(source_type="youtube", external_id=video_id, title=...)
# 2. Mỗi chunk → kb_embeddings với document_id FK
for idx, chunk in enumerate(chunks):
    await insert_kb_embedding(document_id=doc_id, chunk_index=idx, content=chunk, embedding=emb)
```
Giải quyết A2 (video_id vào `documents.external_id TEXT`, không cast UUID nữa).

### 4.5 `youtube_ingest.py` — ghi qua documents + kb_embeddings

```python
# 1. Tạo document row (external_id = video_id, TEXT)
doc_id = await create_document(source_type="youtube", external_id=video_id, title=...)
# 2. Mỗi chunk → kb_embeddings với document_id FK
for idx, chunk in enumerate(chunks):
    await insert_kb_embedding(document_id=doc_id, chunk_index=idx, content=chunk, embedding=emb)
```
Giải quyết A2 (video_id vào `documents.external_id TEXT`, không cast UUID nữa).

---

## 5. Migration tool (F1) — Alembic

```
agenticRAG/agentic_rag_gemini/
  alembic.ini
  alembic/
    env.py                    # async engine, đọc DSN từ config/langgraph.yaml
    versions/
      0001_baseline.py        # schema hiện tại (users/conversations/messages/embeddings/documents)
      0002_redesign.py        # plan này: split kb/memory, user_id, FK, drop dead cols
```

**Tại sao Alembic**: codebase dùng raw asyncpg (không SQLAlchemy ORM) — nhưng Alembic chạy
được standalone với raw SQL trong `op.execute()`. Cho version control + `upgrade`/`downgrade`
+ env reproducibility (Phase 7 dev/staging/prod). Thay thế `init_schema.sql` thủ công.

**Lưu ý**: `requirements-langgraph.txt` đã có `alembic>=1.13.0` (PLAN-v2.4 §16) nhưng chưa
dùng. Đây là lúc kích hoạt.

---

## 6. Thứ tự thực hiện (đề xuất)

Fresh DB (dev) khác Existing DB (nếu có data cần giữ). Vì chưa có prod data → **fresh rebuild**
đơn giản nhất:

```
1. Init Alembic (alembic.ini + env.py async)
2. Viết 0001_baseline (= schema hiện tại, để có điểm xuất phát versioned)
3. Viết 0002_redesign:
   - CREATE users mới (auth cols), conversations (session_id PK), messages (+user_id)
   - CREATE documents (external_id TEXT), kb_embeddings (FK), memory_embeddings (user_id)
   - DROP embeddings cũ, DROP conversations.{id,summary,messages}
4. vector_backend.py: split search_kb / search_memory
5. pgvector_tool.py → search_kb
6. memory.py::_lookup_ltm → search_memory
7. write_session_turn: + LTM embedding write (4.3)
8. youtube_ingest.py: documents + kb_embeddings (4.5)
9. update_user_profile hàm (4.4) — schema + hàm, trigger defer
10. Tạo IVFFlat index trên kb_embeddings + memory_embeddings SAU khi seed data (A5)
11. Test: unit + integration với PostgreSQL thật
```

**Nếu CÓ data cần giữ** (xác nhận với Owner): thêm backfill 0002 — copy embeddings cũ sang
2 bảng mới theo source_type, document orphan handling. Phức tạp hơn nhiều — chỉ làm nếu data
hiện tại có giá trị.

---

## 7. Cái KHÔNG làm trong plan này

- **RLS (Row-Level Security) của PostgreSQL** — tenant isolation bằng app-level `user_id`
  filter là đủ cho MVP. RLS để Phase 7+ nếu cần defense-in-depth tầng DB.
- **Bảng auth_accounts riêng** (multi-provider per user) — gộp vào `users.auth_provider/subject`
  cho đơn giản. Tách khi cần 1 user nhiều login method.
- **Embedding versioning / re-embed** — khi đổi model 384-dim sang khác. Defer.
- **Partition `messages` theo thời gian** — chỉ cần khi >10M rows. Defer.

---

## 8. Quyết định (Owner/N, 29/05) — đã chốt

1. **Data hiện tại**: KHÔNG có data giá trị, toàn bộ là test → **fresh rebuild** (§6, không
   cần backfill migration). Đơn giản hóa: 0002 cứ DROP bảng cũ thẳng.

2. **Hai loại "hiểu biết user", tách 2 cột** (§3.1):
   - `profile` (FACTS) — **endpoint truyền thống** `PATCH /users/{id}/profile`, user tự nhập.
     KHÔNG để LLM ghi thẳng. (§4.4)
   - `ai_understanding` (IMPRESSION) — AI tự đúc kết từ recent history, background throttled
     mỗi 5 turn, advisory. (§4.6)
   - Hệ thống ở mức **advisory wellness**, không phải healthcare chuyên nghiệp → AI đúc kết
     ấn tượng OK; nhưng facts vẫn đi đường endpoint để giữ provenance + user kiểm soát.

3. **LTM granularity**: **embed mỗi Q+A pair** (full text, không summary). Không tốn LLM call
   (chỉ embedding model local), recall chính xác từng câu, đúng với việc PLAN-v2.4 §4.3 đã bỏ
   summary agent. Embed ngay tại `write_session_turn` (§4.3). 1 vector ≈ 1.5KB, user_id filter
   giữ search nhanh.

4. **`conversations` KHÔNG có cột title**. UI session label dùng `first_user_message_preview`
   (~50 ký tự tin nhắn user đầu, đã có trong `list_user_sessions`). Không tốn LLM, không cột
   thừa. Rename session thủ công = feature tương lai, thêm cột lúc đó.

### Hệ quả lên plan
- §6 bước 3: `0002_redesign` DROP thẳng bảng cũ, không backfill.
- §3.1: `users` có 2 cột — `profile JSONB` (facts) + `ai_understanding TEXT` (impression).
- §4.4: `update_user_profile` chỉ gọi từ settings endpoint `PATCH /users/{id}/profile`.
- §4.6: `refresh_ai_understanding` background, throttle K=5, đọc full history (không vector).
- §4.3: embed per-Q&A-pair tại write_session_turn, không summary.
- §3.2: `conversations` không có cột title.
