# Plan: Lưu per-message character (thay `role='assistant'` cứng) + initial message đúng nhân vật

> Author: K | Ngày: 31-08-2026 | Trạng thái: **PLAN ONLY — KHOAN CODE** | Liên quan: `session_store.py:write_session_turn`, `session_store.py:load_session_messages`, `api/main.py:_stream_chat`, `alembic 002/005`, `ChatContext.tsx`

---

## 1. Yêu cầu Owner & Câu hỏi cần làm rõ

> **Quote Owner:** "đầu tiên thì trên backend, chuyển hết role từ assistant sang anne, cái thứ hai, dựa theo nhân vật mà model nhắn hãy dùng nó là role lưu vào message, từ đó khi load, dựa vào role của message đầu tiên mà trả về initial message đúng với role đó. à mà khoan, nếu làm như vậy thì frontend sẽ phải load cả persona của nhân vật đó, đây là một điểm không tốt chút nào."

**Hiểu đúng:**
- (1) Migration dữ liệu cũ: mọi `role='assistant'` hiện có → `role='anne'` (vì `anne` là default character hiện tại, 4 nhân vật sống: `anne/bronya/miki/hatsune-miku`, `anne` là default trong `MotionContext.tsx:60`).
- (2) Từ nay mỗi turn assistant ghi `role = persona_id` thực tế (`anne|bronya|miki|hatsune-miku`) thay vì `'assistant'` cứng. Khi `GET /sessions/{id}` (restore), frontend nhìn `role` của message đầu tiên để biết session này thuộc nhân vật nào, từ đó trả về initial greeting đúng.

**Vấn đề Owner tự thấy:** Nếu initial greeting (`ui_strings.greeting`) phải suy từ `role` của message đầu, frontend phải load `persona/ui_strings` của nhân vật đó — thêm fetch, thêm coupling.

**Câu hỏi Owner:** "`write_session_turn` nhận `persona_id` là sao và `load_session_messages` ở đây là làm gì như cái bạn muốn nói"

→ Trả lời ở §2-3, và plan này đề xuất cách để **không bắt frontend load thêm persona** cho initial message.

---

## 2. Hiện trạng (đã verify)

### 2.1 Schema

```sql
-- alembic/002_m4_fresh_schema.py:73, infra/sql/init_schema.sql:43
role TEXT NOT NULL CHECK (role IN ('user','assistant'))
```
Chỉ cho `user|assistant`. Mọi query (`session_store.py:186`, `234`, `313`) filter `role='user'` hoặc `role='assistant'`.

### 2.2 `write_session_turn(user_id, session_id, user_query, assistant_answer, total_tokens, motion_job_id)`

- **Nguồn:** `session_store.py:333-383`, được gọi ở `api/main.py:587-595` sau khi `graph.astream` cho ra `final_answer`.
- **Hiện:** Không nhận `persona_id`. Insert 2 dòng:
  ```py
  (session_id, "user", user_query)
  (session_id, "assistant", assistant_answer, extras={"motion":{...}})
  ```
- **Thiếu:** `persona_id` có sẵn ở `config["persona_id"]` trong `_stream_chat` (`api/main.py:282`) nhưng không truyền xuống `write_session_turn`. Nên dù `POST /chat {persona_id:"miki"}` thì DB vẫn ghi `assistant`.

### 2.3 `load_session_messages(user_id, session_id, limit, before)`

- **Nguồn:** `session_store.py:261-310`.
- **Làm gì:** `SELECT role, content, extras, created_at FROM messages WHERE session_id=... ORDER BY created_at DESC, seq_id DESC LIMIT ...` → `_shape_message` → trả `{messages:[{role,content,timestamp,motion_job_id}]}`.
- **Dùng ở 2 nơi:**
  1. **STM warm-up** (`api/main.py:264-269` limit 6): khi `/chat` đến mà Redis STM trống (session mới hoặc Lambda khác), backfill 3 cặp Q&A cuối vào Redis.
  2. **Restore history** (`routes_crud.py` → `GET /sessions/{id}`): frontend `ChatContext.tsx:179` gọi `getSession` → `load_session_messages` → `ChatContext` prepend `buildInitialMessages(ui)` (greeting frontend-only) + map DB rows thành `Message[]`. `populate_stm_from_messages:313` cũng dùng để rebuild STM từ DB rows.

### 2.4 Initial message hiện tại

- Không lưu DB. Frontend luôn `buildInitialMessages(ui)` với `ui = uiStringsFor(selectedVrmId)` hiện tại (`ChatContext.tsx:54-63` + `ChatContext.tsx:145-168` sau fix gần nhất: chỉ adopt 1 lần `fallback->anne`, sau đó giữ nguyên). Nên reload sẽ thấy greeting của **character đang chọn**, không phải character lúc tạo session.

---

## 3. Vấn đề nếu làm theo yêu cầu nguyên văn (role = slug)

### 3.1 Nếu chỉ đổi `role` từ `'assistant'` → `slug`:

- **Migration:** `UPDATE messages SET role='anne' WHERE role='assistant'` — OK cho dữ liệu cũ (vì trước đây chỉ có `anne` là default, nhưng nếu có history với `miki` sẽ bị gán sai).
- **Write:** `write_session_turn(..., persona_id="miki")` → insert `role='miki'`.
- **Read cho initial:** Frontend `GET /sessions/{id}` → `messages[0].role == 'miki'` → suy ra initial greeting của `miki` là `ui_strings.greeting` của `miki`.

**Hậu quả Owner lo là đúng:**
- Frontend phải `fetch /characters/miki` hoặc `fetch /characters/miki/avatar-profile` + `ui_strings` để có greeting, dù chỉ để render 1 câu chào. Nếu list `GET /characters` đã trả `ui_strings` thì đỡ, nhưng hiện `routes_characters.py` **không trả `ui_strings` ở list** (chỉ ở detail, và frontend `characters.ts:45` mark `ui_strings?` optional). Nghĩa là sau reload, `ChatContext` chỉ có `selectedVrmId='anne'` (default), không có `miki`'s greeting cho tới khi fetch detail — thêm 1 round-trip, chậm, và nếu session có nhiều character khác nhau (user đổi giữa chừng) thì phải load nhiều persona.

### 3.2 Các phương án thay thế

| # | Phương án lưu character | Ưu | Nhược | Ghi chú |
|---|------------------------|----|-------|---------|
| **A. Đổi `role` thành slug** (yêu cầu nguyên văn) | Đơn giản, không thêm cột | Phá `CHECK` cũ, code hiện tại `if role=='assistant'` vỡ khắp nơi (`session_store:318`, `ChatContext:132`, `populate_stm:320`), mất phân biệt `user vs assistant` (phải `role!='user'` để biết là assistant) | Không khuyến nghị |
| **B. Giữ `role` = `user|assistant`, thêm cột `character_slug`/`persona_id` TEXT | Giữ semantics `user/assistant`, query cũ không vỡ, filter `WHERE role='assistant'` vẫn chạy | Thêm cột, cần migration | **Khuyến nghị** |
| **C. Giữ `role`, nhét `character` vào `extras` JSONB (`extras->'character'`)** như `motion` | Không миграtion cột, tận dụng `extras` (đã có `motion`, `008_messages_extras.py`) | Mỗi read phải `jsonb_extract`, không index được, query `WHERE extras->>'character'='miki'` chậm | Chấp nhận được nếu không cần index |
| **D. Lưu initial greeting luôn như 1 message DB** (`role='assistant', character='miki', content=greeting`) | Khi load, không cần suy greeting, có sẵn | Mỗi session mới phải insert thêm 1 dòng greeting, và khi đổi character giữa session sẽ có nhiều greeting xen kẽ — trùng với vấn đề Owner nói "không có quy luật cứng lưu đổi character" | Không cần cho v1 |

**Khuyến nghị:** **B** (thêm cột) hoặc **C** (extras) — giữ `role` cho logic `user vs assistant`, lưu character riêng. **B** rõ ràng hơn cho query và type.

---

## 4. Plan đề xuất (giữ greeting bất biến, không bắt frontend load thêm persona)

### 4.1 Nguyên tắc

1. **Greeting vẫn frontend-only, không lưu DB** (như hiện tại). Không dùng `role` của message đầu để suy greeting — vì greeting không phải là message đầu trong DB.
2. **Session's character** là metadata của **session**, không phải của từng message. Khi tạo session, nhân vật đang chọn là `initial_character`. Các turn sau có thể đổi nhân vật, nhưng **initial greeting chỉ cần `initial_character`** — frontend đã có sẵn sau khi fetch `/characters` list (chứa `display_name` + `ui_strings` nếu ta cho list trả `ui_strings`).
3. **Để frontend không phải fetch thêm persona cho initial**, ta có 2 cách:
   - (i) **Backend trả initial greeting sẵn** trong `load_session_messages` / `GET /sessions/{id}`: kèm `initial_character` + `initial_greeting` (lấy từ `characters.ui_strings.greeting` ở DB).
   - (ii) Hoặc trả `initial_character` thôi, frontend lookup trong `vrmOptions` đã fetch (`GET /characters`) — đã có `ui_strings` nếu ta mở list trả `ui_strings`. (ii) nhẹ hơn, không cần backend join.

Chọn **(ii) + fallback (i)**: `conversations` thêm cột `initial_character_slug TEXT`, `GET /sessions/{id}` trả thêm `initial_character` và `initial_greeting` (đã resolve theo time slot) để frontend không cần fetch detail.

### 4.2 Schema migration

**Alembic mới `009_character_per_message.py`:**

```sql
-- B: thêm cột character cho per-message
ALTER TABLE messages ADD COLUMN character_slug TEXT; -- nullable, không CHECK để linh hoạt thêm nhân vật
CREATE INDEX idx_messages_character ON messages(character_slug) WHERE character_slug IS NOT NULL;

-- Cho initial_character của session (để load initial greeting không cần scan messages)
ALTER TABLE conversations ADD COLUMN initial_character_slug TEXT;
-- Không cần CHECK cứng, app layer validate persona_id ∈ characters.slug

-- Cập nhật CHECK cũ không cần đổi vì role vẫn user|assistant
-- Nếu chọn A (đổi role), thì phải: ALTER TABLE messages DROP CONSTRAINT ...; ADD CHECK (role IN ('user','anne','bronya',...))
-- Không làm.

-- Backfill dữ liệu cũ: gán 'anne' cho mọi assistant message cũ (vì trước đây chỉ có anne là default)
UPDATE messages SET character_slug='anne' WHERE role='assistant' AND character_slug IS NULL;
-- Backfill conversations: lấy character của message assistant đầu tiên, hoặc 'anne' nếu không có
UPDATE conversations c SET initial_character_slug = COALESCE(
  (SELECT character_slug FROM messages WHERE session_id=c.session_id AND role='assistant' ORDER BY created_at LIMIT 1),
  'anne'
) WHERE initial_character_slug IS NULL;
```

Nếu chọn **C** (extras) thì không cần cột `character_slug`, chỉ `UPDATE messages SET extras = jsonb_set(...)` — nhưng khuyến nghị B vì sạch.

### 4.3 `write_session_turn` nhận `persona_id`

**Signature mới:**

```py
async def write_session_turn(
  user_id: str,
  session_id: str,
  user_query: str,
  assistant_answer: str,
  total_tokens: int = 0,
  grader_result: str = "pass",
  motion_job_id: str | None = None,
  persona_id: str | None = None,  # NEW: slug như 'anne'|'miki'...
) -> None
```

**Impl:**

```py
# 1. Ensure conversations.initial_character_slug được set lần đầu
await pg.execute(
  "UPDATE conversations SET initial_character_slug = COALESCE(initial_character_slug, $2) WHERE session_id=$1::uuid",
  session_id, persona_id or 'anne'
)
# 2. Insert messages với character_slug
await pg.executemany(
  "INSERT INTO messages (session_id, role, content, token_count, extras, character_slug) VALUES ($1::uuid,$2,$3,$4,$5::jsonb,$6)",
  [
    (session_id, "user", user_query, None, None, None),  # user không cần character
    (session_id, "assistant", assistant_answer, total_tokens, json.dumps({"motion":{...}}) if motion_job_id else None, persona_id or 'anne'),
  ]
)
```

**Caller `api/main.py:_stream_chat:587`:**

```py
await write_session_turn(
  user_id=resolved_user_id,
  session_id=req.session_id,
  user_query=req.query,
  assistant_answer=final_answer,
  total_tokens=...,
  motion_job_id=...,
  persona_id=req.persona_id or "anne",  # NEW
)
```

### 4.4 `load_session_messages` trả character, không bắt frontend load persona

**Hiện tại** `_shape_message` chỉ trả `role/content/timestamp/motion`. **Đổi:**

```py
def _shape_message(row, created_at):
  out = {"role": row["role"], "content": row["content"], "timestamp": ..., "character": row["character_slug"]}
  ...
```

**`load_session_messages` SELECT thêm `character_slug`:**

```sql
SELECT role, content, token_count, extras, character_slug, created_at FROM messages ...
```

**Trả thêm `initial_character` cho frontend để render greeting đúng:**

```py
async def load_session_messages(...):
  header = await pg.fetchrow("SELECT updated_at, initial_character_slug FROM conversations WHERE ...")
  rows = await pg.fetch("SELECT ...")
  return {
    "session_id": ...,
    "initial_character": header["initial_character_slug"] or "anne",  # NEW
    "messages": [_shape_message(r) for r in rows],
    ...
  }
```

**Frontend `ChatContext.tsx:179-185`:**

```ts
const data = await getSession(sessionId)
const history = data.messages as SessionMessage[] // SessionMessage thêm field character?: string
const initialCharacter = data.initial_character ?? 'anne'
const initialUi = uiStringsFor(vrmOptions.find(o=>o.id===initialCharacter)?.character)
setMessages([
  ...buildInitialMessages(initialUi), // greeting đúng nhân vật lúc tạo session, không phải selectedVrmId hiện tại
  ...history.map(m => ({ role: m.role, content: m.content, character: m.character, ... })),
])
```

→ **Không cần fetch thêm persona** vì `vrmOptions` (đã fetch `GET /characters`) đã chứa `ui_strings` cho mọi character (nếu ta cho `GET /characters` trả `ui_strings`). Nếu list không trả, chỉ cần 1 fetch `GET /characters/{initialCharacter}` — vẫn rẻ hơn việc suy từ `messages[0].role`.

**Tuỳ chọn trả sẵn greeting:** Nếu muốn zero fetch, backend có thể join `characters.ui_strings` và trả luôn `initial_greeting: getGreeting(characters.ui_strings)` — frontend dùng thẳng, không lookup. Plan khuyến nghị **trả `initial_character` + để frontend lookup** để giữ backend không phải resolve `GreetingSlots` theo time slot (logic `getGreeting` đang ở frontend).

### 4.5 `populate_stm_from_messages` & `save_session`

- `populate_stm_from_messages` không cần character, giữ nguyên.
- `save_session` (legacy `SessionStore.save_session` cho migrate) cũng nên nhận `character_slug` nếu dùng.

### 4.6 Frontend `Message` type

Mở rộng `ChatMessage.tsx:11`:

```ts
role: 'user' | 'assistant' | 'system'
character?: string // slug của assistant turn, ví dụ 'miki' — để render avatar/name nếu cần
```

Nhưng **không dùng `role` để lưu slug** — giữ `role='assistant'` và thêm `character`.

Nếu Owner vẫn muốn `role='miki'` như yêu cầu nguyên văn, thì phải đổi toàn bộ `role CHECK` và `if role=='assistant'` → `if role!='user'` — rủi ro cao, không khuyến nghị.

---

## 5. Tại sao plan này giải quyết lo ngại "frontend phải load persona"

- **Lo ngại gốc:** Nếu `role` của message đầu là `'miki'` và initial greeting phải suy từ `role`, frontend phải biết `miki`'s `greeting` → phải fetch `miki`'s persona.
- **Giải pháp:** Không suy greeting từ `messages[0].role`. Greeting là **session metadata** (`conversations.initial_character_slug`), được trả kèm `GET /sessions/{id}` ngay trong `load_session_messages` header — 1 query, không cần scan messages. Frontend chỉ cần `vrmOptions` đã có (1 fetch `/characters` lúc app mount) để resolve `initial_greeting`. Không thêm fetch per-message.
- **Per-message character** (`messages.character_slug`) chỉ dùng để render avatar/name cho từng bubble (nếu cần), không dùng cho initial greeting.

---

## 6. Migration dữ liệu cũ (1) — "chuyển hết role từ assistant sang anne"

- **Nếu theo B (thêm cột):** Không đổi `role`, chỉ backfill `character_slug='anne'` cho mọi `role='assistant'` cũ — an toàn, không chạm `CHECK`.
- **Nếu theo A (đổi role):** `UPDATE messages SET role='anne' WHERE role='assistant'` — phải nới `CHECK` trước, và mọi code `role=='assistant'` vỡ. **Không làm.**

Khuyến nghị **B**: `role` giữ nguyên, `character_slug` mới chứa `'anne'` cho dữ liệu cũ.

---

## 7. Các bước thực hiện (tách issue)

| # | Issue | File | Mô tả |
|---|-------|------|-------|
| 1 | Migration 009 | `alembic/versions/009_character_per_message.py` | Thêm `messages.character_slug`, `conversations.initial_character_slug`, backfill `anne`, index |
| 2 | Update `write_session_turn` | `db/session_store.py:333` | Thêm `persona_id` param, insert `character_slug`, update `initial_character_slug` |
| 3 | Update `api/main.py:_stream_chat` | `api/main.py:587` | Truyền `persona_id=req.persona_id or "anne"` vào `write_session_turn` |
| 4 | Update `load_session_messages` & `_shape_message` | `db/session_store.py:261` | SELECT thêm `character_slug`, trả `initial_character`, `_shape_message` thêm `character` |
| 5 | Update `routes_crud.py` `GET /sessions/{id}` | `api/routes_crud.py` | Trả thêm `initial_character` (và optionally `initial_greeting`) |
| 6 | Frontend `SessionMessage` & `ChatContext` restore | `lib/api.ts:291`, `contexts/ChatContext.tsx:179` | Nhận `character` per message, `initial_character` để `buildInitialMessages(initialUi)` |
| 7 | (Optional) `GET /characters` trả `ui_strings` ở list | `api/routes_characters.py:92` | Để frontend không cần fetch detail cho initial greeting |

Không cần đụng `ChatDivider` hay `MotionContext`.

---

## 8. Câu hỏi cho Owner trước khi code

1. Chọn **B (thêm cột `character_slug`)** hay **C (nhét vào `extras`)**? Đề xuất B.
2. `initial_character` có cần lưu không, hay chỉ cần per-message `character_slug` và suy initial từ message đầu tiên? Đề xuất lưu `initial_character` riêng để không phải scan `messages` cho greeting.
3. Dữ liệu cũ: backfill `character_slug='anne'` cho mọi `assistant` cũ có OK không, hay cần phân biệt theo lịch sử thực (không có lịch sử, chỉ có thể gán `anne`)?
4. Frontend có cần render per-bubble avatar/name theo `character` không, hay chỉ cần initial greeting đúng?

> Trả lời xong mới code — **KHOAN CODE** như yêu cầu.

