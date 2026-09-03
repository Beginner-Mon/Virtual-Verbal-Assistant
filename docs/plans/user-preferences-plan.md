---
title: User Preferences — Synced vs Device-specific
status: superseded
superseded_by: docs/plans/preferences-v3-plan.md
branch: feature/langgraph-rewrite
author: K (Senior Solution Architect)
date: 2026-09-01
adr: ADR-007 (pending Owner approval)
related:
  - docs/architecture/langgraph-flow-persona.md
  - docs/architecture/schema-redesign.md
  - docs/plans/reupdate-plan.md (D14)
  - agenticRAG/langgraph_agents/alembic/versions/002_m4_fresh_schema.py
---

# User Preferences Plan — v2 (cập nhật sau trả lời Q1–Q12)

> **Owner trả lời 01/09/2026 — đã lock 11/12. Q12 chốt user explicit. Q10 chốt: preferences KHÔNG chứa tiền sử — chỉ UI. Đã sẵn sàng duyệt trước ACC phase.**

---

## 1. Hiện trạng (đã khảo sát)

Không có bảng `preferences`. 3 context đều `localStorage`-only, mất khi đổi device:

| Context | File | Key | Giá trị |
|---|---|---|---|
| Theme | `ECA_UI/frontend/src/contexts/ThemeContext.tsx:7` (`main.tsx:18` override `eca-theme`) | `eca-theme` | `light\|dark` |
| Graphics | `GraphicsContext.tsx:19` | `vva-graphics-settings` | `{ssao,particles,vignette,mtoon,showGrid,showAxes}` |
| AvatarBg | `AvatarBgContext.tsx:12` | `vva_avatar_bg` | 8 màu `slate\|violet\|...` |
| Character | `MotionContext.tsx:39` `selectedVrmId` | — | ephemeral, default `Anne` (`MotionContext.tsx:60` regex `/anne/i`) |

BE: `users(id, auth_provider, auth_subject)` + RLS `app.user_id` (`007_rls.py:64`), không có `users.profile` (đã bỏ ở M.4). `characters(slug, vrm_url, ui_strings)` là catalog global.

---

## 2. Giải thích lại 4 câu Owner chưa hiểu

### Q4. "Device backup" nghĩa là gì?

> *"Nếu theme chỉ nằm cố định trên desktop, có cần lưu lên server để khi user mua laptop mới thì laptop mới tự có theme đó không?"*

* **Nếu KHÔNG backup (chọn):** Mỗi browser/phone là 1 `localStorage` riêng, đổi máy = setup lại từ đầu. Đơn giản, không tốn BE.
* **Nếu CÓ backup:** BE lưu `device_preferences(user_id, device_id)` — laptop mới fetch về được. Tốn thêm bảng + `device_id` UUID.

**Owner chưa trả lời Q4, nhưng Q3 đã nói “theme/graphic chỉ cố định desktop, không lây sang phone yếu hơn” → suy ra KHÔNG cần backup.** Plan chốt **không backup**, giữ B ở `localStorage` only. Nếu sau này Owner đổi ý, chỉ thêm migration `010_device_preferences` (đã thiết kế sẵn, §5.2).

### Q7. "Tần suất ghi" nghĩa là gì?

> *"Graphics setting có bị user kéo liên tục 60 lần/giây không, hay chỉ chỉnh 1 lần lúc setup?"*

* Nếu **liên tục** (vd: slider volume): Mỗi lần kéo = 1 `PATCH /me/preferences` → QPS cao → cần debounce 500ms + DynamoDB hoặc throttle.
* Nếu **hiếm** (chỉnh 1 lần rồi thôi): QPS ~0 → Neon thoải mái.

**Owner: “tùy setup và không thay đổi liên tục” → hiếm.** Vậy debounce nhẹ 300ms là đủ, Neon không lo.

### Q10. "PII/PHI" nghĩa là gì? — ĐÃ CHỐT: preferences KHÔNG chứa tiền sử

> *"Preference có chứa thông tin sức khỏe nhạy cảm không?"*

**Chốt 01/09: KHÔNG.** `user_preferences` chỉ chứa UI thuần túy, tiền sử ở `user_memory` như cũ.

| Loại data | Ví dụ | Lưu ở đâu | Tính tin cậy |
|---|---|---|---|
| **UI preference (chốt)** | `avatar_bg=violet`, `selected_character=bronya`, `display_name`, `locale` | `user_preferences` — tin 100% vì user bấm nút Set as default | Explicit |
| **Tiền sử sức khỏe (không lưu ở preferences)** | `“tôi bị thoát vị L4”`, `“mẹ tôi 65t đau gối”` | `user_memory(fact_text, valid, category)` — advisory | Không tin 100% — user có thể hỏi cho người khác (như Owner nói), hoặc nói đùa. Cần `valid` flag + confirm trước khi dùng làm filter cứng |

*Vì vậy `user_preferences.prefs` CHỈ chứa `notifications/locale` (§5.1), không chứa `injury_history/fitness_level/age`. Nếu sau này Owner muốn “mức tập mong muốn: nhẹ/vừa/nặng” làm preference thì phải làm flow confirm riêng: AI gợi ý → Card “Lưu ‘tập nhẹ’ làm mặc định? [Lưu]” → mới ghi.*

### Q12. "Ai được ghi?" nghĩa là gì?

> *"Chỉ user bấm nút mới đổi preference, hay AI cũng được tự đổi dùm?"*

* **Chỉ user (CHỐT):** `PATCH /me/preferences` chỉ gọi từ UI khi bấm “Set as default”. AI muốn đổi phải hỏi “Bạn có muốn đặt Bronya làm mặc định không? [Đồng ý]” → user bấm mới ghi.
* **AI tự ghi:** Không làm (đã loại).

---

## 3. Quyết định đã chốt sau Q1–Q12

| Quyết định | Chốt | Lý do từ trả lời |
|---|---|---|
| **Q1 Default character** | **Synced (A)** — thêm nút “Set as default” trong `AvatarsPanel.tsx`, lưu `selected_character_slug` vào `user_preferences`, mọi device load character đó thay Anne | Owner yêu cầu rõ |
| **Q2 Offline** | **Không cần offline queue** — chatbot cần mạng mới chat được, localStorage chỉ hiển thị nhân vật vô nghĩa | Owner: “offline không làm được gì, chưa chắc đủ dung lượng” |
| **Q3 Device = desktop/laptop/phone** | **B = per-browser localStorage** — tự nhiên tách: desktop `localStorage` ≠ phone `localStorage`, phone yếu không bị lây `ssao:true` từ desktop | Owner: “phone cấu hình yếu hơn” |
| **Q4 Backup** | **Không backup B lên server (Option A Minimal)** | Suy từ Q3 |
| **Q5 Avatar** | **Enum 8 màu hiện tại, chưa upload file** — cột `avatar_bg TEXT CHECK` | Owner: “hiện tại màu nền, sau này có thể đổi” |
| **Q6 Query by pref** | **Không cần** → JSONB đủ, không index | Owner: “phần lớn không cần” |
| **Q7 Tần suất** | **Hiếm, debounce 300ms** | Owner: “không thay đổi liên tục” |
| **Q8 Delete** | **CASCADE delete hết** — `ON DELETE CASCADE` cả `user_preferences` | Owner: “xóa hết” |
| **Q9 Scale 100 users × 3 devices = 300 rows** | **Neon đủ, không DynamoDB** | Owner: “100 người” |
| **Q10 PHI** | **CHỐT: preferences KHÔNG chứa tiền sử — chỉ UI. Tiền sử ở `user_memory` advisory** | Owner: “preference không có tiền sử” |
| **Q11 FOUC white→dark** | **Phải fix FOUC, không để flash** → thêm blocking script + optional cookie `eca-theme` | Owner: “không chuyên nghiệp” |
| **Q12 Ai ghi** | **CHỐT: Chỉ user explicit (bấm nút)** | Owner confirm 01/09 |

**Stack chốt:** **Neon + Hybrid (cột typed cho stable + JSONB cho extensible) + Backend cho A (UI-only, không PHI) + localStorage cho B + blocking script chống FOUC.**

---

## 4. Phân loại cuối

| Nhóm | Fields | Scope | Lưu ở | Ghi |
|---|---|---|---|---|
| **A. Synced (UI-only, KHÔNG PHI)** | `avatar_bg`, `selected_character_slug`, `display_name`, `prefs.notifications/locale` | `user_id` 1 row | **Neon `user_preferences`** | `PATCH /me/preferences` từ UI explicit |
| **B. Device** | `theme`, `graphics.*`, `cameraMode`, `isMusicPlaying` | per-browser `localStorage` | **`localStorage` only** (`eca-theme`, `vva-graphics-settings`) | local, không gọi BE |
| **C. Tiền sử sức khỏe (tách biệt)** | `fact_text` “thoát vị L4”, `category` | `user_id` N rows | **`user_memory` (cũ, không đụng)** | AI ghi advisory, `valid` flag, không tin 100% |
| **D. Ephemeral** | `webSearch`, `voiceReply`, `isTyping` | memory | — | — |

---

## 5. Schema

### 5.1 `user_preferences` — synced (bắt buộc)

```sql
-- alembic 009_user_preferences.py — UI-only, KHÔNG chứa PHI (tiền sử ở user_memory)
CREATE TABLE user_preferences (
  user_id UUID PRIMARY KEY REFERENCES users(id) ON DELETE CASCADE,
  avatar_bg TEXT NOT NULL DEFAULT 'slate'
    CHECK (avatar_bg IN ('slate','violet','blue','emerald','amber','rose','cyan','indigo')),
  selected_character_slug TEXT REFERENCES characters(slug) ON DELETE SET NULL,
  display_name TEXT,
  prefs JSONB NOT NULL DEFAULT '{}'::jsonb,
  -- prefs CHỈ chứa UI: {"notifications":{"email":true},"locale":"vi"}
  -- KHÔNG chứa injury_history/fitness_level/age — những thứ đó ở user_memory
  version INT NOT NULL DEFAULT 1,
  updated_at TIMESTAMPTZ NOT NULL DEFAULT now(),
  created_at TIMESTAMPTZ NOT NULL DEFAULT now()
);

ALTER TABLE user_preferences ENABLE ROW LEVEL SECURITY;
CREATE POLICY user_prefs_own ON user_preferences
  USING (user_id = current_setting('app.user_id')::uuid)
  WITH CHECK (user_id = current_setting('app.user_id')::uuid);
GRANT SELECT, INSERT, UPDATE, DELETE ON user_preferences TO eca_user;

CREATE INDEX idx_user_prefs_updated ON user_preferences(updated_at);
```

*Seed:* Trong `api/routes_crud.py:98` sau `INSERT INTO users ... ON CONFLICT DO NOTHING`, thêm:
```sql
INSERT INTO user_preferences(user_id) VALUES ($1) ON CONFLICT DO NOTHING
```
trong cùng transaction.

*GDPR:* `api/gdpr.py:delete_user()` cascade tự xóa (FK).

### 5.2 `device_preferences` — per-device (dự phòng, chỉ tạo khi Owner đổi Q4)

```sql
-- 010_device_preferences.py (chưa chạy ở phase đầu)
CREATE TABLE device_preferences (
  user_id UUID NOT NULL REFERENCES users(id) ON DELETE CASCADE,
  device_id UUID NOT NULL,
  device_label TEXT,
  theme TEXT NOT NULL DEFAULT 'light' CHECK (theme IN ('light','dark')),
  graphics JSONB NOT NULL DEFAULT '{"ssao":false,"particles":true,"vignette":true,"mtoon":false,"showGrid":false,"showAxes":false}'::jsonb,
  prefs JSONB NOT NULL DEFAULT '{}'::jsonb,
  updated_at TIMESTAMPTZ NOT NULL DEFAULT now(),
  created_at TIMESTAMPTZ NOT NULL DEFAULT now(),
  PRIMARY KEY (user_id, device_id)
);
```

---

## 6. API

### 6.1 Endpoints (mount vào `api/main.py` + `crud_app.py`)

```
GET   /me/preferences
  -> 200 {avatar_bg, selected_character_slug, display_name, prefs, version, updated_at}
      ETag: W/"3"  Last-Modified: ...

PATCH /me/preferences
  Headers: If-Match: W/"3"  (optional optimistic lock)
  Body: {
    avatar_bg?: "violet",
    selected_character_slug?: "bronya" | null,
    display_name?: string,
    prefs?: {notifications?: {email?: boolean}, locale?: string},  -- merge patch
    version: 3
  }
  -> 200 { ... version:4 } | 409 {error:"version_conflict", current:{...}}
```

*Auth:* `Depends(current_user_id)` (`api/auth.py:207`) → `bind_request_user()` → RLS.
*Validation:* Pydantic `UserPreferencesPatch` với `Literal` enum + FK check `characters.slug`.
*Merge:* `SET avatar_bg=COALESCE($2, avatar_bg), selected_character_slug=COALESCE($3, ...), prefs = prefs || $4::jsonb, version=version+1, updated_at=now() WHERE user_id=$1 AND version=$5`.

### 6.2 Không có endpoint cho B (device) ở phase đầu — B chỉ local.

---

## 7. Frontend

### 7.1 Synced — `useUserPreferences()` (mới)

**File mới:** `ECA_UI/frontend/src/hooks/useUserPreferences.ts` + `src/lib/preferences.ts`

* Fetch: `GET /me/preferences` ngay sau `AuthGuard.tsx:38` `fetchAuthSession()` success, SWR 5m, refetch on `visibilitychange`.
* Optimistic: `setAvatarBg('violet')` → set local state ngay → `PATCH` debounce 300ms → rollback + toast nếu 409.
* Hydrate:
  * `AvatarBgContext.tsx:15` đổi `localStorage.getItem('vva_avatar_bg')` → `prefs.avatar_bg` (fallback `localStorage` cho guest `vva_demo_user` `api.ts:127`).
  * `MotionContext.tsx:39,60` `selectedVrmId` hydrate từ `selected_character_slug`, fallback `Anne` nếu null.
  * `ProfileContent.tsx:63` hiển thị `display_name` từ prefs.

**Nút “Set as default” (Q1):**

`components/panels/AvatarsPanel.tsx:56` `AvatarCard` thêm:

```tsx
{isSelected && !isDefault && <button onClick={() => patchDefault(slug)}>Set as default</button>}
{isDefault && <span>★ Default</span>}
```

`isDefault = slug === preferences.selected_character_slug`. Click → `PATCH {selected_character_slug: slug, version}`.

### 7.2 Device — giữ `localStorage` + fix FOUC (Q11)

**FOUC fix (Owner yêu cầu không flash white→dark):**

Thêm **blocking script** trong `ECA_UI/frontend/index.html` `<head>` trước `main.tsx`:

```html
<script>
  (function(){try{
    var t=localStorage.getItem('eca-theme');
    if(!t) t=matchMedia('(prefers-color-scheme:dark)').matches?'dark':'light';
    document.documentElement.classList.add(t);
  }catch(e){}})();
</script>
```

* Tại sao? `ThemeContext.tsx:15` hiện chạy sau React mount → flash. Script này chạy trước paint.
* **Cookie `eca-theme` (optional, chỉ nếu cần SSR/CloudFront):** Trong `ThemeContext.tsx:25` sau `localStorage.setItem`, thêm `document.cookie="eca-theme="+theme+"; Path=/; SameSite=Lax; Max-Age=31536000"`. Cookie này **không httpOnly, không chứa PII**, chỉ để CDN/server đọc nếu sau này có SSR. Không bắt buộc ở phase đầu — blocking script đã đủ cho SPA Vite.

`GraphicsContext.tsx:19` giữ `vva-graphics-settings` local, không đổi.

### 7.3 Guest (`vva_demo_user`)

Chưa login → `GET /me/preferences` 401 → dùng `localStorage` fallback như cũ, không gọi PATCH.

---

## 8. Phases

| Phase | Việc | Owner | Thời gian |
|---|---|---|---|
| **0. Lock** | ✅ Done 01/09 — Q1-Q12 chốt, `user_preferences` UI-only (không PHI), chỉ user explicit | Owner | — |
| **1. BE** | Alembic 009 (`user_preferences` UI-only), `routes_preferences.py`, seed trong `POST /me/memory`, RLS + `current_setting('app.user_id')`, `pytest -m unit` + integration Neon branch | N | 0.5 ngày |
| **2. FE synced** | `useUserPreferences`, migrate `AvatarBgContext`+`MotionContext` sang BE, nút **Set as default** trong `AvatarsPanel.tsx:56`, blocking script FOUC trong `index.html` | N | 1 ngày |
| **3. FOUC verify** | Test `eca-theme` không flash white→dark trên Chrome/Firefox, Lighthouse CLS=0 | N | 0.5 buổi |
| **4. Hardening** | `DELETE /me` cascade (`user_preferences` ON DELETE CASCADE), `GET /health/detailed`, log `prefs_patch {user_id, keys, version}`, ETag/409 handling | N | 0.5 ngày |
| **(dự phòng) 5. Device BE** | Chỉ khi Owner đổi Q4 → 010 `device_preferences` + `lib/deviceId.ts` + `GraphicsContext` sync | N | 0.5 ngày |

---

## 9. Rủi ro & Mitigation

| Rủi ro | Mitigation |
|---|---|
| 409 version conflict khi 2 device cùng đổi avatar | LWW + toast “Đã cập nhật ở nơi khác, đã tải lại”, không mất data |
| `prefs` JSONB phình | Giới hạn 8KB, validate depth ≤2, reject `__proto__` |
| Neon pooled `SET app.user_id` leak | Dùng `VVA_PG_DSN` direct như `postgres.py:274` đã fix, không pooler |
| FOUC vẫn flash nếu user lần đầu chưa có `localStorage` | Blocking script fallback `prefers-color-scheme` |
| Guest vs authed lẫn lộn | `AuthGuard.tsx:26` purge `localStorage` khi đổi user, không sync guest prefs lên server |

---

## 10. Không làm (out of scope theo Q2/Q9)

* Offline queue / IndexedDB / service worker — không cần (Q2).
* DynamoDB bảng preferences — không cần với 100 users (Q9).
* S3 avatar upload — chưa (Q5 enum).
* AI tự ghi preference — chưa (Q12 chỉ user).

---

## 11. Đã chốt toàn bộ — sẵn sàng ACC

* **Q10:** `user_preferences` **KHÔNG chứa tiền sử** — chỉ `avatar_bg`, `selected_character_slug`, `display_name`, `prefs{notifications,locale}`. Tiền sử ở `user_memory` advisory như cũ.
* **Q12:** **Chỉ user explicit** — AI không tự ghi, phải qua nút “Set as default”.

Không còn câu hỏi mở. ADR-007 coi như **approved** theo trả lời 01/09. N có thể bắt đầu **Phase 1 (BE)** ngay.

---

*K — Senior Solution Architect. Plan v2.1 lock 01/09/2026 — `user_preferences` UI-only. N log work vào `docs/worklogs/DD-MM-YYYY.md` và tạo Alembic 009 trên `feature/langgraph-rewrite`.*
