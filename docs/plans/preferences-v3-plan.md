---
title: Preferences gộp vào users + đường ra prod
status: approved
branch: feat/user-preferences
author: K (Senior Solution Architect)
date: 2026-09-03
adr: ADR-008 (approved 03/09/2026)
supersedes:
  - docs/plans/user-preferences-plan.md
  - docs/plans/character-lite-optimization-plan.md
related:
  - docs/architecture/langgraph-flow-persona.md
  - docs/architecture/schema-redesign.md
tags: [plan, preferences, schema, infra]
---

# Plan v3 — Preferences gộp vào `users` + đường ra prod

> Thay thế [[user-preferences-plan]] (v2.1) và
> [[character-lite-optimization-plan]]. ADR-008.

## Context

Hai plan trước đã thực thi xong (`484bb32`, `232c90f`, `863458d`). Review thiết
kế phát hiện hai nhóm vấn đề đáng sửa **ngay bây giờ**, khi chưa có dữ liệu người
dùng thật và tính năng chưa ra prod — đảo ngược lúc này gần như miễn phí.

**Nhóm 1 — thiết kế:**

- Bảng `user_preferences` riêng cho quan hệ 1:1 với `users` là chuẩn hoá thừa. Nó
  kéo theo: policy RLS thứ hai, seed row thủ công (khiến `GET` trở thành lệnh
  ghi), `display_name` trùng lặp, và một JOIN.
- `version` optimistic lock tạo xung đột ở nơi vốn không có: `PATCH` đã `COALESCE`
  từng field nên hai thiết bị sửa hai field khác nhau không đụng nhau. Rồi 409 lại
  được giải quyết bằng LWW — trả giá để cài đặt đúng thứ sẽ nhận được nếu không có
  `version`.
- PHI guard là blocklist 5 từ khoá, chỉ soi tầng ngoài cùng và chỉ soi tên khoá.
  `{"a":{"injury_history":"x"}}` và `{"note":"thoát vị L4"}` đều lọt.
- `863458d` bỏ `SELECT 1 ... AND is_active`, thay bằng bắt lỗi FK. FK không biết
  `is_active` → set được nhân vật đã tắt làm mặc định.
- FE flash **dark→light** ở lần vào đầu tiên trên máy dark-mode: blocking script
  trong `index.html` fallback `prefers-color-scheme`, còn `main.tsx:18` đặt
  `defaultTheme="light"`. Đúng lỗi Q11 muốn diệt, chỉ ngược chiều.
- `useUserPreferences.ts` (đã có debounce 300ms + rollback + visibilitychange)
  không nơi nào import; dedupe làm bằng `_prefsInflight` dùng chung nhưng vẫn nhận
  `signal` của caller — caller nào abort thì giết luôn caller kia.

**Nhóm 2 — không có đường ra production:**

- `/me/preferences` không tồn tại trong `rest_api_stack.py`. Deployed build gọi vào
  nhận 403 và FE nuốt lỗi im lặng → tính năng chỉ chạy trên localhost.
- Tối ưu lite chỉ áp vào `api/routes_characters.py`, vốn tự khai là dev shim. Prod
  đọc `infra/lambda/characters/handler.py` — vẫn trả 8 cột.

**Kết quả mong muốn:** một bảng ít hơn, một cột duy nhất mở rộng được không cần
migration, hàng rào PHI thật sự chặn, và cả hai tính năng chạy được ở prod.

## Quyết định đã chốt (phiên 03-09-2026)

| # | Quyết định | Lý do |
|---|---|---|
| D1 | Bỏ bảng `user_preferences`, gộp thành **một cột** `users.preferences JSONB` | 1:1; `users` đã có RLS + GRANT (`007_rls.py:66`), row đã được tạo sẵn ở 5 code path, `gdpr.py:144` đã `DELETE FROM users` |
| D2 | Toàn bộ preference nằm trong JSONB, **không cột typed, không FK** | Preference là tập mở. Thêm tính năng sync mới = thêm một field vào Pydantic, không migration |
| D3 | Không FK cho `selected_character_slug` | Catalog dùng **soft delete** (`is_active`), nên `ON DELETE SET NULL` không bao giờ chạy; `slug` là định danh ổn định nên không cần `ON UPDATE CASCADE`; và check `is_active` ở tầng app (D7) **mạnh hơn** FK |
| D4 | `avatar_bg`: bỏ CHECK constraint **và** bỏ enum ở Pydantic; backend chỉ chặn độ dài + pattern | Từ 3 nơi đồng bộ tay xuống **1**: `avatarPalette.ts`. Thêm màu không cần deploy backend, nên không có bẫy thứ tự deploy (FE qua Amplify, BE qua CDK là hai đường độc lập — ship màu mới lên FE trước sẽ khiến user nhận 422). An toàn vì FE tra bảng có fallback: giá trị lạ rơi về `slate` |
| D5 | Bỏ enforcement `version`, LWW thành thật | Merge per-field đã có sẵn; 409 là xung đột nhân tạo |
| D6 | An toàn PHI bằng Pydantic `extra="forbid"` | Model chính là schema: khoá lạ → 422 bất kể tên gì, lồng sâu bao nhiêu. Thay hẳn blocklist |
| D7 | Khôi phục `SELECT 1 FROM characters WHERE slug=$1 AND is_active`, bỏ `except` so chuỗi `"foreign key"` | **Sau D3 đây là hàng rào duy nhất.** FK cũ chỉ biết "slug có tồn tại", không nhìn `is_active` — nên từ `863458d` đã set được nhân vật đã tắt làm mặc định, rồi lần load sau `GET /characters/{slug}` trả 404 và màn hình rơi về Anne trong khi ★ Default trỏ chỗ khác. Nhận diện lỗi bằng so chuỗi thì vỡ khi Postgres đổi cách diễn đạt. "Tiết kiệm 1 query" là lookup dưới 1ms trên hệ 100 user ghi hiếm — không đo được |
| D8 | `theme` + `graphics.*` **giữ device-local** | Owner giữ Q3. FOUC sửa trong phạm vi device-local |
| D9 | **Bỏ `display_name` khỏi API preferences** | Cognito là nguồn sự thật: `ProfileContent.tsx:60` đọc `custom:displayName` từ token, `CreateAccountPage.tsx:36` ghi vào Cognito. `users.display_name` và `user_preferences.display_name` không ai đọc |
| D10 | Giữ tên `/me/preferences`, giữ 2 request nối tiếp | URL ánh xạ đúng một cột. Bước kế tiếp là tải VRM 9-17MB nên thêm ~100ms round-trip là nhiễu, đổi lại giữ được `/characters/{slug}` public + cache 5 phút ở CDN |
| D11 | Plan bao gồm cả wiring ra prod | Không có wiring thì mọi sửa đổi không đánh giá được ở môi trường thật |

## API contract sau thay đổi

```
GET /me/preferences
  -> 200 { preferences: { avatar_bg?, selected_character_slug? }, updated_at }
  Không tạo row. User chưa có row -> trả default, vẫn 200.

PATCH /me/preferences
  body { preferences: { avatar_bg?, selected_character_slug? } }
  -> 200 (cùng shape)
   | 400 unknown or inactive character
   | 422 khoá lạ trong preferences
  Không còn `version`, không còn 409, không còn ETag, không còn display_name.
```

`preferences` merge nông: `preferences = preferences || $2::jsonb`.
Gửi `selected_character_slug: null` tường minh để xoá.

Thứ tự load không đổi:
`Auth → GET /me/preferences → GET /characters/{slug} → vrm_url → render`.
`GET /characters` (list lite) chỉ chạy khi user mở AvatarsPanel.

---

## Phần A — Migration (schema)

**Đã xác nhận 03/09: 009 đã chạy trên Neon** → đi nhánh `010_preferences_into_users.py`,
có carry dữ liệu. (Nhánh còn lại — sửa thẳng 009 khi nó chưa chạy ở đâu — không dùng.)

```sql
ALTER TABLE users ADD COLUMN IF NOT EXISTS preferences JSONB NOT NULL DEFAULT '{}'::jsonb;

-- chuyển dữ liệu nếu 009 đã chạy và có row (bỏ display_name theo D9)
UPDATE users u SET preferences =
    jsonb_strip_nulls(jsonb_build_object(
      'avatar_bg',               p.avatar_bg,
      'selected_character_slug', p.selected_character_slug
    )) || p.prefs
FROM user_preferences p WHERE p.user_id = u.id;

DROP POLICY IF EXISTS user_preferences_owner ON user_preferences;
DROP TABLE IF EXISTS user_preferences;
```

Không cần `ENABLE ROW LEVEL SECURITY`, không `GRANT`, không policy mới: `users` đã
nằm trong `OWNED_DIRECTLY` của `007_rls.py:66` với đủ 4 quyền.

`downgrade()`: dựng lại bảng theo 009, copy ngược từ JSONB, rồi `DROP COLUMN`.

## Phần B — Backend

`agenticRAG/langgraph_agents/api/schemas.py`

- **Xoá `AvatarBg = Literal[...]`** (D4). `avatarPalette.ts` trở thành nơi duy nhất
  định nghĩa bảng màu.
- Thêm model làm **schema of record** cho JSONB:

```python
class SyncedPrefs(BaseModel):
    model_config = ConfigDict(extra="forbid")   # ← thay toàn bộ blocklist PHI
    # Không enum: FE tra AVATAR_BG_OPTIONS và fallback 'slate' nếu không khớp,
    # nên backend chỉ cần chặn rác. Thêm màu = sửa mỗi avatarPalette.ts.
    avatar_bg: Optional[str] = Field(
        default=None, max_length=32, pattern=r"^[a-z][a-z0-9_-]{0,31}$"
    )
    selected_character_slug: Optional[str] = Field(
        default=None, max_length=64, pattern=r"^[A-Za-z0-9_-]{1,64}$"
    )
    # Tính năng sync mới thêm field ở đây — không migration, không đụng DB.
```

- `UserPreferencesOut` / `UserPreferencesPatch`: bỏ `version`, bỏ `display_name`,
  bỏ `avatar_bg`/`selected_character_slug` phẳng; chỉ còn `preferences: SyncedPrefs`
  (+ `updated_at` ở Out).

`agenticRAG/langgraph_agents/api/routes_preferences.py` — viết lại phần thân:

- Xoá `_validate_prefs`, `ALLOWED_AVATAR_BG`, `MAX_PREFS_BYTES`, `MAX_PREFS_DEPTH`,
  `FORBIDDEN_PREFS_KEYS`, `_phi_keys`. `extra="forbid"` thay hết.
- `GET`: một `SELECT preferences, updated_at FROM users WHERE id = $1`.
  **Không INSERT.** Không có row → trả `{}` mặc định, vẫn 200. Dùng
  `pg.transaction()` như hiện tại (tự set `app.user_id`, xem `postgres.py:361`).
- `PATCH`: giữ idiom `INSERT INTO users (id) VALUES ($1::uuid) ON CONFLICT DO
  NOTHING` đã dùng ở `routes_crud.py:114` / `session_store.py:156`, rồi **một**
  `UPDATE users SET preferences = preferences || $2::jsonb, updated_at = now()
  WHERE id = $1 RETURNING preferences, updated_at`. Bỏ nhánh 409 và fetchrow phụ.
- Khi patch có `selected_character_slug` khác null, verify trước UPDATE:
  `SELECT 1 FROM characters WHERE slug=$1 AND is_active` → không có thì 400. Bỏ
  `except Exception` so khớp chuỗi `"foreign key"`.
- Giữ `logger.info("prefs_patched", ...)`, bỏ `version` khỏi extra. Bỏ header ETag.

## Phần C — Frontend

`src/lib/preferences.ts`

- Kiểu mới theo contract trên; bỏ `version`, bỏ `display_name`; `avatar_bg` và
  `selected_character_slug` nằm dưới `preferences`.
- Bỏ `_prefsCache`/`_prefsInflight` **và** bỏ tham số `signal`. Sau khi dedupe ở
  tầng provider (dưới) thì chỉ còn một caller — cache dùng chung hết lý do tồn tại,
  và đó cũng là cách xoá bug abort-giết-caller-kia.

`src/hooks/useUserPreferences.ts` — **hồi sinh, không viết lại**. File đã đúng:
debounce 300ms, refetch theo `visibilitychange`, rollback khi lỗi. Chỉ bỏ nhánh
`status === 409` và bỏ `version` khỏi `patch()`.

`src/layouts/MainLayout.tsx:30` — thêm `PreferencesProvider` bọc ngoài
`MotionProvider`, gọi `useUserPreferences()` **đúng một lần**, cung cấp
`{ data, patch }` xuống. Xoá ba call site `fetchPreferences()` hiện có:
`MotionContext.tsx:85`, `AvatarBgContext.tsx:44`, `AvatarsPanel.tsx:242,262,272`.

`src/contexts/ThemeContext.tsx` — sửa FOUC bằng cách lấy trạng thái ban đầu **từ
chính class mà blocking script đã đặt**, thay vì tự đoán lại:

```ts
const [theme, setTheme] = useState<Theme>(() => {
  const root = document.documentElement
  if (root.classList.contains('dark')) return 'dark'
  if (root.classList.contains('light')) return 'light'
  return (localStorage.getItem(storageKey) as Theme) || defaultTheme
})
```

Một nguồn sự thật, không thể lệch với `index.html`. `main.tsx:18` giữ nguyên.
`GraphicsContext.tsx` không đụng (D8).

## Phần D — Đường ra production

`infra/infra/rest_api_stack.py` — thêm ngay dưới khối `/me/memory` (dòng 234-238),
theo đúng idiom `**authed`:

```python
prefs = me.add_resource("preferences")
prefs.add_method("GET", crud, **authed)
prefs.add_method("PATCH", crud, **authed)
```

`crud_app.py:96` đã mount router nên Lambda đã có code — chỉ thiếu resource.

`infra/lambda/characters/handler.py:58` — áp cùng tập cột lite cho `list`, giữ full
cho `get_by_slug`, khớp `routes_characters.py:38`.

Lambda đóng gói bằng `Code.from_asset(infra/lambda/characters)`
(`character_stack.py:173`) nên **không import chéo được** sang `agenticRAG`. Thay vì
shared module, thêm `tests/infra/test_characters_contract.py`: đọc hai file, parse
hai hằng cột, assert bằng nhau. Rẻ, và chặn đúng loại trôi đã xảy ra.

## Phần E — Tài liệu

- Ba commit ngày 01/09 merge mà không có worklog nào — vi phạm convention trong
  CLAUDE.md. Ghi bù `docs/worklogs/01-09-2026.md`, rồi ghi phần này vào
  `docs/worklogs/03-09-2026.md`.
- ADR-008 trong worklog: ghi D1–D11. Ghi rõ **Q4 chưa từng được Owner trả lời**
  (plan v2 §2 tự thú nhận nhưng §3 lại ghi "chốt"), và D8 giữ device-local là quyết
  định có ý thức chứ không phải suy luận.
- Ghi nhận nợ còn lại (không làm trong plan này): `users.display_name` từ 002 giờ
  chắc chắn không ai đọc — dọn ở một migration riêng sau khi xác nhận không code
  path nào dùng.
- Đánh dấu hai plan cũ `status: superseded`, trỏ sang plan này.

---

## Verification

**Schema**
```
alembic upgrade head          # trên Neon branch, không phải main
\d users                      # có cột preferences jsonb
\d user_preferences           # phải báo không tồn tại
```

**Unit / contract**
```
C:/Miniconda/envs/firstconda/python.exe -m pytest tests/langgraph_agents/test_preferences.py -q
C:/Miniconda/envs/firstconda/python.exe -m pytest tests/infra/test_characters_contract.py -q
```
`test_preferences.py` phải sửa: bỏ 4 test `_validate_prefs` blocklist và test 409,
thay bằng —
`preferences:{injury_history:"x"}` → **422**;
`preferences:{a:{injury_history:"x"}}` → **422** (lỗ nested phải đóng);
`preferences:{avatar_bg:"x"*40}` → **422** (chặn rác bằng độ dài, không bằng enum);
`preferences:{avatar_bg:"neon"}` → **200** (D4: màu lạ được nhận, FE tự fallback
`slate` — test này khoá đúng hành vi "thêm màu không cần deploy backend");
hai PATCH khác field liên tiếp → cả hai cùng sống sót (chứng minh LWW per-field).
Giữ nguyên `test_get_requires_auth` và `test_no_user_id_param_on_routes` — hai test
IDOR này vẫn đúng và vẫn cần.

**Hành vi (localhost, DevTools Network)**
1. Xoá `localStorage`, đặt OS sang dark, hard reload → **không flash** theo chiều
   nào. Lặp lại với OS light.
2. Lần load đầu: đúng **1** `GET /me/preferences`, đúng **1** `GET /characters/{slug}`,
   **không** có `GET /characters`.
3. Mở AvatarsPanel → lúc này mới có `GET /characters`, response không chứa `vrm_url`,
   card vẫn vẽ đủ.
4. Bấm "Set as default" trên nhân vật khác → reload → nhân vật đó lên `<Canvas>`.
5. `UPDATE characters SET is_active=false WHERE slug='x'` rồi PATCH slug đó → **400**,
   không phải 500 và không phải 200.

**Prod**
```
cdk diff VvaRestApiStack      # đúng 2 method mới dưới /me/preferences
# sau deploy:
curl -H "Authorization: Bearer $TOK" $API/me/preferences   # 200, không phải 403
curl $API/characters | jq '.characters[0] | keys'          # không có vrm_url
```

**Đa thiết bị (chứng minh D5)** — trình duyệt A đổi `avatar_bg`, trình duyệt B đổi
`selected_character_slug`, reload cả hai → **cả hai thay đổi cùng sống**, không toast
xung đột, không mất dữ liệu.
