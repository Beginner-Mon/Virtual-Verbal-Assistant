---
title: Characters Lite + Preferences Dedup — Optimization
status: superseded
superseded_by: docs/plans/preferences-v3-plan.md
branch: feat/user-preferences
author: K (Senior Solution Architect)
date: 2026-09-02
depends_on: docs/plans/user-preferences-plan.md v2.1 (ADR-007)
---

# Vấn đề

**1. `GET /characters` trả quá nhiều, FE gọi sai lúc:**
- Cha `GET /characters` hiện trả full `_PUBLIC_COLUMNS` = `slug, display_name, description, vrm_url, thumbnail_url, vrm_metadata, voice_language, sort_order, ui_strings` (`routes_characters.py:39`) cho cả 4 con. Card `AvatarsPanel` chỉ cần `slug, display_name, thumbnail_url, description`.
- `vrm_url` 9-17MB, `vrm_metadata`, `voice_language` không cần để vẽ grid → tốn JSON.
- FE `MotionContext.tsx:52` `fetchCharacters()` gọi list full ngay khi mount `MainLayout` để lấy 1 con `anne`/`preferences.selected_character_slug` lên `<Canvas>` — lãng phí 3 con còn lại. `AvatarsPanel` cũng dùng list đó.

**2. `GET /me/preferences` double:**
- `AvatarBgContext.tsx:28` và `MotionContext.tsx:78` cùng `fetchPreferences()` song song → 2 `SELECT` Neon cùng lúc khi mới load. Đã có `hooks/useUserPreferences.ts` nhưng chưa dùng.

**3. Verify thừa:**
- `routes_preferences.py:88` `SELECT 1 FROM characters WHERE slug=$1` trước `UPDATE user_preferences` — thừa vì FK `selected_character_slug REFERENCES characters(slug)` đã chặn `slug` fake (400), và lazy load `GET /characters/{slug}` 404 cũng đủ. Neon hiện làm 2 queries cho 1 PATCH.

# Giải quyết

**A. `GET /characters` → lite (không thêm endpoint mới):**
- Đổi `_PUBLIC_COLUMNS` list thành `slug, display_name, thumbnail_url, description` (bỏ `vrm_url, vrm_metadata, voice_language, sort_order, ui_strings`). Chúng chỉ trả ở `GET /characters/{slug}`.
- `GET /characters` chỉ gọi khi mở `CharactersPanel` (lazy), không gọi khi mới load web.
- `GET /characters/{slug}` giữ nguyên full. Cache `max-age=300` giữ.

**B. Initial load chỉ 1 con:**
- Flow mới: `AuthGuard` → `GET /me/preferences` (1 lần) → `selected_character_slug` → `GET /characters/{slug}` 1 con để lấy `vrm_url` render. Không `GET /characters` list.
- Giảm từ 1 list 4 full (~10KB) → 1 full 1 con (~2KB).

**C. Dedupe `GET /me/preferences`:**
- Xóa `fetchPreferences()` trong 2 context, chỉ `MainLayout` (hoặc `AuthGuard`) gọi `useUserPreferences()` 1 lần rồi `provide` qua `AvatarBgContext`/`MotionContext` prop. Giảm 6 → 5 calls khi mới load.

**D. Bỏ verify SELECT:**
- Xóa `SELECT 1 FROM characters` trong `PATCH /me/preferences`, chỉ `UPDATE ... SET selected_character_slug=$1` — FK violation → catch → 400 `unknown character`. Giảm PATCH từ 2 queries → 1.

# File chỉnh sửa

| File | Sửa gì |
|---|---|
| `agenticRAG/langgraph_agents/api/routes_characters.py:39` | `_PUBLIC_COLUMNS` list đổi từ 8 cột full xuống 4 cột lite (`slug, display_name, thumbnail_url, description`). |
| `agenticRAG/langgraph_agents/api/routes_characters.py:82` | `GET /characters/{slug}` giữ nguyên full. |
| `ECA_UI/frontend/src/lib/characters.ts` | Giữ `fetchCharacters()` (giờ trả lite), thêm `fetchCharacter(slug)` gọi `GET /characters/{slug}` lấy `vrm_url`. |
| `ECA_UI/frontend/src/contexts/MotionContext.tsx:44` | Bỏ `fetchCharacters()` list khi mount, đổi thành `fetchPreferences()` → `fetchCharacter(slug)` 1 con. `vrmOptions` chỉ set khi mở panel. |
| `ECA_UI/frontend/src/components/panels/AvatarsPanel.tsx` | Khi mở panel mới `fetchCharacters()` lite để render grid (lazy). |
| `ECA_UI/frontend/src/contexts/AvatarBgContext.tsx:28` | Xóa `fetchPreferences()` riêng, nhận `avatar_bg` từ `useUserPreferences` prop. |
| `ECA_UI/frontend/src/hooks/useUserPreferences.ts` | Dùng chung 1 instance ở `MainLayout`, provide xuống 2 context — xóa double fetch. |
| `agenticRAG/langgraph_agents/api/routes_preferences.py:88` | Xóa `SELECT 1 FROM characters` verify, chỉ giữ FK catch. |
| `tests/langgraph_agents/test_preferences.py` | Sửa `test_patch_rejects_unknown_character` mock FK violation thay vì `fetchval`. |

# Không làm

- Không thêm `?view=card`, không thêm endpoint mới, không DynamoDB, không đụng `avatar-profile` (giữ tách).
- `tier_required` cột thêm sau (migration riêng) — plan này chỉ bóp `GET /characters` lite.

# Verify

- `GET /characters` trả 5 cột, không có `vrm_url` → `AvatarsPanel` vẫn vẽ card.
- Mới load web `Network` chỉ 1 `GET /characters/bronya` (theo prefs), không `GET /characters` list.
- `PATCH /me/preferences {slug:"fake"}` → 400 do FK, không 500.
- 1 `GET /me/preferences` khi mới load (không double).
