# Frontend Tasks — 25/08/2026

> Tách từ `tech-debt.md` cho gọn — chỉ việc frontend. Sắp xếp theo **priority** (giá trị user) rồi **difficulty** (rủi ro kỹ thuật).
> Branch hiện tại: `feature/langgraph-rewrite` (đã merge `feature/frontend-fixes` + `feature/motion-frontend` 30/08 — chứa motion replay + handoff 62 lint).
> Mỗi task ghi rõ *hiện trạng → kỳ vọng → acceptance* để N pick vertical slice.

Priority: 🔴 Critical · 🟠 High · 🟡 Medium · ⚪ Low
Difficulty: ★☆☆ Low · ★★☆ Medium · ★★★ High

---

## 🔴 Critical — làm trước

### 1) VRM greeting chỉ chạy lần đầu chọn — ⏸️ REVERT, chuyển cho agent khác
**Hiện trạng:** Plan B1 đã thử (`visitedRef` trong `MotionContext` + `hasGreeted` trong `useFsmTriggers`) nhưng gây `hasPose` race → `transitionTo('greeting')` luôn `false` (log `[greeting] result false`), refresh không greeting, đổi animation cũng kẹt. Đã revert về trước Plan B (`14018f7`) — lõi `AnimationController`/`hasGreeted` về nguyên bản, greeting lại chạy (global 1 lần/session) nhưng chưa phải 1 lần/VRM.
**Kỳ vọng:** Lần đầu chọn `slug` → `greeting` 1 lần; chọn lại cùng `slug` → không chạy lại.
**Priority:** 🟠 High · **Difficulty:** ★☆☆ Low · **Status:** ⏸️ Chưa làm — để agent khác, cần làm mỏng ngoài `useFsmBoot` (đợi `hasPose` + `idle` rồi mới greet, không dùng `attachedIdRef`).
**Phụ thuộc:** `MotionContext` + `AnimationStates`

### 2) Floating navbar mất position khi resize mobile ↔ desktop — ✅ XONG (Plan A)
**Hiện trạng:** Đã fix `14018f7`: `FloatingNavBar.tsx` re-measure `barSize` + clamp `position` + `refs.setReference` khi `resize`/`isMobile` flip; `MobileNavBar.tsx` clamp `menuXRef/Y`.
**Priority:** 🔴 Critical · **Difficulty:** ★★☆ Medium · **Status:** ✅ Xong

### 3) Tin nhắn đầu session bị đổi theo VRM mới chọn — sai dữ liệu ⏳ CHƯA LÀM
**Hiện trạng:** Đã fix hiển thị greeting ở mọi session (không chỉ session mới), nhưng `ui.greeting` lấy từ `characters.ui_strings` hiện tại → đổi VRM làm message đầu của session cũ cũng đổi (session đó dùng model cũ).
**Priority:** 🔴 Critical · **Difficulty:** ★★☆ Medium
**Effort:** ~1 ngày · **Phụ thuộc:** `ChatContext.tsx:150,185,268` + `conversations.persona_id`
**Acceptance:** Lưu `persona_id/slug` vào `conversations` lúc tạo session; khi restore, greeting lấy theo `session.persona_id` lúc tạo, không theo `selectedVrmId` hiện tại.

---

## 🟡 Medium — làm sau critical

### 4) Motion Controls modal đang show đồ dev — ✅ XONG (Plan C v3)
**Hiện trạng:** Đã fix `0d89006`: bỏ `offset X/Y/Z` → thêm `Lock X/Y/Z`, bỏ `Play/Pause/speed/Clip Info` cả dev+user, `Character state`/`Motion File`/`Expressions` chỉ `DEV`, xóa `Speak/Test WAV` + `test.wav`.
**Priority:** 🟡 Medium · **Difficulty:** ★☆☆ Low · **Status:** ✅ Xong

### 5) Click nhân vật trả về bộ phận cụ thể — ✅ XONG
**Hiện trạng:** Đã fix: `raycast` trả về `boneName` đúng (normalize mesh name ↔ `VRMHumanBoneName`), test trên anne/bronya/miku/miki (khác rig). Click effect tách riêng, đã chạy trước đó.
**Priority:** 🟡 Medium (feature) · **Difficulty:** ★★★ High · **Status:** ✅ Xong
**Effort:** ~2-3 ngày · **Phụ thuộc:** `CharacterViewer.tsx` + `three` raycaster + VRM `humanoid` bones + `vrmManifest`
**Acceptance:** Click lên tay/chân/đầu → trả về `boneName` đúng.

### 6) Ghi âm trong chat — chưa thực thi
**Hiện trạng:** UI chat có chỗ cho ghi âm nhưng chưa có `MediaRecorder` flow.
**Priority:** ⚪ Low-Medium (feature) · **Difficulty:** ★★☆ Medium ( + ★★★ nếu kèm STT)
**Effort:** ~2 ngày (chỉ ghi + playback), ~4 ngày nếu kèm STT/Whisper
**Phụ thuộc:** `ChatPanel.tsx`, `useAuth`, browser permission, (optional) `agenticRAG` STT endpoint
**Acceptance:** Bấm mic → `MediaRecorder` → preview → gửi như message type `audio`; fallback khi không có permission; không chặn `streamChat`.

---

## Thứ tự làm đề xuất (còn lại — #2, #4, #5 đã xong)

1. **#3 Session greeting persona** (Critical, Medium) — ⏳ CHƯA LÀM, sai dữ liệu
2. **#1 Greeting 1 lần** (High, Low) — ⏸️ Chờ agent khác
3. **#6 Ghi âm** (Low-Med, Medium) — ⏳ CHƯA LÀM

> #2, #4, #5 đã ✅ xong, #1 ⏸️ revert cho agent khác.

---

## Liên kết

- Gốc: `docs/tracking/tech-debt.md` (438 dòng, nhiều mục infra) — file này chỉ frontend.
- Worklog liên quan: `25-08-2026.md` (single-port + characters local shim) + `29-08-2026-frontend-lint-backlog.md` (62 lint handoff chi tiết).
- Branch: `feature/langgraph-rewrite` (hiện tại, đã gồm `frontend-fixes`) — các fix trên nên làm vertical slice, mỗi task 1 commit.
