# Plan Frontend Fixes — 25/08/2026 (v3 — chốt sau review)

> Branch: `feature/frontend-fixes`. 3 slices độc lập, có thể song song sau duyệt.
> v2 bị chê sơ xài nên v3 đo kỹ hơn, không đoán mò.

---

## Plan A — Floating navbar mất position khi resize mobile ↔ desktop (Critical)

### A1 vs A2 khác nhau chỗ nào?
- **A1 (giữ vị trí hiện tại):** Khi `window.resize` hoặc `isMobile` flip, **re-measure** `barSize` từ DOM + **clamp** `position` hiện tại vào viewport mới + **cập nhật lại** `refs.setReference(newEl)` cho `useFloating`. User kéo navbar tới đâu thì sau resize vẫn ở gần đó. Code nhiều hơn (thêm `resize` listener + debounce + `refs` update).
- **A2 (reset về mặc định):** Khi `isMobile` flip hoặc resize vượt ngưỡng, **bỏ** `position`/`dockedEdge` hiện tại, set lại `position={x:16, y:center}` + `dockedEdge='left'` + re-measure. Đơn giản nhất, không giữ chỗ user đã kéo.

**Tradeoff:** A1 giữ UX (không mất chỗ kéo) nhưng thêm code + phải debounce + xử lý `floatingStyles` stale. A2 gọn, ít bug, nhưng user mất vị trí đã kéo mỗi lần xoay màn hình. Vì ông báo modal nhảy góc trên-trái (triệu chứng của `position` stale), **chọn A1** nếu instrument cho thấy `barSize` stale, chọn A2 chỉ khi `floating ref` quá phức tạp.

### Triệu chứng (đã bổ sung test lại 25/08)
- `desktop→mobile→desktop` rồi bấm icon → modal ở góc trên-trái `(0,0)` thay vì bám `FloatingNavBar`.
- **Mới:** Sau khi về desktop mà **drag navbar sang chỗ khác thì nó gần như biến mất khỏi khung** — `position`/`barSize` sai nên `transform: translate(x,y)` đẩy ra ngoài viewport, `restrictToScreen` không kéo lại.

### Giả thuyết (chưa chốt, cần instrument)
- `barSize` chỉ đo khi `dockedEdge` đổi `FloatingNavBar.tsx:271` → sau resize `barSize` stale (ví dụ vẫn 56×320 default hoặc size mobile), nên `floatingStyles`/`panelDimensionsClass` tính sai, `getSnapPosition` clamp sai, `restrictToScreen:Modifier` `378` với `barSize` cũ đẩy `transform.x/y` ra ngoài.
- `position` `{x:16,y:center}` không re-clamp sau resize → `position` có thể âm/vượt viewport, drag với `restrictToScreen` dùng `barSize` cũ nên `minX/maxX` sai → bar bay ra ngoài.
- `refs.setReference` chỉ `useEffect []` `292` → sau `isMobile` flip, `.floating-nav-bar` DOM mới mount nhưng `useFloating` vẫn giữ ref cũ → `floatingStyles` về `(0,0)` cho cả modal.
- `barRef` null lúc đo `273/292` (chưa paint sau mount) → `barSize` giữ default.

### Instrument trước khi fix
Log `barSize/position/dockedEdge/isMobile/floatingStyles/isPositioned` + `bar.getBoundingClientRect()` + `window.innerWidth/Height` sau mỗi `resize`. Reproduce 4 path: (1) resize không drag, (2) drag→resize→drag (bắt bug biến mất), (3) resize khi panel mở, (4) resize khi `mobileMenuOpen=true`. Ghi log khi dragNavbar biến mất để xem `position` có vượt `maxX/maxY` không.

### Steps A1 (chốt)
1. Thêm `useEffect` `window.addEventListener('resize', debounce(100, remeasure))` → `setBarSize(rect)` + `setPosition(c=>clamp(c, newSize))`.
2. Thêm `useEffect([isMobile])` → khi flip, `refs.setReference(newBarEl)` + re-measure.
3. Test 768↔1024↔1440, cả khi panel mở, không nhảy góc.

---

## Plan B — Greeting chỉ chạy lần đầu chọn VRM (High, Low) — CHỐT B1

**Chốt:** Dùng **B1 session only**, **B2 out scope**. User refresh → greeting lại là chấp nhận được.

**File:** `MotionContext.tsx:102-144` `selectedVrmId`, `CharacterViewer` `attachControllers`.

**Steps B1:**
1. `visitedRef = useRef<Set<string>>(new Set())` trong `MotionProvider`.
2. `useEffect([selectedVrmId, animController])` nếu `!visited.has(id) && isCompatible(character) && animController` → `visited.add(id)` → `await transitionTo('greeting')`.
3. Guard: không greeting nếu `incompatible`, `animController==null`, hoặc `currentState==='greeting'` đang chạy.
4. Test: anne lần 1 chạy, bronya lần 1 chạy, quay lại anne không chạy; reload thì chạy lại (đúng B1).

**Không làm B2** (localStorage persist) — out scope.

---

## Plan C — Motion Controls cleanup (Medium, Low) — VIẾT LẠI THEO SPEC MỚI

### Spec mới ông chốt
- `enablePan` + `followTarget` đã có → **bỏ** 3 thanh `offset X/Y/Z` (không còn quan trọng).
- **Thêm** 3 button `Lock X` / `Lock Y` / `Lock Z` — khi lock, right-click drag không đổi coordinate đó, dễ di chuyển model theo ý.
- **Bỏ** thanh `Play/Pause + Reset + speed 0.5/1.0/1.5/2.0` — cả dev và user đều bỏ, không cần.
- **Bỏ** `Clip Info` — cả dev và user đều bỏ.
- **Expression** chỉ dev thấy, user không. **Bỏ** `Speak` + `Test WAV` bên trong expression, **xóa file** `asset/audio/test.wav` (đã test xong lip-sync theo audio).
- `Motion File (debug)` và `Character state` đang chung 1 chức năng (chỉ khác góc camera) — **chỉ dev thấy**, user không. Và **không để action của Character state lẫn vào Motion File debug**.

### Bảng quyết định

| # | Control (file:line) | Hiện tại | Hành động | Cho ai | Ghi chú |
|---|---|---|---|---|---|
| C1 | `followTarget` checkbox `MotionControlPanel.tsx:157` | Có | **Giữ** | User |  |
| C2 | `enablePan` checkbox `165` | Có | **Giữ** | User |  |
| C3 | `offset X/Y/Z` sliders `178-219` | Có | **Bỏ** | — | Thay bằng Lock buttons |
| C4 | **Lock X / Y / Z** (mới) | Chưa có | **Thêm** | User | Button toggle, khi lock thì `CameraController`/`OrbitControls` không update trục đó khi right-drag |
| C5 | `cameraMode` head/hips `221` | Có | **Giữ** | User |  |
| C6 | `Reset` camera `147` | Có | **Giữ** | User |  |
| C7 | `Character state` selector `234` | Có | **Giữ nhưng Dev-only** | Dev | User không thấy. Chỉ list `stateOptions` (không lẫn file) |
| C8 | `Motion File (debug)` selector `262` | Có | **Giữ nhưng Dev-only** | Dev | User không thấy. Filter: không chứa action của Character state, chỉ list `MOTION_FILES` |
| C9 | `isPlaying` Play/Pause `285` | Có | **Bỏ** | — | Cả dev và user |
| C10 | `handleReset` `295` | Có | **Bỏ** | — | Cùng thanh playback, bỏ luôn |
| C11 | `speed` 0.5-2.0 `306` | Có | **Bỏ** | — | Cả dev và user |
| C12 | `Clip Info` `319` | Có | **Bỏ** | — | Cả dev và user |
| C13 | `Expressions` emotion buttons `339` + intensity/duration `351` | Có, gate `DEV` | **Giữ Dev-only** | Dev | User không thấy (đã đúng) |
| C14 | `Speak` / `Test WAV` `379-401` | Có, trong Expressions | **Bỏ** | — | Cả dev và user, xóa luôn |
| C15 | `asset/audio/test.wav` + import `9` | Có | **Xóa file + import** | — | Đã test xong lip-sync |
| C16 | `avatarMode/lastEmotion` `403` | Có | **Giữ Dev-only** (tùy) | Dev | Nếu giữ thì để trong Dev panel, không cho user |

### Steps C
1. `MotionControlPanel.tsx`: xóa `C3, C9-C12, C14-C15` khỏi JSX; thêm `C4` Lock X/Y/Z (3 toggle, state trong `cameraConfig` hoặc `useState`, truyền vào `CameraController`/`Controls`).
2. Bọc `C7, C8, C13` bằng `import.meta.env.DEV` (như Expressions hiện tại `332`).
3. `C8` filter: `motionFileOptions` chỉ `MOTION_FILES`, không map `stateOptions`.
4. Xóa `import testWavUrl` + `playWavSpeech` + `speak/speakWav` handlers + file `test.wav` (check không còn import).
5. Test: `npm run build` prod không thấy file picker/state/speed/clip/expressions/speak; `npm run dev` thấy `Character state`+`Motion File`+`Expressions` (không có Speak); Lock X/Y/Z bấm right-drag không đổi trục lock.

---

## Tách session (sau duyệt v3)

- Session A: Plan A
- Session B: Plan B (B1)
- Session C: Plan C (bảng trên)

Mỗi session commit riêng, không đụng file nhau (A: FloatingNavBar, B: MotionContext, C: MotionControlPanel).
