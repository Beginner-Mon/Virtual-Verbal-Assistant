# Plan phần còn lại — 24 lỗi `react-hooks` mức trung bình-cao → nguy hiểm nhất — 30/08/2026

> Tiếp nối `frontend-lint-62-fix-plan.md` S0→S3-5 đã xong (81→24, `tsc` 0, `build` ✅, `vitest` 109/109).
> Branch: `feature/langgraph-rewrite` @ `44b8494` + S0-S3-5 (24e/0w còn lại).
> Người lập: K — Mr. Senryuu duyệt.

---

## 1. Còn lại gì (đo 30/08 sau S3-5)

`npm run lint` → **✖ 24 problems (24 errors, 0 warnings)**

| Nhóm | Rule | Count | File chính | Mức đụng |
|------|------|-------|------------|----------|
| R1 | `react-hooks/refs` — `Error: Cannot access refs during render` | 11 | `ProfileContent:110×4`, `ChatContext:124,171,173`, `CharacterViewer:598,601`, `useFsmTriggers:84`, `FloatingNavBar:528` | **Rất cao** — đọc/gán `ref.current` trong render |
| R2 | `react-hooks/set-state-in-effect` — `Error: Calling setState synchronously within an effect` | 5 | `ChatMessage:213`, `FloatingNavBar:267`, `ProfileSettingsModal:24`, `ChatContext:192`, `LoginPage:42` | **Cao** — đổi *khi nào* state set |
| R3 | `react-hooks/immutability` — `This value cannot be modified` + `Cannot modify local variables` | 4 | `CharacterViewer:107,573`, `RendererSetup:49`, `ChatMessage:223` (+ `useFrame` 559) | **Cao-Rất cao** — mutate `gl`/`camera`/`vrm.scene` |
| R4 | `react-hooks/purity` — `Cannot call impure function (Math.random)` | 3 | `CharacterViewer:353,354,355` | **Trung bình-cao** — `Math.random` trong render |
| R5 | `react-hooks/immutability` — `Cannot access variable before declared` (`tick`) | 1 | `ChatMessage:223` | **Cao** — closure `tick`/`requestAnimationFrame` |

Tổng 24 = đúng nhóm R2-R5 handoff §24-RỦI RO (không còn `any`/`refresh`/`deps`). Tất cả nằm trong 3 file vừa sửa ở `frontend-fixes`: `CharacterViewer`, `ChatContext`, `ProfileContent`.

---

## 2. Nguyên tắc cho nhóm này

1. **1 file / 1 commit / 1 lần thử tay.** Handoff đã cảnh báo “đừng sửa hàng loạt” — sửa đúng phải cấu trúc lại `effect`/`ref`, tức đổi hành vi thật.
2. **Instrument trước khi sửa:** thêm `console.debug` tạm cho `hasPose`, `greeting`, `position`, `barSize` để thấy race; quay màn hình hoặc Playwright headed nếu có.
3. **Mỗi slice phải xanh `tsc -b` + `vitest` + `vite build` + thử tay** (login, đổi VRM 4 model, resize, motion replay, HMR).
4. **Ưu tiên thấp → cao:** `purity` (không thấy) → `immutability` ít đụng → `set-state` → `refs` (đụng render pipeline).

---

## 3. Plan — 4 slices tuần tự (không song song)

### S3-1 — `purity` 3 lỗi — `CharacterViewer:353-355` (Trung bình-cao, 0.5d)

**Hiện trạng:**
```ts
const positions = useMemo(() => {
  const arr = new Float32Array(count * 3)
  for (let i=0;i<count;i++) {
    arr[i*3+0] = (Math.random()-0.5)*10 // ← R4
    arr[i*3+1] = (Math.random()-0.5)*10
    arr[i*3+2] = (Math.random()-0.5)*10
  }
  return arr
}, [count])
```
`Math.random` là impure, gọi trong render path (kể cả trong `useMemo` factory thì vẫn tính là render). Rule `react-hooks/purity` báo.

**Cách sửa:**
- Chuyển sang `useState(() => generate(count))` hoặc `useMemo` với seed cố định + `useEffect` không chạy lại — một lần sinh, giữ nguyên. Floating particles không cần random mỗi lần `count` đổi ngoài mount.
- Hoặc giữ `useMemo` nhưng thay `Math.random` bằng `pseudoRandom(i)` deterministic (dựa trên `i`), vẫn trong `useMemo` nhưng không còn impure — rule chỉ cấm `Math.random`, không cấm pure function.

**Verify:** `npm run lint` 24→21, `build` ✅, không thấy khác bằng mắt (particles vẫn đều).

---

### S3-2 — `immutability` 4 lỗi + 1 `useFrame` (Cao, 1d)

**a) `CharacterViewer:107` `vrm.scene.userData.restPoses = restPoses`**
```ts
if (!vrm.scene.userData.restPoses) vrm.scene.userData.restPoses = restPoses // ← R3
```
`vrm` là prop từ `useLoader` (hook return), rule coi như “value returned from a hook cannot be modified”.

**Sửa:** Bọc trong `useEffect(() => { if (!vrm.scene.userData.restPoses) vrm.scene.userData.restPoses = ... }, [vrm])` hoặc `useLayoutEffect` — mutate trong effect là intentional, không phải trong render. Nếu rule vẫn báo, thêm `// eslint-disable-next-line react-hooks/immutability -- restPoses is a one-time cache on the loaded GLTF, not React state` (đã thống nhất cho phép disable có comment cho 2 ca intentional).

**b) `CharacterViewer:573` `pos.x = lockPrevPosRef.current.x` + `RendererSetup:49` `gl.toneMapping =`**
- `pos` là `camera.position` (từ `useThree()`), `target` là `controlsRef.current.target` — mutate trong `useFrame` (559) để lock trục khi `lockX/Y/Z`.
- `gl` là `useThree()` return, mutate trong `useEffect`.

Cả hai là **intentional mutation của Three.js object**, không phải React state. Sửa = giữ nguyên logic nhưng chuyển `gl.toneMapping` vào `useLayoutEffect` đã có, và cho `pos/target` thêm disable có comment: `// eslint-disable-next-line react-hooks/immutability -- Three.js camera is mutable by design; lockX reverts the user's drag`.

**c) `ChatMessage:223` `tick` before declared + `useFrame` 559**
```ts
const tick = useCallback(() => { rafRef.current = requestAnimationFrame(tick) }, []) // ← R3 + R5
```
`tick` tham chiếu chính nó trước khi khai báo.

**Sửa:** Đổi sang `useRef` + `useCallback` không tự tham chiếu:
```ts
const tickRef = useRef<() => void>(() => {})
tickRef.current = () => {
  const a = audioRef.current; if (!a) return
  setProgress((a.currentTime/a.duration)*100||0)
  rafRef.current = requestAnimationFrame(tickRef.current)
}
const tick = useCallback(() => tickRef.current(), [])
```
Hoặc dùng `useEffect` với `requestAnimationFrame` loop.

**Verify S3-2:** `npm run lint` 21→17, `tsc` ✅, thử tay: `Lock X/Y/Z` right-drag không đổi trục lock, `toneMapping` vẫn đúng, `ChatMessage` progress mượt.

---

### S3-3 — `set-state-in-effect` 5 lỗi (Cao, 1.5d) — làm từng ca, không gộp

| File | Dòng | Code | Sửa |
|------|------|------|-----|
| `FloatingNavBar:267` | `useEffect(() => setPosition(...), [])` | Init `position` bằng `useState(() => ({x:16, y: (innerHeight - h)/2}))`, bỏ effect |
| `ChatMessage:213` | `useEffect(() => { setUrl(audioUrl); setPlaying(false)... }, [audioUrl, url])` | Derived state: bỏ `useState(url)` và tính `url = audioUrl` trực tiếp, hoặc `useEffect` chỉ khi `audioUrl` đổi và `url !== audioUrl` với guard |
| `ChatContext:192` | `useEffect(() => setMessages(prev => prev[0].id===GREETING_ID ? ...), [ui])` | Đây chính là bug #3 (greeting đổi theo VRM). Sửa đúng là lưu `persona_id` vào `conversations` và khi restore lấy `session.persona_id` (xem `frontend-tasks.md #3`). Nếu chưa làm #3 thì giữ disable có comment `// eslint-disable-next-line react-hooks/set-state-in-effect -- sync greeting while still the only message, see #3` |
| `ProfileSettingsModal:24` | `useEffect(() => { setSettingsView(ROOT_VIEW); setSelectedProvider(...) }, [type])` | Reset khi prop đổi: đổi sang `key={type}` để remount, hoặc giữ với disable + comment |
| `LoginPage:42` | `useEffect(() => { if (stored==='email_mismatch') setMismatchError(true) }, [])` | Init state: `useState(() => sessionStorage.getItem(AUTH_ERROR_KEY)==='email_mismatch' \|\| searchParam==='email_mismatch')`, bỏ effect; `sessionStorage.removeItem` và `setSearchParams` cho vào `useEffect` chỉ chạy khi init true |

**Mỗi ca 1 commit, kiểm:** `FloatingNavBar` không mất position khi resize, `ChatMessage` không kẹt file cũ, `ChatContext` greeting không đổi khi đổi VRM (kiểm bằng cách tạo session với `anne` rồi đổi sang `bronya` — message đầu vẫn `anne`), `LoginPage` vẫn hiện mismatch banner.

**Verify S3-3:** `npm run lint` 17→12 (còn lại 11 refs + 1 tick đã xong ở S3-2).

---

### S3-4 — `refs` 11 lỗi (Rất cao, 2d) — làm cuối, từng file

**a) `ProfileContent:110×4`**
```ts
const setRef = (id: string) => (el: HTMLDivElement | null) => {
  sectionRefs.current[id] = el // ← R1
}
```
`sectionRefs` là `useRef` object, gán trong render (callback tạo trong render).

**Sửa:** Dùng `useCallback` ref hoặc `useEffect`:
```ts
const setRef = useCallback((id: string) => (el: HTMLDivElement | null) => {
  sectionRefs.current[id] = el
}, [])
```
Nhưng callback này tạo mới mỗi render vẫn gán trong render. Cách đúng: `ref={el => { sectionRefs.current[id] = el }}` là render-time ref access — rule vẫn báo. Chấp nhận `useCallback` + `// eslint-disable-next-line react-hooks/refs -- section anchor, not render state` nếu không có cách khác, hoặc chuyển sang `useState` + `useEffect`.

**b) `ChatContext:124,171,173`**
```ts
const uiRef = useRef(ui); uiRef.current = ui // ← R1
const inputRef = useRef(input); inputRef.current = input
const isGeneratingRef = useRef(isGenerating); isGeneratingRef.current = isGenerating
```
Gán `ref.current` trong render.

**Sửa:** Dùng `useLayoutEffect`:
```ts
const uiRef = useRef(ui)
useLayoutEffect(() => { uiRef.current = ui }, [ui])
```
Tương tự cho `inputRef`, `isGeneratingRef`. Đây là “latest-value ref” pattern — phải sync trong effect, không trong render.

**c) `CharacterViewer:598,601`**
```ts
<RendererSetup vrm={vrmRef.current} /> // ← R1
<SceneLighting vrm={vrmRef.current} />
```
Đọc `ref.current` trong JSX.

**Sửa:** Đưa `vrm` vào state (`const [vrmState, setVrmState] = useState<VRM|null>(null)` + `useEffect(() => setVrmState(vrmRef.current), [vrm])`) hoặc truyền `vrm` prop trực tiếp từ `VRMCharacter` thay vì ref. Hoặc nếu bắt buộc đọc ref, bọc trong `useSyncExternalStore`.

**d) `FloatingNavBar:528` `refs.setFloating`**
```ts
<div ref={refs.setFloating} style={floatingStyles} /> // ← R1
```
`refs` là return của `useFloating`, `refs.setFloating` là ref callback, đọc `refs.current` trong render.

**Sửa:** Đây là pattern chuẩn của floating-ui, rule báo sai. Thêm `// eslint-disable-next-line react-hooks/refs -- floating-ui ref, stable` hoặc tách `const { refs: { setReference, setFloating } } = useFloating(...)` và dùng `setFloating` trực tiếp (không qua `refs.current`).

**e) `useFsmTriggers:84`**
```ts
const controllerRef = useRef(controller); controllerRef.current = controller // ← R1
```
Tương tự ChatContext — chuyển sang `useLayoutEffect`.

**Thứ tự làm trong S3-4:** `useFsmTriggers` (1) → `ProfileContent` (4) → `ChatContext` (3) → `CharacterViewer` (2) → `FloatingNavBar` (1). Mỗi file xong chạy `vitest` + thử tay: đổi VRM, greeting, click body part, resize navbar, idle→bored autoAfter.

**Verify S3-4:** `npm run lint` 12→0, `Frontend Build` xanh.

---

## 4. Tổng hợp thứ tự & effort (phần còn lại)

| Slice | Việc | Lỗi giảm | Effort | Rủi ro | Ghi chú |
|-------|------|----------|--------|--------|---------|
| S3-1 | `purity` 3 (`Math.random`) | 24→21 | 0.5d | Trung bình-cao | Không thấy bằng mắt |
| S3-2 | `immutability` 4+1 (`vrm.userData`, `gl`, `camera`, `tick`) | 21→16 | 1d | Cao | 2 ca intentional → cho phép disable có comment |
| S3-3 | `set-state-in-effect` 5 | 16→11 | 1.5d | Cao | Mỗi ca 1 commit, liên quan bug #3 |
| S3-4 | `refs` 11 | 11→0 | 2d | **Rất cao** | 5 file, từng file 1 commit |

**Tổng phần còn lại: ~5 ngày dev (1 người), 6-7 commit.** Cộng với 2.5d đã làm (S0→S3-5) = ~7.5-8 ngày cho toàn bộ 81→0 như plan gốc.

---

## 5. Tiêu chí nghiệm thu phần này

- [ ] `npm run lint` → `0 problems` (hiện 24)
- [ ] `npm run build` + `tsc -b` + `vitest` 109/109 xanh
- [ ] Thử tay sau mỗi slice:
  - S3-1: particles vẫn đều
  - S3-2: `Lock X/Y/Z` + `toneMapping` + `ChatMessage` audio progress
  - S3-3: greeting không đổi khi đổi VRM, `FloatingNavBar` init, `LoginPage` mismatch banner
  - S3-4: VRM load không T-pose, `SceneLighting` shadow đúng, `useFsmTriggers` greeting 1 lần
- [ ] `Frontend Build` trên GitHub xanh (không còn `react-hooks/*`)

---

## 6. Câu hỏi cần Owner chốt trước khi code S3-4

1. Cho phép `eslint-disable` có comment cho 2 ca `immutability` intentional (`gl.toneMapping`, `camera.position` lock) không, hay bắt buộc refactor để hết rule bằng code?
2. `ChatContext:192` có làm chung với task #3 (`frontend-tasks.md` #3 session greeting persona) không, hay tách?
3. Có muốn giữ `lint` chặn `build` trong lúc S3-4 chưa xong (job vẫn đỏ) hay tách tạm để build lấy lại tín hiệu?

---

## 7. Liên kết

- Plan gốc: `docs/plans/frontend-lint-62-fix-plan.md` (S0→S3-5 đã xong)
- Handoff: `docs/worklogs/29-08-2026-frontend-lint-backlog.md`
- Tracking: `docs/tracking/frontend-tasks.md` (branch đã `feature/langgraph-rewrite`)
- Workflow: `.github/workflows/release-tests.yml:153` (`frontend-build`)
