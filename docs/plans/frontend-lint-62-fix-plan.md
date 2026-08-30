# Plan sửa 62 lỗi lint đang đỏ job `Frontend Build` — 30/08/2026

> Branch: `feature/langgraph-rewrite` @ `44b8494` (đã merge `frontend-fixes` + `motion-frontend`)
> Handoff gốc: `docs/worklogs/29-08-2026-frontend-lint-backlog.md` (62 lỗi, commit `f78c8e2` handoff, `b3a83c5` đã sửa 74→62)
> Đo lại 30/08: `npm run lint` → **81 problems (69 errors, 12 warnings)** trên 60 hit — chênh 19 so với 62 do code mới (`motion` SSE, `.amplify/generated`) + rule `exhaustive-deps`/`empty-object` nổi thêm
> Người lập: K (Senior Solution Architect) — Mr. Senryuu duyệt

---

## 0. Vì sao phải làm

`.github/workflows/release-tests.yml` job `🏗️ Frontend Build` (ubuntu, 19/08 mới có):

```
npm ci → npm run lint → npm run build (vite + tsc -b) → npm run test (vitest) → gate check VITE_API_GATEWAY_URL
```

`lint` đỏ → `build` **không bao giờ chạy**. Muốn xanh phải hết **toàn bộ** lỗi — sửa 1 phần không đổi màu (handoff §"Vì sao job đỏ"). Hiện `tsc --noEmit` sạch, `vitest` 109/109 xanh, `vite build` local thành công (7.85s, 3474 modules), chỉ lint đỏ. Hậu quả: job đỏ đủ lâu để không ai nhìn → lỗi mới lẫn vào không ai thấy (handoff §Cảnh báo).

**Phạm vi:** chỉ `ECA_UI/frontend/**`. Không đụng `agenticRAG`, `SpeechLLm`, CDK, DynamoDB.

---

## 1. Đo hiện trạng (30/08)

Chạy `npm run lint` tại `ECA_UI/frontend`:

**Tổng:** `✖ 81 problems (69 errors, 12 warnings)` — 0 fixable auto với `--fix` (đã thử).

**Phân nhóm (từ `lint.txt` 30/08):**

| Nhóm | Count | Rule | Ghi chú so với handoff 62 |
|------|-------|------|---------------------------|
| A | 32 | `@typescript-eslint/no-explicit-any` | **Khớp** handoff 32 (15 `amplify/functions/*`, 13 `catch (err:any)`, 4 `AuthGuard/ProfileContent/CharacterViewer/amplify-env.d.ts`) |
| B | 11 | `react-hooks/refs` — `Error: Cannot access refs during render` | Handoff 11 — khớp |
| C | 5 | `react-hooks/immutability` — `Error: This value cannot be modified` + `Cannot modify local variables` | Handoff 5 — khớp |
| D | 5 | `react-hooks/set-state-in-effect` — `Error: Calling setState synchronously within an effect` | Handoff 5 — khớp |
| E | 3 | `react-hooks/purity` — `Error: Cannot call impure function (Math.random)` | Handoff 3 — khớp |
| F | 6 | `react-refresh/only-export-components` | **Khớp** handoff 6 |
| G | 6 | `react-hooks/exhaustive-deps` | **Mới** — không có trong 62, do motion code mới thêm (`FloatingNavBar: barSize.height/refs`, `RendererSetup: gfx.mtoon`, `ChatContext: registerSessionMotion`, `useRedirectIfAuthenticated: navigate/to`, `CharacterViewer: camera.position`) |
| H | 6 | `The {} ("empty object") type` | **Mới** — 6 hit tại `.amplify/generated/**` (codegen Amplify) |
| I | 5 | `Unused eslint-disable directive` | **Mới** — 4× `no-restricted-syntax` + 1 `no problems` (vrmManifest) |
| J | 1 | `Definition for rule '@typescript-eslint/no-unused-vars' was not found` | **Mới** — `.amplify/artifacts/cdk.out/.../index.js` bị lint dù là artifact |
| K | 1 | `amplify-env.d.ts:2 no-explicit-any` dư | Đã đếm trong A nhưng gây nhiễu |

**File hit chính (ngoài `.amplify`):** `amplify/functions/*` (7 file), `CharacterViewer.tsx` (7 lỗi), `ChatContext.tsx` (4), `ProfileContent.tsx` (4), `ChatMessage.tsx` (3), `FloatingNavBar.tsx` (3), `MotionContext/GraphicsContext/ThemeContext/button.tsx` (6 refresh), `LoginPage/CreateAccountPage/...` (any).

**Kết luận:** 62 handoff vẫn là lõi (A-F = 62), phần chênh (G-J = 19) là hygiene — nếu không ignore `.amplify/**` thì tổng không bao giờ về 0.

---

## 2. Quyết định thiết kế (ADR mini)

| D | Quyết định | Lý do |
|---|------------|-------|
| D1 | **Không đổi `eslint.config.js` extends** — giữ `tseslint.configs.recommended` + `reactHooks.configs.flat.recommended` + `reactRefresh.configs.vite` | Đây là rule đang làm đỏ; đổi config để xanh là gian lận tín hiệu |
| D2 | **Thêm `globalIgnores` cho `.amplify/**`** (generated + cdk.out) | H-J (13 lỗi) là code máy sinh, không phải việc frontend. Handoff đã ghi 32 any nhưng không tính generated — hiện tại generated làm tổng 62→81 |
| D3 | **Không tách `lint` thành job riêng không chặn build** (tạm thời) | Handoff cảnh báo: tách sẽ làm build xanh giả. Chỉ tách nếu Owner chốt chấp nhận lint đỏ lâu dài |
| D4 | **Thứ tự slice: Hygiene → refresh → any → hooks** | Handoff gợi ý refresh trước vì chữa HMR thật. Đảo `any` và `refresh` cũng được nhưng refresh cơ học nhất, rủi ro thấp nhất, nên làm đầu để lấy tín hiệu HMR xanh sớm |
| D5 | **Mỗi slice phải xanh `tsc -b` + `vitest` + thử tay luồng auth** | Nhóm `any` nằm trong đường đăng nhập; nhóm `hooks` đổi hành vi thật (CharacterViewer/ChatContext). Không test tay = không merge |
| D6 | **Không sửa hàng loạt `react-hooks`** — 1 file/commit, có instrument | Handoff §24-RỦI RO: sửa đúng phải cấu trúc lại effect/ref, dễ đổi hành vi. `CharacterViewer` vừa sửa trong `frontend-fixes` nên càng nhạy |

---

## 3. Plan — 4 slices độc lập (đề xuất làm tuần tự)

### Slice 0 — Hygiene: dọn nhiễu để đếm đúng 62 (Low, 0.5 ngày)

**Mục tiêu:** Đưa tổng 81 → 62 (khớp handoff), không đổi hành vi.

1. `ECA_UI/frontend/eslint.config.js:8` — thêm:
   ```js
   globalIgnores(['dist', '.amplify/**'])
   ```
   Lý do: loại H (6 `{} type` trong `.amplify/generated`) + J (1 `no-unused-vars` trong `cdk.out`) + 4 `Unused eslint-disable` trong generated. Tổng ~11 hit biến mất.

2. Xóa 2 `eslint-disable` thừa:
   - `src/avatar/vrmManifest.ts:1` — `// eslint-disable-next-line` không còn lỗi sau khi ignore (hiện warning `Unused eslint-disable`)
   - Kiểm tra 4× `no-restricted-syntax` disable trong `amplify/functions/*` — nếu sau ignore vẫn còn thì xóa.

3. Verify:
   ```bash
   cd ECA_UI/frontend && npm run lint 2>&1 | tail -5
   # Kỳ vọng: ✖ 62 problems (không còn 81)
   npm run build   # vẫn xanh (tsc -b && vite build)
   npm run test    # 109/109 xanh
   ```

**Acceptance S0:** `npm run lint` báo 62 (không phải 81), `tsc` + `build` + `vitest` xanh, diff chỉ `eslint.config.js` + 1-2 dòng disable.

---

### Slice 1 — `react-refresh/only-export-components` (6 lỗi, Low-Med, 1 ngày)

**Vì sao làm đầu:** Cơ học, không đổi hành vi, chữa dứt lỗi HMR đã làm mất thời gian trong phiên (`useChat must be used within ChatProvider`, phải `rm -rf node_modules/.vite && vite --force`).

**6 file — mỗi file tách thành 2:**

| File hiện tại | Tách thành |
|---------------|------------|
| `src/contexts/ChatContext.tsx` (export `ChatProvider` + `useChat` + `GREETING_ID` const) | `ChatContext.tsx` giữ `createContext` + `Provider` ; `hooks/useChat.ts` export `useChat()` (re-export context) |
| `src/contexts/MotionContext.tsx` | `MotionContext.tsx` + `hooks/useMotion.ts` |
| `src/contexts/GraphicsContext.tsx` | `GraphicsContext.tsx` + `hooks/useGraphics.ts` |
| `src/contexts/ThemeContext.tsx` | `ThemeContext.tsx` + `hooks/useTheme.ts` |
| `src/components/ui/button.tsx` | `button.tsx` chỉ export component ; `lib/buttonVariants.ts` chứa `buttonVariants` const (hiện chung file nên trigger rule) |

**Cách làm:**
- Mỗi context: `export const ChatContext = createContext<...>(null)` giữ trong `ChatContext.tsx`, `export function useChat(){ const c=useContext(ChatContext); if(!c) throw...; return c }` chuyển sang file mới.
- Update import: `import { useChat } from '@/contexts/ChatContext'` → `import { useChat } from '@/hooks/useChat'` (hoặc re-export barrel để không đụng nhiều file — chọn 1, ghi trong commit).
- `button.tsx` — tách `cva` variants ra `buttonVariants.ts`, `button.tsx` chỉ export `Button`.

**Verify S1:**
```bash
npm run lint  # 62 → 56 (trừ 6)
npm run build && npm run test
npm run dev  # sửa ChatContext xong không còn hard-reload, HMR update nóng
```

**Rủi ro:** Thấp — chỉ di chuyển export, không đổi logic. Dễ review.

**Acceptance S1:** `npm run lint` 56 lỗi, `git grep "only-export-components"` 0 hit trong `src/`, HMR sửa context không cần hard-reload.

---

### Slice 2 — `@typescript-eslint/no-explicit-any` (32 lỗi, Med, 2 đợt)

**Chia 2 đợt vì 15 lỗi nằm trong Cognito Lambda (deployment riêng, hỏng là hỏng login).**

#### S2a — `amplify/functions/*` (15 lỗi, 1 ngày, làm trước)

**File:** `amplify/functions/auth-status/handler.ts`, `lookup-email/handler.ts`, `post-confirmation/handler.ts`, `pre-sign-up-handler/handler.ts`, `pre-token-generation/handler.ts`, `set-password/handler.ts`, `shared/cors.ts`

**Cách:**
- Cài `@types/aws-lambda` đã có → thay `event: any` bằng `PreSignUpTriggerEvent`, `PostConfirmationTriggerEvent`, `PreTokenGenerationTriggerEvent` v.v. (check `aws-lambda` exports).
- `handler.ts:55` `result: any` → `result: PreSignUpTriggerEvent['response']` hoặc `unknown` + narrow.
- Nếu type quá rộng, dùng `unknown` + `as` có guard, không dùng `any`.

**Verify S2a:** `npx tsc -p amplify/tsconfig.json` (hoặc `tsc -b` trong frontend nếu có) xanh, `npm run lint` 56→41, **thử login thật**: tạo account email → login Google cùng email → link sạch (worklog 05/08 §9), không 500.

#### S2b — `src/**` + `catch (err:any)` (17 lỗi, 1 ngày)

**13× `catch (err: any)`:** `LoginPage.tsx`, `CreateAccountPage.tsx`, `EnterPasswordPage.tsx`, `SetPasswordPage.tsx`, `AuthGuard.tsx`, `ProfileContent.tsx`, `useRedirectIfAuthenticated.ts` v.v.
- Tạo `src/lib/errors.ts`:
  ```ts
  export function errorMessage(e: unknown): string {
    if (e instanceof Error) return e.message;
    if (typeof e === 'string') return e;
    return 'Đã có lỗi xảy ra';
  }
  ```
- Thay `catch (err: any) { toast(err.message) }` → `catch (err: unknown) { toast(errorMessage(err)) }`

**4× `useState<any>/useRef<any>`:** `AuthGuard.tsx:30` `useState<any>` cho Amplify session → `useState<AuthSession | null>`, `ProfileContent.tsx:18` `useRef<any>` OrbitControls → `useRef<OrbitControls | null>`, `CharacterViewer.tsx:403` `any` cho restPose → đã có `RestPoseMap` từ `b3a83c5` (dùng lại), `amplify-env.d.ts:2` → `Record<string, unknown>` hoặc `typeof amplify_outputs`.

**Verify S2b:** `npm run lint` 41→24, `tsc -b` xanh, `vitest` xanh, thử login + đổi VRM + chat 1 lượt (đảm bảo không vỡ `ChatContext`).

**Acceptance S2:** `npm run lint` còn 24 (đúng nhóm hooks), không còn `any` trong `src/` và `amplify/functions/`.

---

### Slice 3 — `react-hooks/*` (24 + 6 `exhaustive-deps` mới, High, 3-4 ngày)

**Đây là slice RỦI RO — mỗi fix đổi hành vi thật. Làm từng file, từng rule, có instrument.**

**Tồn 30 (24 cũ + 6 mới G):**
- `react-hooks/refs` 11 — `ProfileContent 4, ChatContext 3, CharacterViewer 2, useFsmTriggers 1, FloatingNavBar 1`
- `react-hooks/immutability` 5+1 — `CharacterViewer 3 (107, 573, 49 mod gl/camera)`, `ChatMessage 1 (tick before decl)`, `RendererSetup 1 (gl.toneMapping)`
- `react-hooks/set-state-in-effect` 5 — `ChatMessage 213, FloatingNavBar 267, ProfileSettingsModal 24, ChatContext 192, LoginPage 41`
- `react-hooks/purity` 3 — `CharacterViewer 353-355 Math.random`
- `exhaustive-deps` 6 — `CharacterViewer:446 camera.position`, `FloatingNavBar:268 barSize.height / 296 refs`, `RendererSetup:102 gfx.mtoon`, `ChatContext:296 registerSessionMotion`, `useRedirectIfAuthenticated:11 navigate/to`

**Nguyên tắc (từ handoff):** Đừng sửa hàng loạt. 1 commit = 1 file hoặc 1 rule cohort, chạy `vitest` + thử tay sau mỗi commit.

**Gợi ý thứ tự trong S3:**

1. **S3-1 Purity (3) — `CharacterViewer.tsx:353-355`:** `Math.random` trong render → chuyển vào `useMemo(() => Float32Array, [count])` với seed cố định hoặc `useState(() => generateOnce)`. Đây là floating particles — đổi sang `useMemo` không đổi hành vi thấy được, nhưng hết `purity` error.

2. **S3-2 Immutability mod `gl`/`camera` (4):** `RendererSetup.tsx:49 gl.toneMapping = ...` (mutate `gl` từ `useThree`) → bọc trong `useLayoutEffect` là đúng (đã trong effect nhưng rule vẫn báo vì `gl` là hook return). Thêm `// eslint-disable-next-line react-hooks/immutability -- three gl is mutable intentionally` nếu đã trong effect và không có cách khác, hoặc chuyển sang `useThree` setter. Tương tự `CharacterViewer:573 pos.x = ...` (mutate `camera` trong `useFrame`) — đây là intentional (OrbitControls lockX), nên để `// eslint-disable` có comment giải thích, không cố "fix" bằng cách đổi logic lock.

3. **S3-3 `set-state-in-effect` (5):** Mỗi ca khác nhau:
   - `FloatingNavBar.tsx:267 setPosition` trong `useEffect([])` init → đổi sang `useState(() => ({x:16, y: Math.round((innerHeight - barSize)/2)}))` (initial state function), bỏ effect.
   - `ChatMessage.tsx:213 setUrl` trong effect sync `audioUrl→url` → đây là derived state, nên bỏ `useState(url)` và tính `url = audioUrl` trực tiếp hoặc dùng `useSyncExternalStore` nếu cần.
   - `ChatContext.tsx:192 setMessages` trong effect sync `ui.greeting` → đây là bug #3 (greeting đổi theo VRM) — fix đúng là lưu `persona_id` vào `conversations` (task #3 trong `frontend-tasks.md`), không phải giữ effect. Slice này nên **đợi #3 xong** hoặc làm cùng.
   - `LoginPage.tsx:41 setMismatchError` trong effect đọc `sessionStorage/searchParams` → chuyển sang `useState(() => sessionStorage.getItem(...) === 'email_mismatch')` (init), bỏ effect.
   - `ProfileSettingsModal.tsx:24 setSettingsView` trong `useEffect([type])` → đây là reset khi prop đổi, đúng pattern nhưng rule báo; có thể giữ với comment hoặc đổi sang `key={type}` để remount.

4. **S3-4 `refs` (11):** `ProfileContent.tsx:110 sectionRefs.current[id]=el` trong render → đổi sang `useCallback` ref hoặc `useEffect` để gán; `ChatContext.tsx:124/171/173 uiRef.current = ui` trong render → dùng `useLayoutEffect` hoặc `useRef` với `useEffect` sync, không gán trực tiếp trong render; `CharacterViewer.tsx:598/601 vrmRef.current` trong render → truyền `vrm` qua prop/state thay vì đọc ref trong JSX; `FloatingNavBar.tsx:528 refs.setFloating` trong render → đây là `useFloating` pattern, thường cho phép — nếu rule vẫn báo thì thêm disable có comment.

5. **S3-5 `exhaustive-deps` (6):** Thêm deps thiếu (`barSize.height`, `refs`, `gfx.mtoon`, `registerSessionMotion`, `navigate/to`, `camera.position`) hoặc bọc bằng `useCallback`/`useMemo` để deps ổn định. Mỗi thêm deps phải test lại vì effect sẽ chạy lại nhiều hơn.

**Verify S3:** Sau mỗi sub-slice: `npm run lint` giảm dần 24→0, `npm run build` xanh, `vitest` 109 xanh, **thử tay**: drag navbar resize không mất position (đã fix 14018f7), greeting 1 lần/VRM, click body part, motion replay không mất sau refresh (352e1cf), auth login/logout.

**Acceptance S3:** `npm run lint` 0 errors 0 warnings, `Frontend Build` job xanh (bao gồm `build gate rejects a missing variable`).

---

## 3.5 Xếp theo mức độ đụng hệ thống — thấp → cao (an toàn nhất → nguy hiểm nhất)

> **Nguyên tắc xếp:** đếm *runtime behavior change* thực, không đếm số lỗi. Config/type-only = an toàn; đụng `effect`/`ref`/`render` = nguy hiểm vì đổi thứ tự chạy/sync.

| Hạng | Nhóm (slice) | Số lỗi | Đụng gì | Mức đụng hệ thống | Vì sao ở đây |
|------|--------------|--------|---------|-------------------|--------------|
| **1 — An toàn nhất** | **S0 Hygiene** — `globalIgnores(['.amplify/**'])` + xóa `eslint-disable` thừa | 19 (H+I+J) | Chỉ `eslint.config.js` + comment | **Không đụng runtime**, chỉ đổi cách đếm lint | Xóa nhiễu để 81→62, không đổi JS chạy. Có thể làm đầu, không cần test tay ngoài `npm run lint`. |
| **2 — Rất an toàn** | **S1 `react-refresh`** — tách `Chat/Motion/Graphics/ThemeContext` + `button.tsx` | 6 | Di chuyển `export` giữa file, đổi `import` path | **Cơ học, không đổi logic** | Chỉ đổi nơi `createContext`/`useX` sống. Chữa HMR thật (handoff: `useChat must be used within ChatProvider` do hot-reload). Dễ review, `tsc` bắt sai import ngay. |
| **3 — An toàn** | **S2 `no-explicit-any` toàn bộ 32** | 32 | Chỉ `type` TS, không đổi JS emit | **An toàn về hành vi, nhưng ½ nằm trong đường đăng nhập** | `any → unknown + guard` hoặc `PreSignUpTriggerEvent` không đổi runtime (type bị xóa khi build). Rủi ro duy nhất: gõ sai type trong `amplify/functions/*` làm Cognito Lambda log khó đọc, không gãy luồng. Chia S2a (15 Lambda) trước để thử login riêng. |
| 3a | └ S2b `catch (err:any)` 13 + `useState<any>` 4 | 17 | type + helper `errorMessage(e:unknown)` | Thấp-Medium | Đổi `err.message` → `errorMessage(err)`, hành vi giữ nguyên. |
| **4 — Trung bình** | **S3-5 `exhaustive-deps` 6** | 6 | Thêm deps vào `useEffect`/`useCallback` | **Trung bình** — chạy lại effect nhiều hơn, có thể re-render thêm | Thiếu `barSize.height`/`refs`/`gfx.mtoon`/`registerSessionMotion`/`navigate` làm effect stale; thêm vào thì effect chạy đúng hơn nhưng tốn render. Đụng nhẹ, test bằng resize/nav. |
| **5 — Trung bình-Cao** | **S3-1 `purity` 3** — `Math.random` trong render (`CharacterViewer:353-355`) | 3 | `Float32Array` particles từ `Math.random` | **Trung bình-Cao** — chuyển `Math.random` trong render → `useMemo`/`useState` init | Đổi từ "mỗi render sinh mới" sang "sinh 1 lần". Với particles thì không thấy khác, nhưng là lần đầu đụng render phase. |
| **6 — Cao** | **S3-3 `set-state-in-effect` 5** | 5 | `useEffect(() => setState)` → init state / derived state / `key` remount | **Cao** — đổi *khi nào* state được set | Mỗi ca là 1 pattern khác: `FloatingNavBar` init position, `ChatMessage` derived `url`, `ChatContext` sync `ui.greeting` (liên quan bug #3 session greeting sai dữ liệu), `LoginPage` read `sessionStorage`, `ProfileSettingsModal` reset. Làm sai → kẹt render hoặc greeting sai VRM (đã dính ở #3). |
| **7 — Nguy hiểm nhất** | **S3-2 + S3-4 `immutability` 6 + `refs` 11** | 17 | `gl.toneMapping =`, `camera.position` mutate trong `useFrame`, `ref.current` gán/đọc trong render | **Rất cao** — đụng render pipeline & 3D loop | Tập trung ở **3 file vừa sửa trong `frontend-fixes`**: `CharacterViewer.tsx` (7 lỗi: `vrm.scene.userData.restPoses`, `vrmRef.current` trong JSX, `pos.x` lock, `Math.random` particles) — trực tiếp điều khiển VRM pose/shadow; `ChatContext.tsx` (4 lỗi: `uiRef.current = ui` trong render, `setMessages` sync greeting) — nguồn bug #1/#3 greeting; `ProfileContent.tsx` (4 lỗi: `sectionRefs.current[id]=el` trong render). Handoff cảnh báo "sửa đúng phải cấu trúc lại effect/ref, tức đổi hành vi thật". Đã từng revert `14018f7` vì `hasPose` race → `transitionTo('greeting')` false. **Phải làm cuối cùng, 1 file/commit, instrument log + thử tay + `vitest` sau mỗi commit.** |

**Tóm tắt 1 dòng:**

> **S0 (hygiene) < S1 (refresh) < S2 (any) < S3-5 (deps) < S3-1 (purity) < S3-3 (set-state) < S3-2/4 (immutability+refs)** — càng về sau càng đụng `render`/`effect`/`ref` thật, càng phải làm mỏng và test tay.

**Đề xuất thứ tự thực hiện theo mức an toàn (đã cập nhật §3):** `S0 → S1 → S2a → S2b → S3-5 → S3-1 → S3-3 → S3-2/4` — thay vì gộp cả S3 một block. S3-2/4 để cuối cùng.

---

## 4. Thứ tự & ước lượng tổng (đã sắp theo an toàn)

| Slice | Việc | Lỗi giảm | Effort | Mức đụng | Có thể song song? |
|-------|------|----------|--------|----------|-------------------|
| S0 | Hygiene: ignore `.amplify` + xóa disable thừa | 81→62 | 0.5d | **Không đụng** | — |
| S1 | `react-refresh` 6 | 62→56 | 1d | **Rất an toàn** | Sau S0 |
| S2a | `any` trong `amplify/functions` 15 | 56→41 | 1d | **An toàn** | Sau S1 |
| S2b | `any` trong `src` 17 | 41→24 | 1d | **An toàn** | Sau S2a |
| S3-5 | `exhaustive-deps` 6 | 24→18 | 0.5d | **Trung bình** | Đầu S3 |
| S3-1 | `purity` 3 | 18→15 | 0.5d | **Trung bình-Cao** | Sau S3-5 |
| S3-3 | `set-state-in-effect` 5 | 15→10 | 1d | **Cao** | Sau S3-1 |
| S3-2/4 | `immutability` 6 + `refs` 11 | 10→0 | 2d | **Nguy hiểm nhất** | Cuối cùng, tuần tự từng file |

**Tổng:** ~7-8 ngày dev (1 người), chia 7-8 commit vertical slice. Vì xếp theo an toàn nên không song song S3 — mỗi bước là 1 lần thử tay.

---

## 5. Rủi ro & giảm thiểu

- **Rủi ro lớn nhất S3:** Sửa `refs`/`immutability` sai làm `CharacterViewer` kẹt pose hoặc `ChatContext` mất greeting (đã từng revert 14018f7 vì `hasPose` race). Giảm thiểu: mỗi fix có **instrument log** (`console.debug` tạm) và **screenshot test** trước/sau (Playwright headed nếu có, không thì quay màn hình tay).
- **Auth gãy:** `any` trong `pre-sign-up` sai type thì `AdminLinkProviderForUser` im lặng fail, user bị tạo duplicate `b@` (incident 08-08). Giảm thiểu: sau S2a chạy **login thật** trên sandbox (`npx ampx sandbox` nếu có cred, không thì `amplify_outputs.json` local) với 2 case: email mới + Google cùng email.
- **HMR lại gãy:** S1 tách context sai thì `vite --force` mới chạy (handoff đã mất nửa ngày). Giảm thiểu: sau S1 chạy `npm run dev`, sửa 1 dòng trong `ChatContext` xem HMR có nóng không.
- **Job vẫn đỏ sau S3 vì `exhaustive-deps` mới:** Nếu fix hết 24 cũ mà 6 mới còn, job vẫn đỏ. Giảm thiểu: S3 tính cả 6 G ngay từ đầu (tổng 30), không để sót.

---

## 6. Tiêu chí nghiệm thu chung

- [ ] `cd ECA_UI/frontend && npm run lint` → `0 problems` (không còn 81/62)
- [ ] `npm run build` (tsc -b && vite build) xanh, không cảnh báo `dynamically imported ... statically imported` mới
- [ ] `npm run test` 109/109 xanh (hiện tại 109, có thể tăng sau khi thêm test cho S3)
- [ ] `Frontend Build` trên `origin/release` xanh (check GitHub Actions)
- [ ] Thử tay: login email + Google link, đổi VRM 4 model (anne/bronya/miku/miki) greeting đúng, resize mobile↔desktop navbar không mất, motion replay sau refresh còn, HMR sửa context không cần hard-reload

---

## 7. Việc không làm trong plan này

- Bật `tsconfig.app.json: strict` — gate hiện yếu nhưng bật strict là việc riêng (sẽ sinh thêm lỗi TS).
- Tách `lint` thành job không chặn build — chỉ làm nếu Owner chốt chấp nhận lint đỏ lâu dài (handoff §Cảnh báo).
- Bundle 2.1 MB chunk warning — không liên quan lint, để tech-debt riêng.

---

## 8. Câu hỏi cho Owner (cần chốt trước khi code)

1. **Có cho phép `// eslint-disable` có comment cho 2 ca `immutability` intentional (mutate `gl`/`camera`) không, hay bắt buộc refactor để hết rule?** (Ảnh hưởng S3-2)
2. **S2a có được deploy thử lên sandbox Cognito thật không, hay chỉ `tsc` local?** (Máy K trước đây thiếu cred, giờ Owner đã có `244203483654/admin` — cần xác nhận)
3. **Có muốn giữ `lint` chặn `build` trong CI (hiện tại) hay tách ra để build lấy lại tín hiệu trong khi S3 chưa xong?**

---

## 9. Liên kết

- Handoff: `docs/worklogs/29-08-2026-frontend-lint-backlog.md`
- Tech debt: `docs/tracking/tech-debt.md:308` (`npm run lint đỏ: 11 error / 3 warning` — đã lạc hậu, giờ 81)
- Frontend tasks: `docs/tracking/frontend-tasks.md` (branch đã cập nhật `feature/langgraph-rewrite`)
- Workflow: `.github/workflows/release-tests.yml:153` (`frontend-build`)
- Config: `ECA_UI/frontend/eslint.config.js` (hiện chỉ `globalIgnores(['dist'])`)
