# Plan: Animation FSM Refactor — CharacterViewer (v1.2)

**Status:** ✅ **IMPLEMENTED 30/07/2026 bởi K** — build xanh, 17/17 test xanh (§7 + UI + chat).
Xem §12 để biết những điểm code khác plan và **lý do**.
**Ngày tạo:** 30/07/2026 — K | **Revised:** 30/07/2026 — K (v1.1 đối chiếu code thật → v1.2 sửa
boot sequence + chi phí bảo trì)
**Người implement:** N
**Phạm vi:** ~4 files mới + ~4 files sửa trong `ECA_UI/frontend/src/`
**Effort ước tính:** 2–3 ngày

---

## Changelog v1.0 → v1.2

**v1.2 (30/07)**: thêm lỗi #8 (boot sequence) và #9 (chi phí bảo trì khi thêm animation).

| # | Vấn đề trong v1.0 | Mức | Sửa ở |
|---|---|---|---|
| 1 | **FSM deadlock sau mỗi exercise**: `exercise_cooldown: ['idle']` nhưng §5 lại nói "không cần `transitionTo('idle')` vì cùng clip" → state **kẹt vĩnh viễn** ở `exercise_cooldown`; lượt chat sau gọi `thinking_intro` bị ALLOWED chặn → animation thinking chết | 🔴 | §2.1, §2.4, §3.4, §5 |
| 2 | **Thiếu `thinking_loop → exercise`** — chặn đúng luồng chính. Trigger table nói "SSE motion ready → `transitionTo('exercise')`" nhưng `thinking_loop: ['thinking_outro']` only → motion về trong lúc đang thinking (case hay xảy ra nhất) bị **drop im lặng** | 🔴 | §2.2 |
| 3 | **Xoá `selectedMotionId`/`motionOptions` làm mất đường verify Kimodo offline** (dropdown → chọn `motions/generated/motion_*.bvh`). Đây là regression-test path duy nhất cho pipeline NPZ→BVH mà không cần backend | ⚠️ | §4.1, §4.3 |
| 4 | **Động lực "T-pose bug" đã lỗi thời**: code hiện tại ĐÃ play-before-fade ([CharacterViewer.tsx:242-254](../../ECA_UI/frontend/src/components/CharacterViewer.tsx#L242-L254)) + readiness gate `revealed` (fix trong merge 29-30/07). T-pose là **invariant đã có, refactor phải BẢO TOÀN** — không phải bug cần fix | ⚠️ | §0, §7 |
| 5 | **Thiếu `newAction.reset()`** trước `play()`: `exercise` lần 2 sẽ đứng ở frame cuối (LoopOnce đã dừng + `paused`) → motion không chạy lại | ⚠️ | §3.3 |
| 6 | **`registry.invalidate()` không nói khi nào gọi**: clip đã retarget bám skeleton cụ thể → đổi VRM model mà không invalidate thì animation méo | ⚠️ | §3.1, §4.2 |
| 7 | **Thiếu ghi nhận: Facial/Emotion controller không đồng bộ với CharState.** Sau refactor có 2 state machine độc lập (body FSM ↔ facial engaged/idle) → thân "suy nghĩ" mà mặt tự cười, cười lúc demo bài tập, head-follow đè chuyển động đầu của clip | 📝 | **§9 (mới)** — follow-up, ngoài scope PR này |
| 8 | **Boot sequence sai — mất `greeting` lúc mới vào**: trigger table v1.0/v1.1 ghi `Khởi tạo → transitionTo('idle')`, nhưng code hiện tại boot vào **`action_greeting` rồi mới về idle** ([MotionContext.tsx:93-97](../../ECA_UI/frontend/src/contexts/MotionContext.tsx#L93-L97) ưu tiên `action_greeting` làm `defaultMotionId`; `isAction` = true vì URL chứa `action_` → LoopOnce → `finished` → `handleAnimationFinished` fallback về `standard idle`). Implement đúng plan = **lặng lẽ xoá lời chào** — regression UX, không có test nào bắt được | 🔴 | §2.3, §2.5, §7 #9 |
| 9 | **Chi phí bảo trì khi thêm animation vẫn cao** (Owner đặt câu hỏi): plan có **6 map song song** cùng khoá `CharState` → thêm 1 animation phải sửa **4-7 nơi**, trong đó **3 nơi không có lưới an toàn compile-time** (`ALLOWED_TRANSITIONS` reachability viết tay; `ON_FINISHED` là `Partial<>`; `STATIC_ENTRIES` khai `Record<string,…>`). Đây **đúng error class đã sinh ra cả 2 lỗi 🔴 của v1.0** — refactor mà giữ nó thì chỉ đổi hình dạng lỗi chứ không xoá được lỗi | ⚠️ | **§3.0 (mới)**, §2.2, §2.3, §3.1, §3.2, §3.4, §4.3, §9.4, **§11 (mới)** |

---

## 0. Problem Statement

Code hiện tại quyết định animation behavior bằng **string matching trên URL** rải rác 7 vị trí khác nhau:

| Vị trí | Match | Quyết định |
|---|---|---|
| `CharacterViewer.tsx:231` | `action_`, `random_`, `idle_`, `#intro/#outro` | LoopOnce / LoopRepeat |
| `CharacterViewer.tsx:207` | `motion_`, `smplx` | SMPLX vs STANDARD retarget |
| `CharacterViewer.tsx:470` | `generated`, `built-in` | Camera mode |
| `MotionContext.tsx:93` | `action_greeting`, `standard idle` | Default motion |
| `MotionContext.tsx:143` | `#intro`, `#outro` | Finished callback routing |
| `MotionContext.tsx:173` | `idle`, `random_` | Random idle trigger |
| `MotionContext.tsx:155` | `thinking` | Thinking state sequence |

Hậu quả:
- **Fragile**: đổi tên file → behavior thay đổi ngầm, không có lỗi compile nào báo
- **Không scalable**: thêm animation mới = thêm if-else ở nhiều nơi, dễ quên 1 chỗ
- **Camera không đồng bộ**: camera và animation là 2 hệ flag chạy song song, phải tự tay giữ đồng bộ

> **KHÔNG phải động lực** (sửa v1.1 — lỗi #4): T-pose. Code hiện tại **đã** đảm bảo invariant
> "≥1 action luôn drive skeleton": `action.play()` gọi TRƯỚC khi `fadeOut` action cũ
> ([CharacterViewer.tsx:242-254](../../ECA_UI/frontend/src/components/CharacterViewer.tsx#L242-L254)),
> load bất đồng bộ KHÔNG stop action cũ, cộng readiness gate `revealed`/`onReady`.
> Refactor này **phải bảo toàn** invariant đó — đây là tiêu chí regression số 1 (§7 test #3),
> không phải thứ cần tạo mới.

## 1. Target Architecture

```
MotionContext (React — thin glue layer)
    │  transitionTo(state)              animationRegistry.update(key, url)
    ▼                                          │
┌──────────────────────┐              ┌─────────────────────┐
│ AnimationController  │──────────────│ AnimationRegistry   │
│                      │  get(state)  │                     │
│ • FSM transitions    │──────────────│ • Load (glob/cache) │
│ • cross-fade         │  clip        │ • Retarget (SMPLX)  │
│ • ≥1 action invariant│              │ • Subclip (thinking)│
│ • play/pause/speed   │              │ • Dynamic update    │
└──────┬───────────────┘              └─────────────────────┘
       │  stateChanged event
       ▼
┌──────────────────────┐
│  CameraController    │
│                      │
│ • State → camera map │
│ • Cooldown timer     │
│ • Orbit bone follow  │
└──────────────────────┘
```

**Separation of Concern**:

| Module | Biết về | KHÔNG biết về |
|---|---|---|
| `AnimationRegistry` | URL, retarget type, subclip range, cache key | Mixer, fade, state machine, camera |
| `AnimationController` | AnimationClip, fade config, loop mode, mixer | URL, file format, retarget params, subclip logic |
| `CameraController` | CharState, camera offsets, cooldown timer | AnimationClip, mixer, fade, URL, FSM transitions |

## 2. State Machine Definition

### 2.1 States (7 — bỏ `exercise_cooldown` so với v1.0, xem §2.4)

```typescript
type CharState =
  | 'idle'            // Default: Standard Idle, LoopRepeat, camera=head
  | 'greeting'        // One-shot: action_greeting → idle
  | 'bored'           // One-shot: random_* → idle
  | 'thinking_intro'  // Sequence: → thinking_loop
  | 'thinking_loop'   // Sequence: → thinking_outro (khi !isThinking)
  | 'thinking_outro'  // Sequence: → idle
  | 'exercise'        // One-shot: generated motion → idle (camera hips + cooldown)
```

### 2.2 Transition rule (sửa lỗi 🔴 #2)

Thay vì liệt kê tay (v1.0 thiếu `thinking_loop → exercise`), dùng **nguyên tắc** để xoá cả *lớp* lỗi:

1. **State do user/backend điều khiển → reachable từ MỌI state**: `thinking_intro`, `exercise`.
   Lý do: chat message hoặc motion-ready event đến ở **bất kỳ** thời điểm nào. Chặn chúng = drop im
   lặng tương tác của user — chính là lỗi v1.0.
2. **State nội bộ chuỗi → chỉ từ predecessor**: `thinking_loop` ← `thinking_intro`;
   `thinking_outro` ← `thinking_intro` | `thinking_loop` (câu trả lời có thể về trước khi intro xong).
3. **Mọi state đều về được `idle`** — fallback an toàn khi one-shot kết thúc.
4. `exercise → exercise` **được phép**: motion mới đè motion cũ.

```typescript
const USER_DRIVEN: CharState[] = ['thinking_intro', 'exercise']

const ALLOWED_TRANSITIONS: Record<CharState, CharState[]> = {
  idle:            [...USER_DRIVEN, 'greeting', 'bored'],
  greeting:        [...USER_DRIVEN, 'idle'],
  bored:           [...USER_DRIVEN, 'idle'],
  thinking_intro:  [...USER_DRIVEN, 'thinking_loop', 'thinking_outro', 'idle'],
  thinking_loop:   [...USER_DRIVEN, 'thinking_outro', 'idle'],
  thinking_outro:  [...USER_DRIVEN, 'idle'],
  exercise:        [...USER_DRIVEN, 'idle'],
}
```

> ⚠️ **Bảng trên là để ĐỌC, không phải để code** (sửa lỗi #9): khi implement, ma trận này được
> **sinh ra** từ `STATES[to].reach` qua `canTransition()` — xem §3.0. Viết tay bảng này nghĩa là mỗi
> lần thêm state phải nhớ sửa row của mọi state khác; đó chính là cách lỗi 🔴 #2 xảy ra.

> Test bắt buộc (§7 #1): sau một lần `exercise` hoàn tất, `transitionTo('thinking_intro')` **phải
> trả `true`**. Đây chính là case deadlock của v1.0.

### 2.3 Trigger Points (mọi chỗ gọi transitionTo)

| Trigger | Gọi |
|---|---|
| Khởi tạo (VRM ready) | `transitionTo('greeting')` — xem §2.5 |
| Debug panel: chọn built-in animation | `transitionTo('greeting')` / `('bored')` |
| Debug panel: chọn file motion bất kỳ | `playMotionFile(url)` → `registry.update('exercise', url)` + `transitionTo('exercise')` |
| Random idle timer fires | `transitionTo('bored')` |
| Chat: bắt đầu generating | `transitionTo('thinking_intro')` |
| Chat: token đầu tiên / done | `transitionTo('thinking_outro')` |
| Chat SSE: generated motion ready | `registry.update('exercise', url)` + `transitionTo('exercise')` |
| `finished` event trên mixer (one-shot) | `transitionTo(ON_FINISHED[state])` |

> ⚠️ Bảng `ON_FINISHED` dưới đây cũng là **derived view** (sửa lỗi #9): nó nằm trong trường
> `onFinished` của `STATES` (§3.0), và nhờ discriminated union `loop:'once' ⇒ onFinished: CharState`
> thì **không thể** khai báo một one-shot mà quên đích — `Partial<>` như dưới đây thì quên là im lặng,
> hệ quả: animation chạy xong đứng ở frame cuối, FSM kẹt state đó.

```typescript
// One-shot nào xong thì về đâu. State LoopRepeat không có entry.
const ON_FINISHED: Partial<Record<CharState, CharState>> = {
  greeting:       'idle',
  bored:          'idle',
  thinking_intro: 'thinking_loop',
  thinking_outro: 'idle',
  exercise:       'idle',   // camera tự giữ 'hips' thêm COOLDOWN_MS — §3.4
}
```

(v1.0 có trigger `exercise_cooldown camera timer fires → transitionTo('idle')` — bỏ theo §2.4:
camera timer không được phép lái FSM.)

### 2.4 Vì sao bỏ `exercise_cooldown` (sửa lỗi 🔴 #1)

v1.0 tạo state này chỉ để đạt "clip = idle nhưng camera = hips". Nhưng:

- Nó **không khác `idle` về animation** → pseudo-state, không mang thông tin gì cho FSM.
- Chính nó tạo deadlock: §5 của v1.0 nói *"không cần `transitionTo('idle')` vì cùng clip"* → FSM
  ở lại `exercise_cooldown` mãi; state này chỉ cho đi tới `idle` → **mọi tương tác chat sau đó bị chặn**.
- "Giữ camera rộng thêm 3s" là **hành vi theo thời gian**, đúng bản chất là timer trong
  CameraController — không phải một node trong FSM animation.

Sau khi bỏ: `exercise` xong → `idle` (clip idle chạy ngay, crossfade 0.8s), CameraController thấy
"vừa rời `exercise`" → giữ `hips` thêm `COOLDOWN_MS` rồi về `head`. **User thấy y hệt v1.0**, nhưng
không còn state giả và không còn deadlock.

### 2.5 Boot sequence: greeting → idle (sửa lỗi 🔴 #8)

**Hành vi hiện tại — đã verify trong code, PHẢI bảo toàn**: mới vào app, avatar **chào trước** rồi
mới sang idle.

| Bước | Code hiện tại | Cơ chế |
|---|---|---|
| 1 | [MotionContext.tsx:93-97](../../ECA_UI/frontend/src/contexts/MotionContext.tsx#L93-L97) | `defaultMotionId` ưu tiên `action_greeting`, chỉ fallback `standard idle` nếu không tìm thấy |
| 2 | [CharacterViewer.tsx:231-237](../../ECA_UI/frontend/src/components/CharacterViewer.tsx#L231-L237) | URL chứa `action_` → `isAction` = true → `LoopOnce` + `clampWhenFinished` |
| 3 | [CharacterViewer.tsx:168-173](../../ECA_UI/frontend/src/components/CharacterViewer.tsx#L168-L173) | mixer `finished` → `onAnimationFinishedRef.current()` |
| 4 | [MotionContext.tsx:149-150](../../ECA_UI/frontend/src/contexts/MotionContext.tsx#L149-L150) | nhánh cuối `handleAnimationFinished` → `standard idle` |

FSM tương ứng — **không cần state mới**, `greeting` + `ON_FINISHED['greeting'] = 'idle'` đã đủ:

```typescript
// AnimationController: currentState khởi tạo = 'idle' nhưng CHƯA có clip nào
// (currentAction = null). 'idle' ở đây là gốc của ALLOWED_TRANSITIONS, không phải
// một clip đang chạy — nhờ vậy transitionTo('greeting') hợp lệ ngay từ frame đầu.
private currentState: CharState = 'idle'
private currentAction: THREE.AnimationAction | null = null
```

```typescript
// Boot, gọi 1 lần khi VRM ready (§4.2) — KHÔNG phải transitionTo('idle')
const ok = await controller.transitionTo('greeting')
if (!ok) await controller.transitionTo('idle')   // ← bắt buộc, xem dưới
```

Ba điểm bắt buộc, thiếu là hỏng:

1. **Fallback không được bỏ.** `transitionTo` trả `false` khi `registry.get()` fail (§3.3 bước 5).
   Boot mà không fallback → `currentAction` vẫn `null` → **không action nào drive skeleton → T-pose**,
   đúng cái invariant §0 phải bảo toàn. Code hiện tại miễn nhiễm vì `defaultMotionId` fallback ngay ở
   tầng chọn asset; FSM chuyển việc đó sang tầng transition nên phải viết lại tay.
2. **Không greet lại khi đổi VRM model.** Controller được tạo/dispose theo effect `[vrm]` (§4.2), nên
   boot naive sẽ chào lại mỗi lần user đổi model — code hiện tại **không** như vậy (đổi model chỉ
   reload clip của state đang chạy, `selectedMotionId` không đổi). Giữ một ref ngoài effect:
   ```typescript
   const hasGreetedRef = useRef(false)   // sống qua các lần đổi model
   const boot = hasGreetedRef.current ? 'idle' : 'greeting'
   hasGreetedRef.current = true
   ```
   > StrictMode double-mount (React 19) cũng được ref này chặn luôn — nếu không, avatar chào 2 lần
   > trong dev.
3. **Chào không được chặn tương tác.** `greeting: [...USER_DRIVEN, 'idle']` (§2.2) đã cho phép user
   gõ chat ngay khi avatar còn đang chào → `thinking_intro` cắt ngang. Đây là hành vi đúng, đừng
   "sửa" thành chờ chào xong.

## 3. New Files

### 3.0 `lib/AnimationStates.ts` — SINGLE SOURCE OF TRUTH (sửa lỗi ⚠️ #9)

> **Vì sao có mục này**: v1.1 (trước sửa) mô tả **6 map song song** cùng khoá bằng `CharState`:
> `ALLOWED_TRANSITIONS`, `ON_FINISHED`, `STATIC_ENTRIES`, `TRANSITIONS`, `WIDE_STATES`,
> `STATE_OPTIONS` (+ `FACIAL_POLICY` ở §9). Thêm **một** animation = sửa **4-7 nơi**, và **3 trong số
> đó không có lưới an toàn compile-time** → đúng cái error class đã sinh ra cả 2 lỗi 🔴 của v1.0.
> Mục này gộp tất cả thành **một bảng**; các map ở §2.2/§2.3/§3.1/§3.2/§3.4/§4.3/§9.4 giữ lại trong
> plan để **giải thích**, nhưng khi code chúng là **derived view**, không phải nguồn dữ liệu.

```typescript
type Reach =
  | 'anytime'                    // user/backend lái → vào được từ MỌI state (§2.2 rule 1)
  | 'from-idle'                  // chỉ khi avatar đang rảnh (greeting, bored)
  | { after: CharState[] }       // mắt trong chuỗi → chỉ từ predecessor (§2.2 rule 2)

interface StateBase {
  /** Nguồn clip. 'dynamic' = do registry.update() nạp lúc runtime (chỉ 'exercise'). */
  source:
    | { match: RegExp; retarget: 'smplx' | 'standard' | 'mixamo'; isFbx: boolean
        subclip?: { name: string; start: number; end: number; fps: number } }
    | 'dynamic'
  /** Ai được vào state này. Khai báo TRÊN state đích — xem "vì sao" bên dưới. */
  reach: Reach
  camera: 'head' | 'hips'
  facial: { wander: boolean; hold?: CanonicalEmotion }   // §9.4
  /** Có giá trị = hiện trong debug dropdown. Không có = không chọn tay được. */
  debugLabel?: string
  /** Cross-fade khi RỜI state này (giây). Default 0.3. */
  crossfade?: number
  /** Escape hatch khi một cặp FROM→TO cần khác: đè `crossfade`. */
  crossfadeTo?: Partial<Record<CharState, number>>
}

// Discriminated union: LoopOnce BẮT BUỘC có đích, LoopRepeat BẮT BUỘC không có.
// → "one-shot mà quên onFinished" (state kẹt ở frame cuối) thành LỖI COMPILE.
type StateDef = StateBase &
  ({ loop: 'once'; onFinished: CharState } | { loop: 'repeat'; onFinished: null })

// Record (KHÔNG Partial, KHÔNG Record<string,…>) → thêm CharState mà quên khai báo
// = lỗi compile ngay tại đây. Compiler dẫn đường thay cho checklist trong đầu.
export const STATES: Record<CharState, StateDef> = {
  idle: {
    source: { match: /standard idle/i, retarget: 'standard', isFbx: true },
    loop: 'repeat', onFinished: null,
    reach: 'anytime', camera: 'head', facial: { wander: true },
    debugLabel: 'Standard Idle',
    autoAfterIdle: [60, 120],            // → 'bored', xem dưới
  },
  greeting: {
    source: { match: /action_greeting/i, retarget: 'standard', isFbx: false },
    loop: 'once', onFinished: 'idle',
    reach: 'from-idle', camera: 'head', facial: { wander: true },
    debugLabel: 'Action Greeting', crossfade: 0.5,
  },
  bored: {
    source: { match: /random_bored/i, retarget: 'standard', isFbx: true },
    loop: 'once', onFinished: 'idle',
    reach: 'from-idle', camera: 'head', facial: { wander: true },
    debugLabel: 'Random Bored', crossfade: 0.5,
  },
  thinking_intro: {
    source: { match: /thinking/i, retarget: 'standard', isFbx: true,
              subclip: { name: 'intro', start: 0, end: 38, fps: 30 } },
    loop: 'once', onFinished: 'thinking_loop',
    reach: 'anytime', camera: 'head', facial: { wander: false, hold: 'neutral' },
  },
  thinking_loop: {
    source: { match: /thinking/i, retarget: 'standard', isFbx: true,
              subclip: { name: 'loop', start: 38, end: 75, fps: 30 } },
    loop: 'repeat', onFinished: null,
    reach: { after: ['thinking_intro'] },
    camera: 'head', facial: { wander: false, hold: 'neutral' },
  },
  thinking_outro: {
    source: { match: /thinking/i, retarget: 'standard', isFbx: true,
              subclip: { name: 'outro', start: 75, end: 127, fps: 30 } },
    loop: 'once', onFinished: 'idle',
    reach: { after: ['thinking_intro', 'thinking_loop'] },
    camera: 'head', facial: { wander: false, hold: 'neutral' },
  },
  exercise: {
    source: 'dynamic',                   // registry.update('exercise', url)
    loop: 'once', onFinished: 'idle',
    reach: 'anytime', camera: 'hips', facial: { wander: false, hold: 'neutral' },
    crossfade: 0.8,
  },
}
```

**Reachability là hàm, không phải bảng** — đây là phần xoá được cả *lớp* lỗi:

```typescript
export function canTransition(from: CharState, to: CharState): boolean {
  if (to === 'idle') return true                        // §2.2 rule 3
  const r = STATES[to].reach
  if (r === 'anytime') return true                      // rule 1 (+ rule 4: exercise→exercise)
  if (r === 'from-idle') return from === 'idle'
  return r.after.includes(from)                         // rule 2
}
```

> Vì `reach` khai báo **trên state được vào**, thêm một state mới **không thể** làm hỏng
> reachability của state khác, và **không phải sửa row của ai khác**. Lỗi 🔴 #2 của v1.0 (quên nhét
> `exercise` vào array của `thinking_loop`) **không còn tồn tại được về mặt cấu trúc**.
>
> Đã đối chiếu: `canTransition` sinh ra **đúng y** ma trận `ALLOWED_TRANSITIONS` ở §2.2 (khác duy
> nhất `idle→idle` giờ hợp lệ — no-op vô hại, `prev === newAction` ở §3.3 bước 12 đã xử lý).

**Mọi thứ khác là derived** — không map nào phải bảo trì bằng tay:

```typescript
export const onFinishedOf  = (s: CharState) => STATES[s].onFinished          // thay ON_FINISHED §2.3
export const loopModeOf    = (s: CharState) => STATES[s].loop                // thay string-match `action_`/`random_`/`idle_` (§0 dòng 1)
export const cameraModeOf  = (s: CharState) => STATES[s].camera              // thay WIDE_STATES §3.4
export const facialOf      = (s: CharState) => STATES[s].facial              // thay FACIAL_POLICY §9.4
export const staticSourceOf = (s: CharState) =>
  STATES[s].source === 'dynamic' ? null : STATES[s].source                   // thay STATIC_ENTRIES §3.1

export function crossfadeFor(from: CharState, to: CharState): number {       // thay TransitionConfig.ts §3.2
  return STATES[from].crossfadeTo?.[to] ?? STATES[from].crossfade ?? 0.3
}

export const STATE_OPTIONS = (Object.entries(STATES) as [CharState, StateDef][])
  .filter(([, d]) => d.debugLabel)
  .map(([id, d]) => ({ id, label: d.debugLabel! }))                          // thay STATE_OPTIONS §4.3
```

**Trigger tự động theo thời gian** — dạng trigger duy nhất hiện có là "idle lâu → bored"
([MotionContext.tsx:169-188](../../ECA_UI/frontend/src/contexts/MotionContext.tsx#L169-L188)), nên
khai báo được luôn, không cần code riêng:

```typescript
autoAfterIdle?: [minSec: number, maxSec: number]   // trên StateBase; timer chạy khi ĐANG ở state đó
// idle: autoAfterIdle: [60, 120] → sau 60-120s random → transitionTo(<state 'from-idle' đầu tiên>)
```

> **Không** tổng quát hoá xa hơn thế. Trigger từ **event ngoài** (chat bắt đầu/kết thúc, SSE motion
> ready) mỗi cái một ngữ cảnh khác nhau — giữ là code tường minh ở call-site, nhưng **gom hết vào
> `hooks/useFsmTriggers.ts`** để sau này tìm được bằng một lần mở file, thay vì rải 7 nơi như hiện tại.

### 3.1 `lib/AnimationRegistry.ts`

**Trách nhiệm**: Load → Cache → Retarget → Subclip → Trả về `AnimationClip` hoàn chỉnh.

AnimationController **không biết** clip đến từ đâu, URL là gì, có cần subclip không, retarget kiểu gì. Registry là asset layer duy nhất.

```typescript
export class AnimationRegistry {
  /** 
   * Lấy AnimationClip cho state. Trả về null nếu state chưa được đăng ký.
   * Registry xử lý: load từ URL, retarget (SMPLX/STANDARD), subclip, cache.
   */
  async get(state: CharState): Promise<THREE.AnimationClip | null>

  /** 
   * Cập nhật / đăng ký động clip cho một state. 
   * Dùng cho generated motion — gọi khi SSE trả về motion URL mới.
   */
  async update(state: CharState, url: string): Promise<void>

  /**
   * Hủy tất cả cache entries. PHẢI gọi khi đổi VRM model (sửa lỗi #6):
   * clip đã retarget bám skeleton cụ thể của model đó — dùng lại cho model khác
   * sẽ méo/lệch bone. Gọi từ effect có dep [vrmUrl] trong CharacterViewer (§4.2).
   */
  invalidate(): void
}
```

**Internal implementation**:

> ⚠️ `STATIC_ENTRIES` dưới đây là **derived view** (sửa lỗi #9): dữ liệu thật nằm ở `STATES[s].source`
> (§3.0). Lưu ý kiểu `Record<string, …>` viết dưới đây **mất hết** kiểm tra compile — thêm state mà
> quên entry thì chỉ vỡ lúc runtime (`registry.get()` trả `null` → `transitionTo` trả `false` → im
> lặng không có animation). `Record<CharState, …>` trong §3.0 bắt lỗi ngay lúc build.

```typescript
// Static map: label → { url, retarget, isFbx, subclip? }
// Built từ import.meta.glob một lần khi constructor.
const STATIC_ENTRIES: Record<string, {
  label: string
  url: string
  retarget: 'smplx' | 'standard' | 'mixamo'
  isFbx: boolean
}> = {
  'idle':             { label: /standard idle/i,   url: '', retarget: 'standard', isFbx: true },
  'greeting':         { label: /action_greeting/i, url: '', retarget: 'standard', isFbx: false },
  'bored':            { label: /random_bored/i,    url: '', retarget: 'standard', isFbx: true },
  'thinking_intro':   { label: /thinking/i,        url: '', retarget: 'standard', isFbx: true,
                        subclip: { name: 'intro', start: 0, end: 38, fps: 30 } },
  'thinking_loop':    { label: /thinking/i,        url: '', retarget: 'standard', isFbx: true,
                        subclip: { name: 'loop', start: 38, end: 75, fps: 30 } },
  'thinking_outro':   { label: /thinking/i,        url: '', retarget: 'standard', isFbx: true,
                        subclip: { name: 'outro', start: 75, end: 127, fps: 30 } },
}

// Dynamic cache: state → Promise<AnimationClip>. Chỉ 'exercise' set qua update()
// (generated motion / debug file). exercise_cooldown đã bị bỏ — §2.4.
const dynamicCache = new Map<CharState, Promise<THREE.AnimationClip | null>>()
```

**`get(state)` flow**:

```
1. Check dynamic cache → nếu có, return clip
2. Look up STATIC_ENTRIES[state] → lấy { url, retarget, isFbx, subclip }
3. Load: isFbx ? loadMixamoAnimation(url, vrm) : loadAndRetargetBVH(url, vrm, retarget)
4. Cache qua getAnimationClip(key, factory) — dedup concurrent loads
5. Nếu có subclip range → AnimationUtils.subclip(clip, ...)
6. Trả về AnimationClip
```

**`update(state, url)` flow**:

```
1. Xóa cache cũ của state
2. Load clip từ url với SMPLX retarget
   (generated motion luôn là BVH SMPL-X — output của scripts/kimodo_npz_to_bvh.py)
3. Cache vào dynamicCache
4. Dùng cho state 'exercise' (cả SSE motion lẫn debug file selector — §4.3)
```

### 3.2 ~~`lib/TransitionConfig.ts`~~ → gộp vào §3.0 (sửa lỗi #9)

> **File này KHÔNG còn tồn tại** sau sửa #9: mọi override trong bảng dưới đều có dạng `X → idle` với
> `fadeOut === fadeIn`, nên rút gọn thành **một số** `crossfade` trên state nguồn trong `STATES`
> (+ `crossfadeTo` cho trường hợp cần phân biệt theo đích). `crossfadeFor(from, to)` ở §3.0 thay
> `getTransition()`. Giữ mục này để thấy dữ liệu gốc và lý do chọn từng con số.

Fade time phụ thuộc FROM→TO, không chỉ TO.

```typescript
export interface TransitionSpec {
  fadeOut: number    // seconds to fade out FROM action
  fadeIn: number     // seconds to fade in TO action
}

const TRANSITIONS: Partial<Record<CharState, Partial<Record<CharState, TransitionSpec>>>> = {
  // Default: { fadeOut: 0.3, fadeIn: 0.3 } cho mọi transition không được định nghĩa
  exercise: {
    idle: { fadeOut: 0.8, fadeIn: 0.8 },  // motion dài → về idle mượt (v1.0: → exercise_cooldown)
  },
  greeting: {
    idle: { fadeOut: 0.5, fadeIn: 0.5 },
  },
  bored: {
    idle: { fadeOut: 0.5, fadeIn: 0.5 },
  },
}

const DEFAULT_TRANSITION: TransitionSpec = { fadeOut: 0.3, fadeIn: 0.3 }

export function getTransition(from: CharState, to: CharState): TransitionSpec {
  return TRANSITIONS[from]?.[to] ?? DEFAULT_TRANSITION
}
```

### 3.3 `lib/AnimationController.ts`

**Trách nhiệm**: Nhận `AnimationClip` từ Registry → tạo `AnimationAction` → cross-fade → play. **Không biết gì về URL, load, retarget, subclip, file format.**

```
AnimationClip (từ registry.get)
        ↓
mixer.clipAction(clip)
        ↓
newAction.play()          ← INVARIANT: play TRƯỚC khi động vào action cũ
        ↓
fadeOut old / fadeIn new  ← TransitionConfig lookup
        ↓
stop old (khi weight → 0)
```

KHÔNG BAO GIỜ: Stop old → Load → Play new (gap → T-pose).

```typescript
export class AnimationController {
  /** 
   * Chuyển sang state mới. 
   * 1. Validate ALLOWED_TRANSITIONS
   * 2. registry.get(next) → AnimationClip (đã load, cache, subclip sẵn)
   * 3. mixer.clipAction(clip)
   * 4. newAction.play() TRƯỚC — invariant guarantee
   * 5. Cross-fade (TransitionConfig FROM→TO)
   * 6. Update currentState
   */
  async transitionTo(next: CharState): Promise<boolean>

  /** 
   * Every frame: retire faded actions, update mixer.
   * Gọi từ useFrame() của R3F.
   */
  update(delta: number): void

  /** 
   * Play/pause/speed proxy. 
   * Pause = DEBUG-ONLY — stops action, restores bind pose.
   */
  setPlaying(playing: boolean): void
  setSpeed(speed: number): void

  /** 
   * Event emitter: 'stateChanged'(newState), 'finished'(completedState).
   * 'finished' fires khi action LoopOnce chạy xong → FSM listener tự gọi transitionTo next.
   */
  on(event: 'stateChanged' | 'finished', cb: (state: CharState) => void): () => void

  /** Cleanup mixer + all actions. */
  dispose(): void
}
```

**Internal state** (private, không expose ra ngoài):

```typescript
private mixer: THREE.AnimationMixer
private registry: AnimationRegistry
private currentState: CharState
private currentAction: THREE.AnimationAction | null
private fadingActions: THREE.AnimationAction[]
private loadGen: number          // cancel stale transitions
private listeners: Map<string, Set<Function>>
```

**`transitionTo` implementation detail**:

```
1.  if (!ALLOWED_TRANSITIONS[currentState]?.includes(next)) → return false
2.  loadGen++
3.  const gen = loadGen
4.  const clip = await registry.get(next)
5.  if (!clip || gen !== loadGen) → return false (superseded or failed)
6.  const newAction = mixer.clipAction(clip)
7.  Set loop: LoopRepeat (idle, thinking_loop) / LoopOnce (others), 1 rep
8.  newAction.reset()                          // ← sửa lỗi #5, xem note dưới
9.  newAction.play()                           // ← INVARIANT: action sống trước khi fade
10. const transition = getTransition(currentState, next)
11. const prev = currentAction
12. if (prev && prev !== newAction) {
      if (prev.isRunning()) {
        prev.fadeOut(transition.fadeOut)
      } else {
        // LoopOnce đã dừng ở frame cuối — restore weight để cross-fade mượt
        prev.setEffectiveWeight(1)
        prev.fadeOut(transition.fadeOut)
      }
      newAction.fadeIn(transition.fadeIn)
      fadingActions.push(prev)
    } else if (!prev) {
      newAction.setEffectiveWeight(1)          // action đầu tiên: full weight ngay,
    }                                          // fade-in từ 0 sẽ hở bind pose
13. currentAction = newAction
14. currentState = next
15. emit('stateChanged', next)
16. return true
```

> ⚠️ **`newAction.reset()` ở bước 8 (sửa lỗi #5 — v1.0 thiếu)**: `mixer.clipAction(clip)` trả về
> **cùng một instance** cho cùng clip. Nếu action đó từng chạy xong dạng LoopOnce (ví dụ `exercise`
> lần 2, hoặc `greeting` lần 2), nó đang đứng ở frame cuối với `paused = true` — `play()` **không tự
> rewind**, animation sẽ không chạy lại. `reset()` đưa time về 0 + clear `paused` + restore weight.
>
> ⚠️ Nhánh `else if (!prev)` ở bước 12: chỉ có ý nghĩa cho action **đầu tiên** sau khi mount. Không
> có nhánh này thì `fadeIn` chạy từ weight 0 → hở bind pose đúng khoảng fade (chính là bug
> `revealed`/`onReady` đang chặn ở code hiện tại — phải giữ).

**`setPlaying` / `setSpeed`** (debug controls, giữ lại từ code cũ nhưng đơn giản hơn):

```typescript
setPlaying(playing: boolean) {
  if (playing) {
    const a = this.currentAction
    if (a && !a.isRunning()) { a.reset(); a.play() }
    else if (a) a.paused = false
  } else {
    this.currentAction?.stop()  // DEBUG ONLY
  }
}
```

### 3.4 `lib/CameraController.ts`

Camera hoàn toàn độc lập với animation. Chỉ react với state change event.

```typescript
export type CameraMode = 'head' | 'hips'

/** State cần khung rộng để thấy toàn thân. → derived: cameraModeOf(s) === 'hips' (§3.0). */
const isWide = (s: CharState) => cameraModeOf(s) === 'hips'
/** Giữ khung rộng thêm bao lâu sau khi RỜI wide state (thay exercise_cooldown của v1.0). */
const COOLDOWN_MS = 3000

export class CameraController {
  private currentState: CharState
  private cameraMode: CameraMode
  private cooldownTimer: ReturnType<typeof setTimeout> | null
  private onModeChanged: (mode: CameraMode) => void

  onStateChanged(next: CharState): void { /* ... */ }
  update(delta: number): void   // orbit target follow bone — giữ logic hiện có
  dispose(): void               // clear timer
}
```

**Logic** (sửa lỗi 🔴 #1 — thay `delays` map của v1.0):

1. `next ∈ WIDE_STATES` → set `hips` **ngay**, và **clear** cooldown timer nếu đang chạy.
2. **Rời** wide state (`prev ∈ WIDE && next ∉ WIDE`) → **giữ `hips`**, start timer `COOLDOWN_MS`;
   timer fire → set `head`.
3. `next ∉ WIDE_STATES` và không phải vừa rời wide → set `head` ngay.
4. State đổi trước khi timer fire, hoặc `dispose()` → clear timer (không set camera sau unmount).

Nhờ vậy `exercise → idle` vẫn giữ khung rộng 3s rồi mới về mặt — **đúng UX mà v1.0 muốn**, nhưng
camera timer không còn lái FSM nên không tạo được deadlock.

> Khác biệt then chốt với v1.0: timer ở đây chỉ đổi **camera mode**, tuyệt đối không gọi
> `transitionTo`. Một hệ chỉ được có một nguồn lái state.

## 4. Files to Modify

### 4.1 `contexts/MotionContext.tsx` — Simplified

**Remove:**
- `selectedMotionId` / `setSelectedMotionId` (replaced by FSM state)
- `cameraMode` / `setCameraMode` (moved to CameraController)
- `handleAnimationFinished` callback (replaced by FSM internal transitions)
- `isThinking` state + thinking sequence logic (replaced by FSM states)
- Random idle timer logic (FSM handles via `onFinished`)
- `isPlaying`/`speed` stay (debug controls)

**New:**
```typescript
interface MotionContextType {
  /** FSM — single entry point for all state changes. Trả false nếu transition bị chặn. */
  transitionTo: (state: CharState) => Promise<boolean>

  /** Animation registry — dùng để thay clip động (exercise) */
  animationRegistry: AnimationRegistry

  /** Current state + camera (read-only, cập nhật qua stateChanged event) */
  currentState: CharState
  cameraMode: CameraMode

  /** Controller refs (for useFrame in CharacterViewer) */
  animationControllerRef: React.MutableRefObject<AnimationController | null>
  cameraControllerRef: React.MutableRefObject<CameraController | null>

  /** Debug controls */
  isPlaying: boolean; setIsPlaying: (v: boolean) => void
  speed: number; setSpeed: (v: number) => void

  /** Debug dropdown: các FSM state chọn được bằng tay */
  stateOptions: { id: CharState; label: string }[]

  /**
   * Debug: danh sách FILE motion thật (import.meta.glob) + cách phát chúng.
   * GIỮ LẠI (sửa lỗi ⚠️ #3) — đây là đường verify retarget SMPL-X của motion
   * generated (Kimodo NPZ→BVH) mà không cần backend. Xem §4.3.
   */
  motionFileOptions: { id: string; label: string; url: string }[]
  playMotionFile: (url: string) => Promise<void>

  /** Unchanged */
  vrmOptions: AssetOption[]
  selectedVrmId: string; setSelectedVrmId: (id: string) => void
  avatarRef: React.MutableRefObject<AvatarController | null>
  isMusicPlaying: boolean; toggleMusic: () => void
}
```

`playMotionFile(url)` = `await registry.update('exercise', url)` rồi `await transitionTo('exercise')`.

**Random-idle timer** (v1.0 nói "FSM handles via onFinished" — không đúng, `onFinished` chỉ chạy khi
one-shot kết thúc, còn `idle` là LoopRepeat nên không bao giờ fire): giữ dưới dạng effect trong
MotionContext, nghe `currentState === 'idle'` → sau 60–120s random gọi `transitionTo('bored')`.
Hành vi giống hiện tại, chỉ khác là dựa trên **state** thay vì string-match trên label.

**Usage from ChatPanel**:

```typescript
// Thay setIsThinking(true):
transitionTo('thinking_intro')

// Thay setIsThinking(false):
transitionTo('thinking_outro')

// Khi SSE trả về motion URL:
await animationRegistry.update('exercise', motionUrl)
await transitionTo('exercise')
```

### 4.2 `components/CharacterViewer.tsx` — Major simplification

**VRMCharacter component**: Được thay bằng `AnimationController` instance.

**Remove:**
- Animation refs: `mixerRef`, `actionRef`, `fadingOutRef`, `loadGenRef`, `isPlayingRef`,
  `onAnimationFinishedRef`, `animLoaded` state
- Entire `applyAnimation` useEffect (~90 lines) + cả 3 string-match ở dòng 207 / 231 / 470
- `handleFinished` event listener
- FadeOut retirement logic in useFrame

**GIỮ NGUYÊN — không refactor luôn** (ngoài scope, sửa vào là dễ gây regression):
- `restPoses` cache (retargeter cần bind pose thuần)
- `disposeVRM` + StrictMode `disposeGuardRef` pattern
- **Readiness gate `revealed` / `onReady`** — đây là thứ chặn T-pose lúc mount đầu tiên (§0)
- `AvatarController` (facial/eye/head) lifecycle + `applyRestPose`

**Thêm** (sửa lỗi #6): effect `useEffect(() => registry.invalidate(), [vrmUrl])` — clip đã retarget
không dùng lại được cho model khác.

**New VRMCharacter logic**:
```typescript
function VRMCharacter({ vrmUrl, modelId, isPlaying, speed, animationControllerRef, ... }) {
  // 1. Load VRM (giữ nguyên)
  // 2. Create AnimationController instance on mount
  const controller = new AnimationController(vrm, registry)
  animationControllerRef.current = controller
  
  // 3. useFrame — THỨ TỰ BẮT BUỘC, giữ đúng như code hiện tại:
  //    body (mixer) → face/eye/head (setValue + bone) → vrm.update apply tất cả.
  //    HeadController ghi bone neck/head SAU mixer nên vẫn thắng body animation.
  useFrame((_, delta) => {
    controller.update(delta)                  // body: mixer.update + retire faded actions
    avatarControllerRef.current?.tick(delta)  // facial + eye + head follow
    vrm.update(delta)
  })
  
  // 4. isPlaying / speed → delegate to controller
  useEffect(() => { controller.setPlaying(isPlaying); controller.setSpeed(speed) }, [isPlaying, speed])
  
  // 5. Teardown: controller.dispose()
  return () => controller.dispose()
}
```

**Scene component**: Tương tự, dùng CameraController.

```typescript
function Scene({ ... }) {
  // Camera logic uses CameraController
  const cameraController = new CameraController(camera, controlsRef, DEFAULT_CAMERA_POLICY)
  cameraControllerRef.current = cameraController
  
  useFrame((_, delta) => {
    cameraController.update(delta)
  })
  
  return () => cameraController.dispose()
}
```

### 4.3 `components/panels/MotionControlPanel.tsx` — FSM + debug file path (sửa lỗi ⚠️ #3)

**Hai dropdown, giữ CẢ HAI** — v1.0 chỉ có (1) nên mất đường verify Kimodo:

```typescript
// (1) State selector — đường chính, map FSM state.
// KHÔNG viết tay danh sách này (sửa lỗi #9): import STATE_OPTIONS derived từ STATES (§3.0).
// Thêm `debugLabel` vào một state là nó tự xuất hiện ở dropdown — 0 dòng sửa ở đây.
import { STATE_OPTIONS } from '../../lib/AnimationStates'
<select onChange={(e) => transitionTo(e.target.value as CharState)}>...</select>

// (2) Motion FILE selector — DEBUG. Phát một file BVH/FBX bất kỳ qua state 'exercise'.
<select onChange={(e) => playMotionFile(e.target.value)}>
  {motionFileOptions.map(o => <option key={o.id} value={o.url}>{o.label}</option>)}
</select>
```

**Lý do bắt buộc giữ (2)**: quy trình verify Kimodo offline hiện tại là chọn
`motions/generated/motion_*.bvh` từ dropdown rồi xem avatar có retarget đúng không (SMPL-X mirror +
joint mapping cực dễ sai, và đây là cách duy nhất kiểm tra **không cần backend/GPU**). Bỏ nó đi là
mất luôn regression-test path cho pipeline `kimodo_npz_to_bvh.py` → frontend. Xem §7 test #4.

### 4.4 `components/ChatPanel.tsx` — Uses FSM

```typescript
// Thay setIsThinking(true)  — ChatPanel.tsx:69
transitionTo('thinking_intro')      // khi bắt đầu generating
// Thay setIsThinking(false) — ChatPanel.tsx:105, 133, 142
transitionTo('thinking_outro')      // khi token đầu tiên đến / done / error

// Khi SSE trả motion URL — PHẢI update registry trước, không chỉ transitionTo
await animationRegistry.update('exercise', motionUrl)
await transitionTo('exercise')
```

`isThinking` biến mất khỏi MotionContext — ChatPanel gọi FSM trực tiếp.

## 5. Data Flow: Generated Motion (exercise)

```
Backend SSE → motion_file_url (e.g. /motions/generated/motion_UUID.bvh)
        ↓
// 1. Đăng ký clip động vào Registry
await registry.update('exercise', motionUrl)
        ↓
// 2. Chuyển state một lần duy nhất
await animationController.transitionTo('exercise')
        ↓
// Registry.get('exercise') → load BVH (SMPLX retarget) → AnimationClip
// AnimationController: LoopOnce, fadeIn=0.3
// CameraController: ngay lập tức mode='hips'
        ↓
Mixer 'finished' event (exercise kết thúc)
        ↓
// FSM listener: ON_FINISHED['exercise'] = 'idle' → transitionTo('idle')
// AnimationController: crossfade exercise→idle = fadeOut 0.8 / fadeIn 0.8
// CameraController: prev ∈ WIDE, next ∉ WIDE → GIỮ 'hips', start COOLDOWN_MS timer
        ↓
3s timer fires
        ↓
// CameraController: set mode='head'. FSM đã ở 'idle' từ đầu bước trên.
```

**Khác biệt cốt lõi so với v1.0** (sửa lỗi 🔴 #1): FSM về `idle` **ngay khi motion xong**, nên mọi
tương tác tiếp theo (chat → `thinking_intro`, hoặc motion mới → `exercise`) đều hoạt động. v1.0 kẹt
ở `exercise_cooldown` và chặn hết.

Camera vẫn giữ khung rộng đủ 3s sau khi motion kết thúc — hành vi user thấy không đổi.

## 6. Implementation Order

| Step | File | Description | Effort |
|---|---|---|---|
| **0** | **`lib/AnimationStates.ts`** | **Bảng `STATES` + `canTransition` + derived getters (§3.0). Làm TRƯỚC — mọi step sau import từ đây** | **1h** |
| 1 | `lib/AnimationRegistry.ts` | Load/cache/retarget/subclip; đọc `staticSourceOf()` thay vì tự giữ map | 1h |
| ~~2~~ | ~~`lib/TransitionConfig.ts`~~ | **Bỏ file** — `crossfadeFor()` nằm trong §3.0 | ~~30m~~ |
| 3 | `lib/AnimationController.ts` | Core FSM + invariant enforcement | 3h |
| 4 | `lib/CameraController.ts` | State-reactive camera | 1h |
| 5 | `contexts/MotionContext.tsx` | Simplify, wire up FSM + CameraController | 2h |
| 6 | `components/CharacterViewer.tsx` | Rip out old animation code, use new controllers | 3h |
| 7 | `components/panels/MotionControlPanel.tsx` | State selector + **debug file selector** (§4.3) | 45m |
| 8 | `components/ChatPanel.tsx` | Replace setIsThinking + motion triggers | 30m |
| 9 | Test + debug | Checklist §7 | 2.5h |

Total: ~14.5h ≈ 2-3 working days (thêm step 0 1h, bỏ step 2 30m — coi như không đổi; lợi ích của
§3.0 nằm ở **chi phí thêm animation về sau**, không phải ở effort lần này — xem §11).

## 7. Test checklist (gate — phải xanh hết trước khi coi là xong)

| # | Test | Kỳ vọng |
|---|---|---|
| 1 | **Deadlock regression (lỗi v1.0 #1)**: `exercise` → chờ `finished` → `transitionTo('thinking_intro')` | trả `true`, thinking chạy. Đây là case v1.0 chết |
| 2 | **Motion trong lúc thinking (lỗi v1.0 #2)**: đang `thinking_loop` → `transitionTo('exercise')` | trả `true`, motion chạy |
| 3 | **T-pose invariant** (§0): spam đổi state ~10 lần/giây; đổi VRM model giữa lúc đang fade | không frame nào thấy bind pose |
| 4 | **Kimodo retarget (lỗi v1.0 #3)**: debug file selector → chọn `motions/generated/motion_*.bvh` | avatar chạy đúng động tác, clip info 22 tracks |
| 5 | **`reset()` fix (lỗi v1.0 #5)**: chạy `exercise` 2 lần liên tiếp cùng 1 file | lần 2 chạy lại từ đầu, không đứng ở frame cuối |
| 6 | Camera: `exercise` → `idle` | `hips` suốt exercise + 3s sau đó, rồi mới `head` |
| 7 | **`invalidate()` fix (lỗi v1.0 #6)**: đổi VRM model rồi phát lại animation | không méo/lệch bone |
| 8 | Chat end-to-end | thinking intro→loop→outro→idle mượt, không giật |
| 9 | **Boot sequence (lỗi #8)**: reload trang | avatar chạy `action_greeting` **một lần** rồi crossfade sang idle — không vào idle luôn, không chào 2 lần (StrictMode), không T-pose nếu clip greeting lỗi (giả lập: đổi tên file greeting) |
| 10 | **Không greet lại khi đổi model (§2.5 #2)**: đang idle → đổi VRM sang model khác | vẫn idle, **không** chào lại |

## 8. Risks & Mitigations

| Risk | Mitigation |
|---|---|
| Refactor phá invariant T-pose **đang hoạt động** | Test #3 là gate bắt buộc. Giữ nguyên readiness gate `revealed`/`onReady` (§4.2) |
| Mất đường verify Kimodo offline | §4.3 giữ debug file selector; test #4 |
| Thinking subclip (intro/loop/outro) hiện dùng `#` splitting + frame range cứng | Ba state riêng: `thinking_intro`, `thinking_loop`, `thinking_outro`. Mỗi state dùng cùng một clip gốc nhưng subclip khác nhau qua `THREE.AnimationUtils.subclip()`. AnimationRegistry cần hỗ trợ subclip range. |
| Generated motion có URL động (motion_UUID.bvh) | AnimationRegistry có `update('exercise', url)` để thay clip. Mỗi lần `transitionTo('exercise')` sẽ dùng clip từ registry hiện tại. |
| StrictMode double-mount (React 19) | AnimationController handle lifecycle trong constructor/dispose. VRMCharacter tạo controller trong useEffect, dispose trong cleanup. StrictMode remount tạo instance mới. |
| Clip cache dùng chéo giữa 2 model (lỗi #6) | `registry.invalidate()` trong effect `[vrmUrl]`; test #7 |
| `transitionTo` là async → race khi spam | `loadGen` guard (§3.3 bước 2-5); test #3 |
| Hiệu suất: AnimationController là class instance, không phải React | Không state update trong animation loop (useFrame). Event emitter notify React 1 lần/transition, không phải mỗi frame |
| StrictMode double-mount (React 19) | Controller tạo/dispose trong useEffect cleanup; giữ `disposeGuardRef` pattern hiện có |

## 9. Follow-up bắt buộc: Facial/Emotion controller CHƯA đồng bộ với CharState

> **Không nằm trong scope PR này** (plan này chỉ lo body animation + camera), nhưng **phải ghi lại
> vì refactor này làm vấn đề lộ rõ và đồng thời tạo ra chỗ để sửa nó**. Nếu không xử lý, avatar sẽ
> có "thân" và "mặt" nói hai chuyện khác nhau.

### 9.1 Vấn đề: hai state machine độc lập

Sau refactor sẽ có **2 hệ trạng thái chạy song song, không biết nhau**:

| Hệ | State | Nguồn lái |
|---|---|---|
| **Body** (plan này) | `idle / greeting / bored / thinking_* / exercise` | `transitionTo()` — chat event, timer, SSE motion |
| **Facial** ([facial-animation-plan](../../docs/plans/facial-animation-plan.md)) | `engaged / idle` | Engagement deadline: `setEmotion()`, `startLipSync()`, `notifyEngaged()` — [AvatarController.tick:104-115](../../ECA_UI/frontend/src/avatar/AvatarController.ts#L104-L115) |

Hai hệ dùng **timer riêng, event riêng** → trôi lệch nhau. Facial chỉ biết "có ai gọi setEmotion/TTS
gần đây không", hoàn toàn không biết thân đang làm gì.

### 9.2 Ba biểu hiện cụ thể (đều là bug user thấy được)

1. **Thân đang "suy nghĩ", mặt lại tự cười.** Body ở `thinking_loop`, nhưng nếu không có
   `setEmotion`/TTS trong 3s (`EVENT_GRACE_MS`) thì facial rơi về `idle` → `IdleBehaviorController`
   bắt đầu wander micro-emotion (happy 0.62 / relaxed 0.38) + đảo mắt ngẫu nhiên. Avatar vừa làm
   động tác trầm ngâm vừa nhoẻn miệng cười — thiếu mạch lạc.
2. **Cười trong lúc demo bài tập.** Body ở `exercise` (đang minh hoạ động tác trị liệu), facial vẫn
   có thể idle-wander. Trong bối cảnh **clinical safety** thì biểu cảm ngẫu nhiên lúc hướng dẫn bài
   tập là không phù hợp.
3. **Head-follow đánh nhau với body animation.** `HeadController` ghi bone Neck/Head từ rest
   quaternion **mỗi frame, sau `mixer.update`** → trong `exercise`, chuyển động đầu của clip Kimodo
   bị head-follow ghi đè. Đây chính là gap #1 đã ghi trong
   [HeadController](../../ECA_UI/frontend/src/avatar/HeadController.ts) (dự kiến "Phase 2: additive
   blending") — nhưng bản chất nó là **vấn đề đồng bộ**: khi body sở hữu chuyển động đầu thì
   head-follow phải nhường.

### 9.3 Vì sao refactor này là chỗ đúng để sửa

`AnimationController` đã emit **`stateChanged`** (§3.3) — đúng cái hook mà facial đang thiếu.
`CameraController` subscribe event này; facial hoàn toàn có thể subscribe cùng cách, giữ nguyên
nguyên tắc "một nguồn lái state" (§3.4).

### 9.4 Hướng đề xuất (chốt ở PR sau, không làm bây giờ)

**Đúng tầng: DATA, không phải API.** Emotion trong hệ này vốn đã hoàn toàn declarative ở tầng asset
— `recipes`, `morphRepairMap`, `binaryEmotions` khai báo trong
[profiles/*.ts](../../ECA_UI/frontend/src/avatar/profiles/) theo từng model (ví dụ seele:
`happy → 'にこり'`, `binaryEmotions: ['relaxed']`). Giải pháp đồng bộ phải giữ đúng tinh thần đó,
**không** bolt thêm setter/API vào `AvatarController`.

Nhưng phải tách 2 tầng, kẻo sửa sai chỗ:

| Thứ | Tầng | Ở đâu |
|---|---|---|
| Emotion `X` chạy channel nào, weight bao nhiêu | **Asset** (per-model) | `profiles/*.ts` — ✅ đã data-driven |
| Body state `thinking_loop` thì **được cười hay không** | **App policy** (model-independent) | ❌ chưa có — và **KHÔNG** nhét vào `profiles/*.ts` |

> Nếu đưa policy "thinking thì đừng wander" vào `profiles/`, nó sẽ bị lặp ở `seele.ts` /
> `bronya.ts` / `default.ts`, và mỗi model mới lại phải khai báo lại — trong khi luật đó chẳng phụ
> thuộc model nào. Đây đúng là lý do `WIDE_STATES` của CameraController nằm ở tầng app (§3.4).

Hình dạng đề xuất — **một hằng data + một điều kiện đọc data, không thêm public method nào**:

Sau sửa #9 thì **chỗ đặt đã có sẵn**: trường `facial` trong `STATES` (§3.0) — không cần hằng riêng,
và mỗi state mới **buộc** phải khai báo (`Record` không `Partial`), nên không thể quên:

```typescript
// §3.0 — đã nằm trong STATES, đây chỉ là phần liên quan
thinking_loop: { /* … */ facial: { wander: false, hold: 'neutral' } },
exercise:      { /* … */ facial: { wander: false, hold: 'neutral' } },  // + head-follow nhường body
idle:          { /* … */ facial: { wander: true } },
```

<details><summary>Hình dạng cũ (trước khi gộp vào STATES) — giữ để đối chiếu</summary>

```typescript
const FACIAL_POLICY: Partial<Record<CharState, { wander: boolean; hold?: CanonicalEmotion }>> = {
  thinking_intro: { wander: false, hold: 'neutral' },
  thinking_loop:  { wander: false, hold: 'neutral' },
  thinking_outro: { wander: false, hold: 'neutral' },
  exercise:       { wander: false, hold: 'neutral' },
  // idle / greeting / bored: không có entry → wander bật như hiện tại
}
```
`Partial<>` ở đây là điểm yếu: thêm state mới mà quên khai báo thì nó **im lặng** rơi về wander —
đúng lúc không nên cười (ví dụ thêm state `exercise_correction` sau này).
</details>

Chỗ duy nhất phải sửa control-flow: dòng
[115](../../ECA_UI/frontend/src/avatar/AvatarController.ts#L115) `if (!engaged) this.idle.tick(delta)`
— cho nó đọc policy thay vì chỉ hardcode dựa `engaged`. Tương tự, `HeadController` nhận một
attenuation factor lấy từ cùng policy (state `exercise` → gain 0, vì body animation sở hữu chuyển
động đầu).

> ⚠️ **Không** giải quyết bằng cách gọi `notifyEngaged()` liên tục trong lúc `thinking_*`: nó ép
> facial ở `engaged` như một side-effect, làm engagement deadline mất ý nghĩa và vẫn không nói được
> "được cười hay không". Cần policy tường minh, không mượn cơ chế khác.

**Việc cần làm ở PR này**: chỉ **giữ nguyên** thứ tự `useFrame` (body → facial → `vrm.update`, §4.2)
và **không xoá** `stateChanged` emitter — để PR sau cắm vào được.

---

## 10. Open Items

- **Thinking.fbx subclip range**: `intro [0,38]`, `loop [38,75]`, `outro [75,127]` @30fps — **giữ
  nguyên** giá trị hiện tại trong PR này (đang chạy đúng), N verify lại sau refactor.
- **`COOLDOWN_MS = 3000`**: để const trong CameraController, hay đưa vào `environmentConfig.ts`?
  (Đề xuất: const — chưa có nhu cầu tune runtime.)
- **`bored` chọn clip nào** khi có nhiều `random_*`: Registry tự pick random trong nhóm, hay FSM
  truyền tên cụ thể? (Đề xuất: Registry pick — FSM không nên biết tên file.)
- ~~State transition restriction~~: đã giải quyết bằng nguyên tắc §2.2 (user-driven reachable từ mọi
  state) — không thêm validation nữa, vì chặn thêm chính là nguồn của 2 lỗi 🔴.

---

## 11. Cookbook: thêm một animation mới sau này (sửa lỗi #9)

> Mục này tồn tại vì đây là **câu hỏi Owner hỏi trực tiếp**: thêm animation về sau có dễ bảo trì
> không, và thêm **điều kiện kích hoạt** có phải viết lại không.

### 11.1 Trước / sau khi sửa #9

| | Số nơi phải sửa | Nơi không có compile check |
|---|---|---|
| Code hiện tại (`develop`) | **7** chỗ string-match rải 2 file (§0) | 7/7 — đổi tên file là behavior đổi ngầm |
| Plan v1.1 (6 map song song) | **4–7** | **3** (`ALLOWED_TRANSITIONS`, `ON_FINISHED`, `STATIC_ENTRIES`) |
| **Plan v1.2 (§3.0)** | **1 entry** + 1 dòng trigger nếu cần | **0** — thiếu field nào là lỗi build |

### 11.2 Quy trình thêm animation — ví dụ `stretch_demo`

```
1. Bỏ file vào src/asset/animations/action_stretch.bvh
2. Thêm 'stretch_demo' vào union CharState
   → tsc BÁO LỖI NGAY tại STATES: "Property 'stretch_demo' is missing"
      (compiler dẫn đường — không cần nhớ checklist nào)
3. Điền 1 entry vào STATES:
     stretch_demo: {
       source: { match: /action_stretch/i, retarget: 'standard', isFbx: false },
       loop: 'once', onFinished: 'idle',      // bỏ onFinished → LỖI BUILD
       reach: 'anytime',                       // hoặc 'from-idle' / { after: [...] }
       camera: 'hips',                         // camera tự nới khung + cooldown 3s
       facial: { wander: false, hold: 'neutral' },
       debugLabel: 'Stretch Demo',             // → dropdown debug tự có, 0 dòng sửa UI
       crossfade: 0.6,
     }
4. XONG.
```

**Không phải chạm vào**: `AnimationRegistry`, `AnimationController`, `CameraController`,
`MotionContext`, `MotionControlPanel`, `ChatPanel`, `CharacterViewer`. Bốn controller đều generic —
chúng đọc `STATES`, không biết tên state nào cả.

### 11.3 Còn "điều kiện kích hoạt" thì sao?

Ba dạng, **không dạng nào cần viết lại hệ thống**:

| Dạng trigger | Ví dụ | Phải làm gì |
|---|---|---|
| **Chọn tay (debug/QA)** | test clip mới | **0 dòng** — có `debugLabel` là xong |
| **Theo thời gian / điều kiện nội bộ** | idle lâu → bored | **0 dòng code**: khai `autoAfterIdle: [min, max]` (§3.0) |
| **Event ngoài** | chat bắt đầu, SSE motion ready, TTS xong | **1 dòng** `transitionTo('x')` tại call-site, đặt trong `hooks/useFsmTriggers.ts` |

Dạng thứ 3 **cố ý giữ là code**, không data: mỗi event đến từ một nguồn khác nhau (React event, SSE
payload, promise resolve) với dữ liệu kèm theo khác nhau — nhồi vào bảng config sẽ phải phát minh một
mini-DSL cho điều kiện, tức là thêm abstraction để phục vụ **một** call-site. Đổi lại, tất cả trigger
nằm **cùng một file** nên vẫn trả lời được câu "cái gì kích hoạt animation này" bằng một lần mở file
— thay vì lần theo 7 chỗ string-match như hiện tại.

### 11.4 Giới hạn thật (không hứa quá)

- **Thêm animation cùng "họ" với các state hiện có** (one-shot → idle, loop, có/không subclip):
  đúng 1 entry như trên.
- **Thêm cơ chế mới về bản chất** — ví dụ blend 2 clip đồng thời (upper body cầm cốc + lower body đi),
  hoặc animation phụ thuộc tham số liên tục (nghiêng người theo góc camera) — thì **`STATES` không đủ**:
  đó là additive/layered blending, phải mở rộng `AnimationController` (giống head-follow Phase 2).
  Bảng này giải quyết **"state nào chạy clip nào"**, không giải quyết **"nhiều clip cùng lúc"**.
- `reach: 'anytime'` là mặc định an toàn. Chỉ dùng `'from-idle'` / `{ after }` khi có lý do rõ ràng —
  mỗi ràng buộc thêm là một khả năng drop tương tác của user (bài học lỗi 🔴 #2).

---

## 12. Implementation report (30/07/2026 — K)

### 12.1 Files

| File | Trạng thái | Ghi chú |
|---|---|---|
| [lib/AnimationStates.ts](../../ECA_UI/frontend/src/lib/AnimationStates.ts) | 🆕 | Bảng `STATES` + `canTransition` + derived getters (§3.0) |
| [lib/motionAssets.ts](../../ECA_UI/frontend/src/lib/motionAssets.ts) | 🆕 **ngoài plan** | Nơi DUY NHẤT glob file animation. Không có nó thì cả Registry và panel đều phải tự glob → hai nguồn sự thật về asset |
| [lib/AnimationRegistry.ts](../../ECA_UI/frontend/src/lib/AnimationRegistry.ts) | 🆕 | Load → retarget → subclip → cache |
| [lib/AnimationController.ts](../../ECA_UI/frontend/src/lib/AnimationController.ts) | 🆕 | FSM runtime + invariant T-pose |
| [lib/CameraController.ts](../../ECA_UI/frontend/src/lib/CameraController.ts) | 🆕 | Mode + cooldown, không lái FSM |
| [hooks/useFsmTriggers.ts](../../ECA_UI/frontend/src/hooks/useFsmTriggers.ts) | 🆕 | `useFsmBoot` (§2.5) + `useAutoAfterTrigger` |
| ~~lib/TransitionConfig.ts~~ | ❌ không tạo | Gộp vào `crossfadeFor()` (§3.2) |
| [contexts/MotionContext.tsx](../../ECA_UI/frontend/src/contexts/MotionContext.tsx) | ♻️ | -1 glob, -3 effect string-match; +`transitionTo`/`playMotionFile`/`currentState` |
| [components/CharacterViewer.tsx](../../ECA_UI/frontend/src/components/CharacterViewer.tsx) | ♻️ | Bỏ toàn bộ clip-loading/fade/subclip/`isAction` (≈95 dòng) |
| [components/panels/MotionControlPanel.tsx](../../ECA_UI/frontend/src/components/panels/MotionControlPanel.tsx) | ♻️ | 2 dropdown: state (derived) + file debug |
| [components/ChatPanel.tsx](../../ECA_UI/frontend/src/components/ChatPanel.tsx) | ♻️ | `setIsThinking` → `transitionTo`, có guard `thinkingRef` |

**7 chỗ string-match ở §0 đã biến mất hoàn toàn.** Không còn nơi nào trong code quyết định hành vi
animation dựa trên tên file.

### 12.2 Điểm code KHÁC plan — và lý do

| # | Plan viết | Code làm | Lý do |
|---|---|---|---|
| 1 | `StaticSource` có cả `isFbx: boolean` và `retarget: 'smplx'\|'standard'\|'mixamo'` | Discriminated union theo `loader: 'fbx' \| 'bvh'`, `retarget` **chỉ tồn tại** ở nhánh `bvh` | Bản plan cho phép viết tổ hợp vô nghĩa (`isFbx: true` + `retarget: 'smplx'`). Union chặn ở tầng type |
| 2 | `autoAfterIdle: [min, max]` → chuyển tới "state `from-idle` đầu tiên" | `autoAfter: { to, minSec, maxSec }` | "State from-idle đầu tiên" là magic phụ thuộc thứ tự khai báo trong object — thêm state mới có thể đổi đích một cách vô hình |
| 3 | `finished` → **listener bên ngoài** gọi `transitionTo(ON_FINISHED[...])` | `AnimationController` tự nhảy, **đồng thời** vẫn emit `finished` | `onFinished` là phần của định nghĩa state; để bên ngoài lo nghĩa là mọi call-site đều có thể quên → FSM kẹt ở frame cuối. Observer vẫn nhận event |
| 4 | — | `transitionTo` **no-op** khi vào lại state đang chạy và state đó `loop: 'repeat'` | Restart một clip đang loop gây giật hình. One-shot vẫn restart (`exercise → exercise` = phát motion mới) |
| 5 | — | `mixer.update(0)` ngay trước `onClipApplied` **lần đầu** | Reveal gate phải thấy pose thật trên bone. Code cũ làm việc này ở CharacterViewer; giờ controller sở hữu mixer nên phải chuyển vào trong |
| 6 | §4.1 bỏ `cameraMode` khỏi context | Giữ `cameraMode` (read-only, push từ CameraController) + `setCameraMode` = manual override | Panel của N đang có dropdown camera. Bỏ đi là xoá UI của người khác — ngoài scope refactor |
| 7 | `constructor(private readonly x: T)` | Khai field rồi gán trong constructor | `tsconfig` bật **`erasableSyntaxOnly`** → parameter properties là lỗi biên dịch |
| 8 | — | `window.__fsm` (chỉ DEV) đặt ở `MotionProvider` | Phiên trước mất thời gian vì `window.__avatar` chỉ tồn tại khi panel mở. Handle này ở provider nên luôn có → test tự động không phụ thuộc UI |

### 12.3 Kết quả test

Chạy thật trên `:5173` + backend `:8000` (Playwright, Chromium). **17/17 xanh.**

| Test §7 | Kết quả đo được |
|---|---|
| #1 Deadlock regression | `exercise` → finished → `idle`; `transitionTo('thinking_intro')` = **true** |
| #2 Motion lúc đang thinking | `thinking_loop` → `playMotionFile` = true → `exercise` |
| #3 T-pose invariant | 10 transition liên tiếp (100ms/lần, không await) → **0 frame không có pose** |
| #4 Kimodo retarget | `motion_7b4b8d9e-324.bvh` qua `exercise` = true |
| #5 `reset()` fix | Cùng 1 clip 2 lần: run1 5483ms, run2 **5213ms** (nếu kẹt frame cuối sẽ về ~0ms) |
| #6 Camera cooldown | `idle`=head → `exercise`=hips → idle+0.9s=**hips** → +3.5s=head |
| #7 `invalidate()` | Đổi model → phát lại motion = true, không méo (verify bằng ảnh) |
| #8 Chat end-to-end | Sequence đo được: `["thinking_intro","thinking_loop","thinking_outro"]` → `idle` sau 18.8s |
| #9 Boot sequence | State tại pose đầu tiên = **`greeting`**, history `["idle","idle","greeting","idle"]` → greeting đúng **1 lần** (StrictMode không chào 2 lần) |
| #10 Đổi model không chào lại | greeting count trước/sau = **2 → 2**, state sau khi đổi = `idle` |

Thêm 5 test ngoài checklist: reachability guard (`idle→thinking_loop`/`idle→thinking_outro`/
`thinking→bored` đều **false**, `thinking→exercise` **true**), state dropdown derived đúng 4 mục,
file dropdown vẫn liệt kê 3 file generated, chọn dropdown lái được FSM, file selector → `exercise` +
camera `hips`.

### 12.4 Còn lại

- **§9 facial↔body sync**: chưa làm (đúng như plan — ngoài scope PR này). Chỗ cắm đã sẵn:
  `facialOf(state)` đã có trong `AnimationStates.ts`, `stateChanged` emitter còn nguyên, thứ tự
  `useFrame` (body → facial → `vrm.update`) giữ nguyên.
- **`playMotionFile` chưa được ChatPanel gọi**: backend chưa stream motion URL (P2).
- `npm run build` xanh; `tsc --noEmit` xanh.
