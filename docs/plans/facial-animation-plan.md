# Plan: Facial Animation System cho VRM Avatar (v1.2)

> Architect: K | Developer: N | Owner approval: pending | Last update: 2026-07-24
> Status: **v1.2 — R3F-native, SSE transport (per-turn), channel-based mixer, client-side autonomous idle**
> Base: Proposal gốc của Owner + review pass 1 + quyết định "idle thuộc frontend" của Owner
> Scope: `ECA_UI/frontend` — KHÔNG đụng backend ở phase này

---

## Changelog

### Proposal gốc → v1.1
| # | Thay đổi | Lý do |
|---|----------|-------|
| 1 | WebSocket → SSE named events | ADR-005 revised đã chốt SSE + REST POST |
| 2 | Vanilla Three.js → React Three Fiber | Codebase dùng `@react-three/fiber` v9 + React 19, `useFrame` sẵn trong `CharacterViewer` |
| 3 | Thêm IdleBehaviorController | Owner: backend KHÔNG stream idle state |
| 4 | Thêm BlinkController | Proposal không có module sinh blink |
| 5 | Priority system → Channel-based mixer | Priority per-controller không giải quyết xung đột "happy + nói cùng ghi vào mouth" |
| 6 | Profile map runtime preset names (không map tên raw) | three-vrm v3 tự migrate VRM 0.x → tên preset 1.0 |
| 7 | Capability detection + graceful degradation | `bronya_long.vrm` có 0 blendshape groups |
| 8 | Lip sync chỉ Mode 1 (amplitude) | VieNeu-TTS-GGUF không xuất phoneme timestamps |
| 9 | Frame-based → delta-time interpolation | Frame-step sai khi FPS đổi |
| 10 | Thêm DevPanel + acceptance criteria | N cần verify không phụ thuộc backend |

### v1.1 → v1.2 (K, 23/07 — sau khi verify code thật, không đoán)
| # | Thay đổi | Bằng chứng |
|---|----------|-----------|
| A | **Sửa mô hình transport: SSE là PER-TURN, không phải kết nối thường trực** | `streamChat` ([lib/api.ts](../../ECA_UI/frontend/src/lib/api.ts)) mở khi POST `/chat`, stream đúng lượt đó rồi ĐÓNG ở event `done`. Không có kênh persistent. Xem §0 + §7 đã sửa |
| B | **Transport named-event ĐÃ hoạt động thật** (de-risk §7) | `_parseSSEBlocks` trong api.ts parse `event:`/`data:` đúng chuẩn; bug CRLF boundary vừa fix 23/07. `avatar.emotion`/`tts.audio` dùng đúng path `streamChat(onEvent)` — 0 dòng transport code mới |
| C | **Chú thích lại timing `tts.audio`** | Backend fire TTS async qua Celery, trả `speech_pending` + `speech_task_id` rồi poll — audioUrl CHƯA sẵn lúc stream. Xem §9 note |
| D | Verify toàn bộ VRM facts | seele 17 blendshape (khớp), bronya +custom "Surprised" (presetName `unknown`), bronya_long 0 blendshape (khớp). three-vrm preset: `joy→happy, sorrow→sad, fun→relaxed, a→aa, i→ih, u→ou, e→ee, o→oh` |
| E | **Phát hiện + sửa bind rỗng trên model** (24/07) | seele.vrm: 17 group nhưng **tất cả binds rỗng** — expression đăng ký nhưng không điều khiển morph nào. Bronya: Joy/Angry/Sorrow/Fun/Surprised có binds thật, nhưng blink/visemes cũng rỗng. Xem §6.1 note + §6.3 bind repair |

---

## 0. Nguyên tắc gốc: Local-first Animation

> **Backend gửi intent, frontend sở hữu toàn bộ chuyển động.**

- Backend chỉ phát event **khi có interaction turn** (user gửi message → response + TTS). Event mang intent-level: `emotion`, `intensity`, `duration`.
- Backend **không gửi gì** khi avatar idle. Không heartbeat emotion, không raw blendshape weights.
- Frontend tự vận hành idle behavior (blink, micro-emotion, gaze wander) bằng state machine nội bộ.
- **Transport thực tế (sửa v1.2)**: SSE là **per-turn** — mở khi user gửi message (POST `/chat`), stream các event của lượt đó (`stage`/`token`/`avatar.emotion`/`tts.audio`/`done`) rồi đóng. KHÔNG có kết nối thường trực giữa các lượt. Avatar events (nếu backend emit sau này) đi **ghép vào stream của chính lượt chat đó** — hợp lý vì chúng đều là interaction-scoped. Idle là client-autonomous nên KHÔNG cần connection nào đang mở.

Hệ quả: avatar **không bao giờ đứng im** kể cả khi backend chết — degradation tự nhiên, không cần error handling riêng cho idle.

---

## 1. Operating Modes: ENGAGED vs IDLE

### 1.1 State machine (chạy hoàn toàn ở client)

```
   user gửi msg / avatar event                 hết engaged window
   ────────────────────────►┌──────────┐  ────────────────────►┌────────┐
                            │ ENGAGED  │                        │  IDLE  │
                            │ backend  │◄───────────────────────│ mọi SSE│
                            │ commands │  avatar event refresh   │ avatar │
                            │ +lipsync │       timer             │ event  │
                            └──────────┘                        └────────┘
```

### 1.2 Engagement rule (không cần backend báo "turn kết thúc")

```ts
engagedUntil = max(
  lastAvatarEventAt + EVENT_GRACE_MS,   // vd 3000ms sau event cuối
  ttsAudioEndedAt   + TTS_GRACE_MS,     // vd 1500ms sau khi audio dừng
)
isEngaged = now < engagedUntil
```

- Mỗi SSE event `avatar.*` hoặc TTS đang phát → refresh `engagedUntil`.
- Hết hạn → tự rơi về IDLE. Không cần event `turn_end` từ backend.

### 1.3 Arbitration

| Mode | Nguồn điều khiển | Idle behaviors |
|------|------------------|----------------|
| ENGAGED | Backend emotion events + LipSync + EyeController (mouse) | **Suppress** emotion/gaze wanderer. Blink vẫn chạy (sinh lý) |
| IDLE | IdleBehaviorController + EyeController (mouse) | Active |

Chuyển ENGAGED → IDLE **phải cross-fade** (emotion hiện giảm dần về neutral ~800ms trước khi wanderer nhận quyền), không cắt cụt.

---

## 2. Architecture Layers

```
Backend SSE (per-turn, interaction only)
  event: avatar.emotion  {emotion, intensity, duration}
  event: tts.audio       {audioUrl}   ← xem §9 note về timing
        │
        ▼
AnimationState        ← single source of truth, plain TS object, KHÔNG React state
        │
        ▼
AvatarController      ← tick(delta) duy nhất, gọi từ useFrame hiện có
   ├─ ExpressionController   (emotion recipe + cross-fade)
   ├─ BlinkController        (auto-blink, chạy ở cả 2 modes)
   ├─ IdleBehaviorController (emotion + gaze wanderer, chỉ IDLE)   [Phase B]
   ├─ EyeController          (mouse / camera / AI target)          [Phase B]
   └─ LipSyncController      (amplitude-based, chỉ khi TTS phát)   [Phase C]
        │
        ▼
ExpressionMixer       ← channel-based composition (§5)
        │
        ▼
VRMExpressionAdapter  ← canonical → profile mapping → setValue
        │
        ▼
three-vrm ExpressionManager + vrm.lookAt
```

Rule bất biến: **chỉ VRMExpressionAdapter được chạm VRM**. Business logic (ChatPanel, SSE handler) chỉ nói canonical emotion names.

---

## 3. Modules — `ECA_UI/frontend/src/avatar/`

| File | Trách nhiệm | Phase |
|------|-------------|-------|
| `AvatarController.ts` | Facade duy nhất. `attach(vrm, profile)`, `detach()`, `tick(delta)`, `setEmotion()` | A |
| `AnimationState.ts` | Plain object trạng thái hiện tại. Không React | A |
| `ExpressionController.ts` | `setEmotion(name, intensity, durationMs)` + cross-fade delta-time (interrupt mượt, fade từ giá trị hiện tại không snap 0) | A |
| `BlinkController.ts` | Auto-blink: interval ngẫu nhiên 2–6s, chu kỳ ~150ms, đôi khi double-blink | A |
| `ExpressionMixer.ts` | Channel-based composition (§5) | A |
| `VRMExpressionAdapter.ts` | `write(channels)` → resolve qua profile → `setValue`. Capability detection lúc attach | A |
| `AvatarProfile.ts` | Types + `loadProfile(modelId)`: per-model override fallback về default | A |
| `profiles/default.ts` | Mapping chuẩn cho VRM 0.x models (dùng `.ts` typed const, xem note dưới) | A |
| `profiles/bronya.ts` | Override: `surprised` → custom channel `Surprised` | A |
| `AvatarDevPanel.tsx` | Dev-only UI trigger emotion/slider/blink. Ẩn sau `import.meta.env.DEV` | A |
| `IdleBehaviorController.ts` | Emotion + gaze wanderer, chỉ IDLE (§4) | B |
| `EyeController.ts` | Mouse tracking + center-return | B |
| `LipSyncController.ts` | Mode 1 amplitude-only (§9) | C |

> **Note deviation (K)**: profiles dùng `.ts` export typed const thay vì `.json`. Lý do: tsconfig KHÔNG bật `resolveJsonModule` + `verbatimModuleSyntax` on → import JSON gãy type-check. `.ts` cho type-safety chặt hơn, vẫn data-driven + loadable per-model. Semantics y hệt plan.

Tất cả controller là **framework-agnostic plain TS classes** — không import React. Chỉ `CharacterViewer.tsx` (glue) + `AvatarDevPanel.tsx` đụng React.

---

## 4. IdleBehaviorController (Phase B — quyết định của Owner)

### 4.1 Emotion wanderer
Weighted random, bias mạnh về neutral. Config object duy nhất:
```ts
idleConfig = {
  emotionIntervalMs: [4000, 9000],
  weights: { neutral: 0.6, happy: 0.25, relaxed: 0.15 },
  microIntensity: [0.15, 0.4],   // idle KHÔNG BAO GIỜ full intensity
  transitionMs: 800,
}
```
Luân chuyển qua neutral, không nhảy happy→sad trực tiếp.

### 4.2 Gaze wanderer
Khi mouse tĩnh >3s: saccade ngẫu nhiên mỗi 2–5s tới điểm lân cận (±0.3 normalized), rồi về center. Mouse động → nhường quyền ngay.

### 4.3 Ranh giới với body motion
Body idle (BVH) chạy sẵn qua `AnimationMixer` trong `CharacterViewer` — **ngoài scope**. IdleBehaviorController chỉ lo mặt + mắt.

---

## 5. ExpressionMixer — Channel-based (thay Section 7 proposal)

Mỗi output channel (VRM preset name) có chủ sở hữu tĩnh:

| Channel group | Nguồn | Rule |
|---------------|-------|------|
| Mouth viseme (`aa/ih/ou/ee/oh`) | LipSync (ENGAGED), emotion recipe | LipSync **override** khi audio phát; else emotion sở hữu |
| Mouth khác | Emotion recipe | Emotion sở hữu |
| Eyes (`blink`) | BlinkController | three-vrm tự multiply blink override lên eye-open của emotion (`blinkExpressionNames`) — KHÔNG cần cộng tay |
| Look (`lookUp/Down/...`) | `vrm.lookAt` | **Reserve**: emotion recipe cấm chứa các channel này (validate lúc load profile) |
| Brow/cheek | Emotion recipe | Emotion sở hữu |

**Không còn "priority 1/2/3/4"**. Quyền sở hữu theo channel, xác định tĩnh. Mixer merge vào một frame output → adapter ghi một lần mỗi frame.

LookAt check lúc `attach()`: đọc `vrm.lookAt.applier` type — nếu `VRMLookAtExpressionApplier` (blendshape-based) → reserve look channels; nếu `VRMLookAtBoneApplier` (bone-based) → không cần reserve.

---

## 6. VRM Reality Check + Avatar Profile

### 6.1 Models hiện có (đã inspect GLB JSON chunk thật — 23/07, bổ sung 24/07)

| Model | Spec | blendShapeGroups | Binds thật |
|-------|------|-------------------|------------|
| `seele.vrm` | VRM 0.0 | 17: neutral, a, i, u, e, o, blink, joy, angry, sorrow, fun, lookup/down/left/right, blink_l/r | **0/17 có binds rỗng** — mọi expression register nhưng không điều khiển morph nào |
| `bronya.vrm` | VRM 0.0 | 18: như seele **+ custom "Surprised"** (presetName `unknown`, name `Surprised`, 4 binds) | Joy(3), Angry(2), Sorrow(4), Fun(4), Surprised(4) — emotion hoạt động. Blink, visemes (A/E/I/O/U), Look*: 0 binds |
| `bronya_long.vrm` | VRM 0.0 | **0 (rỗng)** — capability detection phải warn, mọi facial output no-op an toàn | 0 |

> **Note (24/07 — K verify lại binds sau khi test thấy biểu cảm chết):**
> Đợt verify 23/07 mới đếm số lượng groups (khớp) nhưng chưa kiểm tra `binds` bên trong.
> seele.vrm là MMD→PMX→VRM conversion: mesh có 47 morph targets tên tiếng Nhật
> (`"にこり"`, `"哀"`, `"怒り"`, `"まばたき"`, `"あ"...`) nhưng VRM extension không hề
> reference tới morph targets qua `binds`. Biểu hiện: `capability detection` pass
> (`getExpression('happy') != null`) → không warning → setValue/setEmotion chạy
> nhưng `applyWeight()` loop qua 0 binds → mọi biểu cảm invisible.
> **Fix**: profile `morphRepairMap` (§6.3) patch binds theo tên morph vào lúc attach.

three-vrm v3 migrate 0.x → preset 1.0 lúc load: `joy→happy, sorrow→sad, angry→angry, fun→relaxed, a→aa, i→ih, u→ou, e→ee, o→oh, blink→blink`. **Profile dùng tên runtime (post-migration).**

### 6.3 Bind repair (thêm 24/07)

Khi VRM 0.x file có expression group với binds rỗng nhưng mesh có morph targets
đặt tên, profile có thể khai báo `morphRepairMap`: map từ channel runtime name →
tên morph target. `VRMExpressionAdapter.attach()` patch `VRMExpressionMorphTargetBind`
trước capability detection, channel được sửa tính như available.

seele profile (`profiles/seele.ts`):
```ts
morphRepairMap: {
  happy: 'にこり', sad: '哀', angry: '怒り', relaxed: 'なごみ',
  surprised: 'びっくり', blink: 'まばたき',
  aa: 'あ', ih: 'い', ou: 'う', ee: 'え', oh: 'お',
}
```

bronya KHÔNG dùng repair: mesh thiếu `extras.targetNames` → không map theo tên được.
Bù lại emotion groups (Joy/Angry/...) có binds thật → biểu cảm chạy.

### 6.2 Profile format (version 1)
```ts
{
  version: 1,
  modelId: 'seele',
  recipes: {
    happy:     { happy: 1.0 },
    sad:       { sad: 1.0 },
    angry:     { angry: 1.0 },
    relaxed:   { relaxed: 1.0 },
    surprised: { surprised: 1.0 },  // seele KHÔNG có → no-op; bronya override → "Surprised"
    neutral:   {},
  },
  visemes: { A: 'aa', I: 'ih', U: 'ou', E: 'ee', O: 'oh' },
}
```

- Canonical emotion set: `neutral | happy | sad | angry | relaxed | surprised`. Backend chỉ được gửi trong set này (validate ở SSE handler, event lạ → warn + drop).
- Recipe hỗ trợ **composite** (nhiều channel một emotion), weight per channel.
- `loadProfile(modelId)`: thử `profiles/<modelId>` → fallback `default`. Model rỗng → profile rỗng + warn, hệ thống vẫn chạy (blink no-op).

---

## 7. Backend Communication — SSE Events

Transport: **SSE named events trên stream chat per-turn hiện có** (KHÔNG WebSocket, KHÔNG kênh persistent).

```
event: avatar.emotion
data: {"emotion":"happy","intensity":0.8,"duration":1000}

event: tts.audio
data: {"audioUrl":"/api/tts/<id>.wav"}
```

- **De-risk (v1.2)**: đường transport ĐÃ chạy. `streamChat(options, onEvent)` gọi `onEvent(type, data)` cho mọi named event; chỉ cần thêm nhánh `avatar.emotion`/`tts.audio` trong callback ở ChatPanel → gọi `avatarController.setEmotion(...)`. Không viết transport mới.
- `intensity` clamp [0,1]. `duration` = thời gian giữ trước khi tự fade về neutral (backend gợi ý, frontend quyết).
- Phase này backend chưa emit → DevPanel giả lập event để test. Contract này là input cho backend phase sau (Conversation node đính kèm emotion metadata).

---

## 8. R3F Integration Rules (bắt buộc)

1. **Một update loop duy nhất**: `avatarController.tick(delta)` gọi trong `useFrame` hiện có của `VRMCharacter`. Thứ tự **bắt buộc**: `mixer.update(delta)` → `avatarController.tick(delta)` → `vrm.update(delta)`. Lý do: `setValue` phải xảy ra TRƯỚC `vrm.update` (nó gọi `expressionManager.update()` nội bộ để apply weights).
2. Controllers khởi tạo **1 lần** (`useRef`/`useMemo`), không tạo lại mỗi render.
3. **Cấm đưa frame data qua React state/context.** ChatPanel/SSE handler gọi method trực tiếp trên controller instance qua ref (theo pattern `onResetRef` hiện có → thêm `avatarControllerRef` vào `MotionContext`).
4. Lifecycle: model load xong → `attach(vrm, profile)`; đổi model/unmount → `detach()` + dispose. AvatarsPanel đổi model không được leak.
5. Interpolation **delta-time-based**: `progress += delta / durationSec`. Easing `easeInOutCubic`.
6. Không allocation trong tick: cache object ngoài loop (theo pattern `followPos`/`deltaVec` đã có).

---

## 9. Lip Sync (Mode 1 only — Phase C)

Pipeline: `tts.audio` → `HTMLAudioElement` → `MediaElementAudioSourceNode` → `AnalyserNode` → RMS mỗi frame → viseme weights (low-mid energy → `aa/ou`, silence → 0) → smooth attack ~50ms / release ~150ms.

- **Autoplay policy**: `AudioContext` phải resume sau user gesture đầu (click send đủ điều kiện). Unlock một lần, reuse.
- Audio clock là source of truth cho sync.
- `VisemeSource` interface giữ chỗ cho Mode 2 (phoneme) — **không implement** cho tới khi VieNeu-TTS xuất phonemes.
- Audio ended → viseme decay về 0 → engagement grace bắt đầu đếm.

> **⚠️ Note timing `tts.audio` (chú thích v1.2 — K)**: backend HIỆN fire TTS **async qua Celery**, trả `speech_pending` + `speech_task_id` trong stream rồi client **poll** kết quả — audioUrl **CHƯA sẵn** tại thời điểm stream turn. Nên một event `tts.audio {audioUrl}` đồng bộ trong turn KHÔNG khớp cơ chế hiện tại. Hệ quả cho các phase sau:
> - **Phase C** (giả lập): test bằng file `.wav` tĩnh, bỏ qua chuyện timing — chấp nhận được.
> - **Phase D** (contract thật): phải chốt cách deliver audioUrl — hoặc (a) đổi TTS sang trả URL đồng bộ được, hoặc (b) LipSync lắng nghe event `speech_ready` phát ra khi poll xong (một round-trip riêng, ngoài stream turn). Đây là **open question cho backend**, ghi vào api-contract.md ở Phase D.

---

## 10. Phases + Acceptance Criteria

### Phase A — Core (không cần backend) ✅ scope hiện tại
1. `AnimationState`, `VRMExpressionAdapter`, `AvatarProfile` + `profiles/default.ts`
2. `ExpressionController` (cross-fade delta-time) + `ExpressionMixer`
3. `BlinkController`
4. `AvatarController` facade + glue vào `CharacterViewer` + `AvatarDevPanel`

**Accept:** DevPanel trigger happy 0.8/500ms trên seele → mượt; đổi sang sad giữa chừng không giật; blink chạy song song; đổi model sang bronya_long → warn console, không crash, blink no-op; transition 500ms hoàn tất ±1 frame ở 60fps.

### Phase B — Idle + Eyes
5. `IdleBehaviorController` + engagement timer
6. `EyeController` (mouse + center-return)

**Accept:** để yên 30s: avatar blink, thỉnh thoảng happy nhẹ rồi về neutral, mắt saccade; mouse động → mắt theo ngay; giả lập SSE event → wanderer suppress, hết grace → cross-fade về idle mượt.

### Phase C — Lip sync
7. `LipSyncController` + audio pipeline + `tts.audio` handler giả lập (file tĩnh)

**Accept:** phát wav mẫu → miệng mở theo amplitude, không vượt emotion channels khác; audio ended → miệng đóng ≤200ms; FPS khi lip sync không tụt quá 5%.

### Phase D — Backend contract (handoff)
8. Document SSE schema vào `docs/architecture/api-contract.md`: `avatar.emotion` emitter phía Conversation node + giải quyết open question timing `tts.audio` (§9 note).

---

## 11. Open Questions cho Owner

1. **Emotion source of truth**: backend Reasoning/Synthesizer tự quyết emotion từ nội dung, hay Conversation node gán theo persona? (Ảnh hưởng backend phase sau, không block frontend.)
2. `surprised` của bronya là custom 0.x — giữ custom name + per-model profile (đã làm ở `profiles/bronya.ts`), hay bỏ `surprised` khỏi canonical set cho tới khi có model 1.0 chuẩn?
3. Gaze wanderer có nên tránh nhìn xuống (trông như buồn) trong clinical context không?
4. **(Mới — v1.2)** `tts.audio` timing: giải quyết theo hướng (a) TTS trả URL đồng bộ hay (b) event `speech_ready` riêng sau poll? (§9 note — quyết ở Phase D.)

---

## 12. Out of Scope

- Full facial tracking / webcam-driven expression
- Phoneme-based lip sync (Mode 2) — chỉ interface
- Body motion / gesture (BVH + Kimodo flow hiện có)
- Backend SSE emitter implementation (Phase D chỉ là contract doc)
- Mobile touch-based eye tracking tuning (mouse trước, touch reuse cùng input path)
