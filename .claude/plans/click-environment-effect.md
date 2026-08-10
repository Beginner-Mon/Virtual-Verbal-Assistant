# Plan: Click Environment Effect — 3D Canvas

**Status:** DRAFT — chờ owner approve  
**Ngày tạo:** 09-08-2026 — K  
**Người implement:** N  
**Phạm vi:** 1 file mới + 1 file sửa trong `ECA_UI/frontend/src/`  
**Effort ước tính:** 0.5–1 ngày

---

## 0. Problem

Click vào canvas 3D → **không có phản hồi gì**. User kỳ vọng có visual feedback đơn giản xác nhận tương tác.

---

## 1. Architecture

```
User click trên <div> (screen space)
        │
        ▼
CharacterViewer.handleClick(e)
        │  convert clientX/Y → {x, y} relative to canvas
        ▼
React state: clickPosition + clickId (key để remount)
        │
        ▼
<ClickRipple key={clickId} position={clickPosition} />
        │  bên trong <Canvas>, dùng useThree() để
        │  unproject screen → world position
        ▼
THREE.RingGeometry / ShaderMaterial ring
  → scale từ 0 → 1.5 trong 0.6s
  → opacity từ 0.6 → 0
  → tự dispose sau animation
```

**Không đụng đến:** FSM, avatarRef, AnimationController, emotion, character.

---

## 2. Implementation

### 2.1 New: `components/scene/ClickRipple.tsx`

Component R3F nhận `position: {x: number, y: number}` (screen space).

- Dùng `useThree()` lấy `camera` + `size`
- `useEffect` trigger khi mount → play animation → tự `dispose` geometry + material
- Sử dụng `THREE.RingGeometry` hoặc `THREE.CircleGeometry` với opacity fade
- Ring đặt ở world position tương ứng với click, mặt phẳng XY (vuông góc camera)
- Cleanup: remove khỏi scene sau 0.8s

**Hoặc simpler:** Dùng `@react-three/drei` `<Html>` để render DOM-based ripple (CSS animation) — không cần dispose logic phức tạp, perf tốt hơn cho effect tạm thời.

### 2.2 Modified: `CharacterViewer.tsx`

- Thêm `onClick` vào outer `<div>` (line 583 area)
- State: `clickEvent: { x: number, y: number, id: number } | null`
- Render `<ClickRipple key={clickEvent.id} position={clickEvent} />` bên trong `<Canvas>`, sau `<Scene>` component
- Throttle 200ms qua `useRef(lastClickTime)`

---

## 3. Checklist

| # | Task | File | Effort |
|---|---|---|---|
| 1 | Tạo `ClickRipple.tsx` — ring expand+fade, tự dispose | NEW | 2-3h |
| 2 | Thêm `handleClick` + state + render ClickRipple trong CharacterViewer | MODIFY | 1h |
| 3 | Manual test desktop + mobile | — | 0.5h |

---

## 4. Design Decision: DOM-based vs WebGL-based

| Approach | Mechanism | Pro | Con |
|---|---|---|---|
| **A: DOM (recommended)** | `<Html>` + CSS `@keyframes` | Không allocate WebGL resource, animation mượt CSS engine, tự cleanup, code ít | Ring không bị occlusion bởi model (luôn trên cùng) |
| **B: WebGL** | `THREE.RingGeometry` + `useFrame` fade | Chính xác trong world space, bị occlusion đúng | Phải manage dispose, thêm geometry/material vào scene tree |

**Recommend A (DOM)** — đơn giản, không leak, không đụng scene graph.

---

## 5. Open Question

1. **Style của ripple?** Đề xuất: ring trắng mờ (#ffffff 0.4 opacity), expand 0→60px, fade out 0.6s, ease-out. Giống mobile tap feedback nhưng tinh tế hơn.
