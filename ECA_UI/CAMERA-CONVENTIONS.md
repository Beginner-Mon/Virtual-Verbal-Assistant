# Camera & Coordinate Conventions — ECA UI (bất biến)

> **Status: verified & correct as of 2026-07-24. Do NOT change without Owner approval.**
>
> File này là rule document — nếu code có thay đổi (refactor, re-architecture, đổi engine), các convention dưới đây PHẢI được giữ nguyên.

---

## 1. World Coordinate System: Z-Up

```ts
// CharacterViewer.tsx, Scene useEffect
camera.up.set(0, 0, 1)
```

- **+Z = up**, +X = right, +Y = ... (derived from right-hand rule).
- Đây là KHÔNG PHẢI Y-up mặc định của three.js. Được chọn để khớp với VRM/MMD coordinate system.
- Ground plane = **XZ plane** (không phải XY). Tất cả grid, wireframe, ground disc, contact shadows đều dùng `rotation={[-Math.PI / 2, 0, 0]}` (xoay XY → XZ) và `position={[0, -1.5, 0]}`.

### Hệ quả

| Element | Convention |
|---|---|
| Ground / Grid | XZ plane, y = -1.5 |
| "Up" trong world | +Z |
| Lights | Position dùng +Z cho "trên cao": `[5, 5, 5]` là forward-right-up |
| AxesHelper (debug) | X=red, Y=green, Z=blue — Z là trục đứng |

---

## 2. Camera Initial Setup

```ts
// CharacterViewer.tsx, Canvas
<Canvas camera={{ position: [0, 2.05, 0], fov: 45 }} ...>
```

- Camera bắt đầu tại `[0, 2.05, 0]`.
- FOV = 45°.
- `alpha: true` + `antialias: true`.

---

## 3. Model Orientation

```tsx
// CharacterViewer.tsx, VRMCharacter return
<group position={[0, -1.5, 0]} rotation={[Math.PI / 2, 0, 0]}>
  <primitive object={vrm.scene} />
</group>
```

- **VRM rest pose** (three-vrm normalized): body up = +Y local, face ±Z local.
- **Group rotation X+90°** (`[π/2, 0, 0]`): maps local +Y → world +Z (body upright trong Z-up world), local +Z → world -Y (face direction).
- Nhờ rotation này, rest pose luôn Z-up + face hướng về camera mà KHÔNG cần BVH.
- **BVH retargeting**: `bvhToVrm.ts` áp dụng `HIPS_ORIENT_COMPENSATION` lên hips bone mỗi frame, đảm bảo BVH animation vẫn đúng dù group rotation đã thay đổi (từ `[0, π, 0]` legacy → `[π/2, 0, 0]`). Xem `bvhToVrm.ts` dòng 18-27.
- Group position `[0, -1.5, 0]`: chân model chạm ground.

---

## 4. Camera Modes & Follow Logic

```ts
// CharacterViewer.tsx
const CAMERA_MODES = {
  head:  { boneName: VRMHumanBoneName.Head, cameraOffset: new THREE.Vector3(0, 0.5, 0) },
  hips:  { boneName: VRMHumanBoneName.Hips, cameraOffset: new THREE.Vector3(0, 2.1, 0) },
}
```

- **Head mode**: camera offset `[0, 0.5, 0]` từ head bone — close-up khuôn mặt.
- **Hips mode**: camera offset `[0, 2.1, 0]` từ hips bone — toàn thân.
- **Follow**: mỗi frame, camera target bám theo bone được chọn. Camera position được shift bởi delta của bone (giữa frame hiện tại và trước) → khoảng cách orbital được bảo toàn ngay cả khi model di chuyển (BVH animation).

### OrbitControls

```tsx
<OrbitControls
  ref={controlsRef}
  enablePan={false}        // cấm pan — chỉ orbit + zoom
  enableZoom={true}
  minDistance={1}
  maxDistance={20}
  target={[0, 0, 0]}       // target ban đầu, bị follow logic override mỗi frame
/>
```

- Pan **bị vô hiệu hóa** — camera luôn quay quanh model.
- Zoom cho phép, khoảng cách 1–20 units.
- `target` bị `useFrame` follow logic ghi đè liên tục.

---

## 5. VRMLookAt / Eye Tracking Sign Convention

Được document chi tiết trong `EyeController.ts` JSDoc. Tóm tắt:

| Input | three-vrm yaw/pitch | Hướng mắt (world) | Hướng mắt (screen, viewer-facing) |
|---|---|---|---|
| Chuột phải (nx=+1) | yaw +22° | model +X local → world -X | **đúng: phải** |
| Chuột trái (nx=-1) | yaw -22° | model -X local → world +X | **đúng: trái** |
| Chuột lên (ny=+1) | pitch **-12°** (negated in EyeController) | gaze +Z world (lên) | **đúng: lên** |
| Chuột xuống (ny=-1) | pitch **+12°** | gaze -Z world (xuống) | **đúng: xuống** |

- three-vrm `VRMLookAt.lookAt(position)` tự tính pitch = `altitudeFrom - altitudeTo`. Target ở trên → altitudeTo > 0 → pitch âm = look UP. Đây là convention của thư viện, KHÔNG thay đổi.
- EyeController negate `ny` để map screen-up → pitch âm → look up.
- Xem `EyeController.ts` để biết chi tiết implementation.

---

## 6. Lighting (Z-up consistent)

```tsx
<ambientLight intensity={...} />
<directionalLight position={[5, 5, 5]} ... />       // forward-right-up
<directionalLight position={[-5, 3, -5]} ... />     // back-right-down
<spotLight position={[0, 5, 0]} ... />              // front-right, at height
```

Tất cả position dùng Z-up: `position.z` = chiều cao.

---

## 7. Không được thay đổi

| # | Item | Lý do |
|---|------|-------|
| 1 | `camera.up = (0,0,1)` | Z-up là identity của toàn bộ scene — ground, model, lights, eye tracking đều dựa trên nó |
| 2 | Group rotation `[0, π, 0]` | Model orientation — eye tracking, camera angle, BVH retarget đều giả định model face -Z world |
| 3 | Group position `y = -1.5` | Ground plane alignment |
| 4 | OrbitControls `enablePan: false` | UX: camera luôn quay quanh model |
| 5 | `PITCH_MAX_DEG = 12`, `YAW_MAX_DEG = 22` | Đã calibrate cho model VRM hiện có — thay đổi cần test lại toàn bộ |
| 6 | EyeController pitch negation | Khớp với three-vrm `VRMLookAt` convention (negative pitch = look up) |
| 7 | `CAMERA_MODES` offsets | Calibrate cho model kích thước hiện tại |
