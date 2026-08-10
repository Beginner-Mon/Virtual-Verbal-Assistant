# Plan: Click Character Face → Trigger Reaction

**Status:** DRAFT — chờ owner approve  
**Ngày tạo:** 09-08-2026 — K | **Revised:** 09-08-2026 (scope narrowed to face only)  
**Người implement:** N  
**Phạm vi:** 2 files mới + 1 file sửa trong `ECA_UI/frontend/src/`  
**Effort ước tính:** 1–1.5 ngày  
**Follow-up plan:** Click body parts (torso, arm, leg) → plan riêng sau

---

## 0. Problem

Click vào mặt nhân vật 3D → cần phản ứng facial emotion khác nhau tùy vị trí (đầu, mắt, miệng). Không đụng body, không FSM.

---

## 1. Architecture

```
User click trên face mesh (R3F pointer event)
        │
        ▼
VRMCharacter: onPointerDown trên <primitive object={vrm.scene}>
        │  R3F raycast → event.object = mesh bị click
        │  drag guard: distance < 3px + time < 300ms
        │
        ▼
identifyFaceRegion(intersectedMesh, vrm)
        │  walk up parent chain → VRMHumanBoneName
        │
        ▼  FaceRegion ─────────────────────────────
        │  'crown'          Head
        │  'forehead'       Head (upper half of bounds)
        │  'left_eye'       LeftEye
        │  'right_eye'      RightEye
        │  'nose'           Head (center, between eyes & jaw)
        │  'mouth'          Jaw
        │  'chin'           Jaw (lower half)
        │  'cheek'          Head (side, not nose/eye)
        │
        ▼
FACE_REACTIONS[region]
        │
        └── avatarControllerRef.current.setEmotion(emotion, intensity, durationMs)
```

---

## 2. Face Region Detection

### 2.1 Bone Mapping (`lib/faceRegionDetector.ts` — NEW)

VRM cung cấp các bone vùng mặt:

| VRMHumanBoneName | Vị trí giải phẫu |
|---|---|
| `Head` | Toàn bộ đầu (crown, forehead, cheek, nose) |
| `LeftEye` | Mắt trái |
| `RightEye` | Mắt phải |
| `Jaw` | Miệng + cằm |
| `Neck` | Cổ |

Vấn đề: `Head` bone cover quá rộng. Cần phân biệt forehead vs cheek vs nose trong cùng bone `Head`.

**Giải pháp:** Dùng local position của hit point relative to `Head` bone để chia nhỏ:

```
Lấy headBone.worldToLocal(intersection.point) → (x, y, z) trong head space
  → y > threshold_upper   → 'crown'       (đỉnh đầu)
  → y > threshold_mid     → 'forehead'    (trán)
  → y > threshold_lower   → 'nose' | 'cheek' (giữa mặt)
  → else                  → 'chin'        (cằm, đã thuộc Jaw bone)
```

Threshold được calibrate từ head bone bounds (dùng bounding box của head mesh hoặc hardcode tỉ lệ).

### 2.2 FaceRegion type

```ts
type FaceRegion =
  | 'crown'
  | 'forehead'
  | 'left_eye'
  | 'right_eye'
  | 'nose'
  | 'mouth'
  | 'chin'
  | 'cheek'
  | 'unknown'
```

**Detection priority:** bone match trước (LeftEye, RightEye, Jaw) → nếu là Head thì dùng local Y position.

---

## 3. Reaction Map

```ts
const FACE_REACTIONS: Record<FaceRegion, { emotion: string; intensity: number; durationMs: number }> = {
  crown:     { emotion: 'surprised', intensity: 0.6, durationMs: 400 },
  forehead:  { emotion: 'surprised', intensity: 0.5, durationMs: 350 },
  left_eye:  { emotion: 'surprised', intensity: 0.9, durationMs: 300 },
  right_eye: { emotion: 'surprised', intensity: 0.9, durationMs: 300 },
  nose:      { emotion: 'surprised', intensity: 0.7, durationMs: 350 },
  mouth:     { emotion: 'angry',     intensity: 0.6, durationMs: 400 },
  chin:      { emotion: 'surprised', intensity: 0.4, durationMs: 300 },
  cheek:     { emotion: 'happy',     intensity: 0.5, durationMs: 400 },
  unknown:   { emotion: 'surprised', intensity: 0.4, durationMs: 300 },
}
```

---

## 4. Hit Particle tại Face

Particle burst tại `intersection.point` (world space), màu theo region:

| Region | Particle color |
|---|---|
| eye | vàng nhạt (`#fde68a`) |
| mouth/chin | đỏ nhạt (`#fca5a5`) |
| cheek | hồng (`#f9a8d4`) |
| crown/forehead/nose | trắng-xanh (`#bae6fd`) |

---

## 5. Drag vs Click Guard

Chỉ trigger nếu:
- `pointerdown → pointerup` distance < 3px VÀ
- `pointerdown → pointerup` time < 300ms

Nếu vượt ngưỡng → coi là OrbitControls drag → không trigger reaction.

---

## 6. Files Changed

| # | File | Mới/Sửa | Nội dung | Effort |
|---|---|---|---|---|
| 1 | `lib/faceRegionDetector.ts` | **NEW** | Walk up parent chain → bone → map Y position → FaceRegion | 2.5h |
| 2 | `lib/clickReactions.ts` | **NEW** | FaceRegion → { emotion, intensity, durationMs } map | 0.5h |
| 3 | `CharacterViewer.tsx` (VRMCharacter) | **SỬA** | onPointerDown + drag guard + detect + trigger emotion | 2h |

**Không cần HitParticle riêng** nếu owner đồng ý — facial reaction đã đủ feedback. Particle có thể thêm sau.

---

## 7. Risks

| Risk | Mitigation |
|---|---|
| Head bone local Y threshold khác giữa model (Anne/Bronya/Seele) | Dùng bounding box của head mesh để tính tỉ lệ động, không hardcode mm |
| R3F pointer event không fire trên VRM `<primitive>` | Test early. Nếu fail → fallback sang manual raycaster + `pointerdown` capture trên canvas |
| VRM mesh có nhiều sub-mesh (tóc, lông mi,...) → bone walk-up ra sai bone | Chỉ match bone name với `VRMHumanBoneName` enum, ignore mesh không thuộc skeleton |
| Emotion thiếu trong AvatarProfile model nào đó | Fallback về `'surprised'` nếu emotion đích không tồn tại |

---

## 8. Open Questions

1. **Reaction map §3 ổn chưa?** (chạm mắt → surprised mạnh, miệng → angry, má → happy, còn lại surprised nhẹ)
2. **Có cần HitParticle không?** Hay facial emotion là đủ feedback?
3. **Cần phân biệt cheek trái/phải không?** Hay gộp chung `'cheek'`?
