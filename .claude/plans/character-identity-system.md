# Plan: Character Identity System — VRM S3 + Persona DB + Subscription Gate

**Status:** v1.0 — DRAFT, chưa triển khai
**Ngày tạo:** 11/08/2026 — K
**Người implement:** N
**Phạm vi:** Backend (DB schema + API + persona refactor) + Frontend (VRM loading + profile refactor) + Infra (S3 bucket + upload script)
**Effort ước tính:** 3–4 ngày

---

## 0. Problem Statement

Hiện tại 4 khía cạnh của một "nhân vật" nằm rải rác ở 4 nơi khác nhau, không có ràng buộc nào giữa chúng:

| Khía cạnh | Vị trí hiện tại | Định dạng |
|---|---|---|
| **VRM model** | `ECA_UI/frontend/src/asset/models/*.vrm` — bundle cùng Vite | `.vrm` (GLB binary) |
| **Avatar profile** (emotion mapping) | `ECA_UI/frontend/src/avatar/profiles/*.ts` — hardcoded TypeScript | `.ts` const |
| **Persona** (identity, voice, behavior) | `agenticRAG/langgraph_agents/personas/*.md` — filesystem MD | Markdown |
| **VRM manifest** (blendShape stats) | `ECA_UI/frontend/src/avatar/vrmManifest.ts` — generated static | `.ts` const |

Hệ quả:
- **Không thể gating subscription**: VRM bundle sẵn trong build → user nào cũng có tất cả model.
- **Persona không gắn với model cụ thể**: `persona_id` truyền từ frontend nhưng không có quan hệ nào đảm bảo "bronya.vrm" đi với persona "eca_friendly".
- **Thêm nhân vật = sửa 4 file ở 2 repo khác nhau**, không có single source of truth.
- **Không có metadata model**: không biết model nào có bao nhiêu spine, có blink không, có bao nhiêu emotion → không thể tự động filter model không tương thích.

---

## 1. Target Architecture

```
┌─────────────────────────────────────────────────────────┐
│                    PostgreSQL                           │
│  ┌───────────────────────────────────────────────────┐  │
│  │                characters table                   │  │
│  │  slug, display_name, description                 │  │
│  │  vrm_s3_url          ← S3 URL to .vrm file       │  │
│  │  vrm_metadata (JSONB)← joints, spine, blendshapes│  │
│  │  avatar_profile (JSONB)← emotion recipes, visemes│  │
│  │  persona (JSONB)      ← identity, rules, safety  │  │
│  │  voice_provider, voice_id, voice_language        │  │
│  │  subscription_tier    ← "free" | "premium"       │  │
│  │  thumbnail_s3_url, sort_order, is_active         │  │
│  └───────────────────────────────────────────────────┘  │
│                                                         │
│  ┌───────────────────────────────────────────────────┐  │
│  │           user_subscriptions (future)             │  │
│  │  user_id → tier, expires_at                      │  │
│  └───────────────────────────────────────────────────┘  │
└─────────────────────────────────────────────────────────┘

┌──────────────────┐       ┌──────────────────┐
│   S3 (R2/MinIO)  │       │  Frontend (Vite) │
│  models/         │       │                  │
│   bronya.vrm    │◄──────│  fetch /api/     │
│   anne.vrm      │  CORS │  characters     │
│   miku.vrm      │       │  → GLTFLoader   │
│   miki.vrm      │       │    load remote  │
└──────────────────┘       └──────────────────┘
```

### Key architectural decisions

| # | Decision | Rationale |
|---|----------|-----------|
| **D1** | **Persona trong DB (JSONB), không filesystem** | Single source of truth cho character. In-memory cache (`_persona_cache`) đã có → latency chỉ ở cache miss đầu tiên (~5ms DB query). MD files giữ lại làm seed source cho migration + dev convenience. |
| **D2** | **Avatar profile trong DB (JSONB), không TypeScript const** | Frontend fetch profile từ API thay vì hardcode. Cho phép thêm/sửa character không cần deploy frontend. Fallback: giữ `defaultProfile` làm emergency fallback nếu API lỗi. |
| **D3** | **VRM public-read trên S3 với CORS, không signed URL** | Subscription gate ở tầng API (chỉ trả về URL những model user được phép thấy), không gate ở tầng CDN. Lý do: S3 signed URL phức tạp hóa caching + renewal. Đơn giản hơn: ai có URL thì load được, nhưng chỉ user đúng tier mới nhận được URL đó từ API. |
| **D4** | **Persona identity gắn với character, không tách rời** | Mỗi character có persona riêng. Không còn `persona_id` truyền độc lập từ frontend. Frontend chọn character → backend tự resolve persona từ DB. |
| **D5** | **`vrm_metadata` extract tự động bằng script, không viết tay** | Script đọc `.vrm` file, trích xuất: joint count, spine count, blendShape group count (emotion/viseme/blink/lookAt/custom), hasBlink. Tránh drift giữa metadata và model thật. |
| **D6** | **Character identity knowledge base (Phase 2, out of scope)** | Mỗi nhân vật có thể có knowledge base riêng (lore, backstory). Sẽ là bảng `character_kb` với pgvector embeddings, query được trong conversation context. **Không triển khai trong plan này.** |

---

## 2. Database Schema

### 2.1 Migration: `characters` table

```sql
-- Migration 002: Character identity system
-- Run: python -m langgraph_agents.db.init_schema

CREATE TABLE IF NOT EXISTS characters (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    slug TEXT UNIQUE NOT NULL,                  -- "bronya", "hatsune-miku"
    display_name TEXT NOT NULL,                 -- "Bronya Zaychik"
    description TEXT,                           -- Marketing copy (~200 chars)

    -- VRM model
    vrm_s3_url TEXT NOT NULL,                   -- Full S3 URL to .vrm file
    vrm_metadata JSONB NOT NULL DEFAULT '{}',   -- Auto-extracted (see §3.2)

    -- Avatar profile (frontend expression mapping)
    avatar_profile JSONB NOT NULL DEFAULT '{}', -- Mirrors AvatarProfile type (§4.2)

    -- Persona (LLM system prompt source)
    persona JSONB NOT NULL DEFAULT '{}',        -- identity, personality, rules, safety_templates

    -- Voice
    voice_provider TEXT DEFAULT 'vieneu',       -- "vieneu" | "elevenlabs" | "coqui"
    voice_id TEXT,                              -- Provider-specific voice ID
    voice_language TEXT DEFAULT 'vi',           -- "vi" | "en"

    -- Access control
    subscription_tier TEXT NOT NULL DEFAULT 'free', -- "free" | "premium" | "enterprise"
    is_active BOOLEAN NOT NULL DEFAULT true,

    -- Display
    thumbnail_s3_url TEXT,
    sort_order INT NOT NULL DEFAULT 0,

    created_at TIMESTAMPTZ NOT NULL DEFAULT now(),
    updated_at TIMESTAMPTZ NOT NULL DEFAULT now()
);

CREATE INDEX IF NOT EXISTS idx_characters_slug ON characters (slug);
CREATE INDEX IF NOT EXISTS idx_characters_tier ON characters (subscription_tier, is_active);
```

### 2.2 `vrm_metadata` JSONB structure

```json
{
  "joint_count": 67,
  "spine_count": 4,
  "has_humanoid_rig": true,
  "blendshape_groups": {
    "total": 18,
    "emotions": 5,
    "visemes": 5,
    "blinks": 3,
    "look_ats": 4,
    "customs": 1
  },
  "has_blink": true,
  "has_look_at": true,
  "incompatible_reasons": [],
  "vrm_version": "0.0",
  "file_size_bytes": 48392011,
  "extracted_at": "2026-08-11T12:00:00Z"
}
```

`incompatible_reasons` được script tự động populate:
- `"spine_count < 3"` → không dùng được cho motion retarget
- `"blendshape_groups.total == 0"` → bronya_long case
- `[]` (empty) → model tương thích hoàn toàn

### 2.3 `avatar_profile` JSONB structure

```json
{
  "version": 1,
  "recipes": {
    "neutral": {},
    "happy": { "happy": 1.0 },
    "sad": { "sad": 1.0 },
    "angry": { "angry": 1.0 },
    "relaxed": { "relaxed": 1.0 },
    "surprised": { "Surprised": 1.0 }
  },
  "visemes": { "A": "aa", "I": "ih", "U": "ou", "E": "ee", "O": "oh" },
  "blinkChannel": "blink",
  "morphRepairMap": {
    "blink": ["43", "44"],
    "aa": "30",
    "ih": "31",
    "ou": "32",
    "ee": "34",
    "oh": "35"
  },
  "binaryEmotions": ["sad"],
  "greetingEmotion": "happy"
}
```

Mirrors trực tiếp TypeScript `AvatarProfile` interface. Frontend nhận JSON này → cast về type đã có.

### 2.4 `persona` JSONB structure

```json
{
  "title": "Bronya — ECA Friendly",
  "identity": "Name: Bronya | Role: Fitness & wellness companion | Avatar: bronya",
  "personality": "Tone: Casual, cheerful, motivating | Formality: Informal",
  "behavioral_rules": "Use casual Vietnamese (\"bạn\"). Add encouragement. Simplify medical terms. Include safety warnings in friendly tone.",
  "response_formatting": "Short paragraphs, conversational style. Use \"→\" for exercise steps. Keep under 200 words.",
  "safety_templates": {
    "red_flag_screen": "⚠️ Ui, triệu chứng này hơi đáng lo á! Bạn nên ngừng tập và đi khám bác sĩ liền nha.",
    "referral_advice": "Bạn nên gặp bác sĩ chuyên khoa để được tư vấn kỹ hơn nha!",
    "scope_disclaimer": "*Nội dung này chỉ để tham khảo thôi, không thay bác sĩ đâu nha!*"
  },
  "voice_identity": {
    "voice_path": null,
    "language": "vi"
  }
}
```

---

## 3. S3 Configuration

### 3.1 Bucket structure

```
s3://vva-assets/
  models/
    anne/
      anne.vrm              ← model file
      thumbnail.png          ← avatar thumbnail (optional)
    bronya/
      bronya.vrm
      thumbnail.png
    hatsune-miku/
      hatsune-miku.vrm
      thumbnail.png
    miki/
      miki.vrm
      thumbnail.png
```

### 3.2 CORS configuration

```json
{
  "CORSRules": [
    {
      "AllowedOrigins": ["https://your-domain.com", "http://localhost:5173"],
      "AllowedMethods": ["GET", "HEAD"],
      "AllowedHeaders": ["*"],
      "ExposeHeaders": ["ETag", "Content-Length", "Content-Type"],
      "MaxAgeSeconds": 3600
    }
  ]
}
```

### 3.3 Upload script

Script `scripts/upload-models-to-s3.py`:
1. Đọc `vrmManifest.ts` → biết danh sách model
2. Với mỗi model, đọc file `.vrm` từ `ECA_UI/frontend/src/asset/models/`
3. Extract metadata (joint count, blendShape stats) bằng `@pixiv/three-vrm` hoặc parse GLB binary
4. Upload lên S3 path `models/{slug}/{slug}.vrm`
5. Insert/update row trong `characters` table với `vrm_s3_url`, `vrm_metadata`
6. Upload thumbnail nếu có

---

## 4. Backend Changes

### 4.1 New API endpoints

#### `GET /api/characters`

Query params:
- `tier` (optional) — filter by subscription tier. Nếu không truyền → trả về tất cả (admin). Nếu có `user_id` header → filter theo tier của user đó.

Response:
```json
{
  "characters": [
    {
      "slug": "bronya",
      "display_name": "Bronya Zaychik",
      "description": "...",
      "vrm_s3_url": "https://s3.../models/bronya/bronya.vrm",
      "thumbnail_s3_url": "https://s3.../models/bronya/thumbnail.png",
      "subscription_tier": "free",
      "vrm_metadata": { ... },
      "compatible": true
    }
  ]
}
```

Note: `avatar_profile` và `persona` **không** trả về ở list endpoint (quá nặng). Frontend gọi list → user chọn character → gọi detail.

#### `GET /api/characters/{slug}`

Response: full character record, bao gồm `avatar_profile` và `persona` (đã parse).

#### `GET /api/characters/{slug}/avatar-profile`

Response: chỉ `avatar_profile` JSONB → frontend cast về `AvatarProfile` type.

### 4.2 Persona loader refactor

File: `langgraph_agents/nodes/_persona_loader.py`

**Before:**
- `get_persona(persona_id)` → đọc MD file từ filesystem
- Cache trong `_persona_cache: dict`

**After:**
```
get_persona(persona_id: str) -> dict
  1. Check _persona_cache → hit → return
  2. Query DB: SELECT persona FROM characters WHERE slug = $1
     - Found → parse JSONB, cache, return
  3. Fallback: đọc MD file (backward compat)
     - Found → cache, return
  4. Fallback: _fallback_persona() (no cache)
```

**Breaking change**: `persona_id` hiện tại là string như `"eca_default"`, `"eca_friendly"`. Sau khi migrate, `persona_id` sẽ map trực tiếp đến `characters.slug`. Các persona MD cũ được giữ lại làm fallback.

Migration mapping:
| persona_id cũ | characters.slug (mới) |
|---|---|
| `eca_default` | `seele` (nếu có model seele) hoặc `anne` |
| `eca_clinical` | TBD — cần quyết định nhân vật nào mang persona clinical |
| `eca_friendly` | `bronya` |

### 4.3 ChatRequest schema update

```python
class ChatRequest(BaseModel):
    query: str
    user_id: str = "anonymous"
    session_id: str = "default"
    character_slug: str = Field(default="anne", ...)  # ← THAY THẾ persona_id
    output_mode: Literal["text", "speech", "both"] = "text"
    ...
```

Flow mới: `character_slug` → DB lookup → lấy `persona` JSONB + `voice_provider`/`voice_id` → inject vào synthesizer + TTS.

### 4.4 Subscription tier resolution (future)

Khi có subscription system:
1. Middleware đọc `user_id` → query `user_subscriptions` → biết `tier`
2. `GET /api/characters` filter: `WHERE subscription_tier <= :user_tier` (theo thứ tự free < premium < enterprise)
3. Nếu user gọi chat với `character_slug` vượt tier → 403

Hiện tại (chưa có subscription): mọi user thấy tất cả character. `subscription_tier` trong DB là placeholder.

---

## 5. Frontend Changes

### 5.1 `MotionContext.tsx` — thay `import.meta.glob` bằng API fetch

**Before:**
```ts
const VRM_ASSET_MODULES = import.meta.glob('../asset/**/*.vrm', {
  eager: true,
  import: 'default',
}) as Record<string, string>
```

**After:**
```ts
// MotionContext initialized with empty options; fetch on mount
useEffect(() => {
  fetch('/api/characters')
    .then(res => res.json())
    .then(data => setVrmOptions(data.characters.map(c => ({
      id: c.slug,
      label: `${c.slug}.vrm`,
      url: c.vrm_s3_url,
    }))))
}, [])
```

### 5.2 `CharacterViewer.tsx` — bỏ `import` static fallback

**Before:**
```ts
import anneUrl from '../asset/models/anne.vrm'
// ...
const vrmUrl = selectedVrm?.url ?? anneUrl
```

**After:**
```ts
const vrmUrl = selectedVrm?.url
// Nếu chưa có url (đang fetch) → hiển thị loading, không render VRM
if (!vrmUrl) return <LoadingOverlay />
```

### 5.3 `AvatarProfile.ts` — fetch profile từ API

**Before:** `loadProfile(modelId)` → lookup trong `PROFILE_REGISTRY` (hardcoded TypeScript)

**After:**
```ts
async function loadProfile(modelSlug: string): Promise<AvatarProfile> {
  // 1. Check runtime cache
  if (profileCache.has(modelSlug)) return profileCache.get(modelSlug)!
  
  // 2. Fetch from API
  const res = await fetch(`/api/characters/${modelSlug}/avatar-profile`)
  if (!res.ok) return { ...defaultProfile, modelId: modelSlug } // fallback
  
  const profile = await res.json()
  profileCache.set(modelSlug, profile)
  return profile
}
```

**Tradeoff**: `loadProfile` từ sync → async. Tất cả call site cần `await`. Impact: ~5 call sites trong CharacterViewer, ExpressionController, AvatarDevPanel.

### 5.4 `vrmManifest.ts` — giữ lại nhưng đổi nguồn data

File này HIỆN TẠI vừa làm 2 việc:
1. Type definitions (`VrmBlendShape`, `VrmManifestEntry`) — giữ nguyên.
2. Static data (blendShape listing per model) — **xóa**, thay bằng fetch từ API.

Type definitions giữ lại vì:
- `VRMExpressionAdapter.ts` dùng `VrmBlendShape` type
- `ExpressionController.ts` đọc emotion list từ manifest

Data thực tế giờ nằm trong `vrm_metadata.blendshape_groups` từ API.

### 5.5 Avatar selector UI

`AvatarsPanel.tsx`:
- Hiện tại: list từ `vrmOptions` (derived từ `import.meta.glob`)
- Sau: list từ API response
- Thêm indicator: model nào locked (subscription tier > user tier) → badge "Premium"
- Thêm filter: model nào `compatible: false` → greyed out + tooltip lý do

---

## 6. Phases

### Phase 1: Database + Migration (Day 1)

| Task | File(s) | Effort |
|------|---------|--------|
| 1.1 | Thêm `characters` table vào `init_schema.sql` | `db/init_schema.sql` | 15m |
| 1.2 | Viết migration script (nếu có data cần migrate) | `db/migrations/002_characters.py` | 30m |
| 1.3 | Viết seed script: extract metadata từ `.vrm` files, map persona cũ → character mới, insert vào DB | `scripts/seed_characters.py` | 1h |
| 1.4 | Chạy migration + seed, verify data trong DB | — | 30m |

### Phase 2: Backend API (Day 1–2)

| Task | File(s) | Effort |
|------|---------|--------|
| 2.1 | `GET /api/characters` endpoint | `api/main.py`, `api/schemas.py` | 1h |
| 2.2 | `GET /api/characters/{slug}` endpoint | same | 30m |
| 2.3 | `GET /api/characters/{slug}/avatar-profile` | same | 15m |
| 2.4 | Refactor `_persona_loader.py`: DB-first, filesystem fallback | `nodes/_persona_loader.py` | 1h |
| 2.5 | Update `ChatRequest.schemas`: `persona_id` → `character_slug` | `api/schemas.py` | 30m |
| 2.6 | Update synthesizer + grader: resolve persona từ character_slug | `nodes/synthesizer.py`, `nodes/grader.py` | 1h |
| 2.7 | Unit tests cho persona loader (DB mock + fallback paths) | `local_tests/` | 1h |

### Phase 3: S3 Upload (Day 2)

| Task | File(s) | Effort |
|------|---------|--------|
| 3.1 | Provision S3 bucket + CORS config (manual hoặc Terraform) | Infra | 30m |
| 3.2 | Viết upload script: extract metadata + upload `.vrm` + update DB | `scripts/upload_models_to_s3.py` | 1.5h |
| 3.3 | Chạy upload, verify URL accessible từ browser | — | 30m |

### Phase 4: Frontend (Day 2–3)

| Task | File(s) | Effort |
|------|---------|--------|
| 4.1 | Refactor `MotionContext.tsx`: thay `import.meta.glob` bằng API fetch | `contexts/MotionContext.tsx` | 1h |
| 4.2 | Refactor `CharacterViewer.tsx`: bỏ `import anneUrl`, handle loading state | `components/CharacterViewer.tsx` | 1h |
| 4.3 | Refactor `AvatarProfile.ts`: `loadProfile` async → fetch API | `avatar/AvatarProfile.ts` | 1.5h |
| 4.4 | Update tất cả `loadProfile()` call sites thành async | `CharacterViewer.tsx`, `AvatarDevPanel.tsx`, `AvatarsPanel.tsx` | 1h |
| 4.5 | Cập nhật `AvatarsPanel.tsx`: hiển thị tier badge + incompatible indicator | `components/panels/AvatarsPanel.tsx` | 1h |
| 4.6 | Giữ lại `vrmManifest.ts` types, xóa static data | `avatar/vrmManifest.ts` | 15m |

### Phase 5: Testing + Cleanup (Day 3–4)

| Task | Effort |
|------|--------|
| 5.1 | End-to-end test: FE → chọn character → chat → persona đúng + voice đúng | 1h |
| 5.2 | Test CORS: load VRM từ S3 trên browser | 30m |
| 5.3 | Test fallback paths: DB down → filesystem persona vẫn hoạt động | 30m |
| 5.4 | Test incompatible model filter: bronya_long (0 blendshape) bị grey out | 30m |
| 5.5 | Update `vrmManifest.ts` regeneration script (nếu còn cần cho dev) | 15m |
| 5.6 | Cleanup: xóa `personas/*.md` cũ? (giữ lại làm fallback — recommend giữ) | 15m |

---

## 7. Decision Records

### D1: Persona in DB vs Filesystem

**Chọn: DB (JSONB) với filesystem fallback.**

| Tiêu chí | DB | Filesystem |
|----------|----|-----------|
| Single source of truth | ✅ Một record = một character hoàn chỉnh | ❌ Persona tách rời khỏi model |
| Admin CRUD | ✅ UI sửa được | ❌ Cần SSH + restart |
| Latency | ⚠️ +5ms cache miss, 0ms cache hit | ✅ 0ms (memory-mapped file) |
| Git versioning | ❌ Không track được thay đổi text | ✅ Diff rõ ràng |
| Dev authoring | ❌ Phải viết SQL/script để update | ✅ Mở MD, gõ, save |

**Mitigation cho DB cons**:
- **Latency**: `_persona_cache` đã có → cache hit = 0ms. Cache miss chỉ xảy ra 1 lần/worker/persona.
- **Git versioning**: Seed script export DB → MD files khi cần audit. MD files vẫn giữ trong repo làm "source of truth" cho migration.
- **Dev authoring**: Vẫn viết persona trong MD → chạy seed script → DB update. Best of both worlds.

### D2: Avatar profile trong DB vs Frontend code

**Chọn: DB (JSONB) với frontend `defaultProfile` fallback.**

Lý do giống D1: thêm nhân vật không cần deploy frontend. Frontend vẫn giữ `defaultProfile` và `bronyaProfile` (v.v.) trong code làm emergency fallback nếu API lỗi.

### D3: VRM access control — S3 signed URL vs API gate

**Chọn: API gate (public-read S3 + CORS).**

| Tiêu chí | API gate | S3 signed URL |
|----------|----------|---------------|
| Implementation | Đơn giản — chỉ filter ở API | Phức tạp — generate, expire, renew |
| Caching | ✅ CDN cache được | ❌ Mỗi URL khác nhau → cache miss |
| Security | ⚠️ URL leak = ai cũng load được | ✅ URL hết hạn sau N phút |
| Subscription check | ✅ Check ở API mỗi lần fetch | ⚠️ Phải check ở API để lấy signed URL, vẫn 1 roundtrip |

Với mô hình hiện tại (không phải content siêu nhạy cảm), API gate là đủ. Nếu sau này cần bảo vệ model khỏi hotlink, thêm CloudFront với signed cookies.

---

## 8. Open Items

| # | Item | Owner | Deadline |
|---|------|-------|----------|
| O1 | Chọn persona → character mapping cuối cùng (anne/bronya/miku/miki → persona nào?) | Owner | Trước Phase 1 |
| O2 | S3 provider: AWS S3 / Cloudflare R2 / MinIO self-hosted? | Owner + K | Trước Phase 3 |
| O3 | Voice ID cho từng character (VieNeu-TTS voice nào cho mỗi nhân vật?) | Owner | Trước Phase 2 |
| O4 | Character identity knowledge base — plan riêng hay gộp vào plan này? | Owner | — |
| O5 | Có cần giữ backward compat cho `persona_id` trong API (để frontend cũ vẫn chạy)? | K + N | Trước Phase 2 |

---

## 9. Rollback Plan

Nếu có vấn đề sau khi deploy:
1. **Frontend**: revert về `import.meta.glob` (giữ nguyên code cũ, feature-flag bằng env var `VITE_USE_S3_MODELS=false`)
2. **Backend**: `_persona_loader` có filesystem fallback → nếu DB lỗi, persona MD vẫn hoạt động
3. **DB**: `characters` table là additive-only → không ảnh hưởng bảng cũ
