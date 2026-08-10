# VVA — Status & Roadmap

> Last update: 2026-08-08 (K) | Branch: `feature/langgraph-rewrite`
> Audience: K/N/Owner takeover after context compaction — đọc mục 0 trước tiên.

---

## 0. TRẠNG THÁI ĐANG TREO (đọc trước — dễ mất khi compact)

- **Service lúc viết file này (08/08)** — kiểm bằng curl, không phải nhớ:
  backend `:8000` ✅ 200 · **TTS `:5000` ✅ 200** · frontend `:5173` ✅ 200 ·
  docker `vva-postgres`/`vva-redis`/`vva-searxng` up 8 ngày ✅
  > `vva-postgres` local **không còn là DB đang dùng** — chỉ là đường lùi.
  > TTS phải chạy bằng conda env **`tts`**, không phải `firstconda`.
  > 🔴 **Backend phải chạy bằng `firstconda`.** Trên máy này `python` trần =
  > Python312, **thiếu `langchain_google_genai`**. Import của nó là *lazy* nên
  > backend vẫn khởi động bình thường và chỉ chết lúc Gemini fallback được gọi —
  > tức đúng lúc DeepSeek đang hỏng. `pytest` cũng phải chạy bằng `firstconda`
  > (Python312 cho 8 test đỏ giả). Worklog 10/08 §0.
- **Config backend: `agenticRAG/.env`** (10/08). Loader duy nhất:
  `langgraph_agents/shared/env.py`. File cũ `agentic_rag_gemini/.env` vẫn được chấp
  nhận nhưng **log WARNING** — chưa xoá, chờ N xác nhận. Mẫu đầy đủ: `agenticRAG/.env.example`.
- **PostgreSQL trên NEON** (`ap-southeast-1`). DSN ở `agenticRAG/agentic_rag_gemini/.env`
  (gitignored), **direct endpoint** không phải `-pooler`.
  - Rollback: xoá dòng `VVA_PG_DSN` → restart backend → về local.
  - ⚠️ **Redis + file motion/audio VẪN Ở LOCAL** — mới chuyển 1/3 kho.
- **Git**: HEAD `b7ed02e` (Tri, 07/08 — đổi model sang Ane/anne).
  **22 file chưa commit** — 7 auth + `ingest_kb_pgvector.py` + health (3) + compose + 4 file
  test/config FE + package(-lock) + 3 doc. K không tự commit.

### 🔴 CHẶN CỨNG — chỉ Owner gỡ được

- **Không có AWS credentials** ⇒ `npx ampx sandbox` chết ở `InvalidCredentialError`.
  Hệ quả: **toàn bộ thay đổi auth backend chưa chạy thật lần nào.** Chống trùng tài khoản
  mới chỉ chứng minh được là *biên dịch được*.
  Gỡ: `cd ECA_UI/frontend && npx ampx configure profile`
  Kèm 2 việc phải làm cùng lúc: đặt `VITE_GOOGLE_CLIENT_ID`, và thêm origin của app vào
  **Authorized JavaScript origins** ở Google Cloud console (GIS chặn theo origin — khác
  danh sách redirect URI của hosted UI).
- **Không có JDK/Android SDK**, iOS bất khả trên Windows ⇒ `src/lib/nativeAuth.ts` chưa
  thực thi lần nào, chưa có thư mục `android/`.
- **Chưa có domain cho App Links** ⇒ mobile auth không hoàn tất được. Checklist:
  `docs/mobile-app-links.md`.
- Danh sách đầy đủ 13 mục: **`docs/tracking/tech-debt.md` mục 🚧** (đọc mục đó trước tiên).

### ✅ Chạy được NGAY, không cần deploy

- **Bug đăng nhập Google chọn nhầm tài khoản rồi kẹt** — 3/4 lớp đã sửa.
  Báo cáo có dẫn nguồn: `docs/auth-google-incident.md`. Gốc rễ: *xoá localStorage bị nhầm
  là đăng xuất* — phiên Cognito là cookie trên domain khác.
  Lớp thứ 4 (chống tạo user trùng) chờ deploy.
- **Ingest không thể xoá KB nữa** — đảo thứ tự (embed trước, `DELETE`+insert trong 1
  transaction) + từ chối chạy khi backend đang bật (exit 2, có `--force`).
  KB verify 2918/2918.
- **TTS + lip sync + session persistence** — worklog §4, §6, §8.
- **`/health/detailed` không còn 503 vì TTS chết** — critical vs optional đã tách.
  Worklog 08/08 §1.
- **Docker tự dậy lại sau reboot** — `postgres` + `redis` có restart policy. Worklog 08/08 §2.
- **FE có test runner** — `npm test` trong `ECA_UI/frontend`: **44 test / 4 file**
  (`AnimationController`, `shadowFit`, `groundClamp`, `googleSignIn`). Worklog 08/08 §3.
- **Mất bảng chọn tài khoản Google sau đăng xuất** (Tri báo 09/08) — ✅ **SỬA XONG**,
  frontend thuần, không cần deploy. `prompt: 'SELECT_ACCOUNT'` trở lại vô điều kiện +
  7 test pin lại. Worklog 09/08.

### ⭐ Việc đang làm — thứ tự ưu tiên đã Owner chốt (30/07)

| # | Việc | Trạng thái |
|---|---|---|
| **P0** | **PostgreSQL → Neon** | ✅ **XONG 05/08** — cutover rồi, chat chạy trên Neon |
| **P0** | **7 lỗi TS ở FE** | ✅ **XONG** — kèm 1 bug thật `getManifest` không bao giờ khớp `Con-Gai-Khang`. Worklog §3 |
| **P0** | **Deploy auth (Đợt 1+2)** | 🔴 **CHẶN** — không có AWS creds. Code xong, chưa chạy thật |
| **P0** | **Bug Google chọn nhầm tài khoản** | ✅ **3/5 lớp XONG**, chạy ngay. `docs/auth-google-incident.md` |
| **P0** | **Chọn nhầm vẫn TẠO user `b@`** (Tri báo 08/08) | 🔴 **CHẶN** — cách sửa đã code xong (luồng link bỏ hosted-UI, `POST /api/user/link-google`), chưa deploy. Worklog 08/08 §5 |
| **P0** | **Ingest xoá sạch KB** | ✅ **XONG** — không thể tái diễn. Worklog §13 |
| **P0** | **KB ingest vào pgvector** | ✅ **XONG** — 2918 rows, giờ nằm trên Neon |
| **P0** | **Deploy + verify ALB Kimodo** | ⏸ **Chờ N/Owner** — K KHÔNG tự làm (cần AWS creds + tốn ~$24/ngày g5.xlarge + $0.75/ngày ALB). Plan `.claude/plans/kimodo-alb-endpoint.md` đã implemented, chỉ cần chạy checklist §5 (**nhớ đổi `ALB_ALLOWED_CIDR` → IP/32 trước deploy**) |
| **P1** | **FSM refactor animation** | ✅ **XONG 30/07** — K implement, **17/17 test xanh**. Xem mục dưới + plan §12 |
| **P2** | Kimodo runtime delivery (generate→convert→serve→SSE→play) | Chờ P0-ALB + P1. Frontend hiện **0 dòng** consume motion từ SSE |
| **P3** | Backend emit `avatar.emotion` | Facial FE sẵn sàng, backend chưa emit |
| **P4** | Head-follow Phase 2 (additive blend), pre-cloud hardening | P2 xong mới biết có cần |

- **Neon migration — XONG 05/08.** Verify thật: 8 bảng · 17 index (đủ 2 HNSW) · KB **2918** ·
  chat `mode: synthesize` 703 token · unit tests **275 passed** · dữ liệu ghi vào Neon (đối chứng
  local không có).
  - **Chi phí DB thật**: **12 query / 1,41 s / lượt = 4,8%** (pool ấm). Đo bằng counter trong
    `PostgresClient` + `/debug/pgstats` (bật bằng `VVA_PG_STATS=1`), **không phải ước lượng** —
    K đã đoán sai 2 lần trước khi đo (chi tiết ở worklog).
  - Lượt ấm **29,3 s ≈ local 28,4 s** → Neon **không** làm chậm đáng kể. Việc đáng làm nhất là
    **giữ pool ấm** (pool nguội 3,29 s vs ấm 1,41 s).
  - ✅ **BẪY NÀY ĐÃ BỊT (08/08)** — trước đây chạy `ingest_kb_pgvector.py` lúc backend đang bật là
    segfault exit 139 (tranh chấp torch), mà `--reset` đã xoá bảng trước ⇒ KB rỗng. K đã dính 05/08.
    Nay script **từ chối chạy** khi thấy `:8000` sống (exit 2, có `--force`), và `DELETE` chỉ xảy ra
    **sau khi** embed xong, trong cùng 1 transaction. Worklog §13.
  - 🔑 **Đổi mật khẩu Neon** trước khi có dữ liệu người dùng thật (DSN đã đi qua kênh chat).
- **P0 KB ingest — ĐÃ XONG, verify end-to-end** (30/07): `kb_embeddings` từ **0 → 2918 rows**.
  Script mới `scripts/ingest_kb_pgvector.py` (thay 2 script legacy target ChromaDB), parse
  `agenticRAG/agentic_rag_gemini/data/knowledge_base/documents.txt` (2918 bài tập, phân cách `---`).
  Chạy lại: `python scripts/ingest_kb_pgvector.py --reset` (idempotent theo `source_type`).
  - **Bug tìm được + fix**: migration 002 tạo HNSW cho `summaries` nhưng **bỏ sót** `kb_embeddings`
    → `kb_search` seq scan. Thêm migration `003_kb_hnsw`, đã `alembic upgrade head`, verify planner
    dùng thật (`Index Scan using idx_kb_emb_embedding`).
  - Verify: `kb_search` trả 5 results (trước 0), sim 0.92 EN / 0.82 VI; pipeline thật →
    `mode: synthesize` (**không còn refuse**), câu trả lời nêu đúng tên bài tập trong DB.
  - ⚠️ **Câu hỏi CHƯA quyết cho Owner**: corpus này là **gym/fitness** (Abdominals, Quadriceps,
    "EZ-Bar Skullcrusher", 17 nhóm cơ), **không phải corpus PT lâm sàng**. Fix xong lỗi "KB rỗng →
    refuse" nhưng chưa làm nó thành KB lâm sàng. 53% record (1550/2918) có `Description: nan`
    (pandas NaN lọt export) → chỉ index được structured fields. Owner đã nói "gác lại, ưu tiên sửa
    plan trước".
- **Kimodo (merge 29-30/07 của Tri)**: đã verify **offline end-to-end PASS**. Pipeline
  `NPZ → scripts/kimodo_npz_to_bvh.py → BVH → frontend SMPLX retarget → play` chạy đúng: demo NPZ
  → 90 frame → 22 tracks/2.97s, avatar múa tự nhiên, **không double-mirror** (converter có mirror+
  `_swap_lr` + orientation correction, frontend có `mirrorZ`+`swapYandZ`+`hipCompensation` — hai lớp
  compose đúng). Cách verify lại: mở `:5173` → nav **Motion** → dropdown Motion Source → chọn
  `motions/generated/motion_*.bvh`.
  - **Gap còn lại**: converter là **CLI thủ công**, chưa ai gọi từ backend; `kimodo_node` trả
    `str(result)` chưa rõ format; frontend chỉ chơi asset bundle sẵn (build-time), chưa fetch runtime.
    Đó là P2.
- **Plan FSM refactor (`.claude/plans/animation-fsm-refactor.md` v1.2)**: K đã sửa **9 lỗi** của
  v1.0, trong đó **2 lỗi 🔴 sẽ gây bug thật** nếu implement nguyên bản:
  1. 🔴 Deadlock: `exercise_cooldown` kẹt vĩnh viễn → chat sau đó không chạy thinking → **bỏ hẳn
     state đó** (8→7), camera cooldown thành timer trong CameraController.
  2. 🔴 Thiếu `thinking_loop → exercise` → motion về lúc đang thinking bị drop → thay allow-list
     bằng **nguyên tắc**: state do user/backend lái (`thinking_intro`, `exercise`) reachable từ MỌI state.
  3. ⚠️ Giữ **debug file selector** (v1.0 xoá) — đó là đường verify Kimodo offline duy nhất.
  4. ⚠️ Bỏ "T-pose bug" khỏi động lực: code **đã** play-before-fade + readiness gate → là invariant
     phải BẢO TOÀN, thành test gate #1.
  5. ⚠️ Thiếu `newAction.reset()` → `exercise` lần 2 đứng ở frame cuối.
  6. ⚠️ `registry.invalidate()` phải gọi khi đổi VRM model (clip đã retarget bám skeleton).
  7. 📝 Thêm **§9**: Facial/Emotion controller **chưa đồng bộ** với CharState (2 state machine độc
     lập) → thân "suy nghĩ" mà mặt tự cười, cười lúc demo bài tập, head-follow đè chuyển động đầu
     của clip. Follow-up, ngoài scope PR. **Giải pháp phải là DATA không phải API** (emotion vốn
     declarative trong `profiles/*.ts`): policy nằm ở trường `facial` trong bảng `STATES` (§3.0)
     + sửa 1 điều kiện ở `AvatarController.ts:115`. Không thêm public method nào.
  8. 🔴 **Boot sequence sai**: plan ghi `Khởi tạo → transitionTo('idle')`, nhưng code hiện tại boot
     vào **`action_greeting` rồi mới về idle** (`MotionContext.tsx:93-97` + `isAction` do URL chứa
     `action_` → LoopOnce → `finished` → idle). Implement nguyên bản = **lặng lẽ xoá lời chào**.
     → thêm **§2.5**: boot = `transitionTo('greeting')`, bắt buộc fallback `'idle'` nếu clip fail
     (không fallback → `currentAction=null` → **T-pose**), + `hasGreetedRef` để không chào lại khi
     đổi VRM model / StrictMode double-mount.
  9. ⚠️ **Chi phí bảo trì khi thêm animation** (Owner hỏi trực tiếp): plan có **6 map song song** cùng
     khoá `CharState` → thêm 1 animation phải sửa **4-7 nơi**, 3 nơi **không có compile check** →
     đúng error class đã sinh ra 2 lỗi 🔴. → thêm **§3.0 `lib/AnimationStates.ts`**: một bảng
     `STATES: Record<CharState, StateDef>` (exhaustive, discriminated union `loop:'once' ⇒ onFinished`),
     reachability thành **hàm** `canTransition()` đọc `STATES[to].reach` (khai trên state ĐÍCH → thêm
     state không phá được state khác), mọi map khác thành derived getter. `TransitionConfig.ts` bị bỏ.
     → thêm **§11 cookbook**: thêm animation = **1 entry + 1 dòng trigger**, 0 chỗ mất compile check;
     controller/context/panel/ChatPanel **không phải sửa**. Giới hạn ghi rõ: bảng này không giải quyết
     blend nhiều clip cùng lúc (đó là additive blending, phải mở `AnimationController`).
  Plan có **§7 test checklist 10 mục** (v1.0 không có), mỗi lỗi trên có 1 test tương ứng.
- **P1 FSM refactor — ĐÃ IMPLEMENT XONG 30/07** (K), worklog `docs/worklogs/30-07-2026.md`, báo cáo
  chi tiết ở plan **§12**. 6 file mới + 4 file sửa:
  - 🆕 `lib/AnimationStates.ts` (bảng `STATES` + `canTransition` + derived getters),
    `lib/motionAssets.ts` (nơi duy nhất glob asset — **ngoài plan**), `lib/AnimationRegistry.ts`,
    `lib/AnimationController.ts`, `lib/CameraController.ts`, `hooks/useFsmTriggers.ts`.
  - ♻️ `MotionContext.tsx` (bỏ `selectedMotionId`/`isThinking`/3 effect string-match),
    `CharacterViewer.tsx` (bỏ ~95 dòng clip-loading/fade/subclip/`isAction`),
    `MotionControlPanel.tsx` (2 dropdown: state derived + file debug), `ChatPanel.tsx`
    (`setIsThinking` → `transitionTo` + guard `thinkingRef`).
  - ❌ `lib/TransitionConfig.ts` **không tạo** — gộp vào `crossfadeFor()`.
  - **Cả 7 chỗ string-match ở §0 plan đã biến mất** — không còn nơi nào quyết định hành vi animation
    dựa trên tên file.
  - **Test 17/17 xanh** (Playwright, `:5173` + backend `:8000`): deadlock regression `true`;
    motion-trong-thinking OK; T-pose invariant **0 frame mất pose** qua 10 transition/1s;
    Kimodo BVH OK; `reset()` fix (cùng clip 2 lần: 5483/**5213**ms); camera cooldown
    head→hips→(+0.9s) hips→(+3.5s) head; đổi model không méo; chat sequence
    `["thinking_intro","thinking_loop","thinking_outro"]`→idle; **boot = `greeting` đúng 1 lần**
    (StrictMode không chào 2 lần); đổi model **không** chào lại. `tsc --noEmit` + `npm run build` xanh.
  - 🔧 Thêm `window.__fsm` (**chỉ DEV**, ở `MotionProvider` nên luôn mount): `.state`, `.hasPose`,
    `.cameraMode`, `.history`, `.transitionTo()`, `.playMotionFile()`, `.setVrm()`. Dùng để test tự
    động **không cần mở panel** — khác `window.__avatar` (chỉ có khi MotionControlPanel mở).
  - **Chưa làm (đúng scope)**: §9 facial↔body sync. Chỗ cắm đã sẵn — `facialOf(state)` trong
    `AnimationStates.ts`, `stateChanged` emitter, thứ tự `useFrame` body → facial → `vrm.update`.
- **Bug "UI chớp đen" — ĐÃ FIX 30/07, thủ phạm là BLOOM** (chi tiết + bảng bisect ở worklog §4).
  Nguyên nhân thật: `EffectComposer` → `<Bloom>` intermittently xuất **1 frame đen đơn lẻ** (~3-5s/lần,
  luma 17.8 giữa các frame ~220), không liên quan sự kiện app. → `bloom.enabled = false`.
  Verify: **0 frame đen / 1148 frame** (headed CDP screencast ~59fps). ⚠️ Đo bằng headless là VÔ NGHĨA
  (không có compositor thật → readback rỗng, screencast 39 frame/20s).
  Ngoài ra `prefetchStatic()` xoá stall load clip lần đầu (111-214ms → 35ms).
- **Shadow frustum auto-fit — 30/07** (`ECA_UI/frontend/src/lib/shadowFit.ts`, worklog §5): frustum
  cũ cố định 5×5 **ở gốc toạ độ** → nhân vật chỉ chiếm 7-8%, lề chỉ 0.43 trong khi subject dịch
  >1.2/clip. Giờ fit theo **skeleton ∪ bóng chiếu xuống sàn** mỗi frame, hình cầu (bất biến khi xoay)
  + snap texel + **giữ nguyên hướng đèn**. Kết quả: texel 0.488 → **0.245-0.359 cm** (sắc hơn
  ~1.8×) với `mapSize` **không đổi**. Config cũ `cameraSize/cameraNear/cameraFar` đã bỏ, thay bằng
  `fitPadding` + `fitGroundZ`. **Đã tái hiện + xác nhận bằng ảnh A/B** (`floor-HEAD.png` bóng bị chém
  thẳng đứng, `floor-FIT.png` bóng nguyên vẹn).
  - ⚠️ **Bài học**: kiểm containment bằng **xương là SAI** — thứ đổ bóng là **mesh** (váy/tóc/ruy-băng
    vươn xa hơn xương), nên box cũ *qua* được test xương trong khi trên màn hình vẫn bị chém. Vì vậy
    `fitPadding` (0.35) là **bù xương→mesh**, không phải trang trí.
  - ✅ **Nguyên nhân THỨ HAI — ĐÃ SỬA 31/07 bằng `lib/groundClamp.ts`** (worklog §5): `motion_b28e8284`
    tụt **−120.7 cm** so với frame 0; `retargetBVHToVRM` neo frame 0 vào chiều cao hông lúc nghỉ rồi
    cộng delta nên **bỏ qua chiều cao tuyệt đối trong BVH** ⇒ ground ở converter **vô dụng**, phải
    clamp ở frontend. Mỗi frame lấy xương thấp nhất, nếu dưới sàn thì nâng group model đúng phần
    thiếu (chỉ nâng, không đẩy xuống → nhảy vẫn rời đất). Kết quả: **289/340 → 0/334** frame dưới sàn.
    Cũng bảo vệ luôn clip stream runtime (P2) mà không cần convert lại.
    `scripts/kimodo_npz_to_bvh.py` giờ đọc thêm định dạng **AMASS/SMPL-X** (2 trong 3 clip đang dùng
    là format này, không phải Kimodo) — output cho file Kimodo bit-identical.
  - ~~🔴 Nguyên nhân THỨ HAI, CHƯA sửa — lỗi dữ liệu motion~~: `motion_b28e8284-328.bvh` cho nhân vật
    **chìm xuyên sàn tới 1.04 m suốt 85% clip** (2 clip còn lại: 0 frame). Sàn z=0 là vật nhận bóng
    nên phần dưới sàn không hắt bóng lên được → bóng bị xén theo giao tuyến thẳng, và **mất hẳn** ở
    cuối clip. Đo được: shadow camera **vẫn bám** nhân vật (lệch ≤0.5, 0 frame bị cắt/3 clip) → không
    phải lỗi hệ bóng. Cần chốt sửa ở `scripts/kimodo_npz_to_bvh.py` (gốc) hay tầng retarget frontend
    (cứu asset đã bundle). Ảnh: `scratchpad/sunk.png`.
  - Nghi phạm phụ chưa đụng: **`<ContactShadows>`** (ô vuông cứng `scale:5`, `far:3`, KHÔNG đi theo
    nhân vật) — cũng cho cạnh thẳng nếu nhân vật ra khỏi ô.
- **Fix shadow ping-pong (cũng 30/07) — thật nhưng KHÔNG phải nguyên nhân chớp đen** (Owner báo sau khi merge FSM). Root cause **có trước refactor**,
  refactor chỉ làm lộ ra nhiều hơn: `ENV_CONFIG.shadows.type = THREE.PCFSoftShadowMap` là hằng
  **deprecated** → `WebGLShadowMap.render()` của three.js warn rồi **tự ghi lại** `this.type =
  PCFShadowMap`; R3F `configure()` (chạy **mỗi lần `<Canvas>` render**) ghi giá trị cũ trở lại, thấy
  `oldType !== newType` → `shadowMap.needsUpdate = true` → **rebuild toàn bộ shadow map** → frame tối.
  - Fix chính: `PCFSoftShadowMap` → **`PCFShadowMap`** (chính là giá trị three.js vẫn dùng → **hình
    ảnh không đổi**). Kèm 4 fix hygiene: hoist `gl`/`camera`/`shadows` của `<Canvas>` ra module const,
    `useMemo` cho `AxesHelper`, `memo()` cho `ScenePostProcessing`, `useMemo` cho context value.
  - Đo được (đếm warning phát ra **từ trong** `WebGLShadowMap.render()` = đúng event rebuild):
    boot **4 → 0**, 4 FSM transition **+4 → +0**, 1 exercise **+2 → +0**. Test lại 11/11 + chat xanh.
  - ⚠️ **Đừng đặt lại `PCFSoftShadowMap`** — bất kỳ ai "sửa cho mượt hơn" sẽ dựng lại bug này.
- 🔎 **Chưa sửa, chờ Owner (đổi hình ảnh)**: SSAO trong `ScenePostProcessing` **không hoạt động** —
  console báo 6x `"Please enable the NormalPass in the EffectComposer in order to use SSAO."`. Đang
  trả giá config mà không nhận occlusion. Hai hướng: bật `enableNormalPass` (thêm 1 pass, có thể xung
  đột outline MToon) hoặc tắt `ssao.enabled`.
- **Docs đã reorg** (đã ổn định từ phiên trước): root `FIX-*.md` cũ giờ ở
  `docs/{architecture,ops,plans,fixes,tracking,archive}/`. File này ở **`docs/tracking/status.md`**.
- **Auth demo bypass**: env-gate `VITE_AUTH_DISABLED=true` trong `ECA_UI/frontend/.env.local` →
  vào thẳng chat không cần Cognito. Production: bỏ/`=false` + `REQUIRE_AUTH=true` + 3 biến Cognito.
- **DeepSeek + Gemini**: từng hết tiền/quota cùng lúc phiên 21/07 (đã verify sống, không phải
  bug) → Owner đã nạp lại DeepSeek. Lỗi `402`/`429` lại = vấn đề tài khoản, không phải code.
- **Gemini context caching** (phiên 22/07, `docs/worklogs/22-07-2026.md`): hạ tầng đã code + test
  xong, nhưng **luôn inert** — free-tier API key có cache-storage quota = 0 (verify live, 429).
  Chỉ áp dụng cho `planner` fallback (prompt tĩnh). Không tối ưu latency hiện tại — chuẩn bị cho
  tính năng BYO-key tương lai.
- **SSE streaming ĐÃ hoạt động thật** (phiên 23/07, `docs/worklogs/23-07-2026.md`): trước đây token
  không stream sống — fix 2 root cause: (A) LangGraph custom-stream drain starvation →
  `await asyncio.sleep(0)` sau mỗi `writer()` trong `synthesizer.py`; (B) CRLF boundary mismatch
  trong `api.ts` (`lastIndexOf('\n\n')` không match `\r\n\r\n`) → regex `/\r?\n\r?\n/g`. Verify:
  token trải đều suốt quá trình sinh, stage indicator "🔍 đang tìm kiếm" hiện đúng.
- **Facial animation avatar** (phiên 23/07, `docs/worklogs/23-07-2026-avatar.md`): Phase A-D xong,
  verify live. Module `ECA_UI/frontend/src/avatar/` (13 file): channel-based expression mixer,
  cross-fade, blink, idle wander, eye gaze, lip-sync amplitude. Độc lập với body motion (Kimodo).
  Default model = seele (`bronya_long` bị filter — 0 blendshape group). Backend chưa emit
  `avatar.emotion` — contract ở `api-contract.md`.
  - ⚠️ **Cập nhật 30/07**: "default body animation đã TẮT" (ghi ở phiên 23/07) **giờ KHÔNG còn đúng**
    — merge của N/Tri đã bật lại: boot = `action_greeting` (LoopOnce) → `finished` → `Standard Idle`
    (LoopRepeat), tức **avatar chào trước rồi mới idle**. Verify: `MotionContext.tsx:93-97`,
    `CharacterViewer.tsx:231-237`. Hành vi này là **cố ý** (thay T-pose) → FSM refactor phải bảo toàn,
    xem plan §2.5.
- **Design treo (không còn cấp bách sau khi streaming fix)**: "bỏ live-stream, chỉ gửi sau grader"
  — Owner chưa chốt. Giờ streaming sống chạy thật nên trade-off K phân tích trước áp dụng đúng
  nguyên văn; bug nối-chữ-khi-grader-retry vẫn có thể xảy ra nếu retry (chưa fix riêng, chờ quyết).
- **Test hiện tại: 312 collected (275 unit pass + 37 integration cần Docker/DeepSeek), 0 fail**
  (verify lại lúc viết file này, dùng đúng `firstconda` env — `python` trên PATH mặc định của
  Bash tool KHÔNG có `langchain_google_genai`, phải gọi thẳng
  `/c/Users/Nguyen/miniconda3/envs/firstconda/python` hoặc `conda activate firstconda` trước).
  Frontend: `npx tsc --noEmit` 0 lỗi, `vite build` 0 lỗi.
- Gọi Owner là **Mr. Senryuu**. Không nhắc chuyện commit trừ khi Owner chủ động.

---

## 1. Tổng quan

Healthcare/wellness AI assistant — physical therapy exercise recommendation, clinical safety
grading, 3D motion avatar (Kimodo), voice I/O (VieNeu, optional/chưa có server). Kiến trúc:
**LangGraph 8-node multi-agent + DeepSeek/Gemini + PostgreSQL/pgvector + Redis + React/Vite FE +
Cognito auth.**

```
memory (STM+LTM) → planner (3-axis intent) → retriever_agent ⇄ tools (cap 2 rounds)
   → kimodo (nếu needs_motion) → synthesizer (persona) → grader (nếu có safety tag) → SSE
```

8 node, 2 cổng routing độc lập, MCP tool servers (web search qua SearXNG, motion qua Kimodo).
Package: `agenticRAG/langgraph_agents/`. **296 test** (259 unit + 37 integration).
README.md (root) đã viết lại đầy đủ — kiến trúc, sơ đồ mermaid, benchmark thật, dùng để
tham khảo CV/portfolio.

---

## 2. Đã hoàn thành ✅ (từ đầu tới nay, gộp các phiên)

### Core + Cụm A/B + Auth + Frontend (trước 02/07 — xem worklog cũ nếu cần chi tiết)
- LangGraph 8-node, 3-axis intent, grader rule-based, GDPR delete+re-summarize, memory tools,
  YouTube transcript tool, Cognito JWT verify (JWKS RS256), frontend React nối backend thật,
  3 bug memory "câm" đã fix, merge frontend của Tri.

### FE debug 02/07
- VRM WebGL context-lost fix (`Environment resolution={64}`), bold-text-ẩn fix (dark:prose-invert),
  port 8080→8000 fix cho chat "something went wrong".

### Health-test dashboard + log setup (phiên này, đầu)
- `ECA_UI/test-ui/health-test/`: port 8000, CORS `null` cho phép mở qua `file://`, sửa dashboard
  đọc trạng thái TTS/SearXNG qua `/health/detailed` (server-side probe, tránh false-negative do
  CORS) thay vì fetch thẳng client-side. Bỏ hẳn card VieNeu TTS (không có server, chỉ có client
  code — báo "down" gây hiểu lầm).
- `vva.log`: backend ghi log ra `agenticRAG/vva.log` (biến `LOG_FILE` trong `.env`, path
  relative theo CWD lúc chạy uvicorn). Chạy qua `cmd /c "... > vva.log 2>&1"` để log sạch (tránh
  PowerShell bọc ErrorRecord khi dùng `*>`).

### Retrieval perf P1/P2/P3 — `docs/fixes/retrieval-perf-p123.md`
- **P1**: embedding load offline thật sự 0-HF-call (set `HF_HUB_OFFLINE=1` ở đầu `api/main.py`
  trước import — set trong hàm load là quá trễ vì huggingface_hub cache cờ lúc import).
- **P2**: hard-cap retriever⇄tools ở 2 vòng trong `routing.py` (state `retriever_rounds`,
  không tin prompt "max 2 rounds").
- **P3**: web_search toggle enforce ở MỌI tầng — prompt có điều kiện (không mời gọi tool khi
  tắt) + guard node bọc `ToolNode` (chặn thật nếu LLM lỡ gọi). Verify live cả 2 trạng thái.

### README.md rewrite
- Viết lại toàn bộ: kiến trúc 8-node đúng thực tế, 2 sơ đồ mermaid (request flow + system stack),
  số liệu đo thật (296 test, ~7.800 LOC backend, ~4.200 LOC FE, 33 quyết định D1-D33), mục
  "Engineering highlights" phục vụ CV/portfolio.

### Benchmark thật + Latency fix #1-4 — `docs/fixes/latency-optimization-1234.md`
- Benchmark 29 request thật → phát hiện 100% chi phí là LLM call, đuôi p90 khủng (planner 24s,
  synth 30s), retriever lãng phí (tool_calls trùng, vòng rỗng).
- **#1** Log cache-hit/miss token DeepSeek (`extract_cache_tokens` trong `llm.py`) → verify live
  **~91% cache-hit rate** trên planner system prompt (M.7 vốn đã hoạt động, giờ đo được).
- **#2** Timeout (fast 20s/heavy 35s) + `max_retries=1` DeepSeek + fallback Gemini một-lần khi
  primary lỗi/timeout (`get_fallback_chat_model`, dùng `GEMINI_API_KEYS` có sẵn trong `.env`).
  K tự phát hiện + sửa 2 bug khi review: (a) streaming dở + fallback ghi đè → chữ trùng lộn xộn
  (guard `already_streamed` trong `synthesizer.py`); (b) Gemini SDK tự retry 36s khi 429 → thêm
  `max_retries=0` cho fallback client (one-shot thật). Verify live dưới double-failure thật
  (DeepSeek 402 + Gemini 429 cùng lúc): fail nhanh 40.8s→4.5s.
- **#3** `max_tokens` theo role (fast 512/heavy 1024) + rút prompt synthesizer "500 từ"→"350 từ".
  Verify live: response dài nhất 243-252 từ, kết câu trọn vẹn không cụt.
- **#4** Dedupe tool_calls trùng hệt (name+args) trong 1 vòng retriever trước khi ToolNode chạy.
  Verify live: không có false-positive (query khác nhau không bị xoá nhầm); true-positive đã
  unit-test riêng.
- **Bonus — bug tìm thấy khi live-test, không nằm trong spec ban đầu**: race-condition trong
  `E5EmbeddingService.model` (singleton lazy-load KHÔNG có lock) — 3 kb_search song song load
  model 3 lần cùng lúc → tràn RAM → **crash backend 2 lần khi benchmark**. Fix: thêm
  `threading.Lock` double-checked locking (`shared/embedding.py`) — giữ nguyên 100% hành vi lazy
  (Owner dặn không bỏ), chỉ serialize lần load đầu tiên khi có nhiều thread đua nhau. Verify:
  test đồng thời 5 thread → còn 1 lần construct; live retry đúng câu từng crash → chạy được.

### Cleanup (dọn dẹp)
- `youtube-transcript-api` pin `<1.0` (1.x bỏ API code đang gọi `get_transcript` → sẽ crash
  runtime nếu không pin).
- `npm run build` sửa 9 lỗi TS/6 file: AuthGuard (unused import + null-safety `tokens?.`),
  MobileNavBar (bỏ prop `onOpenModal` thừa), 4 trang auth (`'select_account'`→`'SELECT_ACCOUNT'`,
  Amplify đổi enum casing). AuthGuard bypass chuyển từ comment-hack → env-gate
  `VITE_AUTH_DISABLED` (xem §0). **Build production giờ chạy được** (`✓ built in ~12s`).
- `.env` (`agenticRAG/agentic_rag_gemini/.env`): xoá config chết kiến trúc cũ (Qdrant, ChromaDB,
  **secret Pinecone đang phơi**, Firebase/Firestore) — giữ đúng những gì code hiện tại đọc.
- 3 file `FIX-*.md` root dời vào `docs/fixes/` (đồng bộ với reorg).

### axios migration (frontend)
- Cài `axios`, tạo `http` instance (`ECA_UI/frontend/src/lib/api.ts`) với request interceptor
  gắn Cognito idToken 1 chỗ. Migrate 6 hàm REST (`listSessions`, `getSession`, `deleteSession`,
  `listUserMemory`, `createUserMemory`, `deleteUserMemory`) sang axios.
  **`streamChat` (SSE `/chat`) CỐ Ý giữ `fetch`** — axios/XHR không stream token tiến dần được.
  ⚠️ Các hàm REST axios **chưa có UI caller** (`ChatSessionsPanel` vẫn là mock tĩnh) — verify
  bằng tsc (0 lỗi) + chat end-to-end vẫn chạy, không verify được qua UI thật vì chưa ai gọi.

### Synthesizer model tier: deepseek-v4-pro → deepseek-v4-flash — `docs/worklogs/21-07-2026.md`
- Live A/B test thật (5 kịch bản × 7 lần/model, không mock): flash thắng **mọi** lần đo (1.5-4x
  nhanh hơn), kể cả worst-case flash < best-case pro. Đọc tay chất lượng output (kể cả kịch bản
  an toàn cao — đau ngực): không thấy khoảng cách, flash còn cụ thể hơn (có số cấp cứu 115).
- `_HEAVY_ROLES` rỗng, tách riêng `_LONG_OUTPUT_ROLES = {"synthesizer"}` để giữ `max_tokens=1024`/
  `timeout=35s` dù đổi model nhẹ (tránh cắt cụt response dài). Fallback Gemini synthesizer đồng bộ
  theo sang flash tier.
- Verify live qua Docker + backend thật: request đau lưng thật, grader reject lần đầu → retry,
  2 lần synthesizer flash (8.57s+7.93s=16.5s) **vẫn nhanh hơn** 1 lần heavy cũ (21.05s benchmark
  trước) dù chạy gấp đôi.
- Follow-up cùng ngày: đổi Gemini fallback model `gemini-2.0-flash` → `gemini-2.5-flash`. Verify
  trực tiếp bằng API thật trước khi chọn — phát hiện `gemini-3.1-flash-lite` trả `.content` dạng
  list cấu trúc (không phải string) qua `langchain_google_genai==4.2.3` hiện cài, sẽ **crash**
  code production đúng lúc fallback cần chạy nhất → loại, chọn `2.5-flash` (verify sạch).

### Gemini explicit context caching (hạ tầng, inert trên free tier) — `docs/worklogs/22-07-2026.md`
- Owner: "cứ tạo đi" — chuẩn bị hạ tầng cho tính năng tương lai (user tự upload API key + chọn
  provider). Research trước khi code: Gemini caching không tự động (khác DeepSeek) — phải tạo
  tường minh `CachedContent` + TTL, ràng buộc `cached_content` không đi kèm `system_instruction`
  riêng. Thử tạo cache thật với đúng prompt planner → **429 quota=0** (giới hạn free-tier, không
  phải lỗi code) — đã honest-disclose, không overclaim đã verify được cache thật giảm latency.
- Scope: chỉ `planner` fallback (prompt tĩnh 100%, giống lý do DeepSeek cache ăn ~91% ở đó).
  Tách `get_warm_gemini_cache()` (tra cứu, không gọi mạng, an toàn dùng trong fallback hot path)
  khỏi `warm_gemini_cache()` (gọi API thật, cố ý KHÔNG auto-invoke từ fallback — tránh lặp lại
  bug "fallback chậm hơn không-fallback" đã sửa trước đó).
  `llm.py` +130 dòng, `planner.py` thêm nhánh thử cached model trước khi cache đã ấm.
- Test: 10 test mới, full suite 270 passed 0 regression. Live thật (không mock): bắt đúng lỗi 429,
  trả `None`, không exception lọt lên trên — xác nhận degrade an toàn trên điều kiện lỗi thật.

### KB-empty web fallback (D34) + SSE stage indicator + 2 fix streaming — `docs/worklogs/23-07-2026.md`
- **D34**: câu hỏi PT phổ thông bị refuse vì KB pgvector RỖNG (0 rows) — không phải grader. Fix:
  tag không-safety-cao + web toggle bật → gọi `kb_search` + `search_medical` SONG SONG round 1
  (P2 hard-cap drop round 2 nên không thể fallback tuần tự). Verify live: trả lời đủ + trích nguồn.
- **Stage indicator**: `ChatPanel.tsx` nghe `stage` event → "🔍 đang tìm kiếm" / "✍️ đang soạn".
- **2 root cause khiến SSE không stream sống** (xem §0): drain starvation (`sleep(0)`) + CRLF
  boundary (`api.ts`). Sau fix, token trải đều thật — verify Playwright screenshot từng giây.
- ⚠️ **Blocker chức năng cốt lõi chưa xử lý**: KB pgvector rỗng — script ingest cũ target ChromaDB
  (kiến trúc cũ), CHƯA có ingest cho schema `documents`/`kb_embeddings`. Mọi câu bài tập phải dựa
  web fallback cho tới khi có ingest. Xem §3.

### Facial animation avatar (Phase A-D) — `docs/worklogs/23-07-2026-avatar.md`
- Module `ECA_UI/frontend/src/avatar/` (13 file), framework-agnostic TS classes. Plan:
  `docs/plans/facial-animation-plan.md` (v1.2, K verify + sửa theo code thật).
- **A** (core): channel-based expression mixer (thay priority), cross-fade delta-time, blink,
  capability detection (degradation an toàn cho `bronya_long` 0-blendshape). **B**: idle wander
  (emotion + gaze) + eye gaze theo mouse + engagement ENGAGED/IDLE. **C**: lip-sync amplitude
  (Mode 1), synthetic-speech test. **D**: contract SSE `avatar.emotion` + giải quyết timing
  `tts.audio` trong `api-contract.md`.
- Verify live (Playwright, đọc weight thật): emotion render + cross-fade S-curve + interrupt
  no-snap + gaze đúng góc + lip-sync mouth theo biên độ. tsc + build 0 lỗi.
- Độc lập hoàn toàn với body motion → khi Kimodo cloud lên, mặt/mắt/miệng chạy y nguyên trên nền đó.
- Default body animation TẮT (`isPlaying=false`, placeholder tạm chờ Kimodo); default model đổi
  về seele (bronya_long render không rõ) → cũng fix luôn issue "cúi đầu lúc load".
- Backend chưa emit `avatar.emotion` — cần Conversation node gán emotion (backend phase sau).

---

### Head-follow + A-pose (25/07) — `docs/worklogs/25-07-2026.md`
- `HeadController.ts`: đầu (Neck+Head bone) xoay theo cùng hướng mắt — đọc góc đã smoothed từ
  `EyeController` (× gain 0.6, smoothing 8/s chậm hơn mắt 10/s → mắt dẫn đầu theo), chia Neck 40% /
  Head 60%, apply trên **normalized humanoid bone** (Euler `YXZ`, compose `rest * offset` → không
  drift). Chỉ đụng bone → độc lập blendshape.
- Verify live: yaw eye 20.9 → head 12.54; pitch -11.4 → -6.84; screenshot 4 hướng đúng (phát hiện +
  lật `PITCH_SIGN` vì ban đầu mouse-lên làm đầu cúi xuống).
- A-pose: `applyRestPose()` hạ arm bone 65° lúc load (VRM bind pose là T-pose). Sau đó N refactor
  BVH → dùng idle clip mặc định, nên A-pose chỉ còn là fallback.

### KB ingest vào pgvector (30/07) — P0
- `scripts/ingest_kb_pgvector.py`: 2918 bài tập từ `documents.txt` → `documents` + `kb_embeddings`
  (384-dim, `embed_passages` batch 64, prefix `passage:` khớp `query:` của `kb_search`).
  `--reset` idempotent theo `source_type='exercise_db'`. **Không** thêm UNIQUE(source_type,
  external_id) vì source có 9 tên trùng.
- Migration `003_kb_hnsw`: bổ sung HNSW index cho `kb_embeddings.embedding` mà migration 002 bỏ sót.
- Verify: `kb_search` 0 → 5 results, `mode: refuse` → `synthesize`. Chi tiết + caveat corpus ở §0.

### Kimodo merge (Tri, 29-30/07) + verify offline (K, 30/07)
- Merge `kimodo-release`: model repo, MCP server, ECS stack, `scripts/kimodo_npz_to_bvh.py`
  (SMPL-X 22-joint, mirror fix + spine damping), animation infra mới (FBX/Mixamo, crossfade,
  motion state machine string-match), scene modules (lighting/env/postprocessing), LoadingOverlay.
- K verify offline PASS: pipeline NPZ→BVH→frontend retarget đúng, không double-mirror. Chi tiết §0.

---

### Animation FSM refactor (30/07 — K)
Thay toàn bộ animation string-match bằng state machine data-driven. Plan
`.claude/plans/animation-fsm-refactor.md` v1.2 (§12 = implementation report).
- **Xoá cả 7 chỗ quyết định hành vi bằng tên file** (`action_`/`random_`/`idle_`/`#intro`/`generated`
  /`motion_`/`thinking`) rải trong `CharacterViewer.tsx` + `MotionContext.tsx`.
- 6 file mới: `AnimationStates.ts` (bảng `STATES` — nguồn chân lý), `AnimationRegistry.ts`,
  `AnimationController.ts`, `CameraController.ts`, `motionAssets.ts`, `useFsmTriggers.ts`.
  `TransitionConfig.ts` trong plan bị bỏ (gộp thành `crossfadeFor()`).
- Xoá ~95 dòng clip-loading/fade/subclip khỏi `CharacterViewer.tsx`; bỏ `selectedMotionId`,
  `isThinking`, `handleAnimationFinished` và 3 effect string-match khỏi `MotionContext.tsx`.
- **Thêm animation mới giờ = 1 entry trong `STATES`** (+1 dòng trigger nếu cần), 0 chỗ mất
  compile check — trước đó là 7 chỗ, không chỗ nào có compile check. Cookbook: plan §11.
- Test **17/17 xanh** (Playwright live): deadlock regression, motion-trong-thinking, T-pose invariant
  (0 frame mất pose qua 10 transition/1s), Kimodo BVH, `reset()` replay (5483/5213ms), camera cooldown
  3s, đổi model không méo, chat `intro→loop→outro→idle`, **boot = greeting đúng 1 lần**, đổi model
  không chào lại, 5 test reachability/UI. `tsc --noEmit` + `npm run build` xanh.
- Handle test: `window.__fsm` (chỉ DEV) — xem §9.

## 3. Còn thiếu / pending (theo mức ưu tiên)

> **Backlog đầy đủ nằm ở [tech-debt.md](tech-debt.md)** (cập nhật 31/07, 27 item mở). Mục này chỉ
> giữ những thứ đang chặn hoặc cần biết ngay; đừng để hai file trôi khỏi nhau.

### 🔴 Chặn trước khi ra mạng thật
| # | Task | Ghi chú |
|---|---|---|
| ~~0~~ | ~~Ingest KB vào pgvector~~ | ✅ **XONG 30/07** — 2918 rows, `scripts/ingest_kb_pgvector.py`. Còn treo: chất lượng corpus (gym/fitness, không phải PT lâm sàng) — xem §0 |
| 1 | Bật auth thật: `REQUIRE_AUTH=true` + config Cognito thật + `VITE_AUTH_DISABLED=false` | cơ chế đã code xong, chỉ chưa bật |
| 2 | Rate limiting cho `/chat` | chưa có gì chặn spam → cháy quota LLM |
| 3 | Secret management (chuyển `.env` key sang secret manager) | trước khi deploy cloud |
| 4 | Gỡ `null` khỏi CORS allow-list | tôi thêm để test `file://`, phải bỏ trước production |

### 🟠 Độ tin cậy / vận hành
- ~~Docker không có restart policy~~ — ✅ **XONG 08/08**. `postgres` + `redis` có
  `restart: unless-stopped` (+ healthcheck `redis-cli ping`). Áp lên container đang chạy bằng
  `docker update`, không tạo lại ⇒ redis không mất STM.
  **Còn treo**: redis chạy **không AOF/RDB** — container sống lại nhưng dữ liệu thì không.
  Bật `--appendonly yes` là đổi hành vi, chờ N chốt.
- ~~`/health/detailed` trả 503 khi thiếu TTS~~ — ✅ **XONG 08/08**. `CRITICAL_CHECKS` vs
  `OPTIONAL_CHECKS`; TTS/SearXNG/MCP hỏng ⇒ `degraded` trên HTTP **200**, instance ở lại LB.
  Check chưa phân loại mặc định là **critical** + log warning (đề phòng quên phân loại thì lỗi
  kêu to, không âm thầm miễn trừ). Worklog 08/08 §1.

### 🟡 Việc code cụ thể
- **17 file chưa commit** (xem §0 và §8): KB ingest (2 file) + FSM refactor (6 file mới, 4 file sửa)
  + plan + worklog. Avatar Phase A-D + head-follow đã được N commit.
- ~~FSM refactor chưa implement~~ — ✅ **XONG 30/07**, 17/17 test xanh (§0 + plan §12).
- **Facial ↔ body state chưa đồng bộ** (§9 của plan FSM) — **việc code tiếp theo rõ ràng nhất**:
  thân "suy nghĩ" mà mặt tự cười; cười lúc demo bài tập; head-follow đè chuyển động đầu của clip
  Kimodo. Sau refactor thì **chỗ cắm đã sẵn**: gọi `facialOf(state)` (đã có trong
  `lib/AnimationStates.ts`, mỗi state đã khai `facial: { wander, hold? }`) từ điều kiện
  `AvatarController.ts:115` `if (!engaged) this.idle.tick(delta)`, + truyền attenuation cho
  `HeadController` (state `exercise` → gain 0). **KHÔNG thêm public method** — đây là data policy.
- **Bug chưa fix, chờ Owner chốt hướng**: grader-retry có thể làm `ChatPanel.tsx` nối chữ 2 lần
  sinh của synthesizer (không tách buffer). Ít cấp bách hơn sau khi streaming fix — Owner đề xuất
  "bỏ live-stream, chỉ gửi sau grader" thay vì vá buffer, chưa chốt.
- **Backend chưa emit `avatar.emotion`**: hệ facial animation FE sẵn sàng nhưng backend Conversation
  node chưa gán emotion metadata → avatar chỉ đổi biểu cảm khi có lệnh (backend phase sau).
- **`get mode(): string`** trong `AvatarController` nên siết thành union `'engaged' | 'idle'` —
  1 dòng, chưa sửa vì nằm trong file N vừa commit.
- Memory FE reset khi đóng sidebar (Owner đã chủ động bỏ persistence — biết và chấp nhận).
- Bundle FE nặng (JS ~1.9MB gzip 549KB + VRM asset 9-29MB bundle thẳng) — chưa lazy-load/CDN.

### 🟢 Tối ưu có dư địa, chưa làm
- Eval dataset ~50 golden case (đo recall/latency trước-sau khi đổi prompt/model).
- CI chạy test mỗi PR + branch protection (có `release-tests.yml`, chưa xác nhận chặn merge).
- Đổi embedding model (đã bàn kỹ — khuyến nghị `gte-multilingual-base` — nhưng cần eval dataset
  trước để đo ROI, chưa làm).

---

## 4. Phase 7 — Hybrid Cloud (ON HOLD, chờ Owner bàn)
- Tri's `infra/` CDK (Python) đã merge: VPC isolated + RDS Proxy + Lambda CRUD + API Gateway.
- Đã chốt: Alembic = nguồn migration duy nhất; `/chat` KHÔNG qua API Gateway (timeout 29s) →
  ECS Fargate; Kimodo host = edge RTX 3060 + SQS pull; voice = push-to-talk (giữ SSE).
- Còn treo (cần Owner): Supabase vs RDS (chi phí ~$80/mo vs ~$30 lean) — chốt trước khi K viết
  spec Phase 7 chi tiết.

---

## 5. Cách chạy (local demo)

```bash
# 1. Docker (postgres local chỉ còn là đường lùi — DB thật đang ở Neon)
docker compose -f docker-compose.langgraph.yml up -d postgres redis searxng
#    DSN Neon nằm ở agenticRAG/agentic_rag_gemini/.env (VVA_PG_DSN, gitignored).
#    Xoá dòng đó = quay về Postgres local.

# 2. Migration (nếu schema đổi)
cd agenticRAG/langgraph_agents && alembic upgrade head

# 3. Backend (:8000 — KHÔNG dùng 8080) — conda env firstconda
cd agenticRAG
python -m uvicorn langgraph_agents.api.main:create_app --factory --port 8000 --host 0.0.0.0

# 4. Frontend React (:5173) — demo mode, VITE_AUTH_DISABLED=true trong .env.local
cd ECA_UI/frontend && npm install && npm run dev

# ⛔ Nạp KB: PHẢI TẮT BACKEND TRƯỚC, nếu không sẽ segfault (exit 139, log rỗng)
#    và `--reset` đã xoá bảng ⇒ KB rỗng, mọi câu hỏi bị từ chối. Xem QUICKSTART §3.
#    python scripts/ingest_kb_pgvector.py --reset

# Tests
python -m pytest tests/langgraph_agents/ -m unit -q   # 275 passed, không cần service sống
python -m pytest tests/langgraph_agents/ -q            # 312 (cần Docker + DeepSeek key thật)
```

## 6. Key files (path MỚI sau reorg — đừng tìm ở root)
| File | Nội dung |
|---|---|
| `docs/plans/reupdate-plan.md` | 33 decisions D1-D33 — nguồn chân lý kiến trúc |
| `docs/plans/facial-animation-plan.md` | Avatar facial animation (Phase A-D, v1.2) — module design |
| `.claude/plans/animation-fsm-refactor.md` | **FSM refactor body animation (v1.2)** — ✅ đã implement, §12 = implementation report, §9 = facial↔body sync (chưa làm), §11 = cookbook thêm animation |
| `ECA_UI/frontend/src/lib/AnimationStates.ts` | **Bảng `STATES` — nguồn chân lý DUY NHẤT của animation FSM.** Thêm animation mới = thêm 1 entry ở đây (xem plan §11) |
| `ECA_UI/frontend/src/lib/AnimationController.ts` | FSM runtime — invariant "không bao giờ T-pose" nằm ở đây (play trước, fade sau) |
| `ECA_UI/frontend/src/lib/{AnimationRegistry,CameraController,motionAssets}.ts` | Asset layer · camera+cooldown · glob file animation |
| `ECA_UI/frontend/src/hooks/useFsmTriggers.ts` | Boot (greeting→idle) + timer trigger, kèm bảng liệt kê MỌI nguồn đổi state |
| `.claude/plans/neon-migration.md` | **Chuyển PostgreSQL lên Neon** — Owner chốt 31/07, chưa thi hành. Có blocker #0: runtime KHÔNG đọc `VVA_PG_DSN` |
| `.claude/plans/kimodo-alb-endpoint.md` | ALB endpoint cho Kimodo MCP — implemented, chờ N deploy |
| `scripts/ingest_kb_pgvector.py` · `scripts/kimodo_npz_to_bvh.py` | KB ingest · Kimodo NPZ→BVH converter |
| `docs/tracking/tech-debt.md` | Việc tồn (nhánh riêng khỏi status.md này) |
| `docs/fixes/*.md` | Spec/handoff từng cụm fix (memory, auth, chatpanel, retrieval-perf, latency) |
| `docs/ops/runbook.md`, `docs/ops/troubleshooting.md` | Chạy + debug chi tiết |
| `docs/architecture/*.md` | Kiến trúc chi tiết theo chủ đề |
| `docs/worklogs/DD-MM-YYYY.md` | Nhật ký từng phiên |
| `README.md` (root) | Đã viết lại đầy đủ — kiến trúc + sơ đồ + benchmark, dùng cho CV/portfolio |
| `.claude/CLAUDE.md` | Roles K/N/Owner, conventions |

## 7. Conventions
- Worklog `docs/worklogs/DD-MM-YYYY.md` mỗi phiên đáng kể. Test phải xanh trước khi coi là xong.
- K = Architect (spec + review; khi việc lớn thì viết spec → spawn subagent Sonnet implement →
  **K tự đọc diff + tự chạy test + tự verify live trước khi báo cáo** — không tin lời subagent).
  N = Developer. Owner = "Mr. Senryuu", chốt vision, không tự commit/push khi chưa được lệnh.
- Code = English, docs = Việt + Anh. UI verify bằng skill `playwright-cli` (không npx). Backend
  port cố định **8000** (8080 = Spring của Owner, không đụng).

## 8. Danh sách file chưa commit (git status lúc viết file này, 30/07 — HEAD `14f7b7d`)
```
?? .claude/plans/animation-fsm-refactor.md      (plan v1.2 — 9 lỗi + §12 implementation report)
?? docs/worklogs/30-07-2026.md                  (worklog phiên 30/07)
?? scripts/ingest_kb_pgvector.py                (P0 KB ingest — đã chạy)
?? agenticRAG/langgraph_agents/alembic/versions/003_kb_embeddings_hnsw.py   (đã alembic upgrade head)
?? ECA_UI/frontend/src/lib/AnimationStates.ts       ?? ECA_UI/frontend/src/lib/AnimationRegistry.ts
?? ECA_UI/frontend/src/lib/AnimationController.ts   ?? ECA_UI/frontend/src/lib/CameraController.ts
?? ECA_UI/frontend/src/lib/motionAssets.ts          ?? ECA_UI/frontend/src/hooks/useFsmTriggers.ts
 M ECA_UI/frontend/src/contexts/MotionContext.tsx
 M ECA_UI/frontend/src/components/CharacterViewer.tsx
 M ECA_UI/frontend/src/components/panels/MotionControlPanel.tsx
 M ECA_UI/frontend/src/components/ChatPanel.tsx
 M ECA_UI/frontend/src/components/scene/ScenePostProcessing.tsx   (fix chớp đen: memo)
 M ECA_UI/frontend/src/config/environmentConfig.ts                (fix chớp đen: shadow type)
 M docs/tracking/status.md
```
Mọi thứ khác **đã commit**: avatar Phase A-D, head-follow (`HeadController.ts`), A-pose, SSE fix,
merge `kimodo-release` + `kimodo_npz_to_bvh.py`, scene modules.
17 file trên chưa commit — chờ lệnh Owner. Lưu ý: FSM refactor **thay thế** animation string-match cũ,
nên 4 file `M` không thể commit rời khỏi 6 file mới.

## 9. Cách verify nhanh (cho phiên sau)

```bash
# 1. KB ingest còn nguyên?
docker exec vva-postgres psql -U vva -d vva -c "SELECT COUNT(*) FROM kb_embeddings;"   # kỳ vọng 2918
# → nếu 0: python scripts/ingest_kb_pgvector.py --reset

# 2. KB thật sự được dùng (không refuse)?
printf '%s' '{"query":"bài tập cho cơ bụng và lưng dưới","session_id":"77777777-7777-7777-7777-777777777777","user_id":"77777777-7777-7777-7777-777777777778","web_search":false}' > /tmp/kb.json
curl -s -m 90 -X POST http://localhost:8000/chat -H "Content-Type: application/json" --data-binary @/tmp/kb.json > /tmp/r.txt
grep -aoE '"mode": "[a-z]+"' agenticRAG/vva_run.log | tail -1     # kỳ vọng "synthesize", KHÔNG "refuse"
```

Verify Kimodo retarget (offline, không cần GPU): `:5173` → nav **Motion** → **Motion file (debug)** →
chọn `motions/generated/motion_*.bvh` → avatar phải múa tự nhiên, camera tự nới sang `hips`.

**Verify animation FSM** — mở `:5173`, DevTools console (chỉ hoạt động ở dev):
```js
__fsm.state            // 'idle' sau khi chào xong
__fsm.history          // ["idle","idle","greeting","idle"] — greeting phải xuất hiện ĐÚNG 1 lần
__fsm.hasPose          // true; false = đang T-pose → bug invariant
await __fsm.transitionTo('thinking_intro')   // true; tự chạy intro→loop
await __fsm.transitionTo('thinking_loop')    // false từ idle (reachability guard)
__fsm.cameraMode       // 'hips' khi đang exercise, giữ thêm 3s sau khi về idle
```
Chạy lại full checklist: script Playwright ở scratchpad phiên 30/07 (`fsm-test.mjs`,
`fsm-ui-test.mjs`, `fsm-chat-test.mjs`) — 17 assertion, xem plan §12.3 để biết kỳ vọng.
