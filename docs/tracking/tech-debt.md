# Tech Debt & Pending Tasks

> Checklist các việc đã biết nhưng CHƯA làm. Cập nhật khi đóng/ mở item.
> Last update: 2026-08-05 (K — thi hành Neon, dọn 7 lỗi TS ở FE, nối lại TTS VieNeu).
> Nguồn trước đó: worklogs 05→12/06-cont; đợt mới: `docs/worklogs/30-07-2026.md`,
> `docs/worklogs/05-08-2026.md`.

Mức: 🔴 critical (phải làm trước Phase 7 deploy) · 🟠 quan trọng · 🟡 nên làm · ⚪ optional

---

## 🔴 Critical — chặn Phase 7

- [ ] **No-auth IDOR — FIX ĐÃ CÓ, đang GATED (phải bật cờ trước deploy)** — `api/auth.py::
      resolve_user_id` verify Cognito ID token (JWKS RS256 + audience + issuer + token_use),
      `user_id = sub`, wired vào mọi endpoint. Cờ `REQUIRE_AUTH` default **false** (demo nội bộ
      vẫn nhận client user_id — không chặn demo). **Hành động bắt buộc trước network deploy: set
      `REQUIRE_AUTH=true` + 3 biến Cognito** → token-less/invalid → 401, IDOR đóng. Còn lại:
      deploy Cognito (`ampx sandbox`/AWS) để có pool thật. (Security review 12/06 Vuln 1;
      integration 18/06, worklog 18/06)

- [x] **Chốt nhà cung cấp DB cloud** — Owner chốt **Neon** (31/07). Plan thi hành: **[`.claude/plans/neon-migration.md`](../../.claude/plans/neon-migration.md)**.
- [x] **Thi hành plan Neon** — ✅ **XONG 05/08** (git: `d7851fb`). Backend `:8000` chạy trên Neon (ap-southeast-1),
      dữ liệu ghi vào Neon (đã đối chứng local). DSN nằm ở `agenticRAG/agentic_rag_gemini/.env`
      (gitignored). Chi phí DB thật: **12 query / 1,4 s / lượt = 4,8%** (đo bằng counter, không đoán).
- [ ] 🔴 **Ingest SEGFAULT khi backend đang chạy** — `scripts/ingest_kb_pgvector.py` chết exit 139,
      **log rỗng, không traceback**. Khoanh vùng được: chết ở `embed_passages()` (inference torch),
      **không** phải DB. Tắt uvicorn thì chạy bình thường → hai tiến trình tranh chấp native runtime
      của torch. Nguy hiểm vì `--reset` xoá bảng TRƯỚC khi crash → **KB rỗng, mọi câu hỏi bị từ chối**.
      Đã dính 05/08 và mất thời gian truy nhầm hướng.
      **Cần làm**: cho script tự phát hiện backend đang chạy và từ chối, hoặc đảo thứ tự để chỉ xoá
      sau khi embed xong. Tạm thời đã ghi cảnh báo ở QUICKSTART §3.
- [ ] **Tối ưu ingest (`unnest` + `executemany`) — đã thử, đã hoàn nguyên** — gộp 2 round-trip mỗi
      batch thay vì mỗi record. Lúc thử thì gặp segfault nên tôi **đổ lỗi nhầm cho `executemany`**;
      thực ra là lỗi tranh chấp torch ở trên. Ý tưởng vẫn đúng và đáng làm lại (ingest ~9 phút →
      ~18 giây), nhưng **phải kiểm thứ tự `RETURNING`** (Postgres không bảo đảm) — bản thử đã có sẵn
      truy vấn đối chiếu `content LIKE 'Exercise: ' || title || '%'`.
- [ ] **Đổi mật khẩu Neon trước khi có dữ liệu người dùng thật** — DSN đã đi qua kênh chat.
- [ ] **Giữ pool DB ấm** — pool nguội vs ấm chênh **3,29 s vs 1,41 s** mỗi lượt, lớn hơn mọi tối ưu
      query khác. Gắn với quyết định scale-to-zero của Neon.
- [ ] **Redis + object storage vẫn ở local** — Postgres đã lên cloud, hai kho này thì chưa. Owner đã
      có sắp xếp riêng cho Redis; object storage đi hướng AWS S3.
      Plan hiện ghi **Supabase** (`.claude/CLAUDE.md` Phase 7, `docs/plans/v2.4-plan.md:1151`), còn
      `docs/plans/reupdate-plan.md:882` vẫn để mở. **"Dời toàn bộ DB" là mô tả chưa đủ** — dữ liệu
      nằm ở 3 kho, Neon chỉ thay được 1:

      | Kho | Chứa | Neon thay được? |
      |---|---|---|
      | PostgreSQL + pgvector | 8 bảng: `users`, `conversations`, `messages`, `summaries`, `user_memory`, `documents`, `kb_embeddings` | ✅ |
      | Redis | STM session (`db/session_store.py`), circuit breaker, Celery broker, TTS task | ❌ → cần Upstash/Redis Cloud |
      | File hệ thống | motion BVH/NPZ, sắp có audio TTS | ❌ → cần R2/S3 (Supabase có Storage sẵn) |

      **Ba rủi ro phải xử lý TRƯỚC khi dời:**
      1. `db/postgres.py` dùng `asyncpg.create_pool` + `register_vector` → endpoint **pooled** của
         Neon/Supabase chạy PgBouncer transaction mode làm vỡ prepared-statement cache của asyncpg.
         Phải dùng direct endpoint hoặc `statement_cache_size=0`.
      2. **Latency**: memory node chạy MỌI request, retrieval là vector search → backend PHẢI cùng
         region với DB. Backend local (VN) + DB cloud = cộng RTT cho từng query, nhiều query/lượt chat.
      3. **Scale-to-zero** của Neon: rẻ nhưng request đầu sau idle cold-start ~0.5-2s → user thấy
         "đang suy nghĩ" lâu hơn. Tắt hoặc chấp nhận.

      **Trước khi cam kết**: spike đo latency thật (đừng quyết bằng cảm tính) + giữ Alembic là nguồn
      migration duy nhất. Lưu ý lý do dời là **độ tin cậy / không phụ thuộc máy cá nhân / có PITR**,
      KHÔNG phải hết dung lượng — KB chỉ ~2918 vector 384 chiều.

## 🟠 Quan trọng

- [ ] **`AdminLinkProviderForUser` gọi trong PreSignUp có mép sắc đã biết** — đây là cách AWS khuyến
      nghị, nhưng nhiều báo cáo cho thấy **lần link đầu tiên** tuỳ cấu hình pool có thể lỗi và user
      phải bấm đăng nhập lại một lần nữa mới vào được. K **chưa verify được** vì máy không có AWS
      credentials. Khi deploy sandbox xong, test đúng kịch bản: tạo tài khoản email → đăng nhập
      Google cùng email → xem lần redirect đầu có vào thẳng không. Nếu có lỗi thì cần bắt và hiển thị
      thông báo "thử lại" thay vì để user thấy lỗi thô. Worklog 05/08 §9.
- [ ] **`loginHint` là no-op với Google** — AWS ghi rõ *"You can't forward login hints to SAML, Apple,
      Login With Amazon, Google, or Facebook (Meta) IdPs"*. Giữ trong `lib/googleSignIn.ts` vì vô hại
      và **sẽ chạy nếu sau này thêm OIDC provider chung**, nhưng **không được tính nó là lớp bảo vệ**.
      Với Google, tài khoản user chọn là **không thể ràng buộc trước** — chỉ kiểm tra được sau
      (`takeExpectedEmail`) và không để nó đẻ tài khoản trùng (PreSignUp linking).

- [ ] **Summarizer E2E với LLM thật** — PR 2 mới có unit test (mock LLM/PG). Cần chạy thủ công:
      hội thoại vượt 10k token → row `summaries` xuất hiện, turn sau memory node load chunk,
      `memory_search` từ session khác cùng user tìm thấy. (worklog 12/06 defer)
- [ ] **`users.auth` cho Phase 7** — bỏ uuid5 coercion ở production, thêm
      `auth_provider/auth_subject` (cột đã có trong schema M.4, chưa có flow).
      uuid5 giữ cho dev/anonymous. (F3 / plan §3.1)
- [ ] **Verify general_query thủ công** — cần LLM + SearXNG chạy thật: hỏi "giá vàng?",
      confirm needs_retrieval=true (3-axis, không còn intent enum), web search fire,
      trả lời hữu ích (không refuse). (worklog 28/05 §5, cập nhật theo 3-axis)
- [ ] **SSAO đang bật nhưng KHÔNG hoạt động** — `ScenePostProcessing.tsx` bật SSAO nhưng
      `<EffectComposer>` thiếu `enableNormalPass`; console báo 6 lần mỗi lần load
      `"Please enable the NormalPass in the EffectComposer in order to use SSAO"`. Đang trả giá
      config + 1 effect pass mà không nhận được occlusion nào. Hai hướng: bật `enableNormalPass`
      (thêm 1 pass, có thể xung đột outline MToon — chính comment trong file cảnh báo) hoặc tắt
      `ssao.enabled`. **Cả hai đều đổi hình ảnh → cần Owner quyết.** (worklog 30/07 §4)
- [ ] **Bloom đang TẮT vì gây frame đen** — `postProcessing.bloom.enabled=false`. Bisect đo được
      Bloom sinh **1 frame đen đơn lẻ mỗi ~3-5s** (luma 17.8 giữa các frame ~220); `mipmapBlur`
      không cứu. Bảng số đo đầy đủ nằm trong comment ở `environmentConfig.ts`. Chỉ bật lại khi
      `@react-three/postprocessing` sửa upstream **và** chạy lại screencast check.
- [ ] **grader-retry có thể làm ChatPanel nối chữ 2 lần** — synthesizer sinh lại nhưng buffer FE
      không tách. Owner đề xuất "bỏ live-stream, chỉ gửi sau grader" thay vì vá buffer — **chưa chốt**.
- [ ] **Backend chưa emit `avatar.emotion`** — hệ facial animation FE đã sẵn sàng (13 module,
      `avatar.emotion` có trong `api-contract.md`), Conversation node chưa gán emotion metadata.
- [ ] **Facial ↔ body state chưa đồng bộ** (§9 của `.claude/plans/animation-fsm-refactor.md`) —
      thân "suy nghĩ" mà mặt tự cười; cười lúc demo bài tập; head-follow đè chuyển động đầu của clip.
      Sau FSM refactor **chỗ cắm đã sẵn**: `facialOf(state)` trong `lib/AnimationStates.ts` (mỗi state
      đã khai `facial: { wander, hold? }`), sửa 1 điều kiện ở `AvatarController.ts:115` + truyền
      attenuation cho `HeadController` (state `exercise` → gain 0). **KHÔNG thêm public method** —
      đây là data policy, Owner đã bắt đúng điểm này.
- [ ] **Kimodo runtime delivery (P2)** — `playMotionFile()` phía FE đã sẵn sàng; backend chưa stream
      motion URL qua SSE. Converter vẫn là CLI thủ công.
- [ ] **Docker không có restart policy** — tắt máy là mất container, phải `docker compose up -d` tay.
- [ ] **`/health/detailed` trả `degraded` khi thiếu TTS** — optional dependency kéo cả status tổng.
      Nên tách critical vs optional trước khi có LB/orchestrator thật.

## 🟡 Nên làm

- [ ] **`ai_understanding` (AI tự đúc kết về user)** — AI-auto trích facts vào `user_memory`
      (background, throttled mỗi 5 turn). Advisory; `valid` flag sẵn cho conflict. (D14 phase sau)
- [ ] **`<ContactShadows>` là ô vuông cứng, KHÔNG đi theo nhân vật** — `scale:5`, `far:3`, cố định
      tại gốc toạ độ. Nhân vật ra khỏi ô → bóng tiếp xúc bị cắt thẳng băng; cao quá `far` → mất bóng.
      Chưa đụng vì cần Owner xác nhận có thấy không. Fix cùng kiểu `lib/shadowFit.ts`, vài dòng.
- [ ] **Shadow fitter trễ 1 frame** — chạy trong `useFrame` của `SceneLighting`, đăng ký **trước**
      `VRMCharacter` (nơi chạy mixer). `fitPadding 0.35` hấp thụ chuyển động thường; chỉ lúc **đổi
      clip** (nhân vật nhảy pose) mới vượt, đúng 1 frame. Đã ghi ở đầu `lib/shadowFit.ts`.
- [ ] **2 clip AMASS dùng BVH sinh bởi tool KHÁC** — `motion_7b4b8d9e`, `motion_b28e8284` có
      **hierarchy khác** `scripts/kimodo_npz_to_bvh.py`. Đã thử regenerate → motion đổi hẳn → phải
      hoàn nguyên. Ai chạy lại converter trên 2 NPZ đó sẽ ra kết quả khác bản đang dùng. Converter
      giờ **đọc được** cả 2 định dạng (`load_motion()`), nhưng **đừng regenerate** 2 file này nếu
      chưa verify lại bằng mắt.
- [ ] **`get mode(): string` trong `AvatarController`** — nên siết thành union `'engaged' | 'idle'`.
      1 dòng.
- [ ] **Head-follow Phase 2 (additive blending)** — `HeadController` hiện ghi đè bone Neck/Head mỗi
      frame nên đè chuyển động đầu của clip. Gộp chung với mục facial↔body sync ở trên.
- [ ] **Chất lượng KB corpus** — 2918 record là **gym/fitness**, không phải PT lâm sàng; 53%
      (1550/2918) có `Description: nan`. Đã fix lỗi "KB rỗng → refuse" nhưng chưa làm nó thành KB
      lâm sàng. Owner đã gác lại.
- [ ] **Chưa verify lại được "frame đen" bằng script tự động** — Playwright headed chạy nền treo 3
      lần liên tiếp ở khâu giải mã ~1100 ảnh JPEG (hạn chế công cụ đo, không phải app). Muốn có số
      phải chạy tiền cảnh. Fix Bloom vẫn nguyên trong config; các thay đổi sau đó (`shadowFit`,
      `groundClamp`) không đụng postprocessing.
- [ ] **Bundle FE nặng** — JS ~2MB (gzip ~580KB) + VRM asset 9-29MB bundle thẳng. Chưa lazy-load/CDN.
- [ ] **FE không có test runner** — `package.json` chỉ có `dev/build/lint/preview`, không có `test`.
      Mọi hồi quy FE hiện chỉ dựa vào `tsc -b`. Backend có 275 test, FE có 0. Cân nhắc vitest +
      vài test cho `AnimationController` / `shadowFit` / `groundClamp` (logic thuần, dễ test).
- [ ] **`npm run lint` đỏ: 11 error / 3 warning** — có sẵn, không chặn `build` nên không ai thấy.
      Gồm `no-explicit-any` ×3 (`lib/bvhToVrm.ts`), `set-state-in-effect` ×2, `refs`-during-render
      (`FloatingNavBar.tsx:457`), import thừa (`LogOut`, `LucideIcon`), `tick` tự tham chiếu trong
      `useCallback` (`ChatMessage.tsx:107`). K không sửa vì nằm ngoài phạm vi "chặn build" — cần
      Owner quyết có đưa `lint` vào `build`/CI không, nếu không nó sẽ cứ trôi tiếp.
- [ ] **Panel Sessions vẫn là empty state hardcode** — `ChatSessionsPanel.tsx` chưa gọi API nào,
      dù `api.ts` đã có `listSessions()`/`deleteSession()` và backend đã có `GET /sessions` +
      `DELETE /sessions/{user}/{session}`. Từ 05/08 session đã giữ qua refresh và có nút "cuộc trò
      chuyện mới", nhưng **session cũ vẫn không có đường quay lại** — chúng nằm trong Postgres,
      chỉ thiếu UI. Kèm: cột `conversations.title` có sẵn nhưng **chưa ai ghi**.
- [ ] **Lịch sử khôi phục không có audio** — WAV nằm trên máy TTS với tên ngẫu nhiên, không lưu kèm
      transcript. Tin nhắn khôi phục phải bấm nút loa để tổng hợp lại. **Owner đã xác nhận (05/08)
      đây là hành vi ĐÚNG**, không cần sửa gấp. Khi đẩy audio lên S3 ở Phase 7 thì lưu URL vào
      `messages` là tự khỏi.
- [ ] **`bronya.vrm` không có morph viseme** — bindCount của A/I/U/E/O đều **0**: nhóm blendShape có
      tồn tại nhưng không bind vào morph nào, nên lip sync chạy đúng mà **miệng vẫn đứng im**. Ba
      model kia đều [1,1,1,1,1]. Mặc định là Con-Gai-Khang nên chưa lộ. **Sửa được bằng cách chỉnh
      file VRM (VRoid/Blender), KHÔNG sửa được bằng code.** Phát hiện 05/08 khi nối lip sync.
- [ ] **Lip sync mới là Mode 1 (biên độ)** — `LipSyncController` suy miệng từ RMS, mở/khép theo to
      nhỏ chứ không theo âm vị. Mode 2 (phoneme viseme) cần timestamp mà VieNeu-GGUF không xuất.
      Muốn khẩu hình đúng chữ thì phải forced-alignment riêng — việc lớn, chưa cần bây giờ.
- [ ] **Cân nhắc bật `voiceReply` mặc định** — K để mặc định TẮT dựa trên số 77s, nhưng số đó lấy từ
      một câu trả lời lâm sàng ~2500 ký tự. Đo thật trên câu trả lời thường (175 và 303 ký tự):
      TTS chỉ **4,5s / 5,5s**, tổng lượt **12,8s / 12,0s** — giọng nói chỉ thêm ~5s. Quyết định của
      Owner.
- [ ] **`/chat` giữ SSE mở tới 130s để chờ TTS** — `_poll_speech_result` chặn `done`. FE đã né bằng
      cách thả UI ở `speech_pending`, nhưng về kiến trúc nên đổi sang: FE nhận `task_id` rồi **poll
      `GET /tts/{task_id}/result`** (endpoint đã có sẵn, comment gọi nó là "fallback"). Được: stream
      đóng sớm, sống sót khi mất kết nối, hợp Phase 7 (cloud + edge). Cần Owner quyết vì đổi luồng.
- [ ] **TTS đặt tên file sai ngôn ngữ** — `services/vieneu_tts/tasks.py` không gửi `language`, nên
      `api_server.py` default `"en"` ⇒ văn bản tiếng Việt ra `vieneu_en_*.wav`. **Chỉ sai tên file**,
      `language` không đi vào `tts.infer()`. Cosmetic, nhưng gây hiểu nhầm khi debug.
- [ ] **File WAV không ai dọn** — mỗi lượt có giọng đẻ ra ~5 MB trong `SpeechLLm/data/temp_audio/`,
      không có TTL/cleanup. Chạy vài trăm lượt là đầy đĩa. Phase 7 sẽ đẩy sang S3 nhưng local vẫn cần
      job dọn.
- [ ] **Commit limit của máy dev gần cạn** — 44,6/44,7 GB khi chạy đồng thời backend + TTS. Lúc đó
      `tsc`/`node` **không khởi động nổi** (`paging file is too small`, `VirtualAlloc failed`), rất
      dễ tưởng nhầm là lỗi code. Cần tăng pagefile hoặc đừng chạy TTS song song lúc build FE.

## ⚪ Optional / Phase sau

- [ ] **User upload tài liệu riêng** — ĐÃ QUYẾT (29/05): **Option 1 — KB chỉ của hệ thống.**
      Bỏ `documents.user_id` (luôn = system KB), `search_kb` giữ "all public" an toàn, không có
      private doc nên không leak. Khi thật sự cần upload riêng → thêm bảng riêng (`user_documents`
      + `user_doc_embeddings`) lúc đó, design đúng kịch bản thật. Không schema cho feature chưa có.
- [ ] **LLM gợi-ý profile** — LLM phát hiện fact (age/injury) → đề xuất → user confirm → ghi
      `user_memory`. Không tự ghi thẳng. (plan §4.4 optional) — FEATURE, cần Owner quyết build.
- [ ] **Profile trigger nâng cao** — ngoài endpoint, cân nhắc trích từ hội thoại. = gộp vào
      `ai_understanding` (🟡). FEATURE, cần Owner quyết.
- [ ] **Eval dataset ~50 golden case** — đo recall/latency trước-sau khi đổi prompt/model. Là điều
      kiện tiên quyết để đánh giá việc đổi embedding model (`gte-multilingual-base`).
- [ ] **CI chạy test mỗi PR + branch protection** — có `release-tests.yml`, chưa xác nhận chặn merge.

> **`vector(384)` hardcode — ĐÃ THỎA, không cần làm**: `E5_DIM=384` (shared/embedding.py) đã là
> single constant bên Python; chỗ `vector(384)` còn lại nằm trong Alembic migration ĐÃ CHẠY —
> đổi model = viết migration mới dù sao. Refactor thêm = cosmetic (karpathy #3). K 13/06.

---

## Đã xong (tham chiếu — không phải pending)

- ✅ **Tin nhắn trả về sai thứ tự (đáp trước hỏi)** — cặp user/assistant ghi cùng batch nên
  `created_at` **bằng nhau tuyệt đối**; `load_session_messages` chỉ `ORDER BY created_at DESC` ⇒
  tie-break tuỳ tiện. Cột `seq_id` có sẵn mà không dùng. Thêm `, seq_id DESC`. Chỉ lộ ra khi bắt đầu
  khôi phục lịch sử — trước đó không ai đọc endpoint này. K 05/08, worklog §6.
- ✅ **`AudioButton` kẹt file cũ khi `audioUrl` đổi** — nay adopt prop mới và vứt `<audio>` cũ.
  K 05/08.
- ✅ **`/health/detailed` rò 2 kết nối Neon mỗi probe** — `check_postgres` dựng `PostgresClient()`
  mới mỗi lần gọi và không đóng. Đo: 5 probe → 8→18 kết nối, nghỉ 10 s vẫn 18. `max_connections=901`
  ⇒ LB probe mỗi 10 s là chết DB sau ~1 giờ. Kèm: budget 2 s so với connect Neon 1,3-1,5 s ⇒ lần đầu
  sau nghỉ `ok=false`. Sửa: dùng `shared.get_pg_client()` + `SELECT 1` thật + budget 5 s.
  Sau sửa: **15 probe, rò 0**, latency **91-172 ms**, `status: ready`. +2 test canh (cả 7 test cũ
  đều mock `check_postgres` nên không canh gì). K 05/08, worklog §5.

- ✅ **HNSW iterative_scan** — pgvector 0.8.2 trên DB; bật `SET hnsw.iterative_scan='relaxed_order'`
  ở pool `_init_conn` (postgres.py, best-effort guarded) → mọi connection có, không cần wrap
  transaction. Recall đúng khi memory_search filter `session_id=ANY`. K 13/06.
- ✅ **Persona cache không cache fallback** (R3 nit #2) — `_fallback_persona` gắn cờ `_fallback`;
  `get_persona` chỉ cache persona thật → flood id xấu không phình cache. +1 assert test A0. K 13/06.
- ✅ **YouTube paste-link Q&A (cụm B)** — `youtube_transcript(url)` tool (KHÔNG ghi KB/LTM,
  reuse `_extract_video_id`, truncate 12k chars, empty≠error D23), trong `RETRIEVER_BASE_TOOLS`
  + prompt retriever. Hướng TOOL thay "planner detect" (D2b/D16 — tái dùng đường evidence sẵn,
  0 đổi graph/state/synthesizer). +12 test. Subagent (Sonnet), K verify 237/237. worklog 13/06.
  Spec `FIX-YOUTUBE-PASTE.md`.
- ✅ **`test-ui/app.js` done-label stale field** — `payload.intent` → `required_outputs` (SSE
  done event không còn `intent`). Phần resume rendering đã đúng shape M.4 (không tham chiếu
  `metadata/intent`). + assert thật cho test cache persona (R3 nit #1). K 13/06.
- ✅ **GDPR re-summarize bug (R1) + acceptance tests (R2)** — `rebuild_dirty_chunk` mới ở
  `nodes/summarizer.py` (tái dùng `_summarize_messages` tách từ `_run_summarize`); api/main.py
  fire đúng signature. +17 test (dirty-window, re-summarize round-trip, R1 regression, A1/A3).
  Subagent code (Sonnet, karpathy-guidelines), K verify code + tự chạy 225/225 pass (PG thật,
  không skip). worklog 13/06.
- ✅ **Path traversal `persona_id` (A0)** — 2 tầng: Pydantic `pattern` ở `ChatRequest` +
  validate regex + `relative_to` containment ở `_persona_loader`. 4 test pass. K verify 13/06.
  worklog 12/06-cont (cụm A).
- ✅ **`user_memory` write path (A3)** — 3 endpoints POST/GET/DELETE `/users/{id}/memory`,
  ownership check ở DELETE. K verify 13/06. worklog 12/06-cont.
- ✅ **Task registry `_pending_summarizer_tasks` (A5)** — chuyển về `nodes/summarizer.py`
  module-level, bỏ lazy-import ngược. worklog 12/06-cont.
- ✅ **Clarify động M.2b — tool emit ambiguity (A1 cụm A)** — `memory_search` (gap sim<0.05) +
  `resume_last_session` (gap thời gian<24h) trả `{ambiguous, candidates}`; synthesizer clarify
  nhận tool_results. K verify 13/06. worklog 12/06-cont. (⚠️ còn thiếu test — R2 ở trên.)
- ✅ **`memory_search` tenant leak (A1 tái xuất)** — SQL thêm `session_id = ANY($ids)` cả 2
  nhánh; test tenant-isolation 2-user chạy PG thật PASS. K verify 12/06. worklog 12/06 (PR 1).
- ✅ **`memory_search` + `resume_last_session` bind vào graph** — vào `RETRIEVER_BASE_TOOLS`,
  scope inject qua `config: RunnableConfig` (LLM không thấy `user_id` — test schema verify).
  worklog 12/06 (PR 1).
- ✅ **Background summarizer M.5** — `nodes/summarizer.py` mới: trigger 10k (D13), nền
  (create_task + strong-ref), CAS `ON CONFLICT uq_chunk`, retry 1×. Hook sau
  `write_session_turn`. 10 unit tests. worklog 12/06 (PR 2). (E2E LLM thật → item 🟠 trên.)
- ✅ **Redis STM key/format lệch** — reader đổi sang `stm:{session_id}` +
  `_normalize_redis_format` chấp nhận cả 2 format. worklog 12/06 (PR 1).
- ✅ **Memory & Intent rebuild (M.9 15 bước)** — schema M.4, 3-axis intent, 8 nodes, TAG_RULES,
  GDPR cascade, e5-small. worklogs 06/06 + 11/06. (Còn sót: summarizer M.5 + bind memory tools —
  xem 🔴 phía trên.)
- ✅ **Migration tool (Alembic)** — `002_m4_fresh` chạy thành công, 7 tables. worklog 11/06.
- ✅ **Integration test với PostgreSQL thật** — 187/187 passed (PG 5433 + Redis + embedding).
  worklog 11/06.
- ✅ **HNSW thay IVFFlat** — schema 002 dùng HNSW mọi bảng vector (A5 đóng). worklog 06/06.
- ✅ **DROP cột chết `conversations`** — fresh schema 002 drop toàn bộ bảng cũ, không còn
  JSONB messages. worklog 06/06.
- ✅ **3 test `test_phase3_api` fail** — Redis mock `AsyncMock` fix. worklog 11/06 (bug #9).
- ✅ **Nút 📎 + `/documents/upload` chết** — đã gỡ nút khỏi UI (quyết: Option 1 KB-hệ-thống,
  user upload không thuộc MVP). tracking/status.md 11/06.
- ✅ **Tenant isolation (item cũ 29/05)** — superseded: cụ thể hóa thành 🔴 "`memory_search`
  tenant leak" phía trên (12/06).
- ✅ Phase 6.10 — 8 tasks (CORS, log rotation, stop-gen, health checks, TTS cleanup, STM
  token budget, messages table, youtube ingest). worklog 27/05.
- ✅ Corrections 28/05 — messages dùng cột riêng (bỏ JSONB), resume POST→GET, STM lazy populate,
  `_to_uuid` import, breaker→degraded. worklog 28/05.
- ✅ `reasoning_output`/`final_answer` dedup. worklog 28/05.
- ✅ general_query support (off-domain). worklog 28/05 §5. (còn cần verify thủ công — xem trên)
