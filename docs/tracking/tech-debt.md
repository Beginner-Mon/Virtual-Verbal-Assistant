# Tech Debt & Pending Tasks

> Checklist các việc đã biết nhưng CHƯA làm. Cập nhật khi đóng/ mở item.
> Last update: 2026-08-21 (K — DELETE account hoãn theo quyết định Owner; xem mục "Chờ quyết định").
> Trước đó: 2026-08-16 (K — Track 2 Lambda catalog: rate limit, concurrency, latency, migration).
> Trước đó: 2026-08-05 (K — Neon, 7 lỗi TS, TTS VieNeu, lip sync, auth 1-user, Capacitor).
>
> ⚠️ **Mục "Thiếu credentials" bên dưới đã lạc hậu**: 16/08 Owner xác nhận máy **có** AWS
> credentials (account `244203483654`, IAM user `admin`), và đã deploy thật `VvaCharacterStack` +
> `VvaAssetStack`. Các item auth trong đó cần kiểm lại chứ không còn bị chặn vì credentials.
> **Đọc mục 🚧 đầu tiên**: đó là những việc K bị chặn cứng, không phải việc chưa kịp làm.
> Nguồn trước đó: worklogs 05→12/06-cont; đợt mới: `docs/worklogs/30-07-2026.md`,
> `docs/worklogs/05-08-2026.md`.

Mức: 🔴 critical (phải làm trước Phase 7 deploy) · 🟠 quan trọng · 🟡 nên làm · ⚪ optional

---

## 🚧 K KHÔNG tự giải quyết được — cần Owner/N

> Danh sách này khác các mục bên dưới: đây **không phải việc chưa làm**, mà là việc
> **K bị chặn cứng**, kèm lý do chính xác. Ghi 05/08 theo yêu cầu của Owner.
> Nguyên tắc: mọi thứ ở đây K đã viết xong code hoặc đã chẩn đoán xong — thứ thiếu là
> **quyền truy cập, phần cứng, hoặc một quyết định**, không phải thời gian.

### Thiếu credentials

- [ ] 🔴 **Toàn bộ auth (Đợt 1 + Đợt 2) CHƯA từng chạy thật** — `npx ampx sandbox` chết ở
      `[InvalidCredentialError] Failed to load default AWS credentials`. `~/.aws/` rỗng, không có
      `AWS_PROFILE`/`AWS_ACCESS_KEY_ID`, không có aws-cli, `amplify_outputs.json` chưa từng tồn tại.
      **Hệ quả**: mục tiêu "không duplicate tài khoản" mới chỉ chứng minh được là **biên dịch được**.
      Chưa biết `AdminLinkProviderForUser` có link sạch không, IAM đủ chưa, `ListUsers` filter có
      khớp không, `set-password` trên user đã neo có chạy không — mỗi thứ đều đủ sức làm hỏng đăng nhập.
      **Gỡ chặn**: `npx ampx configure profile` hoặc đặt biến môi trường AWS.
- [ ] **Tắt scale-to-zero của Neon** — Owner bảo "tắt tạm" nhưng đây là thao tác trên **dashboard
      Neon**, K không có tài khoản. Ảnh hưởng: request đầu sau khi idle bị cold-start.
- [ ] 🔑 **Đổi mật khẩu Neon** — DSN đã đi qua kênh chat. Chỉ Owner làm được.

### Thiếu toolchain

- [ ] 🔴 **Không build/chạy được app mobile** — máy **không có JDK, không có Android SDK**; iOS thì
      **không thể trên Windows** (cần macOS + Xcode). Nên `npx cap add android` chưa chạy, chưa có
      thư mục `android/`, và toàn bộ `src/lib/nativeAuth.ts` **chưa từng thực thi một lần nào**.
      **Gỡ chặn**: cài JDK 17+ và Android Studio; iOS cần máy Mac.
- [ ] **Không xem được bằng mắt** — lip sync, autoplay TTS, hiệu ứng shadow… K chỉ verify được tới
      lớp dữ liệu (SSE trả gì, file tải về bao nhiêu byte, tsc/build xanh). Phần "nhìn thấy đúng
      không" luôn phải Owner làm. Từng thử Playwright nhưng treo ở khâu giải mã ảnh, và có lúc máy
      cạn commit limit khiến `tsc` còn không khởi động nổi.

### Thiếu hạ tầng bên ngoài (Owner sở hữu)

- [ ] 🔴 **App Links cho mobile auth** — cần **một domain thật** ngài sở hữu, rồi host
      `/.well-known/assetlinks.json` (Android) và `/.well-known/apple-app-site-association` (iOS),
      **và** khai cùng redirect URI đó trong **Google Cloud console** (Amplify không đụng tới được).
      Không có domain thì không có đường nào khác: Amplify bản web **từ chối custom scheme**
      (`invalidRedirectException`) và với origin mặc định của Capacitor nó tự chọn nhầm
      `http://localhost:5173/`. Checklist đầy đủ: `docs/mobile-app-links.md`.
      ⚠️ Fingerprint bản **debug và release khác nhau** — khai thiếu một cái là App Link im lặng
      không chạy, cực khó đoán.

### Thiếu file nguồn (không sửa được bằng code)

- [ ] **`bronya.vrm` không có morph viseme** — bindCount A/I/U/E/O đều **0**. Lip sync chạy đúng mà
      miệng vẫn đứng im. Phải chỉnh file VRM bằng VRoid/Blender. Ba model kia đều có.
- [ ] **2 clip AMASS sinh bởi tool khác** — `motion_7b4b8d9e`, `motion_b28e8284` có hierarchy khác
      converter hiện tại. Regenerate là motion đổi hẳn. Cần người **nhìn bằng mắt** để duyệt bản mới,
      K không tự quyết được.

### Chờ quyết định, không chờ code

- [ ] **SSAO**: bật `enableNormalPass` hay tắt hẳn — cả hai đều đổi hình ảnh.
- [ ] **Panel thông báo**: giữ như Tri làm hay khôi phục mockup cũ (Tri xoá có lý do chính đáng —
      4 toggle đó không bấm được).
- [ ] **grader-retry nối chữ 2 lần**: Owner từng đề xuất bỏ live-stream, chưa chốt.
- [ ] **`voiceReply` mặc định bật hay tắt** — đo lại thì giọng nói chỉ thêm ~5s, không phải gấp đôi
      như K báo ban đầu.
- [ ] **TTS: đổi sang FE poll thay vì giữ SSE 130s** — đổi luồng, cần Owner duyệt.

- [ ] 🔴 **DELETE account — Owner hoãn 21/08 để bàn lại về bảo mật.** Không phải việc chưa
      kịp làm: code xoá **đã có** (`api/main.py:337` `DELETE /me` → `db/gdpr.py:123`), cascade
      Postgres **đã đúng** (`alembic/versions/002_m4_fresh_schema.py`, `004_demo_billing.py`).
      Thứ thiếu là một quyết định về phạm vi, vì thứ đang có **xoá thiếu 4 kho dữ liệu**:
      Cognito user (⇒ đăng nhập lại được, dữ liệu tự dựng lại qua `routes_crud.py:113` —
      nên đây là *reset*, gọi là "xoá tài khoản" trong UI là nói sai), DynamoDB
      `UserMappings` (⇒ đăng ký lại bằng email cũ bị `pre-sign-up` nối vào **danh tính cũ** —
      lỗ nghiêm trọng nhất), `EmailLocks` (khoá 24h, không báo lý do), Stripe subscription.
      Còn phải chốt: ai cầm `cognito-idp:AdminDeleteUser` (hay dùng Amplify `deleteUser()`
      bằng access token của chính user), xoá ngay hay grace period 30 ngày (không rollback
      được, không có backup theo user), và chống lạm dụng (hiện **không** rate limit, không
      xác nhận lại mật khẩu, không email — token bị trộm là một POST xoá sạch, im lặng).
      Phân tích đầy đủ: [`docs/plans/langgraph-agent-hosting.md`](../plans/langgraph-agent-hosting.md) §7.
      ⚠️ **Kèm ràng buộc lên việc host agent**: Phase B/C của plan đó đưa `api/main.py` lên
      Lambda, nên `create_app()` nguyên trạng sẽ **vô tình publish `DELETE /me`**. Phase A
      phải gate sau `ENABLE_GDPR_ROUTES` (default `false`) trước khi build image.
      ⚠️ `db/init_schema.sql` thiếu cascade trên `conversations.user_id` — file đó lạc hậu so
      với alembic, đừng đọc như schema hiện hành.

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

### Track 2 — Lambda catalog (ghi 16/08, worklog `16-08-2026.md` §6b)

- [ ] 🔴 **Chưa có rate limit ở bất kỳ tầng nào.** CloudFront **chưa gắn WAF** (`WebACLId` rỗng),
      Lambda **chưa set reserved concurrency**, Function URL không có throttle. Thứ duy nhất đang
      chặn là cache CloudFront (TTL 300s cho `/characters*`, `immutable 1 năm` cho VRM). Budget
      $10/$20 đã có nên sẽ báo sớm. **Chưa khuyến nghị WAF** — WebACL ~$5/tháng + $1/rule +
      $0.60/triệu request, tức ngốn 25-50% ngân sách để chống mối nguy mà cache đã chặn phần lớn.
      Làm khi có traffic thật.
- [ ] 🔴 **Hạn mức Lambda concurrency của account = 10, và là `Unreserved` tức DÙNG CHUNG** với
      toàn bộ Lambda khác, kể cả auth trigger của Amplify (`post-confirmation`,
      `pre-token-generation`). Một đợt burst vào `/characters` về lý thuyết làm nghẽn đăng nhập
      Cognito. Đã đo: 50 request đồng thời → 10× 200, 40× 429. **Phải xin nâng quota trước khi
      chuyển endpoint per-user (`/sessions`, `/users/*/memory`) sang Lambda** — những cái đó không
      cache được nên sẽ ăn thẳng vào pool.
- [x] ✅ **XONG 17/08 (`73ad2ee`) — Warm duration 436 ms, gần như toàn bộ là chặng vượt Thái Bình
      Dương.** Lambda ở `us-east-1`, Neon khi đó ở `ap-southeast-1`: mỗi lần gọi tốn 2 round trip
      (liveness `SELECT 1` + query) × ~215 ms. Query 4 dòng có index chỉ tốn vài ms. Đã chuyển Neon
      sang `us-east-1`, kỳ vọng ~20-30 ms. Cold start trước đó 3019 ms (init 539 ms + bắt tay
      TLS/SCRAM xuyên Thái Bình Dương + Neon scale-to-zero).
      **Chưa đo lại sau khi chuyển** — con số 20-30 ms vẫn là dự đoán, không phải kết quả.
- [ ] 🟡 **Giảm memory Lambda 256 → 128 MB.** Đang dùng tối đa **103 MB**. Lambda cấp CPU tỉ lệ
      memory, nhưng hàm này **chờ mạng chứ không tính toán** nên giảm CPU gần như không làm chậm.
      Cắt đôi GB-giây. Một dòng trong `character_stack.py`.
- [ ] 🟡 **`PriceClass_All` → `PriceClass_200`.** Đang trả tiền cho edge Nam Mỹ/châu Phi không dùng.
      `_200` vẫn phủ châu Á. Miễn phí, đổi một dòng.
- [ ] 🟡 **`AWSLambdaBasicExecutionRole` cấp `logs:*` trên `Resource: "*"`** — managed policy chuẩn
      của AWS, là chỗ duy nhất chưa least-privilege trong toàn bộ Track 2. Muốn chặt thì thay bằng
      inline policy giới hạn `/aws/lambda/vva-characters`. Rủi ro thực tế thấp (chỉ ghi log được).
- [ ] 🟡 **`GET /characters/{slug}` chưa ai gọi.** Xây theo draft plan §4.1 nhưng frontend chỉ dùng
      list + `avatar-profile`. Xoá cho gọn, hoặc giữ nếu sắp làm trang chi tiết nhân vật.
- [ ] 🟡 **CloudFront access log đang tắt.** Không có dữ liệu để điều tra khi có sự cố hoặc để biết
      bao nhiêu request thật sự tới origin. Bật thì tốn phí lưu S3.
- [ ] ⚪ **`aws-cdk-lib` thiếu `lambda:InvokeFunction` cho OAC + Function URL** — đã vá bằng
      `CfnPermission` thủ công trong `asset_stack.py`. **Xoá khi CDK cấp đủ cả hai action.** Theo
      dõi `aws-samples/remote-swe-agents#361`.
- [ ] ⚪ **GLB parser trùng nhau** giữa `ECA_UI/frontend/scripts/extract-vrm-meta.mjs` (Node, cho
      frontend) và `scripts/upload_characters_to_s3.py` (Python, cho DB). Chấp nhận — hai consumer,
      hai bộ field khác nhau, và đã đối chiếu khớp tuyệt đối trên cả 4 model.

### Migration & test (16/08)

- [ ] 🔴 **`asyncpg` không chạy được `op.execute()` nhiều câu lệnh** — đây là lý do thật khiến
      `004_demo_billing` chưa bao giờ áp được lên Neon, chứ không phải ai quên chạy. Đã tách `004`
      và `005` thành mỗi câu một `op.execute()`. **CI nên có bước chạy `alembic upgrade head` trên
      DB tạm** để bắt loại lỗi này ngay khi commit, thay vì phát hiện lúc deploy.
- [ ] 🟠 **12 test đỏ sẵn từ trước — hai nguyên nhân khác nhau.** Đã chứng minh không phải do đợt
      character-identity (stash 2 file backend rồi chạy đối chứng: baseline cũng đúng 12 đỏ).
      Chạy lại với `REQUIRE_AUTH=false` tách bạch được hai nhóm: 7 test SSE **pass**, 5 test grader
      **vẫn fail**.

      - **5 test `test_phase2_5_grader.py` — đỏ ở MỌI nơi, kể cả CI.** `d18e744` (11/08, "add safety
        template for each persona") thêm tham số `config` vào `grader_node` vì `grader.py:342` cần
        `config["configurable"]["persona_id"]`. Test lần cuối được sửa ở `47c67d0` (12/06) nên vẫn
        gọi `grader_node(state)` → `TypeError`. Cả hai file đều `@pytest.mark.unit` nên CI **có**
        chọn chúng ⇒ **CI đang đỏ mà không ai để ý, hoặc không chạy trên nhánh này** — cần xác minh
        (máy chưa cài `gh`). Sửa: truyền `{"configurable": {"persona_id": "eca_default"}}`, 5 dòng.

      - [ ] 🔴 **7 test `test_phase5_sse.py` — bộ test KHÔNG hermetic.** `shared/env.py:120` gọi
        `load_env()` lúc import, nạp `agenticRAG/.env` (gitignore) vào `os.environ`; `auth.py:26`
        đọc `REQUIRE_AUTH` **một lần lúc import module**. `.env:56` đặt `true` nên test POST `/chat`
        không kèm token bị 401. Trên CI không có `.env` → mặc định `false` → **xanh**.
        Cùng một commit, hai kết quả tuỳ máy.

        Đây mới là phần nguy hiểm: bộ test nói dối về trạng thái thật. Đúng loại bệnh mà
        `test_requirements_complete.py` được viết ra để chống ("fails on the machine that adds the
        import rather than on the machine that deploys it") — cùng bệnh, chỗ khác. Sửa ở
        `conftest.py`: đặt `REQUIRE_AUTH=false` **trước khi** app được import, vá cả lớp vấn đề chứ
        không riêng 7 test.

      Owner quyết định (17/08): sửa sau khi merge vào `feature/langgraph-rewrite`.
- [ ] 🟡 **Không chỗ nào trong repo ghi môi trường Python nào dùng cho việc gì.**
      `C:\Miniconda\envs\firstconda` là env backend, `infra/.venv` là env CDK. Thiếu tài liệu này
      đã khiến K kết luận sai "máy chưa có venv dự án". Ghi vào `scripts/QUICKSTART.md`.

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
