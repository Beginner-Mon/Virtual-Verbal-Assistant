# Bàn giao — Phase A, deploy LangGraph agent

> K · 21/08/2026 · branch **`feature/agent-deploy`** (tách khỏi
> `feature/langgraph-rewrite` theo yêu cầu Owner)
> Kế hoạch: [[langgraph-agent-hosting]] · Nhật ký: [[21-08-2026]]
> Tests: **465 passed / 0 failed** (trước khi bắt đầu: 437)

---

## 0. Đọc trước — hai thay đổi làm hỏng thói quen local của N

Cả hai đều **cố ý** và đều theo kế hoạch, nhưng chúng đổi hành vi trên máy N chứ
không chỉ trên cloud. Không biết trước thì sẽ tưởng là bug.

| Cái gì | Trước | Sau | Cách lấy lại |
|---|---|---|---|
| **Giọng nói ở `/chat`** | luôn thử, mặc định `localhost:5000` | **TẮT** — phát `speech_disabled` | đặt `VIENEU_TTS_URL=http://localhost:5000` |
| **`DELETE /me`** | có | **404** | đặt `ENABLE_GDPR_ROUTES=true` |

Cả hai đã ghi vào `agenticRAG/.env.example` kèm lý do. Ba thứ **không** đổi hành
vi local: STM vẫn dùng Redis ở `localhost:6379/0`, web search vẫn dùng SearXNG ở
`localhost:6666`, Postgres không đụng tới.

---

## 1. Đã làm — thay đổi và tác động

### 1.1 Chốt bảo mật `ENABLE_GDPR_ROUTES` (mặc định **off**)

**Đổi**: `api/main.py` chỉ đăng ký `DELETE /me` và
`DELETE /sessions/{sid}/messages/{mid}` khi biến bằng đúng chuỗi `"true"`.

**Tác động**: đây là việc **chặn Phase B**. Phase B đưa `api/main.py` lên Lambda;
`create_app()` nguyên trạng sẽ publish endpoint xoá tài khoản cùng ngày `/chat`
lên sóng — đúng tính năng Owner vừa hoãn vì lý do bảo mật. Không có cổng này thì
việc hoãn của Owner chỉ tồn tại trên giấy.

**Vì sao mặc định off** chứ không phải "nhớ tắt khi lên prod": xoá không rollback
được và không có backup theo user. Và thứ route đó xoá **không phải tài khoản** —
`db/gdpr.py::delete_user` xoá row PostgreSQL, user Cognito sống tiếp nên người đó
vẫn đăng nhập được và `routes_crud.py:113` dựng lại row ở lần ghi kế tiếp.

### 1.2 `shared/stm.py` — một interface, ba backend (**file mới, 295 dòng**)

**Đổi**: `redis://localhost:6379` từng được viết lại **4 lần** ở `api/main.py`,
`db/session_store.py`, `nodes/memory.py`, `services/vieneu_tts/tasks.py`. Nay tất
cả đi qua `get_stm()`, backend chọn bằng `STM_BACKEND`.

**Tác động**:
- **Gỡ chặn quyết định Redis của Owner.** Backend `none` không phải test double
  mà là một deployment được hỗ trợ — agent chạy đúng khi không có cache nào, chỉ
  thêm một lượt đọc PostgreSQL mỗi turn. Nghĩa là **deploy được trước, chọn cache
  sau khi đo**, thay vì phải chọn trước.
- **Xoá một lớp bug đang chờ**: sửa ba chỗ quên một chỗ thì agent nửa nhớ nửa
  quên, và triệu chứng là "mất trí nhớ ngắt quãng" chứ không phải lỗi kết nối.
- **Giảm 4 lần connect/close mỗi turn xuống còn một client dùng chung** cho
  đường Redis.

🔴 **Bẫy TTL của DynamoDB — đã xử lý, và đây là phần dễ bỏ sót nhất.** AWS xoá
item hết hạn *"within a few days"*, thường trong 48 giờ, **không** đúng mốc thời
gian. Bê nguyên `setex` sang thì STM đáng ra chết sau 2 giờ vẫn đọc được tới ~48
giờ — agent trả lời bằng ngữ cảnh lẽ ra đã quên, **không lỗi, không log, chỉ
sai**. Nên `expires_at` ghi như thuộc tính thường và **kiểm khi đọc**; TTL của
DynamoDB chỉ còn dọn kho, không quyết định đúng/sai.

**Kèm**: mọi thao tác nuốt lỗi và log **một lần mỗi op**, không phải mỗi request.
CloudWatch tính tiền theo GB nạp vào, nên một cache chết sẽ thành một hoá đơn —
và chôn mất dòng log duy nhất có ý nghĩa.

### 1.3 Cổng TTS `VIENEU_TTS_URL`

**Đổi**: không đặt ⇒ `/chat` phát `speech_disabled` rồi kết thúc stream, **không**
phát `speech_pending`; `POST /tts` trả **503** thay vì task id.

**Tác động**: `main.py` chờ TTS **130 giây** ngay trong request. Trên Lambda,
theo docs AWS, **streaming tính tiền đủ thời gian kể cả khi user đã đóng tab** —
nên mỗi lượt có giọng nói sẽ tốn hai phút tiền RAM để đi tới một thất bại đã biết
trước. Về phía UI: `speech_pending` là lời hứa rằng `speech_ready`/`speech_failed`
sẽ theo sau, nên phát nó khi không có TTS là để UI quay vòng vĩnh viễn.

### 1.4 Cổng web search `SEARXNG_URL`

**Đổi**: `SEARXNG_URL=""` ⇒ trả `[]`, log **một lần mỗi process**.

**Tác động**: `os.getenv` trả chuỗi rỗng chứ không trả default, nên trước đây đặt
rỗng sẽ khiến code gọi `"/search"` vào hư không, hỏng mỗi truy vấn và log mỗi
lần — **một tính năng bị tắt trông như một tính năng hỏng**.

### 1.5 Test chống tái phát

`test_stm.py::test_no_call_site_hardcodes_a_redis_url` quét toàn package, bỏ qua
dòng comment. **Nó đã làm đúng việc ngay lần chạy đầu**: bắt được
`services/vieneu_tts/tasks.py` mà tôi chưa sửa.

---

## 2. File đã đụng

**Mới**

| File | Dòng |
|---|---|
| `agenticRAG/langgraph_agents/shared/stm.py` | 295 |
| `tests/langgraph_agents/test_stm.py` | 244 (15 test) |
| `tests/langgraph_agents/test_gdpr_route_gate.py` | 110 (9 test) |
| `docs/plans/langgraph-agent-hosting.md` | 566 — thay bản 1 đã sai |
| `docs/worklogs/21-08-2026.md` | 150 |

**Sửa** (`+thêm / −bớt`)

| File | Δ | Nội dung |
|---|---|---|
| `api/main.py` | +135 / −40 | cổng GDPR, `tts_enabled()`, STM, tách Redis khỏi STM |
| `tests/test_phase2_5_memory.py` | +76 / −22 | 4 test cũ → interface mới, +3 test |
| `tests/test_phase5_sse.py` | +47 / −0 | 2 test mới cho nhánh TTS tắt |
| `nodes/memory.py` | +24 / −22 | `_load_recent_raw_redis` → `_load_recent_raw_cache` |
| `mcp/web_search_server.py` | +35 / −0 | `search_enabled()` |
| `services/vieneu_tts/tasks.py` | +21 / −4 | URL đọc từ env, đọc lúc gọi |
| `db/session_store.py` | +19 / −22 | `_append_stm` qua store |
| `docs/tracking/tech-debt.md` | +21 / −1 | mục nợ DELETE account |
| `agenticRAG/.env.example` | +50 | 5 biến mới, kèm lý do |

**Đáng chú ý**: `nodes/memory.py` không đọc `langgraph.memory.redis_url` từ
`config/langgraph.yaml` nữa. Một URL trong file commit được thì không thể khác
nhau giữa các môi trường — mà đó chính là yêu cầu khi lên Lambda.

---

## 3. Biến môi trường mới (5)

| Biến | Mặc định | Ghi chú |
|---|---|---|
| `STM_BACKEND` | `redis` | `redis` / `dynamodb` / `none`; typo ⇒ **fail lúc khởi động**, cố ý |
| `STM_TABLE` | `vva-stm` | chỉ dùng khi `dynamodb` |
| `REDIS_URL` | `redis://localhost:6379/0` | STM + task result của TTS |
| `VIENEU_TTS_URL` | *(rỗng)* | **rỗng = TTS tắt**, kể cả local |
| `ENABLE_GDPR_ROUTES` | `false` | chỉ đúng `"true"` mới mở |

`SEARXNG_URL` đã có sẵn từ trước, nay đặt rỗng là tắt sạch thay vì hỏng ồn.

---

## 4. Kiểm chứng đã chạy

- `pytest -m unit` → **465 passed / 0 failed**, chạy bằng
  `C:\Miniconda\envs\firstconda\python.exe`.
- Thủ công: `STM_BACKEND=none` ghi/đọc trả miss đúng. Backend `redis` trỏ vào
  cổng chết (6399) → `get` trả `None`, `set` không raise, lần `get` thứ hai
  **không** log lại.

**Chưa chạy**: `/chat` thật đầu-cuối. Cần DB sống + khoá LLM, và kế hoạch giao
việc đó cho N (mục §11 Phase A của kế hoạch).

---

## 5. CHƯA LÀM

### 5.1 Trong Phase A — ONNX (mục duy nhất còn lại)

Code viết được, **nghiệm thu thì không**, vì cần hai thứ ngoài quyền K:

1. **Cài `optimum` + `onnxruntime` vào `firstconda`** — sửa môi trường làm việc
   của N, không phải sửa repo. K không tự làm.
2. **Đường truy cập KB trên Neon** — kế hoạch đòi `cosine(torch, onnx) > 0.9999`
   trên 200 mẫu lấy từ 2918 rows, **và** top-5 `kb_search` trùng thứ tự cho 20
   truy vấn thật. DSN nằm trong `.env` (gitignored).

K **không** viết đường ONNX rồi để đó chưa đo, vì mean pooling sai là kiểu hỏng
**im lặng**: vector vẫn ra 384 chiều, vẫn tra được, chỉ recall tụt.

**Tác động của việc chưa làm**: agent vẫn cần torch ⇒ RAM 2048 MB thay vì 1024,
image ~4 GB thay vì ~1,2 GB, cold start ~20s thay vì ~5s, và **ECR tốn ~$2,00
thay vì ~$0,60 mỗi tháng** — tức riêng khoản lưu image đắt hơn toàn bộ chi phí
chạy agent. Chưa có ONNX thì Phase B vẫn làm được, chỉ đắt hơn khoảng gấp đôi.

### 5.2 Phase 0 — spike streaming (chưa bắt đầu)

Docs AWS xác nhận LWA là đường streaming chính thức cho Python, **nhưng chỉ tài
liệu hoá đối chiếu với Function URL**. Với Lambda-proxy STREAM, API Gateway
*"expects a specific response format"* — một prelude chứa status code + headers.
**Không tài liệu nào xác nhận prelude của LWA khớp với prelude API Gateway đòi.**

Đây là **giả định rủi ro nhất của cả kế hoạch**. Nửa ngày spike ở đây tránh việc
phát hiện sau khi đã dựng xong image. Sai thì lùi về HTTP proxy → Function URL và
chịu thêm phần shared secret.

Kèm: xác nhận response streaming có ở us-east-1.

### 5.3 Phase B — image (chưa bắt đầu)

Dockerfile, bake model vào image, đo cold start bằng RIE.
🔴 Nhắc lại lỗi đã tìm ra khi đối chiếu aws-core: **layer không dùng được với
container image**. `CrudApiStack` gắn LWA bằng layer; copy pattern đó sang sẽ
hỏng ở INIT với lỗi không nói gì về nguyên nhân.

### 5.4 Phase C — CDK (chưa bắt đầu)

`agent_stack.py`, `/chat` vào `rest_api_stack.py` với
`ResponseTransferMode.STREAM`, nâng integration timeout tường minh (mặc định
29s), thêm agent vào warmer **có sẵn** — đừng tạo scheduler thứ hai.

### 5.5 Phase D — CI/CD + cutover (chưa bắt đầu)

`deploy-agent.yml`, thêm `lambda:UpdateFunctionCode` vào `GitHubActionsECRRole`,
xoá `VITE_API_BASE_URL` khỏi frontend.

### 5.6 Kimodo (ngoài phạm vi đợt này, theo kế hoạch)

Tunnel qua Lambda trong VPC. `.claude/plans/kimodo-alb-endpoint.md` nay **lỗi
thời** ở phần ALB nhưng **chưa được đánh dấu superseded** — phần chẩn đoán health
check 406 và grace period 600s trong đó vẫn đúng và vẫn cần.

---

## 6. Chỉ Owner/N gỡ được

| Việc | Vì sao chặn |
|---|---|
| 🔴 **Nâng quota Lambda concurrency** | Đang **10**, mặc định AWS là **1.000**. Dùng chung với Cognito trigger của Amplify. CRUD giữ container ~100ms; `/chat` giữ **10-30 giây** ⇒ ba người chat đồng thời khoá 30% pool, và thứ bị throttle là **đăng nhập**. Burst 120 request ngày 20/08 đã tạo 61 throttle. |
| 🔴 **Cài dep ONNX + đường đo parity** | §5.1 |
| 🟠 **Quyết backend STM cuối cùng** | Kế hoạch khuyến nghị chạy `none` một tháng, **đo CU-hours thật** rồi mới chọn. Code đã sẵn sàng cho cả ba. |
| 🟠 **Đo token LLM** | Khoản chi lớn nhất và **không** nằm trong bảng $1,90/tháng. `total_tokens` đã được log sẵn ở `main.py::chat_complete` — chạy 20 lượt thật, lấy trung bình. Làm được **ngay hôm nay**, không cần chờ deploy. |
| 🟡 **N kiểm frontend** | UI xử lý thế nào khi có `speech_pending` mà không bao giờ có `speech_ready`? Nay nhánh đó không còn xảy ra, nhưng chưa ai xác nhận UI đọc được `speech_disabled`. |

---

## 7. Rủi ro còn treo trong code vừa viết

- **`_append_stm` là read-modify-write, không nguyên tử.** Hai turn cùng session
  chạy song song có thể mất một. Chấp nhận: một session là một người đang gõ, và
  kẻ thua là một entry cache mà PostgreSQL dựng lại được. Mua tính nguyên tử sẽ
  phải dùng primitive riêng của Redis — đúng thứ `shared/stm.py` sinh ra để khỏi
  phụ thuộc.
- **`DynamoStore` chưa chạy thật lần nào.** Test dùng `MagicMock` cho boto3.
  Chưa có bảng, chưa có IAM, chưa có một lần `put_item` thật.
- **`RedisStore` cache client theo process.** Nhất quán với `_get_redis()` đã có
  trong `main.py`, và đúng dưới LWA (một event loop cho cả vòng đời process).
  Nhưng đây chính là kiểu hỏng mà `infra/lambda/crud_api/run.sh` cảnh báo với
  Mangum — nếu sau này có ai chạy agent qua adapter tạo loop mỗi invoke, chỗ này
  hỏng trước.
