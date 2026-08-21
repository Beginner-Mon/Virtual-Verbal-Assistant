# Bàn giao — Phase A + Phase 0, deploy LangGraph agent

> K · 21/08/2026 · nhánh **`feature/agent-deploy`** · **chưa push**
> Kế hoạch: [[langgraph-agent-hosting]] · Nhật ký: [[21-08-2026]]
> 5 commit · 31 file · +3824 / −119 · Tests **481 passed / 0 failed** (đầu: 437)

---

## 0. Zip hay container? — gỡ nhầm lẫn trước

Owner hỏi đúng: *"sao giờ lại dùng zip, không phải container à?"*

**Agent vẫn là container.** Zip chỉ dùng cho spike Phase 0, và spike đó **đã bị
xoá khỏi AWS** ngay trong cùng phiên.

| | Spike (Phase 0) | Agent thật (Phase B) |
|---|---|---|
| Đóng gói | **zip** 5,3 MB | **container** ~1,2 GB |
| Nội dung | hello-world FastAPI ~40 dòng | LangGraph + ONNX + model e5 |
| Mục đích | một câu hỏi về transport | chạy `/chat` thật |
| Hiện trạng | **đã destroy** | **chưa build** |

Vì sao zip hợp lệ cho spike: `AWS_LWA_INVOKE_MODE` là **biến môi trường**, và LWA
là **cùng một binary** ở cả hai kiểu đóng gói. Nên câu hỏi "prelude của LWA có
khớp thứ API Gateway đòi không" độc lập với cách đóng gói. Dùng container ở đó là
trộn một ẩn số (streaming) với một ẩn số khác (image 1,2 GB bake model đúng chưa)
— hỏng thì không biết cái nào hỏng.

---

## 1. Trạng thái

| | |
|---|---|
| **Phase A** (gỡ localhost) | ✅ **5/5 XONG** |
| **Phase 0** (spike streaming) | ✅ **XONG — kết quả STREAMED** |
| **Phase B** (container image) | ✅ **XONG** — chưa build lần nào |
| **Phase C** (CDK) | ✅ **XONG** — ECR đã tạo, function chờ image |
| **Phase D** (CI/CD + cutover) | ✅ **XONG** — workflow chờ merge |

### Đã đứng trên AWS

| | |
|---|---|
| ECR `vva-agent` | ✅ tạo 21/08 — IMMUTABLE, scan-on-push, giữ 10 image |
| SSM `/vva/llm/deepseek-api-key` | ✅ SecureString |
| SSM `/vva/llm/gemini-api-keys` | ✅ SecureString (4 khoá, một param) |
| IAM `GitHubActionsECRRole` | ✅ +`VvaAgentDeploy` — `UpdateFunctionCode`/`GetFunction`/`InvokeFunction`, scope **đúng một** function |
| Lambda `vva-agent` | ⬜ chờ image đầu tiên |
| Route `/v1/chat` | ⬜ chờ function |

Spike Phase 0 đã destroy; `describe-stacks` và `get-function` đều xác nhận không
sót gì.

---

## 2. Phase 0 — kết quả quan trọng nhất

Giả định rủi ro nhất của cả kế hoạch: prelude của LWA ở chế độ `response_stream`
có khớp thứ API Gateway Lambda-proxy STREAM đòi không? AWS tài liệu hoá **hai
nửa riêng biệt** và không bao giờ nói chúng khớp nhau — mọi ví dụ streaming của
LWA đều viết cho Function URL.

Deploy thật, đo thật:

```
arrived  2.812s   seq=0    server emitted_at=0.000s   ← cold start
arrived  3.265s   seq=1    server emitted_at=0.518s
arrived  3.765s   seq=2    server emitted_at=1.019s
…
11 events, spread 4.953s
```

**STREAMED.** Thời điểm *đến* bám đúng nhịp 0,5s của thời điểm *phát*. Bản
buffered sẽ dồn cả 11 vào vài mili-giây ở cuối.

⇒ **Bỏ hẳn Function URL và shared secret.** API Gateway gọi thẳng Lambda bằng
IAM. Kiến trúc §1 của kế hoạch đứng vững.

Đo bằng dấu thời gian chứ không nhìn `curl -N`, vì một response buffered mà nhanh
và một response streamed mà chậm **trông giống hệt nhau** trên terminal — và
"trông hơi chậm" đúng là kiểu hỏng spike này sinh ra để loại trừ.

Số phụ đáng giữ: **first byte 2,812s** = cold start của zip 5,3 MB. Mốc dưới để
so khi đo image thật ở Phase B.

Mất **~40 phút**, không phải nửa ngày như ước tính.

---

## 3. Phase A — 5 việc, và tác động

### 3.1 Cổng bảo mật `ENABLE_GDPR_ROUTES` (mặc định **off**)

`api/main.py` chỉ đăng ký `DELETE /me` và `DELETE /sessions/{sid}/messages/{mid}`
khi biến bằng **đúng** chuỗi `"true"`.

**Tác động**: đây là việc **chặn Phase B**. Phase B đưa `api/main.py` lên Lambda;
`create_app()` nguyên trạng sẽ publish endpoint xoá tài khoản cùng ngày `/chat`
lên sóng — đúng tính năng Owner đã hoãn vì lý do bảo mật. Không có cổng này thì
việc hoãn chỉ tồn tại trên giấy.

Mặc định off chứ không phải "nhớ tắt khi lên prod": xoá không rollback được, và
không có backup theo user.

### 3.2 `shared/stm.py` — một interface, ba backend

`redis://localhost:6379` từng được viết lại **4 lần**. Nay đi qua `get_stm()`,
backend chọn bằng `STM_BACKEND`: `redis` / `dynamodb` / `none`.

**Tác động**: **gỡ chặn quyết định Redis của Owner.** Backend `none` là một
deployment được hỗ trợ thật — agent chạy đúng khi không có cache nào, chỉ thêm
một lượt đọc Postgres mỗi turn. Nghĩa là deploy trước, đo, rồi mới chọn cache.

🔴 **Bẫy TTL DynamoDB đã xử lý**: AWS xoá item hết hạn *"within a few days"*,
thường 48 giờ, **không** đúng mốc. Bê nguyên `setex` sang thì STM đáng ra chết
sau 2 giờ vẫn đọc được tới ~48 giờ — agent trả lời bằng ngữ cảnh lẽ ra đã quên,
**không lỗi, không log, chỉ sai**. Nên ghi `expires_at` và kiểm khi đọc.

### 3.3 + 3.4 Cổng TTS và web search

`VIENEU_TTS_URL` không đặt ⇒ phát `speech_disabled`, **không** phát
`speech_pending`; `POST /tts` trả 503. `SEARXNG_URL=""` ⇒ trả `[]`, log một lần.

**Tác động**: `main.py` chờ TTS **130 giây** trong request. Trên Lambda đó là
130s tiền RAM chờ một service không tồn tại — và AWS tài liệu rõ **streaming tính
tiền đủ thời gian kể cả khi user đã đóng tab**, nên `is_disconnected()` không cứu
được.

### 3.5 ONNX — parity đạt trên dữ liệu thật

```
single (200 rows KB)      cosine min 1.00000000
batched, mixed lengths    cosine min 1.00000000
long input (~13.8k token) cosine min 1.00000000
queries (20)              cosine min 1.00000000
top-5 retrieval order     0/20 truy vấn bị đảo
```

**Vẫn để mặc định `torch`.** Parity đạt rồi, nhưng file ONNX là artifact 465 MB
bị gitignore — đổi mặc định sẽ làm hỏng mọi checkout chưa chạy export. Image
Lambda đặt `EMBEDDING_BACKEND=onnx` tường minh.

**Tác động**: RAM 2048 → **1024 MB**, image ~4 GB → **~1,2 GB**, cold start ~20s
→ ~5s, và **ECR $2,00 → $0,60/tháng**. Vì tiền chờ tỉ lệ thuận với RAM, đây là
đòn bẩy chính lên chi phí.

---

## 4. Tám lỗi tìm được, không có lỗi nào tự báo

Không cái nào trong số này làm test đỏ hay ghi log lỗi.

| # | Lỗi | Vì sao im lặng |
|---|---|---|
| 1 | `DELETE /me` sẽ tự publish khi lên Lambda | chưa deploy nên chưa ai thấy |
| 2 | `redis://localhost` viết lại 4 lần | deploy xong vẫn trả lời được, chỉ là đọc Postgres mỗi lần |
| 3 | TTL DynamoDB trễ tới 48 giờ | trả về ngữ cảnh cũ, không có lỗi nào |
| 4 | TTS chờ 130s cho service không tồn tại | trả `speech_failed` sau 2 phút |
| 5 | `SEARXNG_URL=""` hỏng mỗi truy vấn | tính năng bị tắt trông như tính năng hỏng |
| 6 | 🔴 **`kimodo_node` gọi `get_mcp_client()` — hàm chưa từng tồn tại** | `ImportError` rơi vào `except` → RECOVERABLE → graph đi tiếp |
| 7 | `ainvoke({"query": …})` nhưng schema đòi `prompt` | **không chạm tới được** — chết ở lỗi 6 trước |
| 8 | Preflight sẽ báo "MISSING CRITICAL: sentence_transformers" trên image ONNX | alarm giả cho package cố ý không có |

**Lỗi 6 là nghiêm trọng nhất**: `mcp/client.py` chỉ export `get_mcp_tools()` và
`close_mcp_client()`. Mọi caller khác (`graph.py`, `retriever_agent.py`,
`health.py`) đều dùng đúng. Chỉ `kimodo.py` gọi một API **chưa từng tồn tại**,
nên node chết ngay **dòng đầu tiên** khối try. **Motion chưa từng chạy một lần
nào.**

Lọt được vì `test_phase3_mcp_kimodo.py` chỉ test **mock server** trực tiếp —
kiểm thứ ở đầu kia sợi dây, không kiểm sợi dây. `kimodo_node` không có test nào.

---

## 5. Sáu chỗ tôi tự sửa chính mình

Ghi lại vì chúng cho biết nên tin báo cáo của tôi tới đâu.

| Tôi từng nói | Sự thật |
|---|---|
| "ONNX chặn vì thiếu `optimum`+`onnxruntime` và thiếu đường vào Neon" | `onnxruntime` đã có, `.env` đã có DSN, model đã cache — **chưa từng bị chặn** |
| "Docker daemon không chạy ⇒ chặn Phase 0" | Phase 0 dùng zip, **không cần Docker** |
| Test parity in *"long input (225 chars, > 512 tokens)"* và **PASS** | 225 ký tự ≈ 60 token — **báo đạt cho việc chưa từng chạy** |
| Kế hoạch ghi nhánh `release` | đang trên `feature/langgraph-rewrite`; tôi lẫn "nhánh CI" với "nhánh đang làm" |
| "Kimodo ghi `.mp4` lên S3" | nó trả **NPZ** qua route HTTP của chính nó |
| "`optimum[exporters]` là đủ" | optimum 2.x tách exporter sang `optimum-onnx` |

Hai test guard của repo (`test_requirements_complete`, `test_preflight`) bắt được
`onnxruntime`/`tokenizers` chưa khai báo — đúng loại test tôi viết cho STM, giờ
bắt chính tôi. Và `test_stm.py::test_no_call_site_hardcodes_a_redis_url` bắt được
`services/vieneu_tts/tasks.py` ngay lần chạy đầu.

---

## 6. Biến môi trường mới (6)

| Biến | Mặc định | Ghi chú |
|---|---|---|
| `STM_BACKEND` | `redis` | `redis`/`dynamodb`/`none`; typo ⇒ fail lúc khởi động |
| `STM_TABLE` | `vva-stm` | chỉ dùng khi `dynamodb` |
| `REDIS_URL` | `redis://localhost:6379/0` | STM + task result TTS |
| `VIENEU_TTS_URL` | *(rỗng)* | **rỗng = TTS tắt**, kể cả local |
| `ENABLE_GDPR_ROUTES` | `false` | chỉ đúng `"true"` mới mở |
| `EMBEDDING_BACKEND` | `torch` | `onnx` cho image Lambda |

🔴 **Hai thay đổi làm hỏng thói quen local của N**: giọng nói ở `/chat` và
`DELETE /me` nay **tắt** trừ khi đặt biến. Đã ghi vào `agenticRAG/.env.example`
kèm lý do.

---

## 7. CHƯA LÀM

### Phase B — image (chưa bắt đầu)

Dockerfile **multi-stage**: stage 1 có torch+optimum để xuất ONNX rồi **bị vứt
đi**, stage 2 chỉ copy file `.onnx`. Gộp một stage thì torch nằm lại và toàn bộ
lý lẽ tiết kiệm mất ý nghĩa.

🔴 Nhắc lại lỗi đã tìm khi đối chiếu aws-core: **layer không dùng được với
container image**. `CrudApiStack` gắn LWA bằng layer; copy pattern đó sang sẽ
hỏng ở INIT với lỗi không nói gì về nguyên nhân.

Build **trên CI**, không cần Docker local (Owner chốt).

### Phase C — CDK (chưa bắt đầu)

🔴 **`DockerImageCode.from_ecr(repo, tag)`, KHÔNG phải `from_image_asset()`** —
cái sau build image lúc synth, tức đòi Docker local. Cần tạo ECR repo
`vva-agent`.

### Phase D — CI/CD + cutover (chưa bắt đầu)

`deploy-agent.yml`, thêm `lambda:UpdateFunctionCode` vào `GitHubActionsECRRole`,
xoá `VITE_API_BASE_URL` khỏi frontend.

### Kimodo (ngoài phạm vi, nhưng đã biết)

NPZ đi đường nào khi tunnel cắt route `/files/` của Kimodo — qua tunnel (trần
6 MB) hay Kimodo ghi thẳng S3. Cần **đo kích thước NPZ thật** trước khi chọn.
`.claude/plans/kimodo-alb-endpoint.md` lỗi thời ở phần ALB, **chưa đánh dấu
superseded**.

---

## 8. Còn lại

### 🔴 Quota Lambda — CHỈ Owner làm được, và không phải vì thiếu quyền

Đã thử bằng API và **AWS từ chối**:

> `IllegalArgumentException: You must provide a quota value greater than the
> default quota value of 1000.0`

Đọc kỹ câu đó: **mặc định của quota này LÀ 1000**, còn account đang ở **10**.
Tức account không "chưa được nâng" mà đã bị AWS **hạ xuống dưới mặc định** —
chuyện AWS làm với account mới. Service Quotas chỉ nhận yêu cầu **tăng trên**
mặc định, nên không có đường đi từ 10 lên 200 hay lên 1000 qua API.

Đường còn lại là Support case, và `aws support` trả:

> `SubscriptionRequiredException: Amazon Web Services Premium Support
> Subscription is required`

Account đang ở gói Basic. ⇒ **Phải làm tay trong AWS Console**: Support Center →
Create case → Service limit increase → Lambda → Concurrent executions.

Vì sao đáng làm, nói gọn: giới hạn 10 **không chặn DDoS**. Nó không ngăn request
tới, không phân biệt kẻ tấn công với người dùng, và khi cạn thì **đăng nhập chết
trước** — pool dùng chung với Cognito trigger. Thứ chặn DDoS là Cognito
authorizer (flood ẩn danh chết ở gateway, Lambda không chạy) và stage throttle.
Nâng quota làm hệ thống **an toàn hơn**, vì docs AWS ghi
`max reservable = account limit − 100`: ở mức 10 thì **không reserve được gì**,
nên hiện tại agent và đăng nhập dùng chung một pool không chia được.

Nâng quota **không làm tăng hoá đơn** — quota là trần, không phải phân bổ.

### Còn lại

| Việc | Ai | Ghi chú |
|---|---|---|
| 🔴 **Merge sang `release`** | Owner | Kích hoạt CI: build image → push ECR → tạo/roll function |
| 🔴 `cdk deploy VvaAgentStack -c agent_image_tag=<sha>` | N | Sau khi CI push image đầu tiên |
| 🟠 **Chốt backend STM** | Owner | Code sẵn cho cả ba. Khuyến nghị chạy `none` một tháng rồi đo CU-hours |
| 🟠 **Đo token LLM** | N | Khoản lớn nhất, **không** nằm trong $1,90/tháng. `total_tokens` đã log sẵn ở `main.py::chat_complete` |
| 🟡 **Kiểm frontend** | N | UI đọc được `speech_disabled` chưa? |
| 🟡 **Push nhánh** | Owner | `feature/agent-deploy` chưa có upstream |

---

## 9. Rủi ro còn treo trong code vừa viết

- **`_append_stm` read-modify-write, không nguyên tử.** Hai turn cùng session
  song song có thể mất một. Chấp nhận: một session là một người đang gõ, và kẻ
  thua là một entry cache mà Postgres dựng lại được.
- **`DynamoStore` chưa chạy thật lần nào.** Test dùng `MagicMock` cho boto3.
  Chưa có bảng, chưa có IAM, chưa một lần `put_item` thật.
- **Motion vẫn chưa chạy end-to-end.** Sửa xong lỗi 6 và 7 nhưng Kimodo cần GPU
  box, và `mcp/kimodo_server.py` là mock trả `mock://`. Cái đổi là node nay hỏng
  vì lý do **đúng**.
- **ONNX chưa chạy trong image.** Parity đo trên máy N; export trong Docker là
  một lần chạy khác. Phải đo lại bên trong image ở Phase B.
