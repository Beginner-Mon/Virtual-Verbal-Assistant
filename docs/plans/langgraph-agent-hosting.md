# Kiến trúc deploy LangGraph Agent — bản 4

> K (Mr. Senryuu) · 21/08/2026
> Đang đứng trên `feature/langgraph-rewrite` — **bản 3 ghi `release`, sai**. CI
> (`release-tests.yml`, `deploy-production.yml`) trigger trên `release`, và tôi
> đã lẫn "nhánh CI chạy" với "nhánh đang làm việc".
> Mọi con số về AWS đã đối chiếu docs qua plugin `aws-core`.

---

## 0. Việc đầu tiên: tách nhánh (Owner yêu cầu 21/08)

Toàn bộ Phase A đang là **thay đổi chưa commit** nằm thẳng trên
`feature/langgraph-rewrite` (9 file sửa, 6 file mới). Không đúng ý Owner.

```bash
# Từ chính nơi đang đứng — switch -c mang theo thay đổi chưa commit,
# nên không cần stash và không có gì bị mất.
git switch -c feature/agent-deploy

git add agenticRAG/langgraph_agents/shared/stm.py \
        agenticRAG/langgraph_agents/api/main.py \
        agenticRAG/langgraph_agents/db/session_store.py \
        agenticRAG/langgraph_agents/nodes/memory.py \
        agenticRAG/langgraph_agents/mcp/web_search_server.py \
        agenticRAG/langgraph_agents/services/vieneu_tts/tasks.py \
        agenticRAG/.env.example \
        tests/langgraph_agents/test_stm.py \
        tests/langgraph_agents/test_gdpr_route_gate.py \
        tests/langgraph_agents/test_phase2_5_memory.py \
        tests/langgraph_agents/test_phase5_sse.py \
        docs/plans/langgraph-agent-hosting.md \
        docs/tracking/agent-deploy-handoff.md \
        docs/tracking/tech-debt.md \
        docs/worklogs/21-08-2026.md
```

Tên `feature/agent-deploy` theo đúng quy ước đang có trong repo
(`feature/capacitor-mobile`, `feature/character-identity`, `feature/frontend`).

Hai commit chứ không phải một, vì chúng có lý do khác nhau và có thể cần revert
độc lập:

1. `feat(agent): make STM, TTS and web search configurable` — `shared/stm.py` +
   4 call site + `.env.example` + test.
2. `fix(security): gate the account-deletion routes behind ENABLE_GDPR_ROUTES` —
   cổng + test. Đây là commit mà một người review bảo mật sẽ muốn đọc riêng.

Docs đi kèm commit tương ứng, **không** gộp thành commit "docs" thứ ba: một thay
đổi hành vi và lời giải thích cho nó mà nằm ở hai commit khác nhau thì `git log`
của file đó không kể được câu chuyện.

⚠️ **Đừng push `feature/langgraph-rewrite` sau khi switch.** Nhánh cũ giờ không
còn các thay đổi này; nó vẫn ở đúng chỗ trước khi bắt đầu.

---

## Context

Bản 1 chọn đúng runtime nhưng mỏng ở bốn chỗ Owner chỉ ra: chi phí **chờ** của
Lambda với workload I/O-bound; TTS chưa được tính; Kimodo phải bật/tắt ALB bằng
tay; và **chưa có CI/CD nào cho langgraph-agent**.

Bản 2 sửa bốn chỗ đó. Vòng đối chiếu `aws-core` này tìm thêm **một lỗi kỹ thuật**
và **hai rủi ro** chưa ai nói tới — mục §1.

**Kết quả mong muốn**: `/chat` chạy trên AWS sau đúng một cửa API Gateway,
`VITE_API_BASE_URL` biến mất khỏi frontend, chi phí định kỳ **~$1/tháng**, không
còn thao tác tay trong đường deploy.

---

## 1. Kết quả đối chiếu aws-core

### ✅ Xác nhận — kiến trúc `/chat` đúng

> *"Lambda supports response streaming on Node.js managed runtimes. For other
> languages, **including Python**, you can use a custom runtime … **or use the
> Lambda Web Adapter**."* — [Response streaming for Lambda functions](https://docs.aws.amazon.com/lambda/latest/dg/configuration-response-streaming.html)

> *"Your Lambda function can also stream response payloads through the Amazon
> API Gateway proxy integration, which uses the `InvokeWithResponseStream` API."*

**`AWS_PROXY` nghĩa là gì, nói cho gọn**: API Gateway có hai kiểu tích hợp —
`AWS_PROXY` gọi thẳng Lambda (gateway gói request HTTP thành JSON đưa cho Lambda),
`HTTP_PROXY` chuyển tiếp tới một URL bất kỳ. `apigw.LambdaIntegration(crud_fn)` ở
`rest_api_stack.py:162` **đã là** `AWS_PROXY`. Nên `/chat` không thêm hình dạng
kiến trúc nào mới — nó vào đúng đường `/sessions` và `/characters` đã vào. Khác
biệt duy nhất là **một tham số**:

```python
apigateway.LambdaIntegration(handler,
    response_transfer_mode=apigateway.ResponseTransferMode.STREAM)
```

Thiếu tham số đó thì SSE vẫn chạy — chỉ là gateway gom hết rồi mới gửi, user chờ
im lặng 20 giây rồi nhận cả cục. Hỏng theo kiểu trông giống "hơi chậm".

⇒ Bỏ hẳn Function URL và shared secret của bản 1. API Gateway gọi thẳng Lambda
bằng IAM. **Thêm một lý do nữa**: *"Lambda function URLs do not support response
streaming within a VPC environment"* — đường Function URL sẽ chết nếu sau này
agent phải vào VPC; đường `AWS_PROXY` thì không.

### 🔴 LỖI trong bản 1 và 2 — layer không dùng được với container image

> *"Max 5 layers/function; **layers don't work with container images**."*
> — `aws-serverless/references/lambda.md`

`CrudApiStack` gắn LWA bằng **layer** (`crud_api_stack.py:164`). Ai copy pattern
đó sang container sẽ hỏng ở INIT với lỗi không nói gì về nguyên nhân. Với image
phải copy binary vào image:

```dockerfile
COPY --from=public.ecr.aws/awsguru/aws-lambda-adapter:0.9.1 \
     /lambda-adapter /opt/extensions/lambda-adapter
ENV PORT=8080
ENV AWS_LWA_INVOKE_MODE=response_stream
```

Và **không** đặt `AWS_LAMBDA_EXEC_WRAPPER=/opt/bootstrap` — biến đó thuộc đường
zip+layer. Hai đường cấu hình khác nhau, `crud_api_stack.py` đang là đường kia.

### 🔴 RỦI RO MỚI 1 — streaming tính tiền **toàn bộ** thời gian, kể cả khi client ngắt

> *"Streamed responses are **not interrupted or stopped when the invoking client
> connection is broken**. Customers are billed for the **full function
> duration**, so customers should exercise caution when configuring long function
> timeouts."*

Đây đánh thẳng vào mối lo số 1 của Owner. `main.py:389` gọi
`await request.is_disconnected()` để dừng sớm — trên Lambda **việc đó không cứu
được tiền**, và có thể không bao giờ kích hoạt.

Hệ quả với thiết kế: **timeout là công cụ kiểm soát chi phí, không phải lưới an
toàn.** Bản 2 đề xuất 300s — quá rộng. Đổi thành **120s**: một turn thật là
10-30s, nên 120s đã là gấp bốn lần trường hợp xấu, và một lần DeepSeek treo chỉ
tốn 120s×1 GB = $0.002 thay vì $0.005.

### 🔴 RỦI RO MỚI 2 — TTL của DynamoDB **không đúng giờ**

> *"DynamoDB automatically deletes expired items **within a few days** of their
> expiration time."* … *"Use filter expressions to remove expired items from Scan
> and Query results."* — [Using TTL in DynamoDB](https://docs.aws.amazon.com/amazondynamodb/latest/developerguide/TTL.html)
> Blog AWS nói rõ hơn: *"usually completes within 48 hours."*

Redis `SETEX` hết hạn **đúng** 7200 giây. DynamoDB TTL là tiến trình nền
best-effort. Nghĩa là một STM đáng ra chết sau 2 giờ **vẫn đọc được tới ~48
giờ** — hội thoại đáng ra đã nguội vẫn giữ trí nhớ ngắn hạn suốt hai ngày, và
hỏng **âm thầm**.

Bắt buộc: ghi thêm thuộc tính `expires_at` (Number, epoch giây) và **kiểm tra khi
đọc** trong `shared/stm.py`. `GetItem` không có filter expression, nên đây là
kiểm tra ở tầng ứng dụng — coi item quá hạn như không tồn tại. TTL của DynamoDB
chỉ còn làm việc dọn rác, không làm việc quyết định đúng/sai.

### 🟠 Ba điều nhỏ hơn, đều phải vào code

- **Integration timeout REST API mặc định 29s**, chỉ nâng được cho Regional/private.
  API của ta là Regional ⇒ nâng được, nhưng **phải nâng tường minh**.
- **Không có WAF inspection với nội dung stream**, không cache, không VTL.
- **Response streaming chưa có ở mọi region** — xác nhận us-east-1 trước Phase 0.

### 🟢 Xác nhận các số đã dùng

- **1 vCPU tại 1.769 MB.** 1024 MB ≈ 0,58 vCPU. Xem §3 — con số này làm rõ đánh đổi.
- **Concurrent executions mặc định 1.000/region (soft).** Account này ở **10** ⇒
  bị hạ rất sâu, và **điều chỉnh được**.
- **On-demand init timeout 10s**, vượt thì Lambda thử lại ở invoke đầu bằng
  timeout của function ⇒ `AWS_LWA_ASYNC_INIT=true` là đúng.
- **Streamed response tối đa 200 MB**, uncapped 6 MB đầu. SSE chữ thì thừa sức.
- **ALB listener rule theo `http-header` + `fixed-response` 403** là pattern có
  tài liệu. Lưu ý: so khớp giá trị **không phân biệt hoa thường** và `*`/`?` là
  ký tự đại diện ⇒ secret không được chứa hai ký tự đó.

---

## 2. Kiến trúc

```
Browser ─┬─ /v1/chat   API GW  AWS_PROXY + ResponseTransferMode.STREAM
         │               └─ vva-agent  Lambda container 1024 MB, timeout 120s
         │                    ├─ Neon (pooled, TLS)     ngoài VPC
         │                    ├─ DynamoDB stm           ngoài VPC, expires_at
         │                    └─ DeepSeek / Gemini      HTTPS
         ├─ /v1/sessions|me|characters|health  → như hiện tại
         └─ .vrm → CloudFront + S3

(tách hẳn, chỉ bật khi demo — KHÔNG có ALB, không có bề mặt public)
  vva-agent ──lambda:Invoke──> kimodo-dispatch (trong VPC) ──> Kimodo g5.xlarge
```

Nguyên tắc giữ từ Track 2: **Lambda ngoài VPC**. Mọi thành phần được chọn để
không kéo NAT gateway $32.85/tháng vào.

---

## 3. Chi phí chờ — đòn bẩy là RAM, không phải runtime

### ONNX làm gì, và KHÔNG làm gì

Owner hỏi có phải ONNX là để tách model ra khỏi image, deploy chỗ khác không.
**Không.** Model vẫn nằm nguyên trong image, vẫn chạy in-process, vẫn cùng một
model `intfloat/multilingual-e5-small`, vẫn 384 chiều.

Thứ bị thay là **runtime**, không phải model:

```
Trước:  agent → sentence-transformers → PyTorch  → e5-small (~470 MB)
                                        ~2 GB thư viện

Sau:    agent → onnxruntime (~50 MB)             → e5-small.onnx (~470 MB)
```

Nghịch lý là ở đó: **thư viện nặng gấp 4 lần chính model nó chạy.** Ta đang đóng
gói 2 GB PyTorch — vốn có đủ thứ để *huấn luyện* mạng nơ-ron — chỉ để làm một
phép nhân ma trận suy luận vài mili-giây mỗi lượt. ONNX Runtime chỉ biết chạy
suy luận, nên nó nhỏ.

Vì sao K **không** đề xuất tách model ra service riêng: embedding được gọi ngay
trên đường nóng (`pgvector_tool.py:68`, mỗi lần tra KB) và mất ~200ms. Tách ra
tức là thêm một network hop và **thêm một cold start nữa nằm giữa đường
retrieval** — trả nhiều hơn để nhận về chậm hơn.

### Vì sao điều đó lại thành tiền

Tiền chờ tỉ lệ thuận với RAM cấp phát. RAM của agent bị torch chiếm, mà torch tồn
tại chỉ để chạy e5-small vài mili-giây mỗi turn — trong khi 20 giây còn lại của
lượt chat là ngồi đợi DeepSeek, và suốt 20 giây đó ta vẫn trả tiền cho 2 GB
PyTorch đang **không làm gì**. Container đã bỏ trần 250 MB của zip, nên lý do bản
1 loại ONNX không còn.

| | torch | ONNX |
|---|---|---|
| Image | ~4,0 GB | ~1,2 GB |
| RAM | 2048 MB | **1024 MB** |
| Cold start | ~20s | **~5s** |
| 30s chờ | $0,0010 | **$0,0005** |
| CI build+push | ~15 phút | **~5 phút** |

**Đánh đổi phải nói rõ** (docs xác nhận 1 vCPU ở 1.769 MB): 1024 MB chỉ được
~0,58 vCPU, nên **embedding chạy chậm hơn ~2 lần**. Chấp nhận được vì embedding
là ~200ms còn chờ mạng là ~20.000ms — tối ưu cho phần dài, không cho phần ngắn.
Ghi chú "over-provisioning memory often lowers cost" trong docs đúng cho việc
CPU-bound, **không đúng cho việc ngồi đợi**.

🔴 **Rủi ro duy nhất của ONNX, và nó im lặng**: `sentence-transformers` làm **mean
pooling theo attention mask + L2 normalize** sau model. Đường ONNX phải tái tạo
đúng hai bước. Sai thì vector vẫn 384 chiều, vẫn tra được, chỉ recall tụt — đúng
dạng hỏng mà `shared/embedding.py` đã cảnh báo với tiền tố `query:`/`passage:`.
Nghiệm thu bằng số: §9.

### Khi nào nên rời Lambda

Hoà vốn với App Runner (~$11/tháng) ở **~22.000 lượt/tháng**. Dự án không tới đó.
**Thứ đẩy ta rời Lambda không phải tiền mà là đồng thời**: Lambda một request một
container; một process asyncio phục vụ hàng trăm request I/O-bound trên một core.
Ngưỡng: **>5 người chat đồng thời thường xuyên**. Vì artifact là container, việc
chuyển sang App Runner/Fargate là đổi cấu hình, không viết lại.

---

## 4. STM — hiện trạng, và vì sao DynamoDB thay được

Owner hỏi STM nằm ở đâu. Đọc từ code:

| | |
|---|---|
| Key | `stm:{session_id}` — **một** key, không phải cấu trúc |
| Value | **một chuỗi JSON**: `[{q, a, ts}, …]` |
| Kích thước | **`_STM_MAX = 3`** cặp Q/A (`session_store.py:20`) ⇒ vài KB |
| TTL | `setex(…, 7200)` = 2 giờ |
| Đọc | `memory.py:109` · `main.py:231` |
| Ghi | `session_store.py:283` `_append_stm` (1 GET + 1 SETEX) |
| Tổng | **~4 thao tác/turn**, single-key trên blob nhỏ |

**Không chỗ nào dùng tính năng riêng của Redis** — không LIST, SORTED SET,
pub/sub, lệnh nguyên tử. Thứ duy nhất dùng là `SETEX`, tức TTL. Và code **đã**
chịu được khi Redis vắng: `_load_recent_raw_redis` trả `None` với mọi exception
rồi rơi về Postgres; `_append_stm` nuốt exception.

### Chi phí thật, đã verify

| | Sàn/tháng | Ghi chú |
|---|---|---|
| **DynamoDB** | **~$0,05** | Ngoài VPC. Repo đã dùng pattern này (`EmailLocks` có `ttl`) |
| Upstash Redis | $0 (free tier) | Ngoài VPC, giữ nguyên code, thêm vendor |
| ElastiCache **Valkey** Serverless | $6,13 **+ NAT $32,85 = $38,98** | $0,084/GB-giờ, tối thiểu **100 MB** |
| ElastiCache **Redis OSS** Serverless | $91,25 **+ NAT $32,85 = $124,10** | $0,125/GB-giờ, tối thiểu **1 GB** |

Đây là lời giải thích cho cú sốc giá: **Redis OSS sàn 1 GB, Valkey sàn 100 MB** —
chênh 15 lần. Nhưng cả hai chỉ sống trong VPC, mà agent cần ra internet ⇒ NAT bắt
buộc. **NAT đắt hơn cache.**

Latency: chênh ~3-5ms/thao tác, ~6ms/turn, trên turn 10.000-30.000ms. Cache ở vị
trí này mua **CU-hours của Neon**, không mua tốc độ cảm nhận được.

**Khuyến nghị: DynamoDB — với `expires_at` kiểm tra khi đọc** (§1, rủi ro 2).
Nếu sau này bật lại TTS thì cân nhắc Upstash: `_poll_speech_result` gõ cache **mỗi
250ms suốt 130 giây** (520 lượt/câu trả lời) — đó mới là hot loop cần Redis thật.

Thi hành: `shared/stm.py`, hai backend chọn bằng env — Redis cho local
(docker-compose đã có), DynamoDB cho cloud. Cùng hình dạng
`infra/lambda/layer/shared/db.py` đang rẽ nhánh theo `DB_MODE`.

---

## 5. Kimodo — giữ MCP, sửa hai thứ đang hỏng

Yêu cầu Owner: **chỉ agent gọi được, user không chạm tới.** Hiện trạng
(`kimodo_ecs_stack.py:44`): `ALB_ALLOWED_CIDR = "0.0.0.0/0"`, và chính comment
trong file ghi *"MCP endpoint không có auth"*. Server là FastMCP
`transport="streamable-http"` port 8000 (`mcp_server.py:413`) ⇒ cần **endpoint
HTTP ổn định có SSE**. SQS không thay được — Owner bác đúng, bản 2 đã sửa.

### Vì sao bỏ hẳn ALB (Owner chốt 21/08)

Hai điều đóng khung bài toán:

1. 🔴 **`ALB_ALLOWED_CIDR` về `/32` không dùng được.** Lambda ngoài VPC dùng
   Hyperplane ENI dùng chung; docs AWS nói thẳng *"you can't use them as static
   IPs"* — **không có IP nào để điền vào allowlist**. Thiết kế bảo mật hiện tại
   không có đường thi hành.
2. 🔴 **ALB không tắt được, chỉ xoá được.** Không có LB nào tính tiền theo lượt
   dùng (NLB cũng $0,0225/giờ). Nên mọi thiết kế còn ALB đều buộc "ngừng trả
   tiền" = "xoá hạ tầng" — mỗi lần bật lại là ~5 phút CFN, mỗi lần xoá là một cơ
   hội để lại drift. Stack này **đã** kẹt `UPDATE_ROLLBACK_COMPLETE` từ 28/07 vì
   đúng loại thao tác đó (xem `deploy.ps1`).

### Kiến trúc: tunnel qua Lambda trong VPC

```
vva-agent (Lambda, NGOÀI VPC)
   │  lambda:Invoke  — IAM, request/response
   ▼
kimodo-dispatch (Lambda, TRONG VPC, private subnet, 128 MB)
   │  HTTP → task :8000, hoàn toàn trong VPC
   ▼
Kimodo ECS task (awsvpc ENI; SG chỉ nhận 8000 từ SG của dispatch)
```

Kết quả: **không ALB, không IP công khai, không domain, không gì phải xoá.**
Kimodo không có bề mặt public nào — user không phải "không nên" truy cập mà là
**không thể**. Và dispatch Lambda chỉ nói chuyện với private IP trong VPC nên
**không cần NAT** (CloudWatch Logs của Lambda đi qua dịch vụ Lambda, không qua
VPC — nên logging vẫn chạy).

### Bốn điểm phải làm đúng

**a. Tìm task ở đâu.** Ưu tiên **ECS Service Discovery (Cloud Map) + private DNS
namespace**: task tự đăng ký/huỷ đăng ký, dispatch gọi
`http://kimodo.vva.local:8000/mcp`, không cần plumbing IP. Chi phí ~$0,50/tháng
cho private hosted zone. Đường $0 tuyệt đối: agent (ngoài VPC, có internet) gọi
`ecs:ListTasks` + `ecs:DescribeTasks` + `ec2:DescribeNetworkInterfaces` để lấy
private IP rồi truyền vào payload — đổi $0,50 lấy thêm code và một chỗ nữa để
cache sai. **Khuyến nghị Cloud Map**; nửa đô một tháng rẻ hơn một buổi debug.

**b. MCP đi qua tunnel bằng một `httpx` transport tùy biến.** Không viết lại
client MCP: cấp cho nó một transport biến mỗi HTTP request thành một
`lambda:Invoke`, dispatch replay nguyên văn (method, path, headers, body) tới
task rồi trả về status + headers + body. Phiên MCP sống trong Kimodo server và
được khoá bằng header `Mcp-Session-Id`, mà header thì đi qua tunnel nguyên vẹn ⇒
**phiên vẫn hoạt động**. Tầng `langchain-mcp-adapters` phía trên không đổi một
dòng.

**c. Giới hạn phải biết — và file NPZ đi đường nào.** Sync invoke của Lambda là
**6 MB request / 6 MB response**, và dispatch **gom** SSE lại thay vì chảy từng
phần. Chấp nhận được vì `tools/call` sinh motion trả **một** kết quả sau 5-10s.

⚠️ Bản 3 viết "`.mp4` ghi lên S3" — **sai**. Kimodo trả **NPZ** (SMPL-X motion
data) và phục vụ chúng bằng route HTTP của chính nó,
`@mcp.custom_route("/files/{filename:path}")` (`mcp_server.py:161`). Tunnel bỏ mọi
bề mặt public ⇒ **route đó chết, browser mất đường lấy file**. Xem §5b lỗi 2.

Hai lối ra, Owner chọn sau khi **đo kích thước NPZ thật**:

- **Qua tunnel**: dispatch trả nội dung file trong payload. Không sửa Kimodo,
  nhưng đụng trần 6 MB — và một chuỗi motion nhiều sample có thể vượt.
- **Kimodo ghi thẳng S3**, trả presigned URL. Không trần, browser lấy qua
  CloudFront (đã có `VvaAssetStack`). Đổi lại: sửa `mcp_server.py` + IAM ghi S3.
  Đây cũng là đường duy nhất còn đúng nếu sau này chạy nhiều task song song.

Nếu sau này cần notification MCP dạng stream dài thì kiến trúc tunnel không phục
vụ được — ghi lại làm giới hạn đã biết.

**d. Bảo mật hai lớp.** Resource policy của dispatch chỉ cho role của agent
invoke. SG của Kimodo chỉ nhận 8000 từ SG của dispatch. Không lớp nào dựa vào một
bí mật đi qua internet.

### Chi phí runtime

| | Khi demo | Khi rảnh |
|---|---|---|
| g5.xlarge | $1,006/giờ | **$0** (ASG `desired=0`) |
| kimodo-dispatch | ~$0,00002/job | **$0** |
| Cloud Map private DNS | — | ~$0,50/tháng |
| **Phải xoá thủ công** | — | **không có gì** |

g5 là toàn bộ chi phí, và nó **scale-to-zero được bằng ASG** — khác hẳn ALB.
Kèm **chốt an toàn**: EventBridge Scheduler ép `kimodo-asg` về `desired=0` lúc
23:00 mỗi ngày (`Asia/Ho_Chi_Minh`), độc lập với nút bật. Quên tắt thì mất một
buổi tối (~$8), không mất một tháng (~$734). Đây là biện pháp kiểm soát chi phí
duy nhất **không phụ thuộc vào việc ai đó nhớ**.

**Ngoài phạm vi đợt này.** `.claude/plans/kimodo-alb-endpoint.md` nay **lỗi
thời** — toàn bộ phần ALB/target group/listener bị thay bởi mục này. Đánh dấu
superseded chứ đừng xoá: phần chẩn đoán health check 406 (§3.5) và grace period
600s (§3.6) vẫn đúng và vẫn cần.

---

## 5b. Luồng thật: khi nào TTS chạy, khi nào Kimodo chạy

Owner hỏi thẳng, và khi truy code tôi tìm ra **hai lỗi mà báo cáo trước chưa
nói**. Chúng đổi cả kết luận của §5.

### Hai thứ này được kích hoạt bởi hai cơ chế KHÁC HẲN nhau

| | Kimodo (motion) | VieNeu (giọng nói) |
|---|---|---|
| Ai quyết | **LLM planner** — `needs_motion` | **Client** — tham số `output_mode` |
| Quyết ở đâu | trong graph (`planner.py:368`) | ngoài graph (`main.py:509`) |
| Cơ chế | **hard edge** của LangGraph | `if` trong hàm stream SSE |
| Chạy lúc nào | **giữa** graph, trước synthesizer | **sau** khi graph xong |
| Gọi qua | MCP tool `generate_motion` | `asyncio.create_task` + HTTP |
| Chặn câu trả lời? | **Có** — synthesizer đợi | Không — chữ đã stream xong rồi |

Nói gọn: **Kimodo là một bước trong dây chuyền; TTS là việc làm thêm sau khi đã
trả lời xong.** Kimodo hỏng thì câu trả lời chậm/thiếu; TTS hỏng thì chỉ mất
tiếng.

```
POST /v1/chat {query, output_mode, session_id}
  │
  ├─ STM warm-up ─────────────► get_stm()  (Redis local / DynamoDB cloud)
  │
  ▼  graph.astream(...)  ── SSE: stage / token ─────────────────────────►
  memory ──► planner ──┬─ needs_motion=false ─► retriever_agent ⇄ tools
                       │                              │  (pgvector, web_search)
                       └─ needs_motion=TRUE ─────► kimodo
                                                      │  MCP generate_motion
                                                      ▼
                                                 synthesizer ──► grader ──► END
  │
  ▼  graph xong, final_answer đã có, chữ đã stream hết
  if output_mode in (speech, both):
      ├─ VIENEU_TTS_URL rỗng ─► speech_disabled ─────────────────────────►
      └─ có URL ─► create_task(synthesize) ─► speech_pending ────────────►
                        │  POST {VIENEU_TTS_URL}/synthesize
                        │  ghi Redis task_result:{id}, TTL 1h
                        └─ poll Redis mỗi 250ms, tối đa 130s
                                          └─► speech_ready {url} ────────►
  ▼
  done
```

### 🔴 Lỗi 1 — Kimodo node gọi sai tên tham số, và chưa ai từng chạy nó

`nodes/kimodo.py:79` gọi `motion_tool.ainvoke({"query": resolved_query})`.

Nhưng **cả hai** MCP server đều khai tham số tên `prompt`:
- mock: `inputSchema.required = ["prompt"]` (`mcp/kimodo_server.py:56`)
- thật: `def generate_motion(prompt: str)` (`text-to-motion/kimodo/mcp_server.py:180`)

`query` không có trong schema, `prompt` bắt buộc lại thiếu ⇒ tool call fail
validation ⇒ rơi vào `except Exception` ở `kimodo.py:98` ⇒ ghi một lỗi
RECOVERABLE và đi tiếp. **Motion im lặng không bao giờ chạy.**

Vì sao lọt: `tests/langgraph_agents/test_phase3_mcp_kimodo.py` chỉ test **mock
server** trực tiếp (`_generate_motion_mock`). **`kimodo_node` không có test nào.**
Tức là cái được kiểm là thứ ở đầu kia sợi dây, không phải sợi dây.

Sửa một dòng, nhưng phải kèm test cho `kimodo_node`, nếu không khe hở y hệt mở
lại lần sau.

### 🔴 Lỗi 2 — Kimodo trả **NPZ qua HTTP của chính nó**, không phải mp4

Kế hoạch bản 3 §5c viết ".mp4 ghi lên S3". Sai hai chỗ:

1. `generate_motion` trả **JSON chứa đường dẫn file NPZ** (SMPL-X motion data để
   avatar 3D diễn lại), kèm `expires_at` — không phải video.
2. File được phục vụ bởi **route HTTP riêng của Kimodo**:
   `@mcp.custom_route("/files/{filename:path}")` (`mcp_server.py:161`), có TTL
   cleanup (`_ttl_cleanup_loop`).

**Điều này đâm thẳng vào kiến trúc tunnel ở §5.** Tunnel bỏ mọi bề mặt public của
Kimodo — nghĩa là **trình duyệt không còn đường lấy file NPZ**. Route `/files/`
sẽ chết cùng lúc với ALB.

Hai lối ra, cần Owner chọn (chi tiết ở §5c đã sửa):
- **NPZ đi qua tunnel** — đơn giản nhất, nhưng đụng trần **6 MB** của Lambda sync
  invoke. Phải đo NPZ thật trước khi chọn.
- **Kimodo tự ghi thẳng lên S3**, trả presigned URL. Không đụng trần, browser lấy
  qua CloudFront. Đổi lại: Kimodo cần quyền IAM ghi S3 và phải sửa `mcp_server.py`.

**Cả hai lỗi này đều ngoài phạm vi đợt deploy agent** — Kimodo vốn đã ngoài phạm
vi. Ghi ở đây để lần chạm vào Kimodo không phải khám phá lại.

---

## 6. TTS — hoãn, nhưng hoãn cho sạch

Owner chốt hoãn. Rủi ro nếu chỉ "không làm gì": `main.py:489` chờ TTS **130 giây**
ngay trong request `/chat`. Trên Lambda đó là 130s tiền RAM chờ một service không
tồn tại — và theo §1 rủi ro 1, **tính tiền đủ 130 giây kể cả khi user đã đóng
tab**.

Phase A thêm `VIENEU_TTS_URL`: không đặt ⇒ bỏ hẳn nhánh TTS, phát
`speech_disabled`, **không** phát `speech_pending`.

🔴 **N kiểm tra trước**: frontend xử lý thế nào khi có `speech_pending` mà không
bao giờ có `speech_ready`? Nếu UI kẹt ở "đang tạo giọng" thì đây là điều kiện
tiên quyết, không phải việc dọn sau.

Giới hạn phải ghi để không bị hiểu là hồi quy: **bản cloud đầu chat bằng chữ, có
nhớ, có tra KB — không giọng nói, không motion, không web search** (SearXNG ở
`localhost:6666`; `web_search_server.py:91` đã degrade sạch).

---

## 7. CI/CD — không phải làm từ đầu

Repo **đã có** pattern chạy được: `.github/workflows/deploy-production.yml` dùng
GitHub OIDC → `arn:aws:iam::244203483654:role/GitHubActionsECRRole` → buildx →
ECR, tag `github.sha`. Nhân bản, không thiết kế lại.

`deploy-agent.yml`, trigger `push` vào `release`, `paths: agenticRAG/**`:

1. `needs:` job unit test — `release-tests.yml` đã chạy `-m unit` cho LangGraph
   Service trên đúng branch này. Không deploy khi test đỏ.
2. Build + push `vva-agent:${{ github.sha }}`.
   🔴 **Tag bất biến, không `:latest`.** `kimodo:latest` trong
   `deploy-production.yml` là phản ví dụ: không biết đang chạy bản nào, không
   rollback được.
3. Deploy bằng `aws lambda update-function-code --image-uri` (vài giây), **không**
   `cdk deploy` (vài phút, cần synth toàn bộ, mà synth lại đòi `crud_api.zip`).
   **CDK sở hữu hình dạng hạ tầng; CI sở hữu mã.** Ranh giới phải rõ, không thì
   hai đường deploy ghi đè nhau.
4. ECR lifecycle giữ 10 image gần nhất — 1,2 GB/build, không dọn thì phình.

Quyền cần thêm cho `GitHubActionsECRRole`: `lambda:UpdateFunctionCode`. Đây là
thay đổi IAM — cần Owner duyệt.

---

## 8. Các bước

### Phase 0 — spike, làm trước mọi thứ (nửa ngày)

Docs xác nhận LWA là đường streaming chính thức cho Python, **nhưng** chỉ tài
liệu hoá đối chiếu với **Function URL**. Với Lambda-proxy STREAM, API Gateway
*"expects a specific response format"* — prelude chứa status code + headers.
**Chưa có tài liệu nào xác nhận prelude của LWA khớp với prelude API Gateway
đòi.** Đó là giả định rủi ro nhất của cả kế hoạch.

Deploy một Lambda hello-world: image + LWA + FastAPI trả SSE, sau API Gateway với
`ResponseTransferMode.STREAM`. Xác nhận token đến **nhỏ giọt**. Sai thì lùi về
HTTP proxy → Function URL (bản 1) và chịu thêm phần shared secret. Nửa ngày ở đây
tránh việc phát hiện sau khi đã dựng xong image 1,2 GB.

Kèm: xác nhận response streaming có ở us-east-1.

### Phase A — gỡ localhost (không đụng AWS)

1. `shared/embedding.py`: thêm đường ONNX sau một cờ, **giữ đường torch** để so
   sánh. Mean pooling + L2 normalize phải khớp.
2. `shared/stm.py`: interface + backend Redis (local) / DynamoDB (cloud), **kèm
   kiểm tra `expires_at` khi đọc** (§1). Sửa 4 call site: `main.py:93`,
   `session_store.py:19`, `memory.py:50`, `services/vieneu_tts/tasks.py:19`.
3. `VIENEU_TTS_URL` / `SEARXNG_URL` không đặt ⇒ tắt nhánh, không log ồn.
4. 🔴 Gate `DELETE /me` + `DELETE /sessions/{sid}/messages/{mid}` sau
   `ENABLE_GDPR_ROUTES` (default `false`). Phase B đưa `api/main.py` lên cloud;
   `create_app()` nguyên trạng sẽ **vô tình publish endpoint xoá tài khoản** mà
   Owner đã hoãn vì lý do bảo mật. Chi tiết: `docs/tracking/tech-debt.md`.
5. Test mới: không call site nào còn hardcode `redis://localhost` — cùng tinh
   thần `test_requirements_complete.py`, để khe hở không mở lại âm thầm.

**Phase A đứng độc lập**, có giá trị kể cả khi hoãn deploy.

### Phase B — image

6. `agenticRAG/Dockerfile` trên `public.ecr.aws/lambda/python:3.12`.
   🔴 LWA bằng `COPY --from=…/aws-lambda-adapter /lambda-adapter
   /opt/extensions/lambda-adapter` — **không dùng layer** (§1). Không đặt
   `AWS_LAMBDA_EXEC_WRAPPER`.
7. 🔴 **Bake model e5-small ONNX vào image.** `main.py:31-33` ép
   `HF_HUB_OFFLINE=1` nên container không tải được lúc chạy; quên bake thì INIT
   chết với lỗi nói về HuggingFace cache chứ không nói về Dockerfile.
8. `AWS_LWA_INVOKE_MODE=response_stream`, `AWS_LWA_ASYNC_INIT=true` (INIT chỉ có
   10s), `AWS_LWA_READINESS_CHECK_PATH=/health`, `PORT=8080`. Đo cold start bằng
   RIE tại chỗ.

### Phase C — CDK

9. `infra/infra/agent_stack.py` theo hình dạng `CrudApiStack`:
   `DockerImageFunction`, **1024 MB**, **timeout 120s** (§1 rủi ro 1), cùng
   `_DEFAULT_DSN_PARAM` pooled, cùng env Cognito, cùng IAM SSM + KMS scoped
   `kms:ViaService`. Thêm quyền DynamoDB cho bảng STM.
10. `rest_api_stack.py`: `/chat` bằng
    `LambdaIntegration(fn, response_transfer_mode=ResponseTransferMode.STREAM)`
    + `**authed`, và **nâng integration timeout tường minh** (mặc định 29s).
11. Warmer: thêm `/health` của agent vào lịch **có sẵn** `vva-crud-api-warmer`.
    Đừng tạo scheduler thứ hai — hai lịch lệch nhau là hai hoá đơn Neon.
12. `app.py`: dựng `AgentStack`, truyền `fn` vào `RestApiStack`.

### Phase D — CI/CD + cutover

13. `deploy-agent.yml` theo §7. Thêm `lambda:UpdateFunctionCode` vào role.
14. `api.ts`: `streamChat()` đổi `API_BASE` → `API_GATEWAY`. Xoá
    `VITE_API_BASE_URL` khỏi `api.ts`, `.env.example`, `vite.config.ts`, Amplify
    Console. Ghi chú "và later /chat" trong `apiBase.ts` — bỏ chữ *later*.
15. Cập nhật `infra/README.md` + `docs/tracking/status.md`.

---

## 9. Chi phí tổng

Giả định nói rõ để ai cũng kiểm lại được: **1.000 lượt chat/tháng, trung bình 20
giây/lượt, 200 cold start**, us-east-1.

### Thêm mới bởi kế hoạch này

| Khoản | Cách tính | $/tháng |
|---|---|---|
| vva-agent compute | 1000 × 20s × 1 GB × $0,0000166667 | 0,33 |
| Cold start | 200 × 6s × 1 GB | 0,02 |
| Warmer (dùng chung lịch sẵn có) | 2640 × 0,3s × 1 GB | 0,01 |
| Lambda requests | 3.640 × $0,20/triệu | 0,001 |
| **ECR** | 1,2 GB × **5** image giữ lại × $0,10/GB | **0,60** |
| DynamoDB STM | ~4.000 WRU + ~1.500 RRU | 0,01 |
| API Gateway REST | ~10.000 req × $3,50/triệu | 0,04 |
| CloudWatch Logs + X-Ray | ingest <1 GB; 1000 trace < 100k free | ~0,05 |
| Cloud Map private DNS (Kimodo, §5a) | private hosted zone | 0,50 |
| **Cộng** | | **~$1,56** |

**Khoản cố định lớn nhất là ECR, không phải compute** — $0,60 để giữ 5 image cho
việc rollback. Mỗi image giữ thêm là +$0,12/tháng. Đây là lý do §3 chọn ONNX cũng
đáng tiền: đường torch 4 GB sẽ là **$2,00/tháng** cho cùng 5 image.

### Đang chạy sẵn, kế hoạch này không đổi

| | $/tháng |
|---|---|
| VvaCharacterStack, VvaCrudApiStack + warmer | ~0,30 |
| VvaAssetStack — S3 ~50 MB + CloudFront (1 TB đầu miễn phí) | ~0,05 |
| VvaVpcStack — `nat_gateways=0`, không interface endpoint | **0** |
| DynamoDB `UserMappings` + `EmailLocks` | ~0,01 |
| Neon | 0 — trong ngân sách, **đang dùng 55/100 CU** |
| **Cộng** | **~$0,36** |

### Tổng

| | $/tháng |
|---|---|
| **Định kỳ, mọi thứ trên AWS** | **~$1,90** |
| Kimodo g5 — chỉ giờ thật chạy | +$1,006/giờ |

Kimodo khi rảnh là **$0** (ASG `desired=0`, không còn ALB). Chốt tự tắt 23:00 đặt
trần rủi ro cho một lần quên ở **~$8**, thay vì $734 nếu chạy cả tháng.

### 🔴 KHÔNG nằm trong bảng trên: token LLM

Đây gần như chắc chắn là **khoản lớn nhất**, và nó không phải hoá đơn AWS. Mỗi
lượt chat gọi LLM **ba lần** (`planner`, `retriever_agent`, `synthesizer` —
`graph.py`), chưa kể vòng lặp `tools` và `summarizer` chạy nền.

Đừng ước lượng — **đo**. `total_tokens` đã được theo dõi sẵn trong state và ghi
ra log ở `chat_complete` (`main.py:493-498`). Chạy 20 lượt thật, lấy trung bình,
nhân với đơn giá DeepSeek đang dùng. Việc này làm được **ngay hôm nay**, không
cần chờ deploy, và nó quyết định ngân sách thật chứ không phải $1,90 kia.

### Độ nhạy theo lưu lượng

| Lượt/tháng | AWS $/tháng | Ghi chú |
|---|---|---|
| 1.000 | ~1,90 | giả định của bảng trên |
| 10.000 | ~5,50 | vẫn rẻ hơn ALB đơn lẻ |
| 100.000 | ~39 | **App Runner/Fargate bắt đầu cạnh tranh** |

Điểm hoà vốn với App Runner (~$11/tháng) rơi vào khoảng **20-30 nghìn lượt/tháng**.
Nhưng §3 đã nói: thứ đẩy ta rời Lambda là **đồng thời**, không phải tiền.

---

## 10. Rủi ro

🔴 **Concurrency của account là 10, dùng chung với Cognito trigger của Amplify**
(mặc định AWS là **1.000**, đã verify — account này bị hạ rất sâu và **điều chỉnh
được**). CRUD giữ container ~100ms; `/chat` giữ **10-30 giây**. Ba người chat
đồng thời khoá 30% pool suốt nửa phút, và thứ bị throttle là **đăng nhập**. Burst
120 request ngày 20/08 đã tạo 61 throttle. **Xin nâng quota trước khi deploy
`/chat`.** Nâng xong thì `reserved_concurrent_executions` mới hợp lệ.

🔴 **Prelude của LWA vs API Gateway chưa được xác nhận.** Phase 0 tồn tại vì điều
này.

🔴 **Streaming tính tiền đủ thời gian dù client ngắt** ⇒ timeout 120s là biện
pháp kiểm soát chi phí, không phải lưới an toàn.

🔴 **TTL DynamoDB trễ tới ~48 giờ** ⇒ bắt buộc kiểm `expires_at` khi đọc.

🔴 **Kimodo node gọi sai tên tham số** (`query` thay vì `prompt`) và **không có
test nào cho `kimodo_node`** ⇒ motion chưa từng chạy end-to-end. §5b lỗi 1.

🔴 **Kimodo trả NPZ qua route HTTP của chính nó**, mà tunnel sẽ cắt mất ⇒ phải
chọn đường cho file trước khi bỏ ALB. §5b lỗi 2.

🟠 **ONNX pooling sai thì im lặng.** §11 nghiệm thu bằng số.

🟠 **Throttle 5 rps stage-wide** chọn cho catalog; `/chat` khác hình dạng hẳn.

🟠 **Không có WAF inspection với nội dung stream**, không cache, không VTL.

🟡 **Cold start ngoài giờ hành chính ~5-8s** (tốt hơn ~20s của đường torch).

🟡 **Build image cần Docker trên máy N** — `CharacterStack` đã cần rồi.

---

## 11. Nghiệm thu

**ONNX (chặn Phase B):**
```powershell
python scripts/verify_onnx_parity.py --sample 200
```
Đạt khi `cosine(torch_vec, onnx_vec) > 0.9999` trên 200 mẫu lấy từ 2918 rows KB,
**và** top-5 của `kb_search` trùng thứ tự cho 20 truy vấn thật.

**STM hết hạn (chặn Phase C):** ghi một item `expires_at` ở quá khứ, đọc lại —
phải coi như không tồn tại, **không** chờ TTL dọn.

**Phase A (local, không cần AWS):**
```powershell
docker stop vva-redis
# firstconda, không phải python trần — status.md mục 0
uvicorn langgraph_agents.api.main:create_app --factory --port 8000
curl -N -X POST localhost:8000/chat -H "Authorization: Bearer $(scripts/dev_token.ps1)" `
     -H "Content-Type: application/json" `
     -d '{"query":"đau lưng dưới nên tập gì","session_id":"...","output_mode":"text"}'
curl -X DELETE localhost:8000/me -H "Authorization: Bearer ..."   # kỳ vọng 404
```
Đạt khi: token ra dần, log `stm_unavailable` đúng **một** lần, không có
`speech_pending` treo, `pytest -m unit` xanh (baseline 331 passed).

**Phase C:**
```bash
aws logs tail /aws/lambda/vva-agent --since 10m | grep INIT_DURATION
curl -N "$REST_API_URL/chat" -H "Authorization: Bearer $ID_TOKEN" ...
```
Đạt khi token đến **nhỏ giọt** chứ không đổ một cục — một cục nghĩa là
`ResponseTransferMode` chưa ăn, và nó trông giống hệt "chỉ hơi chậm".

**Phase D:** build frontend **không** có `VITE_API_BASE_URL`; chat trên bản
deploy; DevTools → Network chỉ thấy một origin.
