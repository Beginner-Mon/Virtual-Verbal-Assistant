# VVA — Re-Architecture Plan (Phase 6.10 + Phase 7)

> Architect: K | Date: 2026-05-28
> Audience: N (Developer), T (Reviewer), Owner
> Scope: Phase 6.10 = pre-deploy hardening (code changes only, no new infra). Phase 7 = hybrid cloud deployment.

---

## Context

Tài liệu này tổng hợp các quyết định kiến trúc sau session debug ngày 26/05/2026 (worklog `docs/worklogs/26-05-2026.md`) và thảo luận ngày 28/05/2026. Mục đích chính là giải thích **tại sao** cho từng quyết định để team có thể review và phản biện — không phải spec code chi tiết (spec riêng sẽ có từng Phase).

---

## Phase 6.10 — Pre-deploy Hardening (hiện tại, trước Phase 7)

### 6.10.1 Database: Normalize Messages Table

**Vấn đề hiện tại**

`conversations.messages` là một cột JSONB lưu toàn bộ lịch sử hội thoại dưới dạng JSON array. Mỗi lần có tin nhắn mới, PostgreSQL phải thực hiện `messages || new_messages::jsonb` — server-side append nhưng vẫn là O(n) vì DB phải decode blob cũ, concat, encode lại toàn bộ. Với session 50-100 turns thì TOAST overhead tăng dần theo số lần ghi. Ngoài ra, không thể paginate lịch sử chat mà không pull toàn bộ blob về.

**Giải pháp: Tách bảng `messages`**

```
conversations (session header)
  session_id  UUID PK
  user_id     UUID FK → users(id)
  created_at  TIMESTAMPTZ
  updated_at  TIMESTAMPTZ

messages (one row per message)
  id          UUID PK
  session_id  UUID FK → conversations(session_id)
  role        TEXT  CHECK (role IN ('user', 'assistant'))
  content     TEXT
  metadata    JSONB   -- intent, tokens, v.v.
  created_at  TIMESTAMPTZ

INDEX messages(session_id, created_at)  -- covering index cho pagination
```

Mỗi lần ghi: 1 upsert vào `conversations` (update `updated_at`) + 2 INSERT vào `messages` (user + assistant). Tất cả O(1), không đụng data cũ.

**Cursor-based Pagination**

Load lịch sử chat dùng cursor (timestamp hoặc UUID) thay vì `OFFSET/LIMIT`. Lý do: `OFFSET n` yêu cầu DB scan qua n rows đầu tiên trước khi trả kết quả — chậm dần khi session dài. Cursor-based luôn O(1) vì dùng index:

```sql
-- Load 20 tin nhắn trước cursor (scroll up)
SELECT * FROM messages
WHERE session_id = $1
  AND created_at < $2   -- cursor = created_at của tin nhắn cũ nhất đang hiển thị
ORDER BY created_at DESC
LIMIT 20
```

UI load 20 tin nhắn gần nhất lúc đầu. User scroll lên → gọi tiếp với cursor = `created_at` của tin nhắn cũ nhất đang có → load 20 tin tiếp. Giống pattern của mọi chat app lớn (Slack, Messenger).

**JSONB cho User Profile và Document Metadata**

Hai trường hợp dùng JSONB với lý do khác nhau:

- **User profile**: Các field thay đổi theo thời gian (`age`, `injury_history`, sau này có thể thêm `medication`, `fitness_level`). Dùng cột cố định thì mỗi lần thêm field phải `ALTER TABLE` + migration script. JSONB cho phép thêm field không cần đụng schema. GIN index cho phép query `WHERE profile @> '{"has_injury": true}'` hiệu quả.

- **Document metadata**: Mỗi loại source có metadata khác nhau — research paper có `{doi, author, year}`, video transcript có `{youtube_id, timestamp}`, exercise guide có `{difficulty, equipment}`. Fixed columns sẽ tạo ra hàng loạt `NULL`. JSONB giải quyết heterogeneous schema tự nhiên.

Nguyên tắc chung: field nào **luôn có và query thường xuyên** (`session_id`, `role`, `created_at`) → cột riêng với type chuẩn và B-tree index. Field nào **tùy biến hoặc thay đổi theo thời gian** → JSONB.

**Short-Term Memory (STM)**

Hiện tại STM cứng 3 Q&A pairs trong Redis. Phần này cần thảo luận thêm trước khi thay đổi. "Quản lý động" là hướng đúng về mặt lý thuyết nhưng cần define rõ: số lượng pairs thay đổi theo intent? Theo độ phức tạp câu hỏi? Hay theo token budget? Giữ nguyên 3 pairs cho đến khi có use case cụ thể.

**Thứ tự thực hiện**

```
1. Viết migration SQL: tạo messages table + index
2. Viết migration script: backfill data từ conversations.messages JSONB → messages rows
3. Sửa session_store.py: write_session_turn, load_session_messages, list_user_sessions
4. Test: pytest -m integration
5. Sau khi test xanh: DROP COLUMN messages từ conversations
```

Note: codebase dùng raw `asyncpg`, không có SQLAlchemy. Không cần sửa ORM model.

---

### 6.10.2 Pre-deploy Checklist (bắt buộc trước Phase 7)

Các item này đã được document trong `ARCHITECTURE-FULL-FLOW-PREDEPLOY.md` nhưng chưa implement:

| Item | File cần sửa | Lý do bắt buộc |
|------|-------------|----------------|
| Lock CORS origins | `api/main.py` | Hiện `allow_origins=["*"]` — bất kỳ domain nào cũng gọi được `/chat`. Security risk rõ ràng trước khi expose ra internet. |
| Log file rotation | `shared/logging.py` hoặc config | Không có rotation → log file phình vô hạn trên server. |
| TTS audio cleanup cron | Script mới | VieNeu ghi file audio ra disk, không có cleanup. Server disk đầy theo thời gian. |
| SpeechLLm + SearXNG health checks | `api/health.py` | `/health/detailed` hiện không check 2 services này → ops blind spot khi chúng down. |

---

### 6.10.3 Stop Generation (SSE disconnect detection)

Hiện tại nếu user đóng tab hoặc muốn dừng LLM giữa chừng, graph vẫn chạy hết. Cần thêm `await request.is_disconnected()` check vào vòng lặp `_stream_chat`. Khi client disconnect → break loop → graph bị cancel qua LangGraph cancellation.

---

## Phase 7 — Hybrid Cloud Deployment

> Các mục dưới đây là định hướng kiến trúc. Spec chi tiết và action items sẽ được viết thành từng PHASE-7.x.md riêng khi team bắt đầu Phase 7.

### 7.1 Quyết định giữ SSE, không chuyển WebSocket

**Câu hỏi được đặt ra**: Có nên đổi từ SSE sang WebSocket không, vì sau này có voice command và stop generation?

**Quyết định: Giữ SSE.** Lý do:

**Stop generation** không cần WebSocket. Client đóng `EventSource` → server detect qua `request.is_disconnected()` → cancel graph. Bidirectional channel không cần thiết cho usecase này.

**Voice command**: Câu hỏi then chốt là push-to-talk hay continuous streaming?

- *Push-to-talk* (user nhấn mic, nói, thả ra): Audio được record thành file → POST lên SpeechLLm STT → nhận text → POST `/chat` như bình thường. SSE hoàn toàn đủ.
- *Continuous streaming* (user nói liên tục, server STT real-time): Cần bidirectional → WebSocket phù hợp hơn.

Với PT assistant (healthcare domain), **push-to-talk hợp lý hơn** vì user cần thời gian diễn đạt câu hỏi y tế rõ ràng. Continuous streaming dễ bị noise, phức tạp hơn đáng kể, và không align với usecase thực tế.

**Chi phí của việc đổi sang WebSocket**:
- Load balancer phải support sticky session (hoặc Redis pub/sub fan-out)
- `EventSource` auto-reconnect mất → phải tự viết reconnect logic ở client
- Phase 7 CloudFront: WebSocket qua CDN phức tạp hơn SSE nhiều
- Phải rewrite `api/sse.py`, `ECA_UI/api.js`, và toàn bộ SSE test suite

**Kết luận**: Nếu Phase 7+ có use case conversational voice real-time (user nói, AI ngắt lời, back-and-forth), lúc đó đủ lý do đánh đổi. Có thể build endpoint `/voice` WebSocket riêng mà không đụng `/chat` SSE hiện tại. Không cần quyết định ngay.

---

### 7.2 Phân tách Compute (Service Decomposition)

**Vấn đề**: Các API nhẹ (CRUD sessions) chạy chung container với LangGraph agent. LangGraph container cần GPU-accessible memory, Python deps nặng (langgraph, langchain-openai, asyncpg, v.v.). Chạy một `GET /sessions` đơn giản trên container đó là lãng phí tài nguyên.

**Lưu ý thuật ngữ**: Đây là *service decomposition by compute weight*, không phải CQRS theo nghĩa nghiêm túc. CQRS (Command Query Responsibility Segregation) là pattern tách command model vs query model ở data layer — phức tạp hơn nhiều và không cần thiết ở đây.

**Đề xuất phân tách**:

```
┌─────────────────────────────────────────────────────┐
│  Client (Browser)                                   │
└──────────────┬──────────────────────────────────────┘
               │
    ┌──────────▼──────────┐
    │  AWS API Gateway    │  ← CRUD endpoints (serverless)
    │  + Lambda           │    GET /sessions
    │  (pay-per-request)  │    DELETE /sessions
    └─────────────────────┘    POST /sessions/messages (fetch history)

    ┌──────────────────────┐
    │  Application Load    │  ← Heavy endpoints (containerized)
    │  Balancer (ALB)      │    POST /chat (LangGraph agent)
    │  → ECS/EKS container │    GET /health
    └──────────────────────┘
```

**Tại sao CRUD endpoints lên Lambda**:
- Pay-per-request: gần free ở MVP scale
- Auto-scale về 0 khi không có traffic
- Không cần maintain container cho logic đơn giản

**Tại sao `/chat` không qua API Gateway**:
- API Gateway có timeout cứng 29 giây. LangGraph agent chạy 30-60s cho knowledge queries → timeout trước khi xong
- ALB không có timeout giới hạn này, routing trực tiếp đến container

**Prerequisite**: Lambda cần kết nối RDS → phải đặt Lambda trong cùng VPC → cần **RDS Proxy** để tránh connection exhaustion (Lambda stateless = new DB connection mỗi invocation). Không khó nhưng cần setup.

---

### 7.3 TTS: asyncio.create_task → SQS + Worker

**Hiện tại**: `asyncio.create_task` với strong ref (đã fix 26/05). Đủ tốt cho single-server, <100 req/min.

**Vấn đề khi scale**: Khi deploy nhiều ECS task instances, TTS task chạy in-process trên instance nào thì kết quả Redis chỉ có ý nghĩa với request đó. Nếu load balancer route `/tts/{id}/result` poll sang instance khác → 404. Phải dùng sticky session hoặc centralized queue.

**Giải pháp Phase 7: SQS + Worker**

```
Agent (ECS) → SQS queue → TTS Worker (EC2/ECS, pull jobs) → Redis task_result
```

**Tại sao SQS thay vì Kafka hoặc EventBridge**:

*Kafka*: Thiết kế cho throughput cực cao (millions msg/sec), event replay, event sourcing. MVP này TTS queue có thể vài chục request/giờ. Kafka yêu cầu quản lý cluster (hoặc trả tiền Confluent/MSK đắt hơn SQS nhiều lần). Overkill hoàn toàn.

*EventBridge*: Event router/bus — dùng để trigger Lambda khi S3 có file mới, fan-out event sang nhiều service. Không phải task queue — không có visibility timeout, không có Dead Letter Queue tích hợp, không có backpressure. Không phù hợp cho pattern "agent bắn job → worker pick up → retry nếu fail".

*SQS* là đúng tool vì:
- **Visibility timeout**: Worker đang xử lý → message ẩn với worker khác → không bị double-process
- **Dead Letter Queue (DLQ)**: TTS fail 3 lần → tự chuyển sang DLQ → ops có thể debug
- **Managed, pay-per-message**: Không quản lý broker, gần free ở MVP scale
- **Native integration với ECS**: Worker poll SQS bằng boto3, AWS setup sẵn

`celery_app.py` skeleton đã giữ trong codebase từ v2.4.1 chính xác cho usecase này. Phase 7 reactivate, thay broker từ Redis → SQS.

---

### 7.4 Infrastructure as Code — AWS CDK

**Tất cả hạ tầng Phase 7 phải được quản lý qua CDK, không thao tác thủ công trên AWS Console (click-ops).**

Lý do:
- **Reproducibility**: Dev/Staging/Prod đều từ cùng code → không có "works in staging, broken in prod" do config drift
- **Version control**: Infra thay đổi có PR, review, rollback như code
- **`cdk diff`**: Xem chính xác những gì sẽ thay đổi trước khi deploy — tương đương `terraform plan`

**Tại sao CDK thay vì alternatives**:

*Terraform*: Multi-cloud, nhưng cú pháp HCL là một ngôn ngữ riêng cần học. Project đã quyết định AWS-first → CDK native hơn, không cần abstraction layer thêm.

*CloudFormation raw YAML*: CDK compile ra CloudFormation cuối cùng, nhưng viết YAML tay cho VPC + ECS + ALB + SQS + RDS dài, verbose, và error-prone. CDK high-level constructs có security defaults baked in (ví dụ: `ApplicationLoadBalancedFargateService` tự tạo VPC, security group, IAM role đúng minimal-privilege).

**Scope CDK Phase 7**:
```
infra/
  lib/
    vpc-stack.ts         -- VPC, subnets, security groups
    database-stack.ts    -- RDS PostgreSQL + pgvector, RDS Proxy
    cache-stack.ts       -- ElastiCache Redis
    agent-stack.ts       -- ECS Fargate (LangGraph agent container)
    lambda-stack.ts      -- Lambda functions (CRUD endpoints)
    queue-stack.ts       -- SQS queues (TTS jobs, DLQ)
    cdn-stack.ts         -- CloudFront + ALB
  bin/
    app.ts               -- Stack entry point, env params
```

**Tạo `infra/` folder trong repo hiện tại** (không tạo repo riêng) để CDK code và application code versioned cùng nhau.

---

## Tóm tắt Action Items

### Phase 6.10 (làm ngay, trước Phase 7)

| # | Task | File(s) |
|---|------|---------|
| 1 | Migration SQL: tạo `messages` table | `db/migrations/001_normalize_messages.sql` |
| 2 | Migration script: backfill JSONB → rows | `db/migrations/migrate_messages.py` |
| 3 | Sửa session_store.py (4 hàm) | `db/session_store.py` |
| 4 | Cursor-based pagination cho load history | `db/session_store.py`, `api/main.py` |
| 5 | Lock CORS origins | `api/main.py` |
| 6 | Log file rotation | `shared/logging.py` |
| 7 | TTS audio cleanup cron | script mới |
| 8 | SpeechLLm + SearXNG health checks | `api/health.py` |
| 9 | Stop generation: disconnect detection | `api/main.py` |

### Phase 7 (sau 6.10 xong, infra setup riêng)

| # | Task |
|---|------|
| 1 | Init `infra/` CDK project (TypeScript) |
| 2 | VPC + RDS + ElastiCache stacks |
| 3 | ECS Fargate stack cho LangGraph agent |
| 4 | Lambda stack cho CRUD endpoints |
| 5 | SQS + TTS worker (reactivate `celery_app.py` với SQS broker) |
| 6 | CloudFront + ALB stack |
| 7 | DNS + SSL |

---

## Open Questions (cần Owner quyết định trước Phase 7)

1. **Voice feature**: Push-to-talk hay continuous streaming? (Ảnh hưởng đến quyết định SSE vs WebSocket)
2. **LLM provider Phase 7**: Giữ DeepSeek hay thêm Claude/Gemini fallback? (Ảnh hưởng đến latency budget và cost)
3. **Database host**: Supabase managed hay self-hosted RDS? (Supabase dễ setup hơn, RDS linh hoạt hơn về pgvector version)
4. **STM dynamic sizing**: Nếu muốn thay đổi số Q&A pairs trong STM, cần define use case cụ thể trước khi implement
