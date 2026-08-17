---
date: 2026-08-16
tags: [ops, neon, postgres, migration, latency]
author: K
to: Tri, N
---

# Chuyển Neon sang `us-east-1` — báo cáo trước khi làm

## Kết luận ngắn

**Làm đi.** N đã chốt: giai đoạn này là lên cloud, backend trên máy chỉ còn để
test. `us-east-1` khớp với chỗ Cognito và Amplify đang nằm.

Phần đo bên dưới **không phải lý do để dừng** — nó ở đây để khi lượt chat local
chậm hẳn đi, không ai mất một buổi đi tìm "bug hiệu năng" mà thật ra là địa lý.

Số đo trên máy N, hôm nay:

| Đích | TCP connect (trung vị) |
| --- | --- |
| Neon `ap-southeast-1` (hiện tại) | **59 ms** |
| AWS `us-east-1` | **326 ms** (min 259) |
| AWS `ap-southeast-1` | 56 ms |

Neon `SELECT 1` với pool ấm: **53,4 ms**.

Một lượt chat tốn **12 query tuần tự** (đo bằng `/debug/pgstats`, không phải ước
lượng). Chuyển sang `us-east-1`:

```
12 × (260 − 53) ms ≈ +2,5 giây mỗi lượt chat
```

**Cộng vào mọi lượt dev trên máy, ngay lập tức.**

Lý do: thứ đang nói chuyện với Neon là **backend chạy trên laptop N ở Việt Nam**.
Trên AWS `us-east-1` hiện chỉ có frontend (Amplify) và mấy lambda auth — mà lambda
auth dùng **DynamoDB + Cognito, không đụng Neon**. Nên hôm nay **không có gì ở
us-east-1 gọi Neon cả**; đổi region chỉ kéo dài đường dây từ Việt Nam.

Khi backend LangGraph lên `us-east-1`, DB cùng region còn ~1-2 ms/query thay vì
53 ms — **tiết kiệm ~600 ms mỗi lượt** so với ap-southeast-1 hiện tại. Đó là thứ
đang mua. Cái giá là +2,5 s/lượt trên máy local, tự hết khi backend lên cloud.

**Việc còn lại quan trọng hơn migration này**: backend chưa được deploy. Chừng
nào chưa, không có gì hưởng lợi từ region mới — chỉ có local chịu phạt. Nếu hai
việc cách nhau xa, cân nhắc làm migration ngay trước lúc deploy backend.

---

## Neon không "chuyển region" được

Region cố định lúc tạo project. Đây là **tạo project mới rồi dựng lại**, không
phải di chuyển. Project cũ nên giữ tới khi cái mới chạy ổn.

---

## Dữ liệu đang có (đo hôm nay, không phải nhớ)

| Bảng | Rows | Mất thì sao |
| --- | ---: | --- |
| `documents` | 2 918 | Dựng lại từ `data/knowledge_base/documents.txt` (**có trong git**) |
| `kb_embeddings` | 2 918 | Dựng lại cùng lệnh trên, ~9 phút |
| `users` | 8 | Mock |
| `conversations` | 14 | Mock |
| `messages` | 48 | Mock |
| `summaries` | 0 | — |
| `user_memory` | 0 | — |

`pgvector` **0.8.0**.

**Không cần dump/restore.** Toàn bộ dữ liệu thật (KB) sinh lại được từ file trong
repo; phần còn lại là mock. Đây là lý do việc này rẻ — và cũng là lý do nó sẽ
**không** còn rẻ sau khi có người dùng thật.

Coi như **mất sạch** là đúng tinh thần: 8 user, 14 hội thoại, 48 tin nhắn biến
mất. Không tiếc, nhưng nói ra để không ai chờ chúng xuất hiện lại.

**Không nằm trong Postgres, nên không bị ảnh hưởng:** Redis (STM + Celery), file
motion/audio, model embedding cache — tất cả vẫn ở local.

### ⚠️ Neon hiện tại đang chậm hơn `head` một migration

`alembic heads` = `004_demo_billing`, nhưng bảng `billing_accounts` **không tồn
tại** trên DB hiện tại ⇒ nó đang ở `003_kb_hnsw`.

DB mới sẽ được dựng thẳng ở `head`, tức **có thêm bảng billing mà DB cũ không
có**. Nếu vì lý do gì phải trỏ ngược về DB cũ, nhớ chạy `alembic upgrade head`
trước.

---

## Tri cần gửi N những gì

**Không file nào.** Mọi thứ để dựng lại DB đều đã ở trong repo:

| Thứ | Ở đâu |
| --- | --- |
| Schema | `agenticRAG/langgraph_agents/alembic/versions/` (3 revision) |
| KB corpus | `data/knowledge_base/documents.txt` (1,1 MB, đã commit) |
| Script nạp | `scripts/ingest_kb_pgvector.py` |

**Thứ duy nhất cần trao tay: DSN mới.** Gửi qua trình quản lý mật khẩu, **không
qua chat** — DSN Neon chứa mật khẩu trong URL.

> DSN hiện tại đã từng đi qua cửa sổ chat. Nó phải được **đổi mật khẩu trước khi
> có dữ liệu người dùng thật** — mục này đã nằm trong `status.md` và vẫn chưa làm.
> Tạo project mới là dịp gọn nhất để khép lại nó.

---

## Các bước

Ai làm gì: **Tri** tạo project + gửi DSN. **N** chạy migration + ingest (máy N đã
có sẵn model embedding cache).

1. **Tri** — Neon console → New project → region **AWS us-east-1**,
   **Postgres 17** (project cũ đang chạy **17.10**).

   Không cần đặt gì thêm:
   - **Extension `vector`**: migration `002` có `CREATE EXTENSION IF NOT EXISTS vector`.
     Project cũ đang dùng pgvector **0.8.0**; Neon cấp bản mới nhất nó hỗ trợ, không
     cần khớp chính xác.
   - **Tên database / role**: để mặc định (`neondb` / `neondb_owner`). Không có gì
     trong code hardcode tên đó — tất cả đọc từ DSN.
   - **Branch**: dùng `main` mặc định. Repo không dùng tính năng branching của Neon.

2. **Tri** — copy **DIRECT connection string**, không phải `-pooler`.
   Pooler không hỗ trợ prepared statement mà asyncpg dùng. Chuỗi phải có
   `?sslmode=require`.

   > Gửi **nguyên chuỗi**, không cần tách. N chỉ dán vào một chỗ duy nhất.
   > **Không** đặt DSN vào `config/langgraph.yaml` — file đó **được commit**,
   > dòng `dsn:` trong đó là fallback local (`vva:vva_dev@localhost:5433`) và
   > phải giữ nguyên như vậy.

3. **N** — sửa `agenticRAG/.env`:
   ```
   VVA_PG_DSN=postgresql://...@ep-xxx.us-east-1.aws.neon.tech/...?sslmode=require
   ```
   Giữ lại dòng cũ dưới dạng comment để lùi được.

4. **N** — dựng schema:
   ```bash
   cd agenticRAG/langgraph_agents && alembic upgrade head && cd ../..
   ```
   Phải đứng đúng thư mục đó — `script_location` trong `alembic.ini` là đường dẫn
   tương đối theo CWD.

5. **N** — **tắt backend**, rồi nạp KB:
   ```bash
   python scripts/ingest_kb_pgvector.py --reset
   ```
   Script tự từ chối chạy khi thấy `:8000` còn sống (exit 2). ~9 phút.

6. **N** — bật backend, kiểm:
   ```bash
   curl -s localhost:8000/health/detailed | grep -o '"postgres":[^}]*}'
   ```

---

## Chỗ dễ sập

| Bẫy | Triệu chứng | Cách tránh |
| --- | --- | --- |
| Dùng endpoint `-pooler` | Lỗi prepared statement lúc chạy, **không** lỗi lúc kết nối | Lấy đúng "Direct connection" |
| Quên `alembic upgrade head` | `relation "documents" does not exist` | Bước 4 |
| Backend đang chạy lúc ingest | Trước đây: segfault exit 139 **sau khi `--reset` đã xoá bảng** ⇒ KB rỗng | Script đã chặn, đừng dùng `--force` |
| Máy chưa có model embedding | Ingest chết vì `HF_HUB_OFFLINE=1` đặt sẵn đầu script | Chạy `HF_HUB_OFFLINE=0 python scripts/ingest_kb_pgvector.py --reset` lần đầu |
| Đổi model embedding giữa chừng | Recall tụt **âm thầm**, không lỗi | Vẫn `intfloat/multilingual-e5-small`, 384 chiều. Prefix `query:`/`passage:` là **bắt buộc** — cả hai phía phải đi qua `shared/embedding.py` |
| Sửa `.env` mà quên restart | Backend giữ DSN cũ | `shared/env.py` đọc lúc khởi động |
| Không có `VVA_PG_DSN` | Backend **vẫn chạy**, âm thầm rơi về Postgres local | Log giờ có cảnh báo `falling back to the LOCAL database` — đọc nó |

---

## Knowledge base — phần duy nhất là dữ liệu thật

Đây là thứ duy nhất mất đi mà **phải dựng lại**, không phải chấp nhận mất.

| | |
| --- | --- |
| Nguồn | `data/knowledge_base/documents.txt`, 1,1 MB, **đã commit** |
| Số bản ghi | 2 918 bài tập, phân cách bằng dòng `---` |
| Model | `intfloat/multilingual-e5-small`, **384 chiều** |
| Prefix | `query:` / `passage:` — **bắt buộc**, cả hai phía đi qua `shared/embedding.py` |
| Bảng | `documents` (metadata) + `kb_embeddings` (vector), FK CASCADE |

### ⏱ Ingest sang us-east-1 sẽ lâu hơn nhiều — đừng tưởng nó treo

Script insert **tuần tự 2 câu lệnh mỗi bản ghi** (một `INSERT ... RETURNING id`,
một insert embedding) = **5 836 round-trip**.

| Đích | Ước tính |
| --- | --- |
| ap-southeast-1 (đo thật) | ~9 phút |
| **us-east-1** | `5836 × 0,26 s ≈ **25 phút**` chỉ riêng phần đi-về mạng, cộng thời gian embed ⇒ **~30 phút** |

Chạy thử 50 bản ghi trước cho chắc, rồi mới chạy full:

```bash
python scripts/ingest_kb_pgvector.py --reset --limit 50
```

> Có thể rút xuống còn vài giây bằng cách gộp insert (`executemany` hoặc `COPY`).
> Chưa làm — nêu ra để không ai tưởng 30 phút là chuyện bình thường.

### Nghiệm thu KB: đếm dòng KHÔNG đủ

2918 dòng vẫn có thể là 2918 vector rác (sai model, thiếu prefix, sai chiều).
Ba thứ phải kiểm:

```sql
-- 1. Đúng chiều
SELECT vector_dims(embedding) FROM kb_embeddings LIMIT 1;        -- phải = 384

-- 2. Index HNSW có thật (thiếu nó thì kb_search seq scan — vẫn chạy, chỉ chậm,
--    và đã từng bị sót đúng ở bảng này trong migration 002)
SELECT indexname FROM pg_indexes WHERE tablename = 'kb_embeddings';
--    phải có idx_kb_emb_embedding

-- 3. Metadata sinh đúng
SELECT count(*) FROM documents WHERE metadata->>'has_description' = 'true';
--    phải = 1368
```

Rồi hỏi thật một câu qua `/chat` — phải ra `mode: synthesize` có trích
`exercise_db`. Ra `refuse` nghĩa là KB rỗng hoặc recall hỏng.

> **1 368 / 2 918** bản ghi có description thật; phần còn lại `Description: nan`
> lọt từ export pandas, chỉ index được structured field. Con số này **giữ nguyên
> sau khi dựng lại** — nếu ra khác là ingest có vấn đề, không phải corpus.
>
> Riêng chuyện corpus này là **gym/fitness chứ không phải PT lâm sàng** thì vẫn là
> câu hỏi mở cho Owner, không liên quan tới việc chuyển vùng.

## Những thứ KHÔNG phải đụng tới

Hỏi trước cho khỏi phải hỏi:

| | Có phải đổi không |
| --- | --- |
| Amplify Console (env vars, secrets) | **Không.** Lambda auth dùng DynamoDB + Cognito, không hề gọi Postgres |
| `amplify/backend.ts`, bảng DynamoDB | **Không** |
| Redis | **Không.** Vẫn ở local, không liên quan region Neon |
| `config/langgraph.yaml` | **Không.** Dòng `dsn:` là fallback local, giữ nguyên |
| `pool_min` / `pool_max` (2/10) | **Không** cần đổi. Neon free tier chịu được. Nếu sau này thấy lỗi hết connection thì mới hạ |
| Code ứng dụng | **Không một dòng nào.** Chỉ đổi một biến môi trường |

## Xoá project cũ khi nào

**Không xoá cùng ngày.** Giữ tới khi:

1. `/health/detailed` → `postgres ok`
2. Một câu hỏi thật qua `/chat` trả `mode: synthesize` có trích `exercise_db`
3. Chạy được ít nhất một buổi dev không sự cố

Rồi mới xoá — **Tri xoá**, vì Tri là người tạo. Neon tính tiền theo storage nên
để thêm vài ngày gần như miễn phí.

## Nghiệm thu

```
documents        2918
kb_embeddings    2918
pgvector         0.8.0
alembic current  004_demo_billing (head)
```

Và một câu hỏi thật qua `/chat` phải trả lời `mode: synthesize` có trích nguồn
`exercise_db` — nếu ra `refuse` thì KB rỗng.

Đo lại `SELECT 1` sau khi chuyển. Nếu vẫn đang dev ở local, con số đó sẽ là
~260 ms thay vì 53 ms. **Đó là kết quả đúng như dự đoán, không phải lỗi cấu hình.**
