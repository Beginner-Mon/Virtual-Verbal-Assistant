---
date: 2026-08-16
tags: [ops, neon, postgres, migration, latency]
author: K
to: Tri, N
---

# Chuyển Neon sang `us-east-1` — báo cáo trước khi làm

## Kết luận ngắn

**Đích đến đúng, nhưng làm bây giờ là lỗ.** Số đo trên máy N, hôm nay:

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

**Đáng làm khi**: backend LangGraph được deploy lên `us-east-1`. Lúc đó
DB cùng region là ~1-2 ms/query thay vì 53 ms, tiết kiệm ~600 ms/lượt — và
`ap-southeast-1` mới là bên chịu phạt.

Nếu vẫn muốn làm trước để "khỏi phải làm hai lần": được, nhưng phải biết là đổi
lấy **+2,5 s mỗi lượt** trong suốt thời gian dev còn ở local. Đề xuất: làm cùng
lúc với việc deploy backend, không làm riêng.

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

1. **Tri** — Neon console → New project → region **AWS us-east-1**, Postgres cùng
   phiên bản với project cũ. Bật extension không cần làm tay: migration `002` có
   `CREATE EXTENSION IF NOT EXISTS vector`.

2. **Tri** — copy **DIRECT connection string**, không phải `-pooler`.
   Pooler không hỗ trợ prepared statement mà asyncpg dùng. Chuỗi phải có
   `?sslmode=require`.

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
