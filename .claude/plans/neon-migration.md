# Plan: Chuyển PostgreSQL lên Neon

**Status:** v4.0 — **ĐÃ CUTOVER 31/07**. Backend `:8000` chạy trên Neon, dữ liệu ghi vào Neon
(đã đối chứng với local). Chi phí DB thật: **12 query / 1,4 s / lượt (4,8%)** — xem §3.8.
**Ngày:** 31/07/2026 — K
**Người implement:** N (hoặc K nếu Owner giao)
**Effort ước tính:** ~4h (không kể thời gian chờ provision)
**Bối cảnh:** Owner + bạn Owner chốt hướng Neon (31/07). Phân tích so sánh Neon/Supabase và 3 rủi ro
đã ghi ở [tech-debt.md](../../docs/tracking/tech-debt.md) mục 🔴 Critical.

---

## 0. Phạm vi — đọc trước, tránh hiểu nhầm

Plan này **chỉ chuyển PostgreSQL**. Dữ liệu VVA nằm ở 3 kho:

| Kho | Trạng thái sau plan này |
|---|---|
| **PostgreSQL + pgvector** (8 bảng) | ✅ lên Neon |
| **Redis** — STM session, circuit breaker, Celery broker, TTS task | ❌ **vẫn ở local**. Neon không chạy Redis. Cần plan riêng (Upstash / Redis Cloud) |
| **File hệ thống** — motion BVH/NPZ, sắp có audio TTS | ❌ **vẫn ở local**. Cần plan riêng (R2/S3) |

> Nói "dời toàn bộ DB lên Neon" là chưa đủ. Sau plan này hệ thống **vẫn còn phụ thuộc máy local**
> ở 2 chỗ trên. Đừng coi là đã xong việc thoát khỏi máy cá nhân.

---

## 1. Số liệu thật (đo 31/07, DB đang chạy)

```
Extensions:  plpgsql 1.0 · vector 0.8.2        ← không có extension lạ nào
Tổng dung lượng DB:  21 MB
```

| Bảng | Rows | Size |
|---|---:|---:|
| `kb_embeddings` | 2 918 | 12 MB |
| `documents` | 2 918 | 984 kB |
| `messages` | 12 | 176 kB |
| `conversations` | 3 | 80 kB |
| `users` | 1 | 80 kB |
| `summaries` | 0 | 184 kB |
| `user_memory` | 0 | 48 kB |

**Kết luận quan trọng**: đây **không phải** một cuộc di trú dữ liệu. 21 MB, và dữ liệu người dùng
thật gần như bằng 0 (1 user, 3 hội thoại, 12 tin nhắn — toàn dữ liệu dev). Phần lớn khối lượng là
**knowledge base sinh lại được bằng một lệnh**. Rủi ro nằm ở **cấu hình kết nối**, không ở dữ liệu.

Index phải có sau khi chuyển (2 HNSW + 7 btree/unique):

```
idx_kb_emb_embedding      hnsw (embedding vector_cosine_ops)
idx_summaries_embedding   hnsw (embedding vector_cosine_ops)
idx_conversations_user_updated · idx_kb_emb_document · idx_messages_session_seq
idx_summaries_session · idx_user_memory_user (partial WHERE valid = true)
+ các unique: uq_chunk, users_auth_provider_auth_subject_key, …
```

> Bài học 30/07: migration 002 **bỏ sót** `idx_kb_emb_embedding` và không ai phát hiện vì bảng đang
> rỗng — mọi `kb_search` chạy seq scan. Sau khi chuyển Neon **phải đếm lại index**, không tin là
> "alembic chạy xong thì chắc đủ".

---

## 2. 🔴 Blocker #0 — phải sửa TRƯỚC, nếu không sẽ split-brain

**Runtime KHÔNG đọc `VVA_PG_DSN`; chỉ Alembic đọc.**

| Đường | Nguồn DSN |
|---|---|
| Alembic (`alembic/env.py:27`) | `VVA_PG_DSN` → `config/langgraph.yaml` → default localhost |
| Runtime (`db/postgres.py:22-23`) | **chỉ** `config/langgraph.yaml` → default localhost |

Hệ quả nếu không sửa: đặt `VVA_PG_DSN` trỏ Neon → **schema lên Neon, còn app vẫn ghi vào Postgres
local**. Không có lỗi nào báo. Đây là loại bug tốn cả buổi để nhận ra.

Kèm theo: DSN của Neon **chứa mật khẩu**, mà `config/langgraph.yaml` **nằm trong git**. Không được
đặt DSN thật vào đó.

**Sửa**: `db/postgres.py` đọc env trước, yaml sau — cùng thứ tự ưu tiên với `alembic/env.py`:

```python
_DEFAULT_DSN = os.environ.get("VVA_PG_DSN") or _PG_CFG.get("dsn", "postgresql://vva:vva_dev@localhost:5433/vva")
```

Giữ giá trị localhost trong yaml làm mặc định cho dev — không ai phải cấu hình gì để chạy local.

---

## 3. Rủi ro kỹ thuật + cách xử lý

### 3.1 🔴 `sslmode` vs `ssl` — một DSN KHÔNG dùng chung được cho cả 2 đường

Neon bắt buộc TLS. Nhưng hai đường kết nối hiểu tham số khác nhau:

| Đường | Chấp nhận | Nếu sai |
|---|---|---|
| Raw asyncpg (`db/postgres.py`) | `?sslmode=require` ✅ | — |
| SQLAlchemy + asyncpg (Alembic) | `?ssl=require` | `sslmode` → `TypeError: connect() got an unexpected keyword argument 'sslmode'` |

**Sửa**: `_to_asyncpg()` trong `alembic/env.py` dịch luôn tham số:

```python
def _to_asyncpg(dsn: str) -> str:
    ...
    # SQLAlchemy asyncpg dialect không hiểu `sslmode` — chỉ hiểu `ssl`.
    return converted.replace("sslmode=", "ssl=")
```

Nhờ vậy chỉ cần **một** biến `VVA_PG_DSN` duy nhất cho cả runtime lẫn migration.

### 3.2 🔴 pgvector phải ≥ 0.8 — nếu không, recall giảm ÂM THẦM

`db/postgres.py:72` chạy `SET hnsw.iterative_scan = 'relaxed_order'` (pgvector ≥ 0.8.0). Nó nằm
trong `try/except: pass`, nên trên pgvector cũ hơn sẽ **không báo lỗi** — chỉ là
`memory_search` (có `WHERE session_id = ANY(...)`) mất recall khi filter cắt bớt ứng viên HNSW.

**Bắt buộc kiểm ngay sau khi tạo DB Neon:**

```sql
SELECT extversion FROM pg_extension WHERE extname = 'vector';   -- phải ≥ 0.8.0
SET hnsw.iterative_scan = 'relaxed_order';                      -- phải chạy KHÔNG lỗi
```

✅ **ĐÃ KIỂM 31/07 trên endpoint thật: `vector` 0.8.0, đã installed sẵn; `SET hnsw.iterative_scan`
chạy không lỗi.** Cổng chặn này **đã qua**.

### 3.3 🟠 Endpoint pooled làm vỡ prepared statement của asyncpg

Neon cấp 2 endpoint: **direct** (`ep-xxx.<region>.aws.neon.tech`) và **pooled**
(`ep-xxx-pooler...`). Pooled chạy PgBouncer **transaction mode** → hỏng statement cache của asyncpg
và có thể phá `register_vector` (đăng ký codec theo từng connection).

⚠️ **DSN Owner cấp đang là endpoint `-pooler`.**

Đo 31/07: prepared statement **chạy được** trên pooled trong phép thử đơn giản (1 connection,
tuần tự). Nhưng phép thử đó **không chứng minh được an toàn** — lỗi kinh điển
(`prepared statement "__asyncpg_stmt_N__" already exists`) chỉ xuất hiện khi pool có churn /
đồng thời, đúng lúc đang chạy thật.

Và **latency hai endpoint bằng nhau** (p50 53 ms cả hai — §3.6), nên pooled **không đem lại lợi
ích gì** ở quy mô này.

**Quyết định**: dùng **direct endpoint** — bỏ `-pooler` khỏi hostname:
```
ep-lingering-salad-a1qixkrs-pooler.ap-southeast-1.aws.neon.tech   ← pooled (Owner cấp)
ep-lingering-salad-a1qixkrs.ap-southeast-1.aws.neon.tech          ← direct (DÙNG CÁI NÀY)
```
Đổi 0 chi phí, loại bỏ hẳn một lớp rủi ro. Nếu vì lý do nào đó phải dùng pooled thì bắt buộc
thêm `statement_cache_size=0` vào `asyncpg.create_pool`.

### 3.4 🟠 Latency — rủi ro thực tế lớn nhất

Memory node chạy **mọi request**; retrieval là vector search. Backend đang chạy trên máy Owner (VN).

- Neon region gần nhất: **ap-southeast-1 (Singapore)** → RTT từ VN ~30-60 ms.
- Một lượt chat có **nhiều** query (memory load → retrieval → memory write) → cộng dồn.

⚠️ **Ước lượng ở v2.0 (+0,3-0,6 s) là SAI — sai khoảng 10 lần.** Đo end-to-end thật ở §3.7 cho
**+7,9 s (+28%)** mỗi lượt chat.

> Vì sao sai: tôi đếm **số chỗ gọi trong code** (6-8 `pg.fetch`) rồi coi đó là số round-trip lúc
> chạy. Sai — vòng lặp và truy vấn theo từng dòng nhân số đó lên. +7,9 s ÷ 53 ms ≈ **150
> round-trip tuần tự** mỗi lượt. Con số 150 này **chưa được instrument trực tiếp**, mới là suy ra
> từ tổng thời gian — cần đếm thật trước khi tối ưu.

### 3.5 🟡 Scale-to-zero

Neon free/launch tier tự ngủ sau ~5 phút không dùng; request đầu cold-start ~0.5-2 s. Với chat SSE
thì user thấy "đang suy nghĩ" lâu bất thường ở lần hỏi đầu tiên.

**Quyết định**: giai đoạn demo cứ để bật (tiết kiệm). Trước khi có người dùng thật thì tắt, hoặc
thêm ping giữ ấm.

---

### 3.6 📊 Số đo thật trên endpoint Owner cấp (31/07, từ máy Owner tại VN)

| | POOLED | DIRECT |
|---|---|---|
| Postgres | **17.10** | 17.10 |
| pgvector | **0.8.0** (đã installed) | 0.8.0 |
| `SET hnsw.iterative_scan` | ✅ ok | ✅ ok |
| `channel_binding=require` | ✅ asyncpg chấp nhận | ✅ |
| RTT p50 | **53 ms** | 53 ms |
| RTT p95 | 85-116 ms | 80-137 ms |
| Connect (nguội) | **1 254 ms** | — |
| Connect (ấm) | 373 ms | 384-538 ms |

**Quy ra mỗi lượt chat**: memory node gộp 3 query vào 1 `asyncio.gather` (memory.py:248), phần
còn lại (`session_store`, `kb_search`, `memory_search`) chạy **tuần tự** → khoảng **6-8 round-trip
nối tiếp** → **+0,3-0,6 s** ở p50.

**Lưu ý PG 17, không phải 16** (plan v1.0 ghi 16). Không sao với đường đã chọn (alembic + ingest).
Chỉ quan trọng nếu sau này dùng `pg_dump`: dump từ local pg16 → restore vào pg17 là **được**;
chiều ngược lại thì không.

---

### 3.7 📊 Đo end-to-end thật (31/07) — KHÁC HẲN dự đoán

Cùng một câu hỏi, chạy xen kẽ hai backend trong cùng khoảng thời gian để loại trừ biến động LLM:

| | Lần 1 | Lần 2 | TB |
|---|---:|---:|---:|
| **LOCAL** (Postgres Docker) | 29,5 s | 27,2 s | **28,4 s** |
| **NEON** (Singapore, direct) | 40,8 s | 31,8 s | **36,3 s** |
| | | | **+7,9 s (+28%)** |

(Lượt đầu tiên trên Neon mất 36 s **kèm cold-start**; các lượt trên đã ấm.)

`/health/detailed` → `postgres.latency_ms`: local **111 ms** vs Neon **1 724 ms**. Chỉ số này đo
chủ yếu **chi phí mở kết nối** chứ không phải query — `api/health.py:44` tạo `PostgresClient()`
mới mỗi lần gọi, tức pool mới + bắt tay TLS mới.

**Ý nghĩa**: +28% là chi phí **thật và nhìn thấy được**, không phải 2-3% như v2.0 ước lượng.
Nguyên nhân gần như chắc chắn là **số round-trip tuần tự quá lớn**, và đó là thứ **sửa được** —
xem §10.

---

## 4. Các bước

### Bước 1 — Sửa code (làm trước, test bằng DB local)

| # | File | Sửa |
|---|---|---|
| 1 | `agenticRAG/langgraph_agents/db/postgres.py` | DSN đọc `VVA_PG_DSN` trước, yaml sau (§2) |
| 2 | `agenticRAG/langgraph_agents/alembic/env.py` | `_to_asyncpg()` dịch `sslmode=` → `ssl=` (§3.1) |
| 3 | `scripts/QUICKSTART.md` · `.env.example` | Ghi `VVA_PG_DSN` là cách trỏ DB, kèm cảnh báo không commit |

**Verify ngay tại local, chưa cần Neon:**

```bash
# Không đặt env → vẫn phải chạy như cũ
python -m pytest tests/langgraph_agents/ -m unit -q

# Đặt env trỏ chính DB local → app phải dùng đúng DSN đó
VVA_PG_DSN="postgresql://vva:vva_dev@localhost:5433/vva" \
  python -m uvicorn langgraph_agents.api.main:create_app --factory --port 8000
curl http://localhost:8000/health/detailed     # postgres.ok = true
```

> Bước này là **điều kiện tiên quyết**. Không được provision Neon trước rồi mới sửa code.

### Bước 2 — ✅ Project Neon: ĐÃ CÓ

Owner đã cấp endpoint 31/07: `ap-southeast-1`, PG 17.10, pgvector 0.8.0, database `neondb`,
user `neondb_owner`. Các cổng chặn ở §3.2 **đã kiểm và đạt**.

Việc còn lại của bước này: **đổi `-pooler` → direct** trong DSN (§3.3).

### Bước 3 — ✅ ĐÃ XONG: Schema + dữ liệu

```bash
export VVA_PG_DSN="postgresql://<user>:<pass>@ep-xxx.ap-southeast-1.aws.neon.tech/vva?sslmode=require"

# 3a. Schema
cd agenticRAG/langgraph_agents && alembic upgrade head && cd ../..
#     → phải ra 003_kb_hnsw (head)

# 3b. Knowledge base
python scripts/ingest_kb_pgvector.py --reset
```

**Vì sao ingest lại thay vì `pg_dump`**: dữ liệu người dùng hiện tại là **rác dev** (1 user, 12
messages), còn KB thì sinh lại được và cách này **kiểm chứng luôn** đường ingest chạy được với Neon.
Nhanh hơn cả việc lo lệch phiên bản `pg_dump`.

> **Khi cutover production thật (Phase 7)** — lúc đã có dữ liệu người dùng thật — thì **phải** dùng
> `pg_dump`, không được ingest lại:
> ```bash
> pg_dump --no-owner --no-acl -Fc -d "postgresql://vva:vva_dev@localhost:5433/vva" -f vva.dump
> pg_restore --no-owner --no-acl -d "$VVA_PG_DSN" vva.dump
> ```
> Kèm cửa sổ ngừng ghi (đọc thêm ở §6).

### Bước 4 — ✅ ĐÃ VERIFY (kết quả thật ở §11)

| # | Kiểm | Lệnh | Kỳ vọng |
|---|---|---|---|
| 1 | pgvector | `SELECT extversion FROM pg_extension WHERE extname='vector'` | ≥ 0.8.0 |
| 2 | Số bảng | `\dt` | 8 bảng |
| 3 | **Index** | `SELECT indexname FROM pg_indexes WHERE schemaname='public'` | **≥ 17**, có đủ **2 HNSW** |
| 4 | KB | `SELECT COUNT(*) FROM kb_embeddings` | **2918** |
| 5 | HNSW được dùng thật | `EXPLAIN SELECT ... ORDER BY embedding <=> $1 LIMIT 5` | `Index Scan using idx_kb_emb_embedding` — **không** phải Seq Scan |
| 6 | `iterative_scan` | `SET hnsw.iterative_scan='relaxed_order'` | không lỗi |
| 7 | App sống | `curl :8000/health/detailed` | `postgres.ok=true` |
| 8 | **End-to-end** | POST `/chat` "bài tập cho cơ bụng và lưng dưới" | `mode: synthesize` — **không** `refuse` |
| 9 | Test suite | `pytest tests/langgraph_agents/ -q` | như baseline hiện tại |
| 10 | **Latency** | so p95 `/chat` giữa local và Neon, ≥10 lượt | ghi số vào worklog, Owner quyết chấp nhận hay không |

> #3 và #5 là bài học trực tiếp từ bug 30/07 — index thiếu mà vẫn "chạy được", chỉ chậm dần.
> #8 là bài học từ bug KB rỗng — schema đúng nhưng dữ liệu rỗng thì mọi câu hỏi bị refuse.

### Bước 5 — ⏸ Cutover: CHỜ OWNER (xem §3.7)

1. Đặt `VVA_PG_DSN` ở nơi chạy backend (env của shell/service, **không** vào git).
2. Restart backend, chạy lại §4 #7-#8.
3. **Giữ Docker Postgres local thêm ít nhất 1 tuần** — chưa `docker compose down -v`.

> 🔑 **Về mật khẩu**: DSN chứa mật khẩu đã đi qua kênh chat, nên coi như **đã lộ ra ngoài phạm vi
> máy Owner**. Bắt buộc: (a) đặt ở env/secret, **không** vào `config/langgraph.yaml` (file này
> nằm trong git); (b) **đổi mật khẩu trên dashboard Neon trước khi có dữ liệu người dùng thật**.
> Giai đoạn dev với DB rỗng thì chưa gấp.

---

## 5. Rollback

Đơn giản vì không có thay đổi một chiều nào:

1. Bỏ `VVA_PG_DSN` (hoặc trỏ lại localhost) → restart backend.
2. Postgres local vẫn còn nguyên dữ liệu (đừng xoá volume).

Điều kiện để rollback dễ: **không xoá volume local** và **không sửa schema chỉ-có-trên-Neon**. Mọi
thay đổi schema vẫn phải đi qua Alembic để hai bên còn tương thích.

---

## 6. Việc CHƯA nằm trong plan này

| Việc | Vì sao tách |
|---|---|
| **Redis lên cloud** (Upstash/Redis Cloud) | Neon không làm Redis. STM session + circuit breaker vẫn phụ thuộc máy local |
| **Object storage** (R2/S3) cho motion BVH + audio TTS | P2 Kimodo runtime delivery sẽ cần. Neon không có Storage — đây chính là điểm Supabase hơn |
| **Backend lên cloud cùng region** | Nếu §4 #10 cho thấy latency không chấp nhận được thì đây là việc bắt buộc tiếp theo, không phải tuỳ chọn |
| **Cutover có dữ liệu thật + downtime window** | Khi nào có người dùng thật. Lúc đó dùng `pg_dump` (§ bước 3) và cần cửa sổ ngừng ghi |
| **Bật `REQUIRE_AUTH=true`** | Đang là 🔴 Critical trong tech-debt. DB lên cloud mà auth còn tắt thì IDOR thành lỗ hổng thật |

> ⚠️ Mục cuối cần nhấn: **hiện `REQUIRE_AUTH` mặc định `false`**. Chừng nào DB còn ở localhost thì
> rủi ro giới hạn trong máy; đưa lên cloud là đổi mô hình đe doạ. Nên bật auth **cùng đợt** hoặc
> ngay sau, đừng để trôi.

---

## 7. Ước lượng

| Bước | Effort |
|---|---|
| 1. Sửa code (2 file) + test local | 1h |
| 2. Provision Neon + kiểm pgvector | 30m |
| 3. Schema + ingest KB | 30m |
| 4. Verify 10 mục | 1h |
| 5. Cutover + theo dõi | 30m |
| Dự phòng | 30m |
| **Tổng** | **~4h** |

Chi phí Neon: 21 MB dữ liệu — nằm gọn trong free tier. Chi phí không phải yếu tố cân nhắc ở quy mô này.

---

## 8. Open items

**Owner đã chốt (31/07):**

- [x] **Region: `ap-southeast-1` (Singapore)** — dùng cho bước 2.
- [x] **Redis**: Owner đã có sắp xếp riêng, tính sau. Không nằm trong plan này.
- [x] **Nhà cung cấp: Neon.** Supabase chỉ hơn ở một điểm duy nhất là **có Storage sẵn**
      (xem §9); mọi thứ khác nó gói kèm (Auth, Realtime) thì dự án này hoặc đã có, hoặc không dùng.

**Còn chờ:**

- [ ] **Scale-to-zero**: bật (rẻ, cold-start 0.5-2s) hay tắt?
- [ ] **Bao giờ đưa backend lên cloud?** Backend còn ở máy Owner thì mỗi query cộng RTT tới
      Singapore — chấp nhận tạm cho demo, không phải trạng thái cuối. §4 bước 4 #10 sẽ cho số.
- [ ] **Object storage dùng gì** — xem §9.

---

## 9. Object storage — việc kế tiếp, không phải việc của plan này

Đây là **điểm duy nhất** Supabase hơn Neon cho dự án này, nên ghi rõ để khỏi bàn lại:

- Neon **chỉ có Postgres**. P2 (Kimodo runtime delivery) sẽ sinh file BVH lúc chạy, và VieNeu-TTS
  sẽ sinh audio → cần chỗ chứa + phục vụ file cho trình duyệt. Neon không làm việc đó.
- Supabase có **Storage** (S3-compatible) trong cùng project → 1 nhà cung cấp thay vì 2.
- Những thứ Supabase gói kèm khác **không phải lý do** ở đây:
  - **Auth** — đã dùng Cognito (`api/auth.py` verify JWKS RS256, frontend Amplify). Dùng Supabase
    Auth nghĩa là làm lại từ đầu.
  - **Realtime** — đã có SSE tự làm trong FastAPI.
  - **Edge Functions** — đã có FastAPI.
  - **pgvector / pooler / region** — hai bên tương đương, không phải yếu tố phân biệt.

Đổi lại, **Neon hơn ở branching**: tạo nhánh DB từ bản sao dữ liệu thật để thử `alembic upgrade`
trước khi chạy lên nhánh chính. Với dự án vừa dính bug migration thiếu index (30/07) thì đây là
lợi ích thật, không phải tính năng cho vui.

**Khi cần object storage, đề xuất Cloudflare R2** (S3-compatible, **không tính phí egress**) —
hợp vì file motion/audio được trình duyệt tải mỗi lần phát. S3 cũng được nhưng có phí egress.
Quyết định này **không chặn** plan Neon; làm sau khi P2 khởi động.

---

## 10. Việc tiếp theo bắt buộc: giảm số round-trip

+28% latency không phải "Neon chậm" — RTT chỉ 53 ms. Vấn đề là **app nói chuyện với DB quá
nhiều lần theo kiểu tuần tự**. Local che mất chuyện này vì mỗi round-trip chưa tới 1 ms.

Bằng chứng rõ nhất gặp ngay khi ingest: `scripts/ingest_kb_pgvector.py` insert **từng dòng**
(2 query × 2918 dòng ≈ 5 800 round-trip) → mất **~9 phút** trên Neon, so với vài giây ở local.

Hướng xử lý, theo thứ tự đáng làm:

1. **Đếm thật** số query mỗi lượt chat (bật log của asyncpg hoặc đếm trong `PostgresClient`)
   trước khi tối ưu. Đừng sửa mò — chính tôi vừa sai vì đoán.
2. Gộp query trong `memory_search` / `session_store` (`asyncio.gather` như `memory.py:248` đã làm,
   hoặc gộp thành một câu SQL).
3. `executemany` / `COPY` cho ingest thay vì insert từng dòng.

Việc này **có lợi cả khi ở local**, và là điều kiện để đưa DB ra khỏi máy mà không mất trải nghiệm.

---

## 11. Kết quả verify thật (31/07)

| # | Kiểm | Kết quả |
|---|---|---|
| 1 | pgvector | ✅ 0.8.0 |
| 2 | Số bảng | ✅ **8** |
| 3 | Index | ✅ **17**, có đủ `idx_kb_emb_embedding` + `idx_summaries_embedding` |
| 4 | KB rows | ✅ **2918** |
| 5 | `SET hnsw.iterative_scan` | ✅ ok |
| 6 | `/health` | ✅ `{"status":"ok"}` |
| 7 | **Chat end-to-end** | ✅ **`mode: synthesize`** (không refuse), token stream đúng |
| 8 | Unit tests sau khi sửa DSN | ✅ **275 passed** |
| 9 | Env thắng yaml | ✅ đã kiểm cả 2 chiều |
| 10 | Latency | ⚠️ **+28%** — xem §3.7, cần Owner quyết |

---

## 12. Đo cuối cùng: DB tốn bao nhiêu trong một lượt chat (§3.8)

Đo bằng counter thật trong `PostgresClient` (`VVA_PG_STATS=1`) + endpoint `/debug/pgstats`,
không phải suy từ tổng thời gian nữa.

| | Lượt 1 (pool nguội) | Lượt 2 (pool ấm) |
|---|---:|---:|
| Tổng lượt chat | 39,0 s | **29,3 s** |
| Số query DB | **12** | **12** |
| Thời gian DB | 3,29 s | **1,41 s** |
| DB chiếm | 8,4% | **4,8%** |
| TB / query | 274 ms | **117 ms** |

Chi tiết lượt ấm: `fetch` ×7 = 1 008 ms (vector search), `execute` ×2 = 159 ms,
`executemany` ×1 = 80 ms, `fetchrow` ×1 = 79 ms, `fetchval` ×1 = 80 ms.

**Lượt ấm 29,3 s ≈ local 28,4 s.** Nghĩa là con số "+28%" ở §3.7 phần lớn là **cold pool +
biến động LLM**, không phải chi phí thường trực của Neon.

### Tôi đã suy luận sai HAI lần — ghi lại để không lặp

| Lần | Tôi nói | Cách suy | Thực tế |
|---|---|---|---|
| 1 | +0,3-0,6 s (2-3%) | đếm **số chỗ gọi** trong code (6-8) | thiếu |
| 2 | ~150 round-trip tuần tự | chia +7,9 s cho RTT 53 ms | thừa hơn 12 lần |
| **Đo** | **12 query, 1,4 s, 4,8%** | counter trong `PostgresClient` | — |

Bài học: query **ít mà chậm** (117 ms/query so với RTT 53 ms), không phải **nhiều mà nhanh**.
Cả hai lần sai đều do suy từ một con số tổng thay vì đo trực tiếp.

### Vì vậy: KHÔNG cần đợt tối ưu round-trip lớn

4,8% một lượt chat không đáng đánh đổi rủi ro. Việc còn lại đáng làm, theo thứ tự:

1. **Giữ pool ấm** — chênh lệch nguội/ấm (3,29 s vs 1,41 s) lớn hơn mọi thứ khác. Liên quan
   trực tiếp tới quyết định scale-to-zero.
2. `fetch` ×7 = 1 008 ms là phần lớn nhất — nếu tối ưu thì nhắm vào đây, nhưng chỉ sau khi
   có lý do thật.

---

## 13. ⚠️ Cảnh báo: `executemany` + pgvector làm Python SEGFAULT

Đã thử gộp insert của `scripts/ingest_kb_pgvector.py` (2 round-trip mỗi **batch** thay vì mỗi
**record**): `unnest` cho `documents` + `executemany` cho `kb_embeddings`.

**Kết quả: Python chết bằng segfault (exit 139)** — crash trong native code, không phải exception.
Vì `--reset` đã xoá bảng trước đó, KB **rỗng hoàn toàn** và app từ chối mọi câu hỏi cho tới khi
nạp lại. Đã hoàn nguyên về bản per-row.

Nghi phạm: codec `vector` của `pgvector.asyncpg` khi đi qua `executemany` với mảng numpy.
**Chưa xác minh** — chỉ biết chắc là nó crash.

Nếu ai muốn thử lại:
- Ép `vector.tolist()` sang list Python thuần trước khi truyền.
- Thử `unnest($1::vector[])` trong một câu INSERT thay vì `executemany`.
- **Luôn thử với `--limit 5` trước**, và **đừng dùng `--reset`** cho tới khi chạy được.

Chi phí thực của việc để nguyên: ingest mất ~9 phút, nhưng đó là thao tác **một lần**. Không
đáng để đánh đổi việc xoá sạch KB thêm lần nữa.
