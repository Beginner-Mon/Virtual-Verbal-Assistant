# QUICKSTART — chạy VVA ở máy local

> Cập nhật **31/07/2026**, viết lại theo kiến trúc **LangGraph** hiện tại.
> Bản cũ mô tả stack đã bị thay: DART, ChromaDB, orchestrator :8080, UI :3000,
> `run_stack.py`, Streamlit :8501 — **không cái nào còn dùng nữa**, đừng làm theo
> hướng dẫn cũ nếu bạn tìm thấy ở đâu đó.

Chỉ có **2 tiến trình** phải tự chạy: **backend** và **frontend**. Phần còn lại nằm trong Docker.

---

## 0. Yêu cầu

| Thứ | Ghi chú |
|---|---|
| Docker Desktop | Postgres + Redis + SearXNG. Phải **mở app** trước, chờ ~5-30s cho engine lên |
| Conda env `firstconda` | Backend Python. Dùng đúng env này — xem cảnh báo ở §6 |
| Node 18+ / npm | Frontend Vite |
| API key LLM | `DEEPSEEK_API_KEY` (chính) và/hoặc `GEMINI_API_KEYS` (fallback) |

**Không cần**: WSL, CUDA, ffmpeg, Redis cài máy. Kimodo (text-to-motion) chạy riêng trên cloud và
**không bắt buộc** để chat hoạt động.

---

## 1. Cổng

| Service | Cổng | Nguồn |
|---|:---:|---|
| **Frontend (Vite)** | **5173** | `npm run dev` |
| **Backend (FastAPI + LangGraph)** | **8000** | `uvicorn` |
| PostgreSQL + pgvector | 5433 | Docker `vva-postgres` |
| Redis | 6379 | Docker `vva-redis` |
| SearXNG (web search) | 6666 | Docker `vva-searxng` |
| VieNeu-TTS (giọng nói) | 5000 | `SpeechLLm/api_server.py` — **tuỳ chọn** |

> ⚠️ **Tuyệt đối không dùng cổng 8080.** Đó là service Spring của Owner trên máy này.
> Backend luôn là **8000**.

---

## 2. Chạy

```bash
# ── 1. Hạ tầng ────────────────────────────────────────────────
docker compose -f docker-compose.langgraph.yml up -d postgres redis searxng
# (docker-compose.yml ở root là của ChromaDB thời cũ — KHÔNG dùng)

# ── 2. Schema (lần đầu, hoặc khi migration mới) ────────────────
# PHẢI đứng đúng thư mục này — `script_location` trong alembic.ini là đường dẫn
# tương đối theo CWD, nên `alembic -c langgraph_agents/alembic.ini` từ agenticRAG
# sẽ báo "Path doesn't exist: alembic".
cd agenticRAG/langgraph_agents && alembic upgrade head && cd ../..

# ── 3. Backend :8000 ──────────────────────────────────────────
conda activate firstconda
cd agenticRAG
python -m uvicorn langgraph_agents.api.main:create_app --factory --port 8000 --host 0.0.0.0

# ── 4. Frontend :5173 (terminal khác) ─────────────────────────
cd ECA_UI/frontend
npm install          # lần đầu
npm run dev
```

Mở **http://localhost:5173** → nhấn nút **Chat** ở thanh điều hướng nổi bên trái.

### 2b. Giọng nói (tuỳ chọn)

Bỏ qua bước này thì chat vẫn chạy bình thường, chỉ là không có tiếng.

```bash
# ⚠️ env là `tts`, KHÔNG phải `firstconda` — firstconda không có gói `vieneu`.
cd SpeechLLm
C:/Users/Nguyen/miniconda3/envs/tts/python.exe api_server.py    # :5000
```

Rồi trong khung chat: menu **`+`** → **"Trả lời bằng giọng nói"**. Mặc định tắt vì lượt có giọng
mất ~**77 s** so với ~**33 s** chỉ chữ (VieNeu chạy CPU, ~18 ms mỗi ký tự).

Lần synthesize đầu tiên phải nạp model GGUF 223 MB nên chậm hơn hẳn — đừng tưởng treo.

> ⚠️ Chạy backend + TTS cùng lúc ăn gần hết **commit limit** của máy này. Lúc đó `npm run build`
> có thể chết với `paging file is too small` / `VirtualAlloc failed` — **không phải lỗi code**.
> Tắt bớt một service rồi build.

---

## 3. Nạp knowledge base (bắt buộc cho lần cài mới)

Bảng `kb_embeddings` rỗng thì **mọi câu hỏi chuyên môn đều bị từ chối** (`mode: refuse`) —
đây từng là một bug tốn nhiều thời gian truy, nên đừng bỏ qua bước này.

> ⛔ **DỪNG backend trước khi chạy lệnh này.** Nếu uvicorn đang chạy, tiến trình ingest sẽ
> **segfault** (exit 139) — hai tiến trình cùng nạp native runtime của torch. Nó chết **im lặng,
> không có traceback**, và vì `--reset` đã xoá bảng trước đó nên **KB rỗng hoàn toàn** → mọi câu hỏi
> bị từ chối. Đã bị dính 31/07.

```bash
# 1. Ctrl+C ở terminal backend (hoặc kill tiến trình cổng 8000)
# 2. Nạp KB
python scripts/ingest_kb_pgvector.py --reset      # ~2918 bài tập, ~9 phút nếu DB ở cloud
# 3. Bật lại backend
```

Kiểm tra:

```bash
docker exec vva-postgres psql -U vva -d vva -c "SELECT COUNT(*) FROM kb_embeddings;"
# kỳ vọng 2918 — nếu 0 thì chạy lại lệnh trên
```

---

## 4. Biến môi trường

**Backend** — đặt trong shell hoặc `.env` ở thư mục gốc:

| Biến | Bắt buộc | Mặc định / ghi chú |
|---|:---:|---|
| `DEEPSEEK_API_KEY` | ✅ | LLM chính |
| `GEMINI_API_KEYS` | ➖ | Fallback, phân tách bằng dấu phẩy |
| `VVA_PG_DSN` | ➖ | `postgresql://vva:vva_dev@localhost:5433/vva` |
| `SEARXNG_URL` | ➖ | `http://localhost:6666` |
| `REQUIRE_AUTH` | ➖ | `false` khi dev. Production phải `true` |
| `LOG_LEVEL` / `LOG_FILE` | ➖ | Ghi log |

**Frontend** — `ECA_UI/frontend/.env.local` (copy từ `.env.example`):

```env
VITE_API_BASE_URL=http://localhost:8000
VITE_AUTH_DISABLED=true      # bỏ qua Cognito, vào thẳng chat
```

> Trước khi deploy production: bỏ `VITE_AUTH_DISABLED` (hoặc `=false`), đặt `REQUIRE_AUTH=true`
> và điền 3 biến Cognito.

---

## 5. Kiểm tra nhanh

```bash
curl http://localhost:8000/health              # {"status":"ok"}
curl http://localhost:8000/health/detailed     # từng thành phần
docker ps --format "{{.Names}} {{.Status}}"    # 3 container phải Up
```

Thử một câu hỏi thật (streaming SSE):

```bash
printf '%s' '{"query":"bài tập cho cơ bụng và lưng dưới","session_id":"77777777-7777-7777-7777-777777777777","user_id":"77777777-7777-7777-7777-777777777778","web_search":false}' > /tmp/q.json
curl -N -X POST http://localhost:8000/chat -H "Content-Type: application/json" --data-binary @/tmp/q.json
```

Chạy test:

```bash
python -m pytest tests/langgraph_agents/ -m unit -q   # 275 passed, không cần service sống
python -m pytest tests/langgraph_agents/ -q            # 312, cần Docker + API key thật
```

---

## 6. Lỗi hay gặp

| Triệu chứng | Nguyên nhân & cách xử lý |
|---|---|
| `ModuleNotFoundError: langchain_google_genai` khi chạy test | **Sai Python.** `python` mặc định trên PATH không phải env `firstconda`. Gọi thẳng: `/c/Users/Nguyen/miniconda3/envs/firstconda/python -m pytest ...` |
| `docker compose` báo lỗi pipe | Docker Desktop chưa khởi động xong — mở app rồi chờ ~5-30s |
| Chat luôn trả lời từ chối (`refuse`) | KB rỗng → chạy §3 |
| Cổng 8000 bận | Còn tiến trình uvicorn cũ. **Đừng** chuyển sang 8080 |
| Frontend gọi sai cổng | Thiếu `.env.local` → mặc định vẫn là `:8000`, kiểm tra `VITE_API_BASE_URL` |
| Avatar là màn hình trắng vài giây | Bình thường: model `seele.vrm` nặng 21 MB, cold load ~10-13s. Có overlay loading |
| `402` / `429` từ LLM | Hết tiền/quota tài khoản, không phải lỗi code |
| `/health/detailed` trả `degraded` | **Bình thường khi chưa chạy TTS** — `speechllm` là optional dependency nhưng vẫn bị tính vào status tổng. `/health` (không `/detailed`) mới là cái để kiểm tra sống/chết |
| `alembic` báo `Path doesn't exist: alembic` | Chạy sai thư mục — xem chú thích ở bước 2 |
| Ingest chết `Segmentation fault` / exit 139, log rỗng | **Backend đang chạy.** Tắt uvicorn rồi chạy lại. Kiểm tra `SELECT COUNT(*) FROM kb_embeddings` — nếu 0 thì `--reset` đã xoá sạch, phải nạp lại |
| Script Python ghi file vào `/tmp/...` mà không thấy đâu | Python bản Windows hiểu `/tmp` là `C:\tmp` (không tồn tại). Dùng đường dẫn Windows đầy đủ |

---

## 7. Tuỳ chọn — Kimodo (text-to-motion)

Chat **không cần** Kimodo. Việc sinh động tác 3D chạy trên cloud và hiện chưa nối runtime.

Chuyển động tác đã sinh (NPZ → BVH) để xem offline:

```bash
python scripts/kimodo_npz_to_bvh.py <file.npz> [output.bvh]
```

Script đọc cả hai định dạng: **Kimodo** (`local_rot_mats` + `root_positions`) và
**AMASS/SMPL-X** (`poses` + `trans`).

Xem thử trong UI: `:5173` → nav **Motion** → dropdown **Motion file (debug)** → chọn
`motions/generated/motion_*.bvh`.

---

## 8. Dừng

```bash
# Ctrl+C ở cả 2 terminal (backend, frontend)
docker compose -f docker-compose.langgraph.yml down     # thêm -v để xoá luôn dữ liệu Postgres
```

---

## Tài liệu liên quan

- [`docs/tracking/status.md`](../docs/tracking/status.md) — trạng thái hiện tại, việc đang treo, ưu tiên
- [`README.md`](../README.md) — kiến trúc tổng quan
- [`docs/ops/runbook.md`](../docs/ops/runbook.md) · [`docs/ops/troubleshooting.md`](../docs/ops/troubleshooting.md) — vận hành chi tiết
- [`docs/worklogs/`](../docs/worklogs/) — nhật ký từng phiên
