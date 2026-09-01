# Phase 6.10 — Pre-Deploy Hardening

> Author: K | Date: 2026-05-28
> Branch: `feature/langgraph-rewrite`
> **Prerequisite for Phase 7 deployment. All items below must be green before Phase 7 starts.**

---

## Status snapshot (đầu ngày 28/05)

Những việc K đã làm hôm nay — N không cần làm lại:

| ✅ Done | File |
|---------|------|
| Delete `nodes/conversation.py` (dead code) | deleted |
| Migrate `api/main.py` sync Redis → `redis.asyncio` | `api/main.py` |
| `health.py` `check_redis` → native `await redis.ping()` | `api/health.py` |
| `_stream_chat` capture `intent` từ planner output | `api/main.py` |
| `_route_after_synthesizer` → intent-based (không dựa vào ToolMessage history) | `graph.py` |
| Grader: thêm no-evidence rule cho `knowledge_query` | `nodes/grader.py` |
| Rewrite `test-ui` cho LangGraph architecture | `ECA_UI/test-ui/` |

---

## Việc N cần làm

### Task 1 — Normalize Messages Table

**Tại sao**: Cột `conversations.messages JSONB` hiện dùng `||` để append — PostgreSQL vẫn phải decode + encode lại toàn bộ blob mỗi lần ghi. Session 50+ turns = TOAST overhead tăng dần. Không thể paginate mà không pull toàn bộ.

**Giải pháp**: Tách bảng `messages`, mỗi row = một tin nhắn, ghi O(1).

---

#### 1a. Tạo file `db/migrations/001_normalize_messages.sql`

```sql
-- Step 1: create messages table
CREATE TABLE IF NOT EXISTS messages (
    id          UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    session_id  UUID NOT NULL REFERENCES conversations(session_id) ON DELETE CASCADE,
    role        TEXT NOT NULL CHECK (role IN ('user', 'assistant')),
    content     TEXT NOT NULL,
    metadata    JSONB DEFAULT '{}',
    created_at  TIMESTAMPTZ DEFAULT now()
);
CREATE INDEX IF NOT EXISTS idx_messages_session_created
    ON messages (session_id, created_at);

-- Step 2: add migration flag so backfill script is idempotent
ALTER TABLE conversations
    ADD COLUMN IF NOT EXISTS _migrated BOOLEAN DEFAULT false;

-- Step 3: note — DROP COLUMN messages chạy SAU khi test xanh (xem cuối Task 1)
```

---

#### 1b. Tạo file `db/migrations/migrate_messages.py`

Script chạy một lần, backfill data từ `conversations.messages` JSONB sang bảng `messages`.

```python
"""One-time migration: conversations.messages JSONB → messages table.

Usage:
    python -m langgraph_agents.db.migrations.migrate_messages
"""
import asyncio
import json
import os
from pathlib import Path

import asyncpg
from dotenv import load_dotenv

load_dotenv(Path(__file__).resolve().parents[3] / ".env")

_DSN = os.getenv(
    "DATABASE_URL",
    "postgresql://postgres:postgres@localhost:5432/vva"
)


async def main():
    conn = await asyncpg.connect(_DSN)
    rows = await conn.fetch(
        "SELECT session_id, messages FROM conversations WHERE _migrated = false"
    )
    print(f"Migrating {len(rows)} sessions…")
    migrated = 0
    for row in rows:
        msgs = row["messages"]
        if isinstance(msgs, str):
            try:
                msgs = json.loads(msgs)
            except json.JSONDecodeError:
                msgs = []
        if not msgs:
            await conn.execute(
                "UPDATE conversations SET _migrated = true WHERE session_id = $1",
                row["session_id"],
            )
            continue

        records = [
            (
                row["session_id"],
                m["role"],
                m.get("content", ""),
                json.dumps(m.get("metadata", {})),
                m.get("timestamp"),  # may be None → asyncpg uses DEFAULT
            )
            for m in msgs
            if m.get("role") in ("user", "assistant") and m.get("content")
        ]
        if records:
            await conn.executemany(
                """INSERT INTO messages (session_id, role, content, metadata, created_at)
                   VALUES ($1, $2, $3, $4::jsonb, COALESCE($5::timestamptz, now()))
                   ON CONFLICT DO NOTHING""",
                records,
            )
        await conn.execute(
            "UPDATE conversations SET _migrated = true WHERE session_id = $1",
            row["session_id"],
        )
        migrated += 1

    await conn.close()
    print(f"Done. {migrated}/{len(rows)} sessions migrated.")


if __name__ == "__main__":
    asyncio.run(main())
```

---

#### 1c. Sửa `db/session_store.py` — 4 hàm

**`write_session_turn`**: thay JSONB append bằng INSERT vào `messages`.

```python
async def write_session_turn(
    user_id: str,
    session_id: str,
    user_query: str,
    assistant_answer: str,
    intent: str,
    tokens: int,
) -> None:
    user_id = _to_uuid(user_id)
    pg = get_pg_client()
    await pg.connect()
    ts = datetime.now(timezone.utc).isoformat()

    # Auto-create user (FK guard)
    await pg.execute(
        "INSERT INTO users (id) VALUES ($1::uuid) ON CONFLICT (id) DO NOTHING",
        user_id,
    )
    # Upsert session header
    await pg.execute(
        """INSERT INTO conversations (id, user_id, session_id, created_at, updated_at)
           VALUES (gen_random_uuid(), $1::uuid, $2::uuid, now(), now())
           ON CONFLICT (session_id) DO UPDATE SET updated_at = now()""",
        user_id, session_id,
    )
    # Insert 2 message rows — O(1), no blob read
    await pg.executemany(
        """INSERT INTO messages (session_id, role, content, metadata, created_at)
           VALUES ($1::uuid, $2, $3, $4::jsonb, $5::timestamptz)""",
        [
            (session_id, "user",      user_query,       json.dumps({}),                          ts),
            (session_id, "assistant", assistant_answer, json.dumps({"intent": intent, "tokens": tokens}), ts),
        ],
    )
    await _append_stm(session_id, user_query, assistant_answer, ts)
```

**`load_session_messages`**: SELECT từ `messages`, trả về list có cursor support.

```python
async def load_session_messages(
    user_id: str,
    session_id: str,
    limit: int = 50,
    before: str | None = None,   # cursor: ISO timestamp của tin nhắn cũ nhất đang hiển thị
) -> dict | None:
    user_id = _to_uuid(user_id)
    pg = get_pg_client()
    await pg.connect()

    header = await pg.fetchrow(
        "SELECT updated_at FROM conversations WHERE user_id=$1::uuid AND session_id=$2::uuid",
        user_id, session_id,
    )
    if not header:
        return None

    # Cursor-based: load `limit` messages trước cursor (scroll-up pattern)
    if before:
        rows = await pg.fetch(
            """SELECT role, content, metadata, created_at
               FROM messages
               WHERE session_id = $1::uuid AND created_at < $2::timestamptz
               ORDER BY created_at DESC LIMIT $3""",
            session_id, before, limit,
        )
        rows = list(reversed(rows))   # trả về chronological order
    else:
        rows = await pg.fetch(
            """SELECT role, content, metadata, created_at
               FROM messages
               WHERE session_id = $1::uuid
               ORDER BY created_at DESC LIMIT $2""",
            session_id, limit,
        )
        rows = list(reversed(rows))

    messages = [
        {
            "role":      r["role"],
            "content":   r["content"],
            "metadata":  _coerce_metadata(r["metadata"]),
            "timestamp": r["created_at"].isoformat(),
        }
        for r in rows
    ]
    return {
        "session_id": session_id,
        "messages":   messages,
        "updated_at": header["updated_at"],
        "has_more":   len(rows) == limit,   # UI dùng để biết còn tin nhắn cũ hơn
        "next_cursor": rows[0]["created_at"].isoformat() if rows else None,
    }
```

Cần thêm helper `_coerce_metadata` (copy từ `vector_backend.py`):

```python
def _coerce_metadata(value) -> dict:
    if value is None:              return {}
    if isinstance(value, dict):    return value
    if isinstance(value, str):
        try:    return json.loads(value)
        except: return {}
    return {}
```

**`list_user_sessions`**: LATERAL JOIN thay `jsonb_array_elements`.

```python
async def list_user_sessions(user_id: str, limit: int = 50) -> list[dict]:
    user_id = _to_uuid(user_id)
    pg = get_pg_client()
    await pg.connect()
    rows = await pg.fetch(
        """
        SELECT c.session_id::text,
               c.created_at,
               c.updated_at,
               COALESCE(first_msg.content, '(empty)')  AS first_user_message_preview,
               COALESCE(msg_count.cnt, 0)::int          AS message_count
        FROM conversations c
        LEFT JOIN LATERAL (
            SELECT content FROM messages
            WHERE session_id = c.session_id AND role = 'user'
            ORDER BY created_at LIMIT 1
        ) first_msg ON true
        LEFT JOIN LATERAL (
            SELECT COUNT(*)::int AS cnt FROM messages
            WHERE session_id = c.session_id
        ) msg_count ON true
        WHERE c.user_id = $1::uuid
        ORDER BY c.updated_at DESC
        LIMIT $2
        """,
        user_id, limit,
    )
    return [
        {
            "session_id":                 r["session_id"],
            "created_at":                 r["created_at"].isoformat(),
            "updated_at":                 r["updated_at"].isoformat(),
            "first_user_message_preview": r["first_user_message_preview"],
            "message_count":              r["message_count"],
        }
        for r in rows
    ]
```

**`populate_stm_from_messages`**: nhận `list[dict]` với key `role/content` (format mới) — logic giữ nguyên, không cần sửa nếu caller truyền đúng format.

---

#### 1d. Cập nhật `api/main.py` — endpoint `/sessions/{id}/resume`

Thêm query param `limit` và truyền xuống `load_session_messages`:

```python
@application.post("/sessions/{session_id}/resume", response_model=SessionResumeResponse)
async def resume_session(session_id: str, user_id: str = Query(...), limit: int = 50):
    row = await load_session_messages(user_id=user_id, session_id=session_id, limit=limit)
    if not row:
        raise HTTPException(404, "Session not found")
    messages = row["messages"] or []
    await populate_stm_from_messages(session_id, messages)
    return SessionResumeResponse(
        session_id=session_id,
        messages=messages,
        stm_populated=True,
        last_updated=row["updated_at"].isoformat(),
    )
```

---

#### 1e. Thứ tự chạy

```bash
# 1. Áp dụng schema mới
psql $DATABASE_URL < agenticRAG/agentic_rag_gemini/langgraph_agents/db/migrations/001_normalize_messages.sql

# 2. Backfill data cũ
cd agenticRAG/agentic_rag_gemini
python -m langgraph_agents.db.migrations.migrate_messages

# 3. Verify
psql $DATABASE_URL -c "SELECT COUNT(*) FROM messages;"

# 4. Chạy test
pytest tests/langgraph_agents/ -m integration -v

# 5. Sau khi test xanh — drop cột cũ
psql $DATABASE_URL -c "ALTER TABLE conversations DROP COLUMN IF EXISTS messages;"
psql $DATABASE_URL -c "ALTER TABLE conversations DROP COLUMN IF EXISTS _migrated;"
```

---

### Task 2 — STM Token-based Sizing

**Tại sao**: Cứng 3 Q&A pairs không tối ưu — greeting query ngắn không cần 3 pairs, clinical query dài có thể bị cut-off. Dùng token budget cho phép include nhiều context khi có room, ít context khi query ngắn.

**File cần sửa**: `nodes/memory.py`

Thêm helper sau phần `_STM_MAX = 3`:

```python
_STM_TOKEN_BUDGET = 1500   # default tokens dành cho STM context
                            # rough estimate: 1 token ≈ 4 ký tự


def _select_stm_pairs(pairs: list[dict], budget: int = _STM_TOKEN_BUDGET) -> list[dict]:
    """Include Q&A pairs từ mới nhất ngược về cũ nhất cho đến khi đạt token budget.

    Estimate token count bằng len(text) // 4 — đủ chính xác cho mục đích giới hạn
    context, không cần tokenizer thực sự.
    """
    selected = []
    used = 0
    for pair in reversed(pairs):   # newest first
        cost = (len(pair.get("q", "")) + len(pair.get("a", ""))) // 4
        if used + cost > budget:
            break
        selected.insert(0, pair)
        used += cost
    return selected
```

Sửa `_read_stm` để dùng `_select_stm_pairs` thay vì `[-_STM_MAX:]`:

```python
async def _read_stm(session_id: str) -> list[dict]:
    try:
        import redis.asyncio as aioredis
        r = aioredis.from_url(_REDIS_URL, decode_responses=True, socket_connect_timeout=1)
        try:
            raw = await r.get(_STM_KEY.format(session_id=session_id))
            if not raw:
                return []
            all_pairs = json.loads(raw)
            return _select_stm_pairs(all_pairs)   # ← thay [-_STM_MAX:]
        finally:
            close_fn = getattr(r, "aclose", None) or r.close
            await close_fn()
    except Exception:
        return []
```

Redis vẫn lưu `_STM_MAX = 3` pairs tối đa (FIFO write không đổi). Chỉ thay đổi bao nhiêu pairs được include vào context khi đọc.

---

### Task 3 — Lock CORS Origins

**Tại sao**: `allow_origins=["*"]` cho phép bất kỳ domain nào gọi `/chat`. Phải lock xuống trước khi expose ra internet.

**File**: `api/main.py`

Thay:
```python
allow_origins=["*"],
```

Bằng:
```python
_ALLOWED_ORIGINS = [
    o.strip()
    for o in os.getenv("ALLOWED_ORIGINS", "http://localhost:3000,http://localhost:8080").split(",")
    if o.strip()
]
# ...
allow_origins=_ALLOWED_ORIGINS,
```

Thêm vào `.env`:
```
ALLOWED_ORIGINS=http://localhost:3000,http://localhost:8080
```

Phase 7: đổi thành domain thực tế của production.

---

### Task 4 — Log File Rotation

**Tại sao**: `configure_root_logger` hiện chỉ ghi ra `stderr` (StreamHandler). Trên production server, log cần được ghi ra file có rotation để tránh disk full. Hiện có `eca.log` ở root nhưng không có rotation.

**File**: `shared/logging.py`

Sửa `configure_root_logger`:

```python
import logging.handlers
import os

def configure_root_logger(level: str = "INFO") -> None:
    root = logging.getLogger()
    root.setLevel(level)
    for h in root.handlers[:]:
        root.removeHandler(h)

    formatter = JsonFormatter()

    # Console handler (luôn có)
    stream_handler = logging.StreamHandler()
    stream_handler.setFormatter(formatter)
    root.addHandler(stream_handler)

    # File handler (chỉ khi LOG_FILE set)
    log_file = os.getenv("LOG_FILE")
    if log_file:
        file_handler = logging.handlers.RotatingFileHandler(
            log_file,
            maxBytes=10 * 1024 * 1024,   # 10 MB per file
            backupCount=5,               # giữ 5 file cũ: eca.log.1 … eca.log.5
            encoding="utf-8",
        )
        file_handler.setFormatter(formatter)
        root.addHandler(file_handler)
```

Thêm vào `.env`:
```
LOG_FILE=eca.log
```

---

### Task 5 — TTS Audio Cleanup Cron

**Tại sao**: VieNeu TTS ghi audio file ra disk sau mỗi request, không có cleanup. Server disk đầy theo thời gian.

**Tạo file mới**: `scripts/cleanup_tts_audio.py`

```python
"""Cleanup VieNeu TTS audio files older than TTL.

Usage:
    python scripts/cleanup_tts_audio.py               # delete files older than 1 hour (default)
    python scripts/cleanup_tts_audio.py --ttl 3600    # explicit TTL in seconds
    python scripts/cleanup_tts_audio.py --dry-run     # preview only

Chạy định kỳ bằng Task Scheduler (Windows) hoặc cron (Linux):
    # Linux cron — chạy mỗi 30 phút
    */30 * * * * /path/to/venv/bin/python /path/to/scripts/cleanup_tts_audio.py

    # Windows Task Scheduler: trigger = every 30 min, action = python cleanup_tts_audio.py
"""
import argparse
import os
import time
from pathlib import Path


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--ttl",     type=int, default=3600, help="Max age in seconds (default 3600)")
    parser.add_argument("--dir",     type=str, default=None, help="Audio dir (default: auto-detect)")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    # Auto-detect VieNeu audio dir relative to this script
    if args.dir:
        audio_dir = Path(args.dir)
    else:
        script_dir = Path(__file__).resolve().parent
        audio_dir  = script_dir.parent / "agenticRAG" / "agentic_rag_gemini" / \
                     "langgraph_agents" / "services" / "vieneu_tts" / "outputs"

    if not audio_dir.exists():
        print(f"Audio dir not found: {audio_dir}")
        return

    cutoff = time.time() - args.ttl
    deleted = 0
    freed   = 0

    for f in audio_dir.glob("*.wav"):
        if f.stat().st_mtime < cutoff:
            size = f.stat().st_size
            if args.dry_run:
                print(f"[dry-run] Would delete: {f.name} ({size // 1024} KB)")
            else:
                f.unlink()
                deleted += 1
                freed   += size

    if args.dry_run:
        print("Dry run complete.")
    else:
        print(f"Deleted {deleted} file(s), freed {freed // 1024 // 1024:.1f} MB")


if __name__ == "__main__":
    main()
```

---

### Task 6 — SpeechLLm + SearXNG Health Checks

**Tại sao**: `/health/detailed` hiện không check VieNeu TTS (:5000) và SearXNG (:6666). Khi 2 services này down, backend logs không có cảnh báo sớm.

**File**: `api/health.py`

Thêm 2 hàm sau `check_mcp`:

```python
async def check_speechllm(timeout: float = 2.0) -> CheckResult:
    """Check VieNeu TTS service at VIENEU_URL/health."""
    import os
    base = os.getenv("VIENEU_URL", "http://localhost:5000").rstrip("/")
    t0 = time.perf_counter()
    try:
        import httpx
        async with httpx.AsyncClient() as client:
            res = await asyncio.wait_for(
                client.get(f"{base}/health"),
                timeout=timeout,
            )
        elapsed_ms = round((time.perf_counter() - t0) * 1000, 1)
        ok = res.status_code == 200
        return CheckResult(name="speechllm", ok=ok, latency_ms=elapsed_ms,
                           detail=None if ok else f"HTTP {res.status_code}")
    except Exception as exc:
        elapsed_ms = round((time.perf_counter() - t0) * 1000, 1)
        return CheckResult(name="speechllm", ok=False, latency_ms=elapsed_ms, detail=str(exc))


async def check_searxng(timeout: float = 2.0) -> CheckResult:
    """Check SearXNG at SEARXNG_URL/healthz (returns HTML 200 when up)."""
    import os
    base = os.getenv("SEARXNG_URL", "http://localhost:6666").rstrip("/")
    t0 = time.perf_counter()
    try:
        import httpx
        async with httpx.AsyncClient() as client:
            res = await asyncio.wait_for(
                client.get(f"{base}/healthz"),
                timeout=timeout,
            )
        elapsed_ms = round((time.perf_counter() - t0) * 1000, 1)
        ok = res.status_code == 200
        return CheckResult(name="searxng", ok=ok, latency_ms=elapsed_ms,
                           detail=None if ok else f"HTTP {res.status_code}")
    except Exception as exc:
        elapsed_ms = round((time.perf_counter() - t0) * 1000, 1)
        return CheckResult(name="searxng", ok=False, latency_ms=elapsed_ms, detail=str(exc))
```

Sửa `run_all_checks` để include 2 checks mới:

```python
async def run_all_checks(graph, redis_client) -> dict:
    results: list[CheckResult] = await asyncio.gather(
        check_redis(redis_client),
        check_postgres(),
        check_graph(graph),
        check_llm(),
        check_mcp(),
        check_speechllm(),   # ← thêm
        check_searxng(),     # ← thêm
        return_exceptions=True,
    )
    # ... rest giữ nguyên
```

---

### Task 7 — Stop Generation (Disconnect Detection)

**Tại sao**: Khi user đóng tab hoặc nhấn Stop, graph vẫn chạy hết. Lãng phí LLM tokens + CPU. Cần detect client disconnect và cancel graph.

**File**: `api/main.py`

Sửa `_stream_chat` signature để nhận `request`:

```python
# Trong create_app(), sửa route:
@application.post("/chat")
async def chat(req: ChatRequest, request: Request, background_tasks: BackgroundTasks):
    # ...
    async def event_generator():
        with with_request_id(request_id):
            async for sse_event in _stream_chat(req, request_id, config, state, background_tasks, request):
                yield sse_event
    return stream_response(event_generator())
```

Sửa `_stream_chat`:

```python
async def _stream_chat(req, request_id, config, state, background_tasks, request=None):
    # ...
    async for mode, payload in graph.astream(
        state, config, stream_mode=["updates", "custom"]
    ):
        # Disconnect check — mỗi chunk
        if request is not None and await request.is_disconnected():
            logger.info("client_disconnected", extra={"request_id": request_id})
            return

        # ... rest của loop giữ nguyên
```

Thêm import ở đầu file:
```python
from fastapi import FastAPI, HTTPException, BackgroundTasks, Query, Request
```

---

### Task 8 — YouTube Ingestion Pipeline

**Tại sao**: Retriever hiện chỉ search knowledge base nội bộ (document, humanml3d). Cần support YouTube transcript để PT assistant có thể retrieve bài tập từ video.

**Flow**: URL YouTube → `youtube-transcript-api` extract transcript → chunk → embed → lưu pgvector `source_type="youtube"` + metadata JSONB `{youtube_id, title, channel_url, chunk_index}`.

Thêm dependency vào `requirements-langgraph.txt`:
```
youtube-transcript-api>=0.6.0
```

**Tạo file mới**: `tools/youtube_ingest.py`

```python
"""YouTube transcript ingestion → pgvector.

Usage:
    python -m langgraph_agents.tools.youtube_ingest \
        --url "https://www.youtube.com/watch?v=VIDEO_ID" \
        --title "Bài tập đau lưng"
"""
import argparse
import asyncio
import re
from urllib.parse import urlparse, parse_qs

from youtube_transcript_api import YouTubeTranscriptApi

from langgraph_agents.shared import get_pg_client, get_embedding_service
from langgraph_agents.db.vector_backend import VectorBackend


_CHUNK_SIZE   = 500   # ký tự per chunk
_CHUNK_OVERLAP = 50


def _extract_video_id(url: str) -> str:
    """Extract video ID từ youtube.com/watch?v= hoặc youtu.be/ URLs."""
    parsed = urlparse(url)
    if parsed.netloc in ("youtu.be",):
        return parsed.path.lstrip("/")
    qs = parse_qs(parsed.query)
    ids = qs.get("v", [])
    if not ids:
        raise ValueError(f"Cannot extract video ID from: {url}")
    return ids[0]


def _chunk_text(text: str, size: int = _CHUNK_SIZE, overlap: int = _CHUNK_OVERLAP) -> list[str]:
    chunks = []
    start = 0
    while start < len(text):
        end = min(start + size, len(text))
        chunks.append(text[start:end])
        start += size - overlap
    return chunks


async def ingest_youtube(url: str, title: str = "", channel_url: str = "") -> int:
    """Ingest YouTube video transcript vào pgvector. Returns số chunks đã insert."""
    video_id = _extract_video_id(url)

    # Fetch transcript (try Vietnamese first, fallback to English)
    try:
        transcript = YouTubeTranscriptApi.get_transcript(video_id, languages=["vi", "en"])
    except Exception as exc:
        raise RuntimeError(f"Cannot fetch transcript for {video_id}: {exc}") from exc

    full_text = " ".join(entry["text"] for entry in transcript)
    chunks    = _chunk_text(full_text)

    pg  = get_pg_client()
    svc = get_embedding_service()
    vb  = VectorBackend(pg)

    inserted = 0
    for idx, chunk in enumerate(chunks):
        embedding = await asyncio.to_thread(svc.embed_texts, chunk)
        if isinstance(embedding, list) and embedding and isinstance(embedding[0], list):
            embedding = embedding[0]

        metadata = {
            "youtube_id":   video_id,
            "title":        title or video_id,
            "channel_url":  channel_url,
            "chunk_index":  idx,
            "total_chunks": len(chunks),
        }
        await vb.insert(
            content=chunk,
            embedding=embedding,
            source_type="youtube",
            source_id=video_id,
            metadata=metadata,
        )
        inserted += 1

    return inserted


async def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--url",     required=True)
    parser.add_argument("--title",   default="")
    parser.add_argument("--channel", default="")
    args = parser.parse_args()

    n = await ingest_youtube(args.url, title=args.title, channel_url=args.channel)
    print(f"Inserted {n} chunks for {args.url}")


if __name__ == "__main__":
    asyncio.run(main())
```

**Retriever không cần sửa** — `pgvector_search` tool tự search `source_type="document"` by default. Nếu muốn search cả YouTube, planner cần thêm `source_type="youtube"` vào `search_strategy`. Đây là Phase 6.11+ work, không block 6.10.

**`pgvector_tool.py`**: thêm `source_type` vào docstring để LLM biết có option `"youtube"`:

```python
# Sửa docstring của pgvector_search:
"""Search internal medical knowledge base for exercises, treatments, and PT theory.

Use for knowledge_query and exercise_recommendation intents.
Returns documents ranked by cosine similarity (highest first).

Args:
    query: Semantic search query (use expanded_query from planner)
    top_k: Number of results to return (default 5)
    source_type: One of "document", "humanml3d", "youtube" (default "document")
"""
```

---

## Thứ tự thực hiện đề xuất

Làm theo thứ tự này để tránh dependency issue:

```
Task 3  — Lock CORS (1-line change, test ngay)
Task 4  — Log rotation (1-function change)
Task 7  — Stop generation (thêm Request param)
Task 6  — Health checks SpeechLLm + SearXNG
Task 5  — TTS audio cleanup script
Task 2  — STM token-based sizing
Task 1  — DB normalization (task lớn nhất, cần migration, làm cuối)
Task 8  — YouTube ingestion (standalone, làm sau Task 1 xong)
```

---

## Test checklist

```bash
# Sau mỗi task:
pytest tests/langgraph_agents/ -x -q

# Sau Task 1 (DB migration):
pytest tests/langgraph_agents/ -m integration -v

# Full suite trước khi đánh dấu Phase 6.10 done:
pytest tests/ -v --tb=short
```

Expected: tất cả test xanh. Task 1 có thể cần update `test_phase5_sse.py` nếu mock session_store dùng JSONB messages.

---

## .env additions (tổng hợp)

```bash
# Task 3
ALLOWED_ORIGINS=http://localhost:3000,http://localhost:8080

# Task 4
LOG_FILE=eca.log

# Task 6
VIENEU_URL=http://localhost:5000
SEARXNG_URL=http://localhost:6666
```

---

---

## Corrections (28/05 — sau khi N implement xong)

Ba thay đổi dưới đây là corrections từ review architecture session. N cần update lại phần Task 1 + resume endpoint.

---

### Correction 1 — `messages` table: bỏ `metadata JSONB`, dùng cột riêng

**Lý do**: `intent`, `tokens`, `grader_result` là 3 field cố định luôn có ở assistant rows — đây là trường hợp dùng cột riêng, không phải JSONB. JSONB hợp lý khi schema heterogeneous (document metadata). NULL cho user rows hoàn toàn bình thường.

**Sửa `db/migrations/001_normalize_messages.sql`** — thay `metadata JSONB`:

```sql
CREATE TABLE IF NOT EXISTS messages (
    id             UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    session_id     UUID NOT NULL REFERENCES conversations(session_id) ON DELETE CASCADE,
    role           TEXT NOT NULL CHECK (role IN ('user', 'assistant')),
    content        TEXT NOT NULL,
    intent         TEXT,          -- assistant only, NULL for user rows
    tokens         INT,           -- assistant only
    grader_result  TEXT,          -- assistant only: 'pass' | 'pass_with_warning' | 'retry'
    created_at     TIMESTAMPTZ DEFAULT now()
);
CREATE INDEX IF NOT EXISTS idx_messages_session_created
    ON messages (session_id, created_at);
```

**Sửa `db/migrations/migrate_messages.py`** — backfill insert:

```python
records = [
    (
        row["session_id"],
        m["role"],
        m.get("content", ""),
        None,   # intent — không có trong JSONB cũ, để NULL
        None,   # tokens — không có trong JSONB cũ, để NULL
        None,   # grader_result — không có trong JSONB cũ, để NULL
        m.get("timestamp"),
    )
    for m in msgs
    if m.get("role") in ("user", "assistant") and m.get("content")
]

await conn.executemany(
    """INSERT INTO messages (session_id, role, content, intent, tokens, grader_result, created_at)
       VALUES ($1::uuid, $2, $3, $4, $5, $6, COALESCE($7::timestamptz, now()))
       ON CONFLICT DO NOTHING""",
    records,
)
```

**Sửa `db/session_store.py` — `write_session_turn`**: thêm params + dùng cột riêng:

```python
async def write_session_turn(
    user_id: str,
    session_id: str,
    user_query: str,
    assistant_answer: str,
    intent: str,
    tokens: int,
    grader_result: str = "pass",
) -> None:
    # ...
    await pg.executemany(
        """INSERT INTO messages (session_id, role, content, intent, tokens, grader_result, created_at)
           VALUES ($1::uuid, $2, $3, $4, $5, $6, $7::timestamptz)""",
        [
            (session_id, "user",      user_query,       None,   None,   None,          ts),
            (session_id, "assistant", assistant_answer, intent, tokens, grader_result, ts),
        ],
    )
```

**Sửa `db/session_store.py` — `load_session_messages`**: thay `metadata` bằng 3 cột riêng:

```python
rows = await pg.fetch(
    """SELECT role, content, intent, tokens, grader_result, created_at
       FROM messages WHERE session_id = $1::uuid
       ORDER BY created_at DESC LIMIT $2""",
    session_id, limit,
)

messages = [
    {
        "role":          r["role"],
        "content":       r["content"],
        "intent":        r["intent"],
        "tokens":        r["tokens"],
        "grader_result": r["grader_result"],
        "timestamp":     r["created_at"].isoformat(),
    }
    for r in reversed(rows)
]
```

**Xóa** `_coerce_metadata` helper — không còn dùng.

**Caller `api/main.py`**: truyền thêm `grader_result` vào `write_session_turn`. Đọc từ `final_state.get("grader_result", "pass")`.

---

### Correction 2 — `POST /sessions/{id}/resume` → `GET /sessions/{id}`

**Lý do**: Resume endpoint hiện có side effect (populate Redis STM) — vi phạm HTTP semantics của GET. Side effect đó thuộc về `/chat` flow, không phải resume.

**Sửa `api/main.py`**: đổi route + bỏ STM population:

```python
# Trước:
@application.post("/sessions/{session_id}/resume", ...)
async def resume_session(session_id: str, user_id: str = Query(...), limit: int = 50):
    row = await load_session_messages(...)
    await populate_stm_from_messages(session_id, messages)  # ← bỏ dòng này
    ...

# Sau:
@application.get("/sessions/{session_id}", response_model=SessionResumeResponse)
async def get_session(session_id: str, user_id: str = Query(...), limit: int = 50):
    row = await load_session_messages(user_id=user_id, session_id=session_id, limit=limit)
    if not row:
        raise HTTPException(404, "Session not found")
    return SessionResumeResponse(
        session_id=session_id,
        messages=row["messages"] or [],
        stm_populated=False,   # STM sẽ populate lazy tại /chat
        last_updated=row["updated_at"].isoformat(),
    )
```

**test-ui `app.js`**: đổi `resumeSession()` gọi `GET /sessions/{id}?user_id=...` thay vì POST.

---

### Correction 3 — STM populate lazy tại `/chat`

**Lý do**: STM populate là prerequisite của graph, không phải feature của resume. Lazy populate tại `/chat` giữ logic ở chỗ đúng, không làm resume phức tạp hơn cần thiết.

**Sửa `api/main.py` — đầu hàm `chat()`**, trước khi invoke graph:

```python
@application.post("/chat")
async def chat(req: ChatRequest, request: Request, background_tasks: BackgroundTasks):
    # ...

    # Lazy STM populate: nếu STM rỗng (session mới hoặc sau resume), load từ PostgreSQL
    stm_key = f"stm:{req.session_id}"
    stm_raw = await _get_redis().get(stm_key)
    if not stm_raw:
        recent = await load_session_messages(
            user_id=req.user_id,
            session_id=req.session_id,
            limit=6,   # 3 Q&A pairs = 6 messages
        )
        if recent and recent["messages"]:
            await populate_stm_from_messages(req.session_id, recent["messages"])

    # invoke graph như bình thường ...
```

`populate_stm_from_messages` không cần thay đổi gì.

---

## Definition of Done

Phase 6.10 complete khi:

- [ ] Task 1–8 implement xong
- [ ] `pytest tests/` xanh 100%
- [ ] `GET /health/detailed` trả `"status": "ready"` với tất cả checks green (kể cả speechllm + searxng)
- [ ] CORS không còn `allow_origins=["*"]`
- [ ] Log file rotation hoạt động (kiểm tra `eca.log.1` xuất hiện sau khi log đạt 10MB)
- [ ] `scripts/cleanup_tts_audio.py --dry-run` chạy không lỗi
- [ ] Chat SSE stream hoạt động với Stop button
- [ ] `test-ui` Session tab list/resume/delete hoạt động với schema mới
