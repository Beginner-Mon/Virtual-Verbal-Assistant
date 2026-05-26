"""Memory node — STM (Redis, 3 Q&A FIFO) + Conditional LTM (pgvector).

STM: always reads last 3 Q&A pairs from Redis. Write happens at FastAPI layer.
LTM: only runs when recall keywords detected. PostgresSQL + pgvector.
"""

import asyncio
import json
import re
import time
from datetime import datetime, timezone
from pathlib import Path

from langgraph_agents.state import AgentState, ErrorSeverity
from langgraph_agents.shared import get_embedding_service, get_pg_client
from langgraph_agents.shared.logging import get_logger

logger = get_logger("langgraph.memory")


def _load_memory_config() -> dict:
    import yaml
    config_path = Path(__file__).resolve().parents[4] / "config" / "langgraph.yaml"
    if config_path.exists():
        with open(config_path, "r", encoding="utf-8") as f:
            return yaml.safe_load(f).get("langgraph", {}).get("memory", {})
    return {}


_MEM_CFG = _load_memory_config()
_REDIS_URL = _MEM_CFG.get("redis_url", "redis://localhost:6379")
_STM_KEY = "stm:{session_id}"
_STM_MAX = 3

# ── Recall keyword detection ───────────────────────────────────────────

_RECALL_PATTERNS = [
    re.compile(r"\bcon\s+nho\b", re.IGNORECASE),
    re.compile(r"\bnho(\s+lai)?\b", re.IGNORECASE),
    re.compile(r"\blan\s+truoc\b", re.IGNORECASE),
    re.compile(r"\btruoc\s+do\b", re.IGNORECASE),
    re.compile(r"\b(tuan|hom\s+qua|thang)\s+truoc\b", re.IGNORECASE),
    re.compile(r"\bhom\s+qua\b", re.IGNORECASE),
    re.compile(r"\bda\s+(noi|hoi|trao\s+doi|lam)\b", re.IGNORECASE),
    re.compile(r"\b(remember|last\s+time|previously)\b", re.IGNORECASE),
]


def _needs_recall(query: str) -> bool:
    return any(p.search(query) for p in _RECALL_PATTERNS)


# ── STM read (Redis, max 3 Q&A, read-only) ─────────────────────────────

async def _read_stm(session_id: str) -> list[dict]:
    """Returns list of up to 3 Q&A pairs, oldest first."""
    try:
        import redis.asyncio as aioredis
        r = aioredis.from_url(_REDIS_URL, decode_responses=True, socket_connect_timeout=1)
        try:
            raw = await r.get(_STM_KEY.format(session_id=session_id))
            if not raw:
                return []
            return json.loads(raw)[-_STM_MAX:]
        finally:
            # redis-py 5.x: aclose() preferred; close() emits DeprecationWarning
            close_fn = getattr(r, "aclose", None) or r.close
            await close_fn()
    except Exception:
        return []


# ── LTM lookup (PostgreSQL + pgvector, conditional) ────────────────────

async def _lookup_ltm(user_id: str, query: str) -> dict:
    """Run only if _needs_recall(query). Returns one of:
      {found: False}
      {found: True, results: [...]}        # 1 session matched
      {ambiguous: True, sessions: [...]}    # 2+ sessions
    """
    try:
        pg = get_pg_client()
        await pg.connect()

        # 1. Find candidate sessions by timestamp (last 14 days)
        rows = await pg.fetch(
            """
            SELECT session_id, summary, created_at
            FROM conversations
            WHERE user_id = $1
              AND created_at >= now() - interval '14 days'
            ORDER BY created_at DESC
            LIMIT 5
            """,
            user_id,
        )
    except Exception:
        return {"found": False}

    if not rows:
        return {"found": False}

    if len(rows) >= 2:
        return {
            "ambiguous": True,
            "sessions": [
                {"session_id": str(r["session_id"]), "summary": r["summary"],
                 "created_at": r["created_at"].isoformat()}
                for r in rows
            ],
        }

    # Exactly 1 session — pgvector search within that session
    target_session = rows[0]["session_id"]
    try:
        svc = get_embedding_service()
        embedding = await asyncio.to_thread(svc.embed_texts, query)

        from langgraph_agents.db.vector_backend import VectorBackend
        vb = VectorBackend(pg)
        results = await vb.search(
            query_embedding=embedding[0] if isinstance(embedding, list) and len(embedding) > 0 and isinstance(embedding[0], list) else embedding,
            top_k=5,
            source_type="conversation",
            source_id=str(target_session),
        )
        return {"found": True, "results": results}
    except Exception:
        return {"found": False}


# ── User profile ───────────────────────────────────────────────────────

async def _get_user_profile(user_id: str) -> dict:
    """Get user profile from PostgreSQL users table."""
    try:
        pg = get_pg_client()
        row = await pg.fetchrow("SELECT profile FROM users WHERE id = $1", user_id)
        return dict(row["profile"]) if row and row.get("profile") else {}
    except Exception:
        return {}


# ── Node ────────────────────────────────────────────────────────────────

async def memory_node(state: AgentState, config) -> dict:
    """STM (always) + LTM (conditional on recall keywords). All three reads run in parallel."""
    t0 = time.perf_counter()
    request_id = config["configurable"].get("request_id", "-")
    user_id = config["configurable"]["user_id"]
    session_id = config["configurable"]["session_id"]
    query = config["configurable"]["query"]

    needs_recall = _needs_recall(query)
    logger.info("node_start", extra={
        "node": "memory", "request_id": request_id,
        "needs_recall": needs_recall, "session_id": session_id,
    })

    # All independent — gather concurrently.
    stm_task = _read_stm(session_id)
    profile_task = _get_user_profile(user_id)

    if needs_recall:
        ltm_task = _lookup_ltm(user_id, query)
    else:
        async def _skipped() -> dict:
            return {"found": False, "skipped": True}
        ltm_task = _skipped()

    short_term, long_term, user_profile = await asyncio.gather(
        stm_task, ltm_task, profile_task,
        return_exceptions=False,
    )

    elapsed_ms = round((time.perf_counter() - t0) * 1000)
    logger.info("node_complete", extra={
        "node": "memory", "request_id": request_id,
        "elapsed_ms": elapsed_ms,
        "stm_items": len(short_term) if isinstance(short_term, list) else 0,
        "ltm_found": bool(long_term.get("found")) if isinstance(long_term, dict) else False,
    })

    return {
        "memory_context": {
            "short_term": short_term,
            "long_term": long_term,
            "user_profile": user_profile,
        }
    }
