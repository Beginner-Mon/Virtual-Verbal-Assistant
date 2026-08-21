"""Memory node — M.5 (in-session) + M.6 Tier 1 (user_memory facts).

Runs FIRST before planner (D9: coreference resolve needs memory context).
Assembles context window per M.7 layout:

    [user_memory facts (Tier 1)] [summary chunks (frozen)]   ← STATIC (prefix cache)
    ─────────────────────────────────────────────────────────
    [recent raw (Redis/DB)] [current query]                  ← DYNAMIC

Output: prepends SystemMessage + recent chat history into messages.
Does NOT write to a separate "memory_context" field (removed — D20).

M.5 rules:
  - PostgreSQL = source of truth; Redis = write-through cache of recent raw only
  - Summary: chunk-based, frozen on write, CAS idempotent
  - Token threshold: 10k (D13) — single threshold, recent-raw window = chunk_size
  - Edge A: summarize fail → retry + hard cap 2× threshold → fallback
"""

from __future__ import annotations

import asyncio
import json
import time
from pathlib import Path
from typing import Optional

from langchain_core.messages import SystemMessage, HumanMessage, AIMessage
from langchain_core.runnables import RunnableConfig

from langgraph_agents.state import AgentState, ErrorSeverity
from langgraph_agents.shared import get_pg_client
from langgraph_agents.shared.logging import get_logger
from langgraph_agents.shared.stm import get_stm

logger = get_logger("langgraph.memory")


# ── Config ────────────────────────────────────────────────────────────────

def _load_memory_config() -> dict:
    import yaml
    config_path = Path(__file__).resolve().parents[3] / "config" / "langgraph.yaml"
    if config_path.exists():
        with open(config_path, "r", encoding="utf-8") as f:
            return yaml.safe_load(f).get("langgraph", {}).get("memory", {})
    return {}


_MEM_CFG = _load_memory_config()
# _REDIS_URL and _RECENT_RAW_KEY were here. Both moved to shared/stm.py on
# 21-08, which owns the key format and the connection for all four call sites.
#
# The config value they read (`langgraph.memory.redis_url` in
# config/langgraph.yaml) is no longer consulted: a URL in a committed YAML file
# cannot differ per environment, which is the whole requirement once this runs
# on Lambda. The store reads REDIS_URL from the environment instead.
_RECENT_RAW_MAX = 20          # max recent messages to load from cache/DB
_SUMMARY_TOKEN_THRESHOLD = 10_000   # D13: single threshold for summarize trigger
_STM_TOKEN_BUDGET = 1500      # max tokens for recent raw in context window


# ── Tier 1: user_memory facts (always-on, cheap, no vector search) ────────

async def _load_user_facts(user_id: str) -> list[str]:
    """Load Tier 1 facts from user_memory table. Always injected into system prompt."""
    try:
        pg = get_pg_client()
        await pg.connect()
        rows = await pg.fetch(
            "SELECT fact_text FROM user_memory WHERE user_id = $1 AND valid = true",
            user_id,
        )
        return [r["fact_text"] for r in rows]
    except Exception as exc:
        logger.warning("user_facts_load_failed", extra={"error": str(exc)})
        return []


# ── Summary chunks (frozen, append-only, from summaries table) ────────────

async def _load_summary_chunks(session_id: str) -> list[str]:
    """Load active summary chunks for session, ordered by covers_from_seq."""
    try:
        pg = get_pg_client()
        await pg.connect()
        rows = await pg.fetch(
            """
            SELECT summary_text FROM summaries
            WHERE session_id = $1 AND status = 'active'
            ORDER BY covers_from_seq
            """,
            session_id,
        )
        return [r["summary_text"] for r in rows]
    except Exception as exc:
        logger.warning("summary_chunks_load_failed", extra={"error": str(exc)})
        return []


# ── Recent raw (Redis cache or DB fallback) ───────────────────────────────

def _token_estimate(text: str) -> int:
    """Rough token count: ~4 chars per token for Vietnamese/English."""
    return len(text) // 4


async def _load_recent_raw_cache(session_id: str) -> Optional[list[dict]]:
    """Load recent raw messages from the STM cache. None means "no cache entry".

    Adapts writer format [{q, a, ts}] to [{role, content}] for downstream.

    Named `_load_recent_raw_redis` until 21-08, when the backend stopped being
    Redis-specific — see shared/stm.py. Returning None for both "empty" and
    "backend unreachable" is intentional and predates this change: the caller
    falls back to PostgreSQL either way, and the store logs the outage once
    rather than leaving this function to decide what an outage means.
    """
    data = await get_stm().get(session_id)
    if not data:
        return None
    return _normalize_redis_format(data)


def _normalize_redis_format(data: list[dict]) -> list[dict]:
    """Convert Redis STM format to [{role, content}].

    Writer (session_store) writes [{q, a, ts}] pairs.
    Reader expects [{role, content}]. Accept both for forward-compat.
    """
    if not data:
        return []
    # Already in reader format [{role, content}] — pass through
    if "role" in data[0]:
        return data
    # Writer format [{q, a, ts}] — convert
    result = []
    for pair in data:
        if "q" in pair:
            result.append({"role": "user", "content": pair["q"]})
        if "a" in pair:
            result.append({"role": "assistant", "content": pair["a"]})
    return result


async def _load_recent_raw_db(session_id: str, limit: int = _RECENT_RAW_MAX) -> list[dict]:
    """Fallback: load recent messages from PostgreSQL."""
    try:
        pg = get_pg_client()
        await pg.connect()
        rows = await pg.fetch(
            """
            SELECT role, content FROM messages
            WHERE session_id = $1
            ORDER BY seq_id DESC
            LIMIT $2
            """,
            session_id,
            limit,
        )
        # Reverse to chronological order (oldest first)
        return [{"role": r["role"], "content": r["content"]} for r in reversed(rows)]
    except Exception as exc:
        logger.warning("recent_raw_db_failed", extra={"error": str(exc)})
        return []


def _select_recent_raw(pairs: list[dict], budget: int = _STM_TOKEN_BUDGET) -> list[dict]:
    """Select recent messages from newest to oldest until token budget reached."""
    selected = []
    used = 0
    for msg in reversed(pairs):
        cost = _token_estimate(msg.get("content", ""))
        if used + cost > budget:
            break
        selected.insert(0, msg)
        used += cost
    return selected


async def _load_recent_raw(session_id: str) -> list[dict]:
    """Load recent raw messages: cache first, DB fallback. Token-budgeted."""
    cached = await _load_recent_raw_cache(session_id)
    if cached:
        return _select_recent_raw(cached)
    db_data = await _load_recent_raw_db(session_id)
    return _select_recent_raw(db_data)


# ── Context assembly ──────────────────────────────────────────────────────

def _assemble_system_message(
    user_facts: list[str],
    summary_chunks: list[str],
) -> Optional[SystemMessage]:
    """Build static-context SystemMessage per M.7 layout.

    Static part (prefix-cacheable): user_facts + summary chunks.
    Returns None if both are empty (no context to inject).
    """
    parts = []

    if user_facts:
        facts_block = "[USER FACTS]\n" + "\n".join(f"- {f}" for f in user_facts)
        parts.append(facts_block)

    if summary_chunks:
        summaries_block = "[SESSION HISTORY]\n" + "\n---\n".join(summary_chunks)
        parts.append(summaries_block)

    if not parts:
        return None

    return SystemMessage(content="\n\n".join(parts))


def _recent_raw_to_messages(recent: list[dict]) -> list:
    """Convert recent raw dicts to LangChain messages."""
    msgs = []
    for item in recent:
        role = item.get("role", "user")
        content = item.get("content", "")
        if role == "user":
            msgs.append(HumanMessage(content=content))
        elif role == "assistant":
            msgs.append(AIMessage(content=content))
    return msgs


# ── Node ──────────────────────────────────────────────────────────────────

async def memory_node(state: AgentState, config: RunnableConfig) -> dict:
    """Memory node — runs FIRST (before planner, D9).

    Loads user_facts + summary chunks + recent raw.
    Prepends context as SystemMessage + recent chat history into messages.

    No separate "memory_context" dict — context flows through messages (D20).
    """
    t0 = time.perf_counter()
    request_id = config["configurable"].get("request_id", "-")
    user_id = config["configurable"]["user_id"]
    session_id = config["configurable"]["session_id"]

    logger.info("node_start", extra={
        "node": "memory", "request_id": request_id, "session_id": session_id,
    })

    # All three loads are independent — run concurrently
    user_facts, summary_chunks, recent_raw = await asyncio.gather(
        _load_user_facts(user_id),
        _load_summary_chunks(session_id),
        _load_recent_raw(session_id),
    )

    # Build output messages
    out_messages = []

    sys_msg = _assemble_system_message(user_facts, summary_chunks)
    if sys_msg:
        out_messages.append(sys_msg)

    if recent_raw:
        out_messages.extend(_recent_raw_to_messages(recent_raw))

    elapsed_ms = round((time.perf_counter() - t0) * 1000)
    logger.info("node_complete", extra={
        "node": "memory", "request_id": request_id,
        "elapsed_ms": elapsed_ms,
        "user_facts_count": len(user_facts),
        "summary_chunks_count": len(summary_chunks),
        "recent_raw_count": len(recent_raw),
    })

    if out_messages:
        return {"messages": out_messages}
    return {}
