"""PostgreSQL session store — M.4 schema (REUPDATE_PLAN.md §M.4).

Coexists with memory/session_store.py (Firebase) for the old code path.
"""

from __future__ import annotations

import json
import uuid
from datetime import datetime, timedelta, timezone
from typing import Optional

from langgraph_agents.db.postgres import PostgresClient
from langgraph_agents.shared import get_pg_client
from langgraph_agents.shared.stm import get_stm

# Second copy of vva_motion.jobs.TTL_SECONDS, and it cannot be an import.
#
# This module is served by TWO deployments. The agent is a container image that
# COPYs vva_motion in (agenticRAG/Dockerfile:112), but the CRUD Lambda is a zip
# built from agenticRAG/langgraph_agents alone (infra/build_crud_api.py:44) —
# `from vva_motion.jobs import ...` at module scope there is a
# ModuleNotFoundError at cold start, which takes /sessions and /me/memory down
# with it. A lazy import only moves the crash to request time, on the very
# endpoint that needs the value.
#
# Same shape as MODEL in nodes/kimodo.py, which is copied for the same reason:
# the other definition lives on the far side of an image boundary. Unlike that
# one, this pair is pinned by a test —
# tests/langgraph_agents/test_motion_expiry_deadline.py asserts the two are equal,
# so drift fails CI instead of quietly mislabelling every restored motion.
MOTION_TTL_SECONDS = 24 * 3600


def _extras(row) -> dict:
    """`messages.extras` as a dict, whatever the driver handed back.

    asyncpg returns JSONB as a `str` unless a codec is registered; a test or
    another driver may hand back a dict already. NULL — the common case, since
    most messages have no extras at all — becomes `{}`.
    """
    raw = row["extras"] if "extras" in row.keys() else None
    if not raw:
        return {}
    return raw if isinstance(raw, dict) else json.loads(raw)


def _shape_message(row, created_at: Optional[datetime]) -> dict:
    """One history message as the API returns it.

    Storage is one JSONB column; the wire stays flat, because a client wants
    `motion_job_id`, not a shape that mirrors how it happens to be persisted.

    Motion keys appear ONLY when the message has a motion. It is an occasional
    extra, never part of a chat turn, so most rows have none — and a message
    with no motion cannot have an expired one. Emitting them unconditionally
    would put keys describing nothing on the large majority of every history
    payload, and assert something false about each.
    """
    out = {
        "role":      row["role"],
        "content":   row["content"],
        "tokens":    row["token_count"],
        "timestamp": created_at.isoformat() if created_at else None,
    }
    job_id = _extras(row).get("motion", {}).get("job_id")
    if job_id:
        out["motion_job_id"] = job_id
        out["motion_expires_at"] = motion_expires_at(created_at)
    return out


def motion_expires_at(created_at: Optional[datetime]) -> Optional[str]:
    """When this turn's rendered motion stops being fetchable. ISO-8601, UTC.

    A DEADLINE, NOT A VERDICT, and the difference is the whole point. "Has it
    expired" is a question whose answer changes while nobody is looking: a
    payload computed at 10:00 says `false`, and a tab left open until the next
    morning is still holding that `false` long after it stopped being true. An
    absolute instant never goes stale — the client compares it to its own clock
    at the moment it actually needs to decide.

    It is also why this is not simply left to the browser to work out from
    `timestamp`. Doing that puts the 24h in TypeScript as a second copy of a
    constant that already exists twice (see MOTION_TTL_SECONDS above), across a
    language boundary where nothing can pin them together. Sending the instant
    keeps the rule server-side and hands the client an answer it cannot get
    wrong.

    Why a deadline exists at all: `messages.extras` outlives what it points at.
    The job row has a 24h DynamoDB TTL and the .bvh a one-day S3 lifecycle
    rule, so a day after the turn every stored id is a dead pointer — and
    `GET /motion/{job_id}` cannot say so, because a swept row, an expired row
    and an id that never existed all answer 404 identically. The age of the
    message is the only surviving signal, and Postgres is the only place with
    it.

    The S3 rule is the binding clock, not the DynamoDB TTL: the file is what
    the browser fetches, lifecycle deletes it on schedule, and AWS only promises
    a TTL sweep "within a few days" — so the row can outlive the file it
    describes, never the reverse.

    Returns None when the age is unknown, which a client must read as "assume
    gone". The mistakes are not symmetric: treating a live motion as expired
    costs a replay the user can ask for again; treating a dead one as live
    costs a poll that ends in a message saying the render failed when it merely
    got old.
    """
    if created_at is None:
        return None
    # asyncpg returns tz-aware timestamps; a hand-built row or another driver
    # may not. A naive value read as local time shifts by the machine's offset,
    # which moves the deadline by hours.
    if created_at.tzinfo is None:
        created_at = created_at.replace(tzinfo=timezone.utc)
    return (created_at + timedelta(seconds=MOTION_TTL_SECONDS)).isoformat()

# _REDIS_URL used to live here, hardcoded to localhost, and was one of four
# copies of the same string. shared/stm.py owns it now — see that module for why
# the store is swappable and why DynamoDB's TTL needed an application-side check.
_STM_MAX = 3


def _to_uuid(value: str) -> str:
    """Coerce arbitrary user-supplied string into a deterministic UUID string.

    UI sends user_id as plain string (e.g. "anonymous", "user_123") for
    simplicity. DB schema requires UUID. Round-trip:
      - already valid UUID → returned unchanged
      - any other string → uuid5(NAMESPACE_DNS, value) → deterministic UUID
        for the same input string across runs (same user always → same UUID).
    """
    try:
        return str(uuid.UUID(value))
    except (ValueError, TypeError):
        return str(uuid.uuid5(uuid.NAMESPACE_DNS, str(value)))


class SessionStore:
    """Async session persistence backed by M.4 conversations + messages tables."""

    def __init__(self, pg: PostgresClient = None):
        self.pg = pg or PostgresClient()

    async def save_session(
        self,
        user_id: str,
        session_id: str,
        messages: list[dict],
        summary: str = None,
    ):
        """Insert or update a conversation + messages in M.4 schema."""
        await self.pg.connect()

        await self.pg.execute(
            "INSERT INTO users (id) VALUES ($1::uuid) ON CONFLICT (id) DO NOTHING",
            user_id,
        )
        await self.pg.execute(
            """INSERT INTO conversations (session_id, user_id, created_at, updated_at)
               VALUES ($1::uuid, $2::uuid, now(), now())
               ON CONFLICT (session_id) DO UPDATE SET updated_at = now()""",
            session_id, user_id,
        )

        if messages:
            for m in messages:
                role = m.get("role", "user")
                content = m.get("content", "")
                if role in ("user", "assistant") and content:
                    await self.pg.execute(
                        """INSERT INTO messages (session_id, role, content, created_at)
                           VALUES ($1::uuid, $2, $3, now())""",
                        session_id, role, content,
                    )

    async def load_session(self, user_id: str, session_id: str) -> dict | None:
        """Load a single session by user and session ID from M.4 schema."""
        row = await self.pg.fetchrow(
            "SELECT session_id, user_id, created_at, updated_at FROM conversations "
            "WHERE user_id = $1 AND session_id = $2",
            user_id, session_id,
        )
        if not row:
            return None
        msgs = await self.pg.fetch(
            "SELECT role, content, created_at FROM messages "
            "WHERE session_id = $1 ORDER BY created_at",
            session_id,
        )
        return {
            "session_id": str(row["session_id"]),
            "user_id": row["user_id"],
            "messages": [
                {"role": m["role"], "content": m["content"],
                 "timestamp": m["created_at"].isoformat()}
                for m in msgs
            ],
            "created_at": row["created_at"].isoformat(),
        }

    async def list_sessions(self, user_id: str, limit: int = 10) -> list[dict]:
        """List recent sessions for a user from M.4 schema."""
        rows = await self.pg.fetch(
            "SELECT session_id, title, created_at FROM conversations "
            "WHERE user_id = $1 ORDER BY created_at DESC LIMIT $2",
            user_id, limit,
        )
        return [
            {
                "session_id": str(row["session_id"]),
                "title": row["title"],
                "created_at": row["created_at"].isoformat(),
            }
            for row in rows
        ]


# ── Standalone async helpers (Phase 5) ──────────────────────────────


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


async def load_session_messages(
    user_id: str,
    session_id: str,
    limit: int = 50,
    before: str | None = None,
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

    # `seq_id DESC` is the tie-breaker, and it is not optional: a turn's user
    # message and its assistant reply are written in one batch and share an
    # identical `created_at` (verified: both 08:49:10.089752). Ordering on the
    # timestamp alone leaves the pair's order to the planner, and it really does
    # come back answer-before-question — which is exactly how the restored
    # transcript would read.
    if before:
        rows = await pg.fetch(
            """SELECT role, content, token_count, extras, created_at
               FROM messages
               WHERE session_id = $1::uuid AND created_at < $2::timestamptz
               ORDER BY created_at DESC, seq_id DESC LIMIT $3""",
            session_id, before, limit,
        )
        rows = list(reversed(rows))
    else:
        rows = await pg.fetch(
            """SELECT role, content, token_count, extras, created_at
               FROM messages
               WHERE session_id = $1::uuid
               ORDER BY created_at DESC, seq_id DESC LIMIT $2""",
            session_id, limit,
        )
        rows = list(reversed(rows))

    messages = [_shape_message(r, r["created_at"]) for r in rows]
    return {
        "session_id": session_id,
        "messages":   messages,
        "updated_at": header["updated_at"],
        "has_more":   len(rows) == limit,
        "next_cursor": rows[0]["created_at"].isoformat() if rows else None,
    }


async def populate_stm_from_messages(session_id: str, messages: list[dict]) -> None:
    """Pick last 3 Q&A pairs from full message log → write to Redis STM."""
    pairs = []
    pending_user = None
    for m in messages:
        if m["role"] == "user":
            pending_user = m["content"]
        elif m["role"] == "assistant" and pending_user:
            pairs.append({
                "q": pending_user,
                "a": m["content"],
                "ts": m.get("timestamp", ""),
            })
            pending_user = None

    stm = pairs[-_STM_MAX:]

    await get_stm().set(session_id, stm)


async def write_session_turn(
    user_id: str,
    session_id: str,
    user_query: str,
    assistant_answer: str,
    total_tokens: int = 0,
    grader_result: str = "pass",
    motion_job_id: str | None = None,
) -> None:
    """`motion_job_id` is stored inside the `extras` JSONB column, namespaced
    under "motion" — not as a column of its own. Motion is an occasional extra
    on a chat turn, and it is not the last one: TTS wants to record the language
    and voice that answered, and the next feature will want its own field. A
    column each would grow `messages` a tail of nullable columns belonging to
    unrelated subsystems, one migration at a time. See migration 008.

    The parameter stays flat because that is what the caller has.

    `motion_job_id` (R25): the Kimodo job id for this turn, if any. Only
    the `queued`/`cache_hit` states carry one — `busy`/`unavailable` pass
    None, same as a turn with no motion at all. Written on the assistant row
    only; the user row's motion_job_id is always NULL."""
    user_id = _to_uuid(user_id)
    pg = get_pg_client()
    await pg.connect()
    ts = datetime.now(timezone.utc).isoformat()

    await pg.execute(
        "INSERT INTO users (id) VALUES ($1::uuid) ON CONFLICT (id) DO NOTHING",
        user_id,
    )
    await pg.execute(
        """INSERT INTO conversations (session_id, user_id, created_at, updated_at)
           VALUES ($1::uuid, $2::uuid, now(), now())
           ON CONFLICT (session_id) DO UPDATE SET updated_at = now()""",
        session_id, user_id,
    )
    # created_at omitted → DB DEFAULT now() fills it. Ordering within a turn is
    # by seq_id (BIGSERIAL, insert order), not created_at. Passing an ISO string
    # for a timestamptz param fails under executemany() binary binding.
    await pg.executemany(
        """INSERT INTO messages (session_id, role, content, token_count, extras)
           VALUES ($1::uuid, $2, $3, $4, $5::jsonb)""",
        [
            (session_id, "user",      user_query,       None,         None),
            (session_id, "assistant", assistant_answer, total_tokens,
             json.dumps({"motion": {"job_id": motion_job_id}}) if motion_job_id else None),
        ],
    )
    await _append_stm(session_id, user_query, assistant_answer, ts)


async def _append_stm(session_id: str, q: str, a: str, ts: str) -> None:
    """Append one turn to the cache, trimmed to the last _STM_MAX pairs.

    Read-modify-write, and deliberately not atomic. Two turns of the SAME
    session racing here could lose one — but a session is one person typing, and
    the loser is a cache entry that PostgreSQL can rebuild. Buying atomicity
    would mean a Redis-only primitive, which is exactly what shared/stm.py exists
    to avoid depending on.

    No try/except: every method on the store already swallows and reports its own
    failures. Wrapping it again would only hide a bug in this function.
    """
    stm = await get_stm().get(session_id) or []
    stm.append({"q": q, "a": a, "ts": ts})
    await get_stm().set(session_id, stm[-_STM_MAX:])
