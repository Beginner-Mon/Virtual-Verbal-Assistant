"""CRUD routes — sessions and user memory. No graph, no LLM, no embeddings.

These are the routes that only ever touch PostgreSQL, split out so they can be
served by a small Lambda instead of a process that imports torch at boot. They
are defined once, here, and mounted by both apps:

    api/crud_app.py  → the Lambda; these routes and nothing else
    api/main.py      → the agent; these routes plus /chat and friends

Defining them once is the point. The previous attempt at this split hand-wrote
parallel handlers under infra/lambda/, and they drifted from the originals
before ever running: infra/lambda/resume_session/handler.py orders messages by
created_at with no seq_id tie-breaker, which is exactly the bug that
db/session_store.py carries a comment warning about (both rows of a turn are
written in one executemany and share a timestamp, so the reply sorts ahead of
the question).

Identity is never taken from the request. Every route below gets its user from
Depends(current_user_id), which reads it from the verified token — see
api/auth.py. There is deliberately no user_id path or query parameter to trust.
"""

from __future__ import annotations

from fastapi import APIRouter, Depends, HTTPException

from langgraph_agents.api.auth import current_user_id
from langgraph_agents.api.schemas import (
    SessionListItem, SessionListResponse, SessionResumeResponse,
    UserMemoryCreate, UserMemoryItem, UserMemoryListResponse,
)
from langgraph_agents.db.session_store import (
    list_user_sessions, load_session_messages,
)
from langgraph_agents.shared import get_pg_client
from langgraph_agents.shared.logging import get_logger

logger = get_logger("langgraph.api.crud")

router = APIRouter(tags=["crud"])


# ── Sessions ───────────────────────────────────────────────────────────────────


@router.get("/sessions", response_model=SessionListResponse)
async def list_sessions(uid: str = Depends(current_user_id), limit: int = 50):
    rows = await list_user_sessions(user_id=uid, limit=limit)
    return SessionListResponse(
        sessions=[SessionListItem(**r) for r in rows],
        total=len(rows),
    )


@router.get("/sessions/{session_id}", response_model=SessionResumeResponse)
async def get_session(
    session_id: str,
    uid: str = Depends(current_user_id),
    limit: int = 50,
):
    row = await load_session_messages(user_id=uid, session_id=session_id, limit=limit)
    if not row:
        raise HTTPException(404, "Session not found")
    return SessionResumeResponse(
        session_id=session_id,
        messages=row["messages"] or [],
        stm_populated=False,
        last_updated=row["updated_at"].isoformat(),
    )


@router.delete("/sessions/{session_id}")
async def delete_session(session_id: str, uid: str = Depends(current_user_id)):
    """Delete a session. Messages follow via ON DELETE CASCADE.

    Redis STM is deliberately NOT cleared here. This route has to behave the
    same whether it is served by the agent (Redis on localhost) or by the Lambda
    (no Redis at all), and reaching for an unreachable Redis would cost a socket
    timeout on every delete. The orphaned stm:{session_id} key is handled at the
    only place it can do damage instead: the STM warm-up in /chat checks that the
    conversation row still exists before trusting the cache.
    """
    pg = get_pg_client()
    await pg.connect()
    result = await pg.execute(
        "DELETE FROM conversations WHERE user_id = $1::uuid AND session_id = $2::uuid",
        uid, session_id,
    )
    if result == "DELETE 0":
        raise HTTPException(404, "Session not found")
    return {"deleted": session_id, "result": result}


# ── User memory (Tier 1 always-on facts) ──────────────────────────────────────


@router.post("/me/memory", response_model=UserMemoryItem)
async def create_user_memory(body: UserMemoryCreate, uid: str = Depends(current_user_id)):
    """Create a user fact (Tier 1 always-on memory). D14 MVP: user self-reports.

    The two statements run inside one transaction, which matters more than it
    looks. Served through Neon's pooled endpoint, PgBouncer lends a backend per
    transaction — so two separate calls could execute on two different servers,
    with the second seeing a database where the first had not happened. The
    insert into `users` exists precisely to satisfy the foreign key of the
    insert that follows it, so they have to be one unit.
    """
    pg = get_pg_client()

    async with pg.transaction() as conn:
        # Ensure user row exists — the FK on user_memory requires it, and a user
        # authenticated by Cognito may not have been written here yet.
        await conn.execute(
            "INSERT INTO users (id) VALUES ($1::uuid) ON CONFLICT (id) DO NOTHING", uid,
        )
        row = await conn.fetchrow(
            """INSERT INTO user_memory (user_id, fact_text, category)
               VALUES ($1::uuid, $2, $3)
               RETURNING id, created_at""",
            uid, body.fact_text, body.category,
        )

    return UserMemoryItem(
        id=str(row["id"]),
        fact_text=body.fact_text,
        category=body.category,
        valid=True,
        created_at=row["created_at"].isoformat(),
    )


@router.get("/me/memory", response_model=UserMemoryListResponse)
async def list_user_memory(uid: str = Depends(current_user_id)):
    """List user facts (valid=true, newest first)."""
    pg = get_pg_client()
    await pg.connect()

    rows = await pg.fetch(
        """SELECT id, fact_text, category, valid, created_at
           FROM user_memory
           WHERE user_id = $1::uuid AND valid = true
           ORDER BY created_at DESC
           LIMIT 50""",
        uid,
    )
    return UserMemoryListResponse(facts=[
        UserMemoryItem(
            id=str(r["id"]),
            fact_text=r["fact_text"],
            category=r["category"],
            valid=r["valid"],
            created_at=r["created_at"].isoformat(),
        )
        for r in rows
    ])


@router.delete("/me/memory/{fact_id}")
async def delete_user_memory(fact_id: str, uid: str = Depends(current_user_id)):
    """Hard-delete a user fact. The user_id predicate is the ownership check."""
    pg = get_pg_client()
    await pg.connect()

    result = await pg.execute(
        """DELETE FROM user_memory
           WHERE id = $1::uuid AND user_id = $2::uuid""",
        fact_id, uid,
    )
    if result == "DELETE 0":
        raise HTTPException(404, "Fact not found")
    return {"deleted": fact_id}
