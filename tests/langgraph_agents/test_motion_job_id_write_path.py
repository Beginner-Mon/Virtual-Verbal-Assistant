"""Write-path test for motion_job_id (Task 10 follow-up / Ruling R25).

Task 10 wired the read side: GET /sessions/{id} now returns motion_job_id on
each message so a page refresh can resume polling an in-flight Kimodo job.
R25 rules that write_session_turn() must actually persist that id, or the
column always reads back NULL and the read-side fix is inert.

Two tests:
  - the real INSERT, guarded behind a live-Postgres check (mirrors the guard
    in test_memory_regression.py — same DSN, same skip behavior). This is the
    only test that verifies the column round-trips through SQL.
  - a DB-free signature check, so there is still a green assertion when no
    Postgres is available (as in this task's environment) that the kwarg
    exists, is optional, and defaults to None — i.e. the single existing
    caller (api/main.py, which does not pass motion_job_id yet — see
    Task 10's report for why) keeps working unchanged.
"""

import uuid
import pytest


# ── PG availability guard (mirrors test_memory_regression.py) ───────────────
HAS_PG = False
try:
    import asyncio
    import asyncpg

    async def _ping():
        conn = await asyncpg.connect("postgresql://vva:vva_dev@localhost:5433/vva")
        await conn.close()
        return True

    HAS_PG = asyncio.run(_ping())
except Exception:
    pass


@pytest.mark.integration
@pytest.mark.skipif(not HAS_PG, reason="PostgreSQL not available on port 5433")
@pytest.mark.asyncio
async def test_write_session_turn_persists_motion_job_id_on_assistant_row_only():
    from langgraph_agents.db.session_store import write_session_turn, _to_uuid
    from langgraph_agents.shared import get_pg_client

    user_id = "regr-motion-" + str(uuid.uuid4())
    session_id = str(uuid.uuid4())
    pg = get_pg_client()
    await pg.connect()

    try:
        await write_session_turn(
            user_id=user_id,
            session_id=session_id,
            user_query="show me a shoulder stretch",
            assistant_answer="here's a shoulder stretch",
            total_tokens=5,
            motion_job_id="abc123deadbeef",
        )

        rows = await pg.fetch(
            "SELECT role, motion_job_id FROM messages "
            "WHERE session_id = $1::uuid ORDER BY seq_id",
            session_id,
        )
        assert len(rows) == 2, f"expected 2 rows, got {len(rows)}"
        assert rows[0]["role"] == "user" and rows[0]["motion_job_id"] is None
        assert rows[1]["role"] == "assistant"
        assert rows[1]["motion_job_id"] == "abc123deadbeef"
    finally:
        await pg.execute("DELETE FROM conversations WHERE session_id = $1::uuid", session_id)
        await pg.execute("DELETE FROM users WHERE id = $1::uuid", _to_uuid(user_id))


@pytest.mark.unit
def test_write_session_turn_accepts_optional_motion_job_id_kwarg():
    """DB-free contract check: trailing optional kwarg, defaults to None.

    This does not verify persistence (that needs the skipped test above
    against a real database) — it only verifies the signature the single
    caller depends on didn't change in a breaking way.
    """
    import inspect
    from langgraph_agents.db.session_store import write_session_turn

    sig = inspect.signature(write_session_turn)
    assert "motion_job_id" in sig.parameters
    param = sig.parameters["motion_job_id"]
    assert param.default is None
    # trailing: every parameter after it must also have a default (i.e. it
    # doesn't insert itself ahead of a required positional and break the
    # existing positional-style call in api/main.py — though that call site
    # uses kwargs, so this is a belt-and-suspenders check).
    names = list(sig.parameters)
    assert names.index("motion_job_id") == len(names) - 1
