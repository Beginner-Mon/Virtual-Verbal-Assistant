"""The pre-deploy gate: run every CRUD route against Neon's POOLED endpoint.

The CRUD Lambda connects through `-pooler` while local development connects
directly, and that difference is not cosmetic — PgBouncer runs transaction mode,
so anything assuming "the next statement reaches the same backend" behaves
differently there. Without this test the pooled path would first execute in
production.

Marked `integration`: it needs the real database and is excluded from `-m unit`.

    .\\scripts\\run_tests.ps1 langgraph tests/langgraph_agents/test_crud_pooled_integration.py

Identity is supplied with dependency_overrides rather than a Cognito token, so
this runs without a user pool. What it proves is the database path; the token
path is covered by test_auth.py and by the post-deploy curl checks.

The rows it writes are removed at the end, including on failure.
"""

from __future__ import annotations

import uuid
from urllib.parse import urlsplit, urlunsplit

import pytest
from fastapi.testclient import TestClient

# A fixed, obviously-synthetic id. Fixed rather than random so a crashed run
# leaves one identifiable row behind instead of accumulating new ones.
TEST_USER_ID = "00000000-0000-4000-8000-00000000dead"


def _to_pooled(dsn: str) -> str:
    """Add `-pooler` to the endpoint label, keeping every other label.

    The `.c-NN` segment must survive; dropping it along with `-pooler` fails
    authentication with an error that says nothing about the hostname.
    """
    parts = urlsplit(dsn)
    labels = (parts.hostname or "").split(".")
    if labels[0].endswith("-pooler"):
        return dsn
    labels[0] += "-pooler"

    userinfo = ""
    if parts.username:
        userinfo = parts.username
        if parts.password:
            userinfo += f":{parts.password}"
        userinfo += "@"
    port = f":{parts.port}" if parts.port else ""
    return urlunsplit((parts.scheme, f"{userinfo}{'.'.join(labels)}{port}",
                       parts.path, parts.query, parts.fragment))


@pytest.fixture
def pooled_client(pg_dsn_or_skip, monkeypatch):
    """A CRUD app bound to the pooled endpoint, authenticated as TEST_USER_ID."""
    from langgraph_agents import shared
    from langgraph_agents.api.auth import current_user_id, override_user
    from langgraph_agents.api.crud_app import create_crud_app

    monkeypatch.setenv("VVA_PG_DSN", _to_pooled(pg_dsn_or_skip))
    shared._pg_client = None            # force a client that reads the new DSN

    app = create_crud_app()
    # override_user, not `lambda: TEST_USER_ID`: the override replaces
    # everything current_user_id does, and one of those things is binding the
    # user for row-level security.
    app.dependency_overrides[current_user_id] = override_user(TEST_USER_ID)

    with TestClient(app) as client:
        yield client

    _cleanup()
    shared._pg_client = None


def _cleanup() -> None:
    """Remove this test's rows, as the user who owns them.

    RLS applies here too — a delete without a bound user does not quietly remove
    nothing, it raises. Binding the id makes the intent explicit and keeps the
    cleanup honest: it can only reach rows the test itself created.
    """
    import asyncio

    from langgraph_agents.db.postgres import (
        PostgresClient, bind_request_user, get_default_dsn,
    )

    async def run() -> None:
        bind_request_user(TEST_USER_ID)
        pg = PostgresClient(dsn=_to_pooled(get_default_dsn()))
        try:
            await pg.execute("DELETE FROM user_memory WHERE user_id = $1::uuid", TEST_USER_ID)
            await pg.execute("DELETE FROM conversations WHERE user_id = $1::uuid", TEST_USER_ID)
            await pg.execute("DELETE FROM users WHERE id = $1::uuid", TEST_USER_ID)
        finally:
            await pg.close()

    asyncio.run(run())


@pytest.mark.integration
def test_health_db_reaches_the_pooled_endpoint(pooled_client):
    """The warmer's endpoint, which is the one that must wake Neon's compute."""
    response = pooled_client.get("/health/db")
    assert response.status_code == 200, response.text
    assert response.json()["db"] == "ok"


@pytest.mark.integration
def test_every_crud_route_works_through_the_pooler(pooled_client):
    """All six routes, in the order a session would exercise them.

    POST /me/memory is the one that matters most here: it runs two INSERTs whose
    second depends on the first, which is exactly the shape transaction-mode
    pooling breaks when the statements are not held in one transaction.
    """
    assert pooled_client.get("/sessions").status_code == 200

    created = pooled_client.post("/me/memory", json={
        "fact_text": "pooled-endpoint probe", "category": "test",
    })
    assert created.status_code == 200, created.text
    fact_id = created.json()["id"]

    listed = pooled_client.get("/me/memory")
    assert listed.status_code == 200, listed.text
    assert any(f["id"] == fact_id for f in listed.json()["facts"]), (
        "the fact just created is not in the list — the two statements may have "
        "reached different backends"
    )

    assert pooled_client.delete(f"/me/memory/{fact_id}").status_code == 200

    # Session routes: a user with no sessions must 404 rather than error.
    missing = uuid.uuid4()
    assert pooled_client.get(f"/sessions/{missing}").status_code == 404
    assert pooled_client.delete(f"/sessions/{missing}").status_code == 404


@pytest.mark.integration
def test_repeated_calls_do_not_lose_prepared_statements(pooled_client):
    """Hammer one route so asyncpg's statement cache is exercised, not just filled.

    The classic PgBouncer failure needs a second execution of an already-prepared
    statement on a different backend, so a single call cannot see it. Measured on
    18-08 this passes — Neon's pooler re-prepares — and this is what would notice
    if that ever changed.
    """
    for _ in range(30):
        response = pooled_client.get("/me/memory")
        assert response.status_code == 200, (
            f"failed partway through: {response.text}. A message naming "
            f"__asyncpg_stmt_ means prepared statements stopped surviving the "
            f"pooler; set statement_cache_size=0 in db/postgres.py."
        )
