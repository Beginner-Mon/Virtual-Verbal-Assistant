"""Tests for PostgresClient.transaction() and user_scope().

These exist because of a measurement, not a theory. Against this project's Neon
database on 18-08-2026, setting `app.user_id` with a plain `SET` and reading it
back from another client gave:

    direct endpoint : 200 reads, 0 saw another client's id
    pooled endpoint : 200 reads, 98 saw another client's id

asyncpg resets a connection's state when returning it to its own pool, which is
why the direct endpoint is clean. Through PgBouncer that reset lands on whichever
backend is lent next rather than the one holding the state, so it misses. The
CRUD Lambda runs against the pooled endpoint.

The unit tests below need no database. The integration test at the bottom
reproduces the real thing and is marked accordingly.
"""

from __future__ import annotations

import inspect
from unittest.mock import AsyncMock, MagicMock

import pytest


# ── Test doubles ──────────────────────────────────────────────────────────────


class _FakeAcquire:
    """Stands in for pool.acquire(), which is an async context manager."""

    def __init__(self, conn):
        self._conn = conn

    async def __aenter__(self):
        return self._conn

    async def __aexit__(self, *_exc):
        return False


class _FakeTransaction:
    def __init__(self, conn):
        self._conn = conn

    async def __aenter__(self):
        self._conn.transaction_entered = True
        return self._conn

    async def __aexit__(self, *_exc):
        return False


def _client_with_fake_pool():
    """A PostgresClient whose pool hands out one recording connection."""
    from langgraph_agents.db.postgres import PostgresClient

    conn = MagicMock()
    conn.transaction_entered = False
    conn.execute = AsyncMock()
    conn.fetchrow = AsyncMock(return_value=None)
    conn.transaction = MagicMock(side_effect=lambda: _FakeTransaction(conn))

    pool = MagicMock()
    pool.acquire = MagicMock(side_effect=lambda: _FakeAcquire(conn))

    client = PostgresClient(dsn="postgresql://unused/unused")
    client._pool = pool                       # skip connect(); nothing dials out
    client.connect = AsyncMock()
    return client, pool, conn


# ── transaction() ─────────────────────────────────────────────────────────────


@pytest.mark.unit
@pytest.mark.asyncio
async def test_transaction_holds_one_connection_for_the_block():
    """Every statement in the block must reach the same connection.

    The per-call methods (execute/fetch/...) acquire and release each time, which
    through PgBouncer means consecutive statements can run on different backends.
    """
    client, pool, conn = _client_with_fake_pool()

    async with client.transaction() as handle:
        await handle.execute("SELECT 1")
        await handle.execute("SELECT 2")

    assert pool.acquire.call_count == 1, (
        f"acquired {pool.acquire.call_count} connections; the whole point is one"
    )
    assert conn.transaction_entered, "statements ran outside a transaction"


# ── user_scope() ──────────────────────────────────────────────────────────────


@pytest.mark.unit
@pytest.mark.asyncio
async def test_user_scope_sets_the_guc_transaction_locally_and_as_a_parameter():
    """Two properties, both load-bearing.

    `is_local=true` (the third argument) keeps the setting inside the
    transaction, so it cannot outlive the block and be read by the next user of
    that connection. And passing the id as $1 rather than splicing it into the
    SQL means a hostile user id cannot become SQL — which is also why this uses
    set_config() instead of SET LOCAL, since SET LOCAL accepts only a literal.
    """
    client, _pool, conn = _client_with_fake_pool()
    uid = "11111111-1111-1111-1111-111111111111"

    async with client.user_scope(uid):
        pass

    conn.execute.assert_awaited_once()
    sql, *params = conn.execute.await_args.args

    assert "set_config" in sql.lower(), f"expected set_config, got: {sql}"
    assert "true" in sql.lower(), "third argument must be true (transaction-local)"
    assert params == [uid], (
        f"user id must be bound as a parameter, not spliced into SQL. args={params}"
    )
    assert uid not in sql, "the user id was interpolated into the SQL text"


@pytest.mark.unit
@pytest.mark.asyncio
async def test_user_scope_runs_inside_a_transaction():
    """Without a transaction, `is_local=true` has nothing to be local to."""
    client, pool, conn = _client_with_fake_pool()

    async with client.user_scope("22222222-2222-2222-2222-222222222222"):
        pass

    assert conn.transaction_entered
    assert pool.acquire.call_count == 1


@pytest.mark.unit
def test_no_other_api_sets_app_user_id():
    """user_scope must be the only way in.

    A second entry point is how the plain-SET form comes back — and that form
    fails silently, returning another user's rows rather than an error.

    Scans the AST rather than the text. A line-based grep flags the prose that
    explains the danger, and comments are not in the AST at all, so this looks
    only at string literals the code actually executes.
    """
    import ast

    import langgraph_agents.db.postgres as pg_mod

    tree = ast.parse(inspect.getsource(pg_mod))

    docstring_nodes = set()
    for node in ast.walk(tree):
        if isinstance(node, (ast.Module, ast.ClassDef, ast.FunctionDef, ast.AsyncFunctionDef)):
            if node.body and isinstance(node.body[0], ast.Expr):
                first = node.body[0].value
                if isinstance(first, ast.Constant) and isinstance(first.value, str):
                    docstring_nodes.add(id(first))

    offenders = [
        node.value.strip()[:100]
        for node in ast.walk(tree)
        if isinstance(node, ast.Constant)
        and isinstance(node.value, str)
        and id(node) not in docstring_nodes
        and "app.user_id" in node.value
        and "set_config" not in node.value
    ]

    assert not offenders, (
        "app.user_id is set outside user_scope(), which is the leaking form: "
        + str(offenders)
    )


# ── create_user_memory ────────────────────────────────────────────────────────


@pytest.mark.unit
@pytest.mark.asyncio
async def test_create_user_memory_runs_both_inserts_in_one_transaction():
    """The users insert exists to satisfy the FK of the memory insert.

    Split across two pooled calls they can land on two backends, with the second
    seeing a database where the first had not happened.
    """
    from unittest.mock import patch

    from langgraph_agents.api.routes_crud import create_user_memory
    from langgraph_agents.api.schemas import UserMemoryCreate

    client, pool, conn = _client_with_fake_pool()
    conn.fetchrow = AsyncMock(return_value={
        "id": "33333333-3333-3333-3333-333333333333",
        "created_at": __import__("datetime").datetime(2026, 8, 18),
    })

    with patch("langgraph_agents.api.routes_crud.get_pg_client", return_value=client):
        await create_user_memory(
            UserMemoryCreate(fact_text="x"),
            uid="44444444-4444-4444-4444-444444444444",
        )

    assert pool.acquire.call_count == 1, (
        f"the two inserts used {pool.acquire.call_count} connections"
    )
    assert conn.transaction_entered


# ── The real thing ────────────────────────────────────────────────────────────


@pytest.mark.integration
@pytest.mark.asyncio
async def test_guc_does_not_survive_the_transaction(pg_dsn_or_skip):
    """Against a real database: the setting must be gone in the next transaction.

    This is the property the whole design rests on. If it ever fails, RLS is
    filtering by whoever came before.
    """
    from langgraph_agents.db.postgres import PostgresClient

    client = PostgresClient(dsn=pg_dsn_or_skip)
    try:
        async with client.user_scope("55555555-5555-5555-5555-555555555555") as conn:
            inside = await conn.fetchval("SELECT current_setting('app.user_id', true)")
        assert inside == "55555555-5555-5555-5555-555555555555"

        after = await client.fetchval("SELECT current_setting('app.user_id', true)")
        assert not after, (
            f"app.user_id survived the transaction as {after!r} — it is now "
            f"readable by the next request that borrows this connection"
        )
    finally:
        await client.close()
