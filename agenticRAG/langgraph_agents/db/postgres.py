"""Async PostgreSQL client for the LangGraph agent system.

Uses asyncpg connection pool. All queries go through this module.
"""

from __future__ import annotations

import asyncio
import os
import time
from contextlib import asynccontextmanager
from contextvars import ContextVar
from pathlib import Path
from typing import Optional

from langgraph_agents.shared.env import env_source, load_env
from langgraph_agents.shared.logging import get_logger

logger = get_logger("langgraph.db.postgres")

_LOCAL_DSN = "postgresql://vva:vva_dev@localhost:5433/vva"


# Whose rows the next statement may touch. Set once per request by
# api/auth.py::current_user_id and read by PostgresClient._scoped.
#
# A ContextVar rather than a module global because concurrent requests share the
# process: a global would give whichever request wrote last, which is the same
# cross-user leak the row-level security exists to prevent.
#
# It also propagates the way this codebase needs. asyncio.create_task copies the
# current context, so the summariser fired after a response — see
# nodes/summarizer.py, called from the /chat stream — inherits the user it was
# produced for, with nothing threaded through by hand.
_request_user_id: ContextVar[Optional[str]] = ContextVar(
    "vva_request_user_id", default=None,
)


def bind_request_user(user_id: str) -> None:
    """Declare whose data the rest of this request may see.

    Called from the authentication dependency, after the token has been
    verified — never from a handler, and never with a value that came from the
    request itself.
    """
    _request_user_id.set(user_id)


def current_request_user() -> Optional[str]:
    """The bound user, or None outside a request (scripts, migrations)."""
    return _request_user_id.get()


# One loader for the whole service. This module used to carry its own copy of
# the search order, which is how "which .env wins" came to depend on import
# order. See shared/env.py.
load_env()


def _load_pg_config() -> dict:
    import yaml  # lazy — may not be available in test envs
    config_path = Path(__file__).resolve().parents[3] / "config" / "langgraph.yaml"
    if config_path.exists():
        with open(config_path, "r", encoding="utf-8") as f:
            return yaml.safe_load(f).get("langgraph", {}).get("postgres", {})
    return {}


# `VVA_PG_DSN` first because `alembic/env.py` reads that name: if the two
# disagree, migrations land on one database while the app talks to another —
# silently, with no error anywhere. `POSTGRES_DSN` arrived from the other branch
# solving the same problem; it stays as an alias so existing shells keep working.
_DSN_ENV_VARS = ("VVA_PG_DSN", "POSTGRES_DSN")


def _dsn_from_ssm() -> Optional[str]:
    """Read the DSN from an SSM SecureString named by `VVA_PG_DSN_PARAM`.

    For Lambda, where the alternative is a plaintext connection string sitting
    in the CloudFormation template and visible to anyone who can read the stack.
    Mirrors what infra/lambda/layer/shared/db.py already does for the characters
    function, so the two deployments fetch their credential the same way.

    boto3 is imported here rather than at module scope: it ships with the Lambda
    runtime but is not a dependency of running this service anywhere else.
    """
    param = os.environ.get("VVA_PG_DSN_PARAM")
    if not param:
        return None

    import boto3  # lazy — present on Lambda, not required locally

    value = boto3.client("ssm").get_parameter(
        Name=param, WithDecryption=True,
    )["Parameter"]["Value"]
    logger.info("dsn_loaded_from_ssm", extra={"param": param})
    return value


def _resolve_dsn(cfg: dict) -> str:
    """env > SSM > langgraph.yaml > local default.

    The env var MUST win, and must be checked here as well as in
    `alembic/env.py`. Until this existed, only Alembic honoured `VVA_PG_DSN`:
    pointing it at a managed database migrated the schema there while the
    running app kept reading and writing the local container — a split brain
    with no error message anywhere.

    It is also the only place a managed DSN can live safely: `langgraph.yaml`
    is committed, and a hosted connection string carries its password. SSM sits
    below the env var for the same reason — a developer exporting VVA_PG_DSN to
    point at a scratch database should not be silently overridden by whatever
    the deployment happens to be configured with.
    """
    for name in _DSN_ENV_VARS:
        value = os.environ.get(name)
        if value:
            return value

    from_ssm = _dsn_from_ssm()
    if from_ssm:
        return from_ssm

    configured = cfg.get("dsn")
    if configured:
        return configured

    # The failure this whole module is shaped around. Landing here means the app
    # is about to talk to the local container while everything else — Alembic,
    # the ingest script, whoever exported VVA_PG_DSN in another shell — may be
    # on the managed database. It reads as "working" right up until the data is
    # in the wrong place, so it says so out loud.
    logger.warning(
        "No %s set and no dsn in langgraph.yaml — falling back to the LOCAL "
        "database (%s). Configuration source: %s. If you meant to use the "
        "managed database, this is the bug.",
        " / ".join(_DSN_ENV_VARS),
        _LOCAL_DSN,
        env_source(),
    )
    return _LOCAL_DSN


_PG_CFG = _load_pg_config()
_DEFAULT_DSN = _resolve_dsn(_PG_CFG)
_DEFAULT_POOL_MIN = _PG_CFG.get("pool_min", 2)
_DEFAULT_POOL_MAX = _PG_CFG.get("pool_max", 10)


def get_default_dsn() -> str:
    """The DSN the application itself uses.

    Exists so scripts and migrations resolve the database the same way the app
    does instead of inventing their own env var and default — which is how
    `migrate_messages.py` ended up pointed at a database that was not this one.
    """
    return _DEFAULT_DSN


class QueryStats:
    """Counts round-trips to the database, per logical operation.

    Added after moving to a managed Postgres: with the DB on localhost every
    round-trip cost <1 ms, so nobody noticed how many there were. Over a 53 ms
    link the same code paid +28% per chat turn — the cost is the NUMBER of
    sequential queries, not the database.

    Guessing that number from total latency was wrong by ~10x once already, so
    it gets measured instead. Enabled with VVA_PG_STATS=1; a no-op otherwise.
    """

    __slots__ = ("count", "seconds", "by_kind")

    def __init__(self):
        self.reset()

    def reset(self) -> None:
        self.count = 0
        self.seconds = 0.0
        self.by_kind = {}

    def record(self, kind: str, elapsed: float) -> None:
        self.count += 1
        self.seconds += elapsed
        slot = self.by_kind.setdefault(kind, [0, 0.0])
        slot[0] += 1
        slot[1] += elapsed

    def summary(self) -> str:
        parts = ", ".join(
            f"{k}={n}/{t * 1000:.0f}ms" for k, (n, t) in sorted(self.by_kind.items())
        )
        return f"{self.count} queries, {self.seconds * 1000:.0f}ms total [{parts}]"


STATS = QueryStats()
STATS_ENABLED = os.environ.get("VVA_PG_STATS") == "1"


class PostgresClient:
    """Async PostgreSQL connection pool manager."""

    def __init__(self, dsn: str = None):
        # Resolved per instance, not at import: `.env` is loaded lazily by
        # llm.py, so a module-level snapshot would miss `VVA_PG_DSN` whenever
        # this module happens to be imported first — silently falling back to
        # the local container.
        self.dsn = dsn or _resolve_dsn(_PG_CFG)
        self._pool: Optional[asyncpg.Pool] = None
        self._connect_lock: Optional[asyncio.Lock] = None

    async def connect(self):
        """Create connection pool with pgvector codec registered (best-effort).

        The codec registration is skipped silently if either:
        - the `pgvector` Python package is not installed, or
        - the `vector` extension hasn't been created in the database yet
          (e.g. first run, before init_schema). The pool still works; vector
          ops just won't auto-encode/decode until the extension is present.
        """
        # Fast path — pool already alive.
        if self._pool is not None:
            return

        # Lazy-create the lock; safe because the very first connect() call is
        # serial (the loop is awaiting us before anything else can start).
        if self._connect_lock is None:
            self._connect_lock = asyncio.Lock()

        async with self._connect_lock:
            # Double-check after lock — another waiter may have built the pool.
            if self._pool is not None:
                return

            import asyncpg  # lazy — may not be installed

            try:
                from pgvector.asyncpg import register_vector
            except ImportError:
                register_vector = None

            async def _init_conn(conn):
                # Enable HNSW iterative index scan (pgvector >= 0.8) so filtered
                # vector search (memory_search: WHERE session_id = ANY(...)) keeps
                # recall when the filter prunes HNSW candidates. Best-effort:
                # skipped silently on older pgvector / missing extension.
                try:
                    await conn.execute("SET hnsw.iterative_scan = 'relaxed_order'")
                except Exception:
                    pass
                if register_vector is None:
                    return
                try:
                    await register_vector(conn)
                except Exception:
                    # Extension not installed yet — caller will create it via init_schema.
                    pass

            self._pool = await asyncpg.create_pool(
                self.dsn,
                min_size=_DEFAULT_POOL_MIN,
                max_size=_DEFAULT_POOL_MAX,
                init=_init_conn,
            )

    async def close(self):
        """Close connection pool."""
        if self._pool:
            await self._pool.close()
            self._pool = None

    @asynccontextmanager
    async def _scoped(self):
        """A connection for one statement, carrying the caller's identity.

        Since migration 007, the user tables have row-level security and the
        application connects as a role the policies apply to. Every statement
        therefore needs `app.user_id`, and it has to be set inside the same
        transaction — a plain `SET` survives the connection and reaches whoever
        borrows it next (measured through Neon's pooler: 98 of 200 reads saw
        another client's id).

        Doing that here rather than at each call site is the difference between
        one change and twenty-one. It also carries into background work for
        free: asyncio.create_task copies the context, so the summariser fired
        after a response still runs as the user whose turn produced it.

        When no user is bound — a migration, an ingest script, the Stripe
        webhook before it establishes whose account an event belongs to — the
        statement runs unscoped and the policies reject it. That failure is the
        design working; see the note on error shapes in `transaction`.
        """
        async with self.transaction() as conn:
            yield conn

    async def execute(self, query: str, *args):
        """Execute a query (INSERT, UPDATE, DELETE)."""
        started = time.perf_counter() if STATS_ENABLED else 0.0
        try:
            async with self._scoped() as conn:
                return await conn.execute(query, *args)
        finally:
            if STATS_ENABLED:
                STATS.record("execute", time.perf_counter() - started)

    async def executemany(self, query: str, args):
        """Execute a query against many arg tuples (batch INSERT)."""
        started = time.perf_counter() if STATS_ENABLED else 0.0
        try:
            async with self._scoped() as conn:
                return await conn.executemany(query, args)
        finally:
            if STATS_ENABLED:
                STATS.record("executemany", time.perf_counter() - started)

    async def fetch(self, query: str, *args) -> list:
        """Fetch multiple rows."""
        started = time.perf_counter() if STATS_ENABLED else 0.0
        try:
            async with self._scoped() as conn:
                return await conn.fetch(query, *args)
        finally:
            if STATS_ENABLED:
                STATS.record("fetch", time.perf_counter() - started)

    async def fetchrow(self, query: str, *args):
        """Fetch single row."""
        started = time.perf_counter() if STATS_ENABLED else 0.0
        try:
            async with self._scoped() as conn:
                return await conn.fetchrow(query, *args)
        finally:
            if STATS_ENABLED:
                STATS.record("fetchrow", time.perf_counter() - started)

    async def fetchval(self, query: str, *args):
        """Fetch single value."""
        started = time.perf_counter() if STATS_ENABLED else 0.0
        try:
            async with self._scoped() as conn:
                return await conn.fetchval(query, *args)
        finally:
            if STATS_ENABLED:
                STATS.record("fetchval", time.perf_counter() - started)

    # ── Connection-scoped work ──────────────────────────────────────────

    @asynccontextmanager
    async def _raw_transaction(self):
        """One connection, one transaction, no identity. Building block only.

        Everything else goes through `transaction`, which adds the user scope.
        """
        await self.connect()
        async with self._pool.acquire() as conn:
            async with conn.transaction():
                yield conn

    @asynccontextmanager
    async def transaction(self):
        """Hold ONE connection for the whole block, scoped to the bound user.

        Two jobs in one place, because they are needed together.

        *One connection*: the query methods acquire and release per call, which
        is right for one-shot statements and wrong for anything assuming two
        statements land on the same place. That assumption is merely fragile
        against the direct endpoint and simply false through Neon's pooled one —
        PgBouncer lends a backend for the duration of a transaction and takes it
        back, so consecutive statements outside a transaction can run on
        different servers. See create_user_memory in api/routes_crud.py.

        *Scoped*: row-level security reads `app.user_id`, and this is where it
        gets set when a request has bound one. An earlier version left that to
        `user_scope` alone, so a handler that reached for `transaction` got a
        block with no identity and every statement in it failed.

        Without a bound user the block runs unscoped and the policies reject it,
        in one of two shapes worth recognising:

            unrecognized configuration parameter "app.user_id"
            invalid input syntax for type uuid: ""

        The second appears once any transaction on that connection has set the
        parameter: a custom GUC set with is_local reverts to the empty string
        rather than to unset. Both are the same mistake — a query outside
        `user_scope` — and both are errors rather than silently empty results,
        which is the point of the policies not passing `missing_ok`.
        """
        user_id = _request_user_id.get()
        if user_id is None:
            async with self._raw_transaction() as conn:
                yield conn
        else:
            async with self.user_scope(user_id) as conn:
                yield conn

    @asynccontextmanager
    async def user_scope(self, user_id: str):
        """A transaction carrying `app.user_id`, for row-level security.

        THE ONLY SUPPORTED WAY to set that setting. Not a style preference — a
        plain `SET app.user_id = ...` leaks the value to other users, and this
        was measured against this database rather than inferred:

            direct endpoint : 200 reads, 0 saw another client's id
            pooled endpoint : 200 reads, 98 saw another client's id

        asyncpg resets a connection's state when returning it to its own pool,
        which is why the direct endpoint is clean. Through PgBouncer that reset
        is sent to whichever backend is lent next, not the one still holding the
        state, so it misses — and the CRUD Lambda runs against the pooled
        endpoint.

        `set_config(..., true)` is the fix on both counts: the third argument
        makes the setting local to the transaction, so it cannot outlive the
        block, and unlike `SET LOCAL` it takes a bind parameter — `SET LOCAL`
        only accepts a literal, which would mean splicing a user id into SQL
        text.

        Uses `_raw_transaction`, not `transaction`: the latter routes back here
        for a bound user, which would be infinite.
        """
        async with self._raw_transaction() as conn:
            await conn.execute("SELECT set_config('app.user_id', $1, true)", user_id)
            yield conn
