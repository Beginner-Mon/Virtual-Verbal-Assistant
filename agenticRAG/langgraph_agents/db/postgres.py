"""Async PostgreSQL client for the LangGraph agent system.

Uses asyncpg connection pool. All queries go through this module.
"""

from __future__ import annotations

import asyncio
import os
import time
from pathlib import Path
from typing import Optional

_LOCAL_DSN = "postgresql://vva:vva_dev@localhost:5433/vva"


def _load_pg_config() -> dict:
    import yaml  # lazy — may not be available in test envs
    config_path = Path(__file__).resolve().parents[3] / "config" / "langgraph.yaml"
    if config_path.exists():
        with open(config_path, "r", encoding="utf-8") as f:
            return yaml.safe_load(f).get("langgraph", {}).get("postgres", {})
    return {}


def _resolve_dsn(cfg: dict) -> str:
    """env > langgraph.yaml > local default.

    The env var MUST win, and must be checked here as well as in
    `alembic/env.py`. Until this existed, only Alembic honoured `VVA_PG_DSN`:
    pointing it at a managed database migrated the schema there while the
    running app kept reading and writing the local container — a split brain
    with no error message anywhere.

    It is also the only place a managed DSN can live safely: `langgraph.yaml`
    is committed, and a hosted connection string carries its password.
    """
    return os.environ.get("VVA_PG_DSN") or cfg.get("dsn", _LOCAL_DSN)


_PG_CFG = _load_pg_config()
_DEFAULT_DSN = _resolve_dsn(_PG_CFG)
_DEFAULT_POOL_MIN = _PG_CFG.get("pool_min", 2)
_DEFAULT_POOL_MAX = _PG_CFG.get("pool_max", 10)


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

    async def execute(self, query: str, *args):
        """Execute a query (INSERT, UPDATE, DELETE)."""
        await self.connect()
        if not STATS_ENABLED:
            async with self._pool.acquire() as conn:
                return await conn.execute(query, *args)
        started = time.perf_counter()
        try:
            async with self._pool.acquire() as conn:
                return await conn.execute(query, *args)
        finally:
            STATS.record("execute", time.perf_counter() - started)

    async def executemany(self, query: str, args):
        """Execute a query against many arg tuples (batch INSERT)."""
        await self.connect()
        if not STATS_ENABLED:
            async with self._pool.acquire() as conn:
                return await conn.executemany(query, args)
        started = time.perf_counter()
        try:
            async with self._pool.acquire() as conn:
                return await conn.executemany(query, args)
        finally:
            STATS.record("executemany", time.perf_counter() - started)

    async def fetch(self, query: str, *args) -> list:
        """Fetch multiple rows."""
        await self.connect()
        if not STATS_ENABLED:
            async with self._pool.acquire() as conn:
                return await conn.fetch(query, *args)
        started = time.perf_counter()
        try:
            async with self._pool.acquire() as conn:
                return await conn.fetch(query, *args)
        finally:
            STATS.record("fetch", time.perf_counter() - started)

    async def fetchrow(self, query: str, *args):
        """Fetch single row."""
        await self.connect()
        if not STATS_ENABLED:
            async with self._pool.acquire() as conn:
                return await conn.fetchrow(query, *args)
        started = time.perf_counter()
        try:
            async with self._pool.acquire() as conn:
                return await conn.fetchrow(query, *args)
        finally:
            STATS.record("fetchrow", time.perf_counter() - started)

    async def fetchval(self, query: str, *args):
        """Fetch single value."""
        await self.connect()
        if not STATS_ENABLED:
            async with self._pool.acquire() as conn:
                return await conn.fetchval(query, *args)
        started = time.perf_counter()
        try:
            async with self._pool.acquire() as conn:
                return await conn.fetchval(query, *args)
        finally:
            STATS.record("fetchval", time.perf_counter() - started)
