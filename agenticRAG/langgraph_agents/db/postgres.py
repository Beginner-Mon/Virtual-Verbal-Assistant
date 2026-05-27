"""Async PostgreSQL client for the LangGraph agent system.

Uses asyncpg connection pool. All queries go through this module.
"""

from __future__ import annotations

import asyncio
from pathlib import Path
from typing import Optional


def _load_pg_config() -> dict:
    import yaml  # lazy — may not be available in test envs
    config_path = Path(__file__).resolve().parents[3] / "config" / "langgraph.yaml"
    if config_path.exists():
        with open(config_path, "r", encoding="utf-8") as f:
            return yaml.safe_load(f).get("langgraph", {}).get("postgres", {})
    return {}


_PG_CFG = _load_pg_config()
_DEFAULT_DSN = _PG_CFG.get("dsn", "postgresql://vva:vva_dev@localhost:5432/vva")
_DEFAULT_POOL_MIN = _PG_CFG.get("pool_min", 2)
_DEFAULT_POOL_MAX = _PG_CFG.get("pool_max", 10)


class PostgresClient:
    """Async PostgreSQL connection pool manager."""

    def __init__(self, dsn: str = None):
        self.dsn = dsn or _DEFAULT_DSN
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
        async with self._pool.acquire() as conn:
            return await conn.execute(query, *args)

    async def fetch(self, query: str, *args) -> list:
        """Fetch multiple rows."""
        await self.connect()
        async with self._pool.acquire() as conn:
            return await conn.fetch(query, *args)

    async def fetchrow(self, query: str, *args):
        """Fetch single row."""
        await self.connect()
        async with self._pool.acquire() as conn:
            return await conn.fetchrow(query, *args)

    async def fetchval(self, query: str, *args):
        """Fetch single value."""
        await self.connect()
        async with self._pool.acquire() as conn:
            return await conn.fetchval(query, *args)
