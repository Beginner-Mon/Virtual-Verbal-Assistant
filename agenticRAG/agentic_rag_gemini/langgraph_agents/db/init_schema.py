"""Initialize PostgreSQL schema for LangGraph agents.

Usage:
    python -m langgraph_agents.db.init_schema
"""

from __future__ import annotations

import asyncio
from pathlib import Path

from langgraph_agents.db.postgres import PostgresClient

_SQL_FILE = Path(__file__).resolve().parent / "init_schema.sql"


async def _run():
    sql = _SQL_FILE.read_text(encoding="utf-8")
    pg = PostgresClient()
    await pg.connect()
    try:
        async with pg._pool.acquire() as conn:
            await conn.execute(sql)
            print("Schema initialized successfully.")
    finally:
        await pg.close()


if __name__ == "__main__":
    asyncio.run(_run())
