"""Alembic async env for VVA LangGraph agents.

Uses SQLAlchemy async engine (asyncpg driver) for migrations.
Migrations are raw SQL — no ORM metadata needed.
"""

import asyncio
from logging.config import fileConfig

from alembic import context
from sqlalchemy.ext.asyncio import create_async_engine

from pathlib import Path
import yaml

# Alembic Config object
config = context.config

# Logging
if config.config_file_name is not None:
    fileConfig(config.config_file_name)

# ── DSN resolution ──────────────────────────────────────────────────────
# Priority: env var > langgraph.yaml > alembic.ini default
def _resolve_dsn() -> str:
    import os
    env_dsn = os.environ.get("VVA_PG_DSN")
    if env_dsn:
        return _to_asyncpg(env_dsn)

    # <repo_root>/config/langgraph.yaml — env.py lives at agenticRAG/langgraph_agents/alembic/
    config_path = Path(__file__).resolve().parents[3] / "config" / "langgraph.yaml"
    if config_path.exists():
        with open(config_path, "r", encoding="utf-8") as f:
            cfg = yaml.safe_load(f)
            pg_cfg = cfg.get("langgraph", {}).get("postgres", {})
            raw = pg_cfg.get("dsn", "postgresql://vva:vva_dev@localhost:5433/vva")
            return _to_asyncpg(raw)

    return _to_asyncpg("postgresql://vva:vva_dev@localhost:5433/vva")


def _to_asyncpg(dsn: str) -> str:
    """Convert postgresql:// → postgresql+asyncpg://"""
    if dsn.startswith("postgresql://"):
        return dsn.replace("postgresql://", "postgresql+asyncpg://", 1)
    if dsn.startswith("postgresql+asyncpg://"):
        return dsn
    return dsn


DSN = _resolve_dsn()
config.set_main_option("sqlalchemy.url", DSN)

# No ORM metadata — pure SQL migrations
target_metadata = None


def run_migrations_offline() -> None:
    """Run migrations in 'offline' mode (emit SQL to stdout)."""
    context.configure(
        url=DSN,
        target_metadata=target_metadata,
        literal_binds=True,
        dialect_opts={"paramstyle": "named"},
    )
    with context.begin_transaction():
        context.run_migrations()


def do_run_migrations(connection):
    """Execute migrations within a transaction."""
    context.configure(connection=connection, target_metadata=target_metadata)
    with context.begin_transaction():
        context.run_migrations()


async def run_async_migrations() -> None:
    """Create async engine and run migrations."""
    connectable = create_async_engine(DSN, echo=False)
    async with connectable.connect() as connection:
        await connection.run_sync(do_run_migrations)
    await connectable.dispose()


def run_migrations_online() -> None:
    """Online mode — connect to DB and run."""
    asyncio.run(run_async_migrations())


if context.is_offline_mode():
    run_migrations_offline()
else:
    run_migrations_online()
