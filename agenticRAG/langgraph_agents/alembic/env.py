"""Alembic async env for VVA LangGraph agents.

Uses SQLAlchemy async engine (asyncpg driver) for migrations.
Migrations are raw SQL — no ORM metadata needed.
"""

import asyncio
from logging.config import fileConfig
from urllib.parse import parse_qsl, urlencode, urlsplit, urlunsplit

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
# Priority: VVA_PG_DSN_OWNER > VVA_PG_DSN > langgraph.yaml > alembic.ini default
#
# Migrations need the role that OWNS the tables; the application deliberately
# runs as something less privileged. Since 007_rls the application connects as
# `eca_user`, which has no CREATE and is subject to row-level security — running
# `alembic upgrade` as that role fails on the first DDL statement.
#
# So VVA_PG_DSN keeps meaning "the DSN the application uses" and the owner
# credential gets its own name, rather than the two silently competing for one
# variable.
def _resolve_dsn() -> str:
    import os
    for name in ("VVA_PG_DSN_OWNER", "VVA_PG_DSN"):
        env_dsn = os.environ.get(name)
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
    """Convert postgresql:// → postgresql+asyncpg://, and libpq SSL args → asyncpg's.

    The same DSN has to work for two different consumers:

      db/postgres.py   raw asyncpg   understands `sslmode` / `channel_binding`
      here             SQLAlchemy    passes query args straight to asyncpg.connect(),
                                     which has no such keywords → TypeError

    So a managed DSN copied verbatim from a provider dashboard (they all emit
    `?sslmode=require`) runs fine in the app and blows up in `alembic upgrade`.
    Translating here keeps ONE `VVA_PG_DSN` valid for both.
    """
    if dsn.startswith("postgresql://"):
        dsn = dsn.replace("postgresql://", "postgresql+asyncpg://", 1)
    elif not dsn.startswith("postgresql+asyncpg://"):
        return dsn

    parsed = urlsplit(dsn)
    if not parsed.query:
        return dsn

    kept = []
    for key, value in parse_qsl(parsed.query, keep_blank_values=True):
        if key == "sslmode":
            # libpq names → asyncpg's `ssl` argument.
            kept.append(("ssl", "disable" if value in ("disable", "allow") else "require"))
        elif key == "channel_binding":
            continue  # libpq-only; asyncpg negotiates SCRAM binding by itself
        else:
            kept.append((key, value))
    return urlunsplit(parsed._replace(query=urlencode(kept)))


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
