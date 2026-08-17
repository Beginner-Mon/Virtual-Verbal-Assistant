"""Minimal local server for exercising only the sandbox billing UI.

This runner exists for development machines that do not have the full VVA
Postgres/Redis/LLM stack available.  It mounts the real billing router and swaps
only its persistence dependency for a small SQLite adapter stored in the system
temporary directory.  Stripe calls are still real Stripe *test-mode* API calls;
the safety checks in :mod:`langgraph_agents.api.billing` remain in force.

Run from ``agenticRAG`` with::

    python -m uvicorn langgraph_agents.api.billing_local:create_app \
        --factory --host 127.0.0.1 --port 8000

This module must never be used as a production application entry point.
"""

from __future__ import annotations

import asyncio
import os
import sqlite3
import tempfile
import uuid
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Any

from dotenv import load_dotenv


# Load the same ignored local file as the full service before importing auth or
# billing; both modules snapshot security-related settings during import.
_AGENTIC_RAG_DIR = Path(__file__).resolve().parents[2]
load_dotenv(_AGENTIC_RAG_DIR / ".env", override=False)

from fastapi import FastAPI  # noqa: E402
from fastapi.middleware.cors import CORSMiddleware  # noqa: E402

from langgraph_agents.api import billing  # noqa: E402


class LocalBillingPgAdapter:
    """Implement the tiny Postgres surface used by ``billing.py`` with SQLite.

    The adapter intentionally recognizes only billing router statements.  If a
    non-billing query is ever routed here it fails loudly instead of turning
    this test runner into an accidental replacement for the application DB.
    """

    def __init__(self, path: Path):
        self.path = path

    def _open(self) -> sqlite3.Connection:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        connection = sqlite3.connect(self.path, timeout=10)
        connection.row_factory = sqlite3.Row
        return connection

    def _setup(self) -> None:
        with self._open() as connection:
            connection.executescript(
                """
                CREATE TABLE IF NOT EXISTS users (
                    id TEXT PRIMARY KEY
                );

                CREATE TABLE IF NOT EXISTS billing_accounts (
                    user_id TEXT PRIMARY KEY REFERENCES users(id),
                    access_plan TEXT NOT NULL DEFAULT 'DEMO'
                        CHECK (access_plan IN ('FREE', 'DEMO')),
                    stripe_customer_id TEXT,
                    stripe_subscription_id TEXT,
                    stripe_subscription_status TEXT,
                    updated_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP
                );

                CREATE TABLE IF NOT EXISTS billing_webhook_events (
                    event_id TEXT PRIMARY KEY,
                    event_type TEXT NOT NULL,
                    livemode INTEGER NOT NULL DEFAULT 0 CHECK (livemode = 0),
                    processed_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP
                );
                """
            )

    async def connect(self) -> None:
        await asyncio.to_thread(self._setup)

    def _execute_sync(self, query: str, args: tuple[Any, ...]) -> str:
        normalized = " ".join(query.split()).upper()
        with self._open() as connection:
            if normalized.startswith("INSERT INTO USERS"):
                connection.execute(
                    "INSERT OR IGNORE INTO users (id) VALUES (?)", (str(args[0]),)
                )
                return "INSERT 0 1"

            if normalized.startswith("UPDATE BILLING_ACCOUNTS"):
                user_id = str(args[0])
                if "SANDBOX_CHECKOUT_COMPLETE" in normalized:
                    connection.execute(
                        """UPDATE billing_accounts
                           SET stripe_customer_id = ?,
                               stripe_subscription_id = ?,
                               stripe_subscription_status =
                                   'sandbox_checkout_complete',
                               updated_at = CURRENT_TIMESTAMP
                           WHERE user_id = ?""",
                        (args[1], args[2], user_id),
                    )
                else:
                    connection.execute(
                        """UPDATE billing_accounts
                           SET stripe_customer_id = COALESCE(?, stripe_customer_id),
                               stripe_subscription_id = ?,
                               stripe_subscription_status = ?,
                               updated_at = CURRENT_TIMESTAMP
                           WHERE user_id = ?""",
                        (args[1], args[2], args[3], user_id),
                    )
                return "UPDATE 1"

            if normalized.startswith("INSERT INTO BILLING_WEBHOOK_EVENTS"):
                connection.execute(
                    """INSERT OR IGNORE INTO billing_webhook_events
                       (event_id, event_type, livemode) VALUES (?, ?, 0)""",
                    (str(args[0]), str(args[1])),
                )
                return "INSERT 0 1"

        raise RuntimeError("The local billing runner rejected a non-billing query")

    async def execute(self, query: str, *args: Any) -> str:
        return await asyncio.to_thread(self._execute_sync, query, args)

    def _fetchrow_sync(self, query: str, args: tuple[Any, ...]) -> dict[str, Any]:
        normalized = " ".join(query.split()).upper()
        if not normalized.startswith("INSERT INTO BILLING_ACCOUNTS"):
            raise RuntimeError("The local billing runner rejected a non-billing query")

        user_id = str(args[0])
        with self._open() as connection:
            connection.execute("INSERT OR IGNORE INTO users (id) VALUES (?)", (user_id,))
            connection.execute(
                """INSERT OR IGNORE INTO billing_accounts (user_id, access_plan)
                   VALUES (?, 'DEMO')""",
                (user_id,),
            )
            connection.execute(
                """UPDATE billing_accounts SET updated_at = CURRENT_TIMESTAMP
                   WHERE user_id = ?""",
                (user_id,),
            )
            row = connection.execute(
                """SELECT user_id, access_plan, stripe_customer_id,
                          stripe_subscription_id, stripe_subscription_status,
                          updated_at
                   FROM billing_accounts WHERE user_id = ?""",
                (user_id,),
            ).fetchone()

        if row is None:  # pragma: no cover - guarded by the insert above
            raise RuntimeError("Could not initialize the local DEMO billing account")
        return dict(row)

    async def fetchrow(self, query: str, *args: Any) -> dict[str, Any]:
        return await asyncio.to_thread(self._fetchrow_sync, query, args)


def _local_database_path() -> Path:
    configured = os.getenv("BILLING_LOCAL_SQLITE_PATH", "").strip()
    if configured:
        return Path(configured).expanduser().resolve()
    return Path(tempfile.gettempdir()) / "vva-billing-sandbox.sqlite3"


_local_pg = LocalBillingPgAdapter(_local_database_path())

# ``billing.py`` imported the singleton accessor directly, so replacing this
# module-local dependency affects only this development process.
billing.get_pg_client = lambda: _local_pg


async def _resolve_local_demo_user_id(_: Any, fallback_user_id: str | None) -> str:
    """Return a stable UUID without importing the production auth/Redis stack."""
    candidate = fallback_user_id or "anonymous"
    try:
        return str(uuid.UUID(candidate))
    except (ValueError, AttributeError):
        return str(uuid.uuid5(uuid.NAMESPACE_URL, f"vva-local:{candidate}"))


billing.resolve_user_id = _resolve_local_demo_user_id


@asynccontextmanager
async def _lifespan(_: FastAPI):
    await _local_pg.connect()
    yield


def create_app() -> FastAPI:
    app = FastAPI(title="VVA billing sandbox (local only)", lifespan=_lifespan)
    app.add_middleware(
        CORSMiddleware,
        allow_origins=["http://localhost:5173", "http://127.0.0.1:5173"],
        allow_credentials=False,
        allow_methods=["*"],
        allow_headers=["*"],
    )
    app.include_router(billing.router)

    @app.get("/health")
    async def health():
        return {
            "status": "ok",
            "service": "billing-local",
            "test_only": True,
            "real_transactions_enabled": False,
        }

    return app
