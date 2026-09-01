"""The CRUD-only FastAPI app — sessions and user memory, nothing else.

This is what the Lambda runs. It exists so that serving a `GET /sessions` does
not require a process that imports langchain_openai at module scope and
sentence_transformers at boot (torch, ~2 GB) purely to run a few SQL statements.

The routes themselves live in api/routes_crud.py and are shared with
api/main.py, so this app is a packaging boundary rather than a second
implementation.

Hard constraint, enforced by a test: importing this module must not pull in
langgraph_agents.graph, langgraph_agents.nodes, or shared.embedding. If you add
an import here, check what it drags behind it.

Run it locally exactly as the Lambda does:

    uvicorn langgraph_agents.api.crud_app:create_crud_app --factory --port 8001
"""

from __future__ import annotations

import os
from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from langgraph_agents.api.auth import verify_auth_config
from langgraph_agents.api.routes_characters import router as characters_router
from langgraph_agents.api.routes_crud import router as crud_router
from langgraph_agents.api.routes_preferences import router as preferences_router
from langgraph_agents.shared import get_pg_client
from langgraph_agents.shared.env import env_source
from langgraph_agents.shared.logging import configure_root_logger, get_logger

logger = get_logger("langgraph.api.crud_app")

_DEFAULT_ORIGINS = "http://localhost:3000,http://localhost:8080,http://localhost:5173"


def allowed_origins() -> list[str]:
    """CORS origins from ALLOWED_ORIGINS, shared with api/main.py.

    Lives here rather than in main.py because main.py imports the graph and this
    module must not. One definition either way — the browser sends an OPTIONS
    preflight before every POST and DELETE, and two apps disagreeing about which
    origins are allowed is a bug that only shows up in a deployed browser.
    """
    return [o.strip() for o in os.getenv("ALLOWED_ORIGINS", _DEFAULT_ORIGINS).split(",") if o.strip()]


def add_cors(application: FastAPI) -> None:
    application.add_middleware(
        CORSMiddleware,
        allow_origins=allowed_origins(),
        allow_credentials=False,      # "*" + credentials disallowed by spec
        allow_methods=["*"],
        allow_headers=["*"],
    )


@asynccontextmanager
async def _lifespan(_: FastAPI):
    configure_root_logger(level=os.getenv("LOG_LEVEL", "INFO"))

    # First, and allowed to raise: a process that cannot verify tokens should
    # fail its deploy, not answer 401 to everyone and look like a user problem.
    # On Lambda this surfaces as an INIT failure.
    verify_auth_config()

    # Build the connection pool during init rather than on the first request.
    # On Lambda, init time is billed at a discount and — more to the point —
    # happens before the user is waiting. Best-effort: a database that is
    # briefly unreachable at boot should not take the whole function down, the
    # first request will retry.
    try:
        await get_pg_client().connect()
        pool_ready = True
    except Exception as exc:
        logger.warning("pg_preconnect_failed", extra={"error": str(exc)})
        pool_ready = False

    logger.info("crud_startup_complete", extra={
        "pool_ready": pool_ready,
        "config_source": env_source(),
    })
    yield
    logger.info("crud_shutdown")


def create_crud_app() -> FastAPI:
    application = FastAPI(title="VVA CRUD API", lifespan=_lifespan)
    add_cors(application)
    application.include_router(characters_router)
    application.include_router(crud_router)
    application.include_router(preferences_router)

    @application.get("/health")
    async def health():
        """Liveness. Touches nothing, needs no credential.

        Kept free of the database so the load-balancer probe cannot be turned
        into database load, and so the frontend can call it on page load without
        waiting on a connection.
        """
        return {"status": "ok"}

    @application.get("/health/db")
    async def health_db():
        """Readiness, and what the scheduled warmer calls.

        The query is the point. Neon suspends its compute when idle and takes
        0.5-2s to come back, which lands on whoever makes the first request —
        so the warmer has to actually touch the database, not just the function.

        That costs compute-hours, which is why the schedule is a working-day
        window rather than every five minutes forever: measured usage is 0.77 of
        100 CU-hours per month, and at the 0.25 CU minimum, staying awake around
        the clock would be 182 of them. See infra/infra/crud_api_stack.py.
        """
        pg = get_pg_client()
        await pg.connect()
        await pg.fetchval("SELECT 1")
        return {"status": "ok", "db": "ok"}

    return application
