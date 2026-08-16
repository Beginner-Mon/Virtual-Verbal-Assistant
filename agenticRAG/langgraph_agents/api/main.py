"""FastAPI layer for the LangGraph v2.5 pipeline.

POST /chat        → SSE stream (stage + tool + token + done events)
GET  /health      → liveness (no dependency checks)
GET  /health/detailed → readiness (parallel DB/Redis/LLM checks, 3s timeout each)
GET  /tts/{task_id}/result   → poll Redis for TTS result (fallback)
GET  /sessions    → list user sessions
GET  /sessions/{session_id} → load session messages
"""

from __future__ import annotations

import asyncio
import json
import os
import time
import uuid
from contextlib import asynccontextmanager

# P1: force huggingface_hub fully offline BEFORE any HF-importing module loads, so the
# embedding model (intfloat/multilingual-e5-small, already cached) makes ZERO network
# calls on load. huggingface_hub reads HF_HUB_OFFLINE once at import — setting it here (top
# of the uvicorn entrypoint, before `from langgraph_agents...` pulls sentence_transformers)
# is the only reliable point. setdefault → an EMBEDDING_ALLOW_DOWNLOAD=1 run can still
# download on a clean machine (set HF_HUB_OFFLINE=0 explicitly to override).
if os.getenv("EMBEDDING_ALLOW_DOWNLOAD") != "1":
    os.environ.setdefault("HF_HUB_OFFLINE", "1")
    os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")

import redis.asyncio as aioredis
from fastapi import FastAPI, HTTPException, BackgroundTasks, Query, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

from langgraph_agents.api.auth import resolve_user_id
from langgraph_agents.api.billing import router as billing_router
from langgraph_agents.api.schemas import (
    ChatRequest, SessionListItem, SessionListResponse, SessionResumeResponse,
    TTSRequest, TTSTaskResponse,
    UserMemoryCreate, UserMemoryItem, UserMemoryListResponse,
)
from langgraph_agents.api.sse import encode_event, stream_response
from langgraph_agents.graph import build_graph_async
from langgraph_agents.nodes._persona_loader import get_persona, preload_personas_from_db
from langgraph_agents.services.vieneu_tts.tasks import synthesize_speech_async
from langgraph_agents.nodes.summarizer import maybe_summarize
from langgraph_agents.db.session_store import (
    list_user_sessions, load_session_messages,
    populate_stm_from_messages, write_session_turn,
)
from langgraph_agents.shared.logging import (
    configure_root_logger, get_logger, with_request_id,
)
from langgraph_agents.shared import get_pg_client
from langgraph_agents.shared.env import env_source
from langgraph_agents.shared.preflight import run_preflight
from langgraph_agents.api.health import run_all_checks

logger = get_logger("langgraph.api")

# ── App-scoped singletons (initialized in lifespan, shared across requests) ──
# Module globals are correct here: graph + redis client live for the FastAPI
# process lifetime, not per-request. ContextVar is wrong scope and breaks
# cross-task visibility (lifespan sets in its own context → request tasks
# spawned later get the default None).

_graph = None
_redis: aioredis.Redis | None = None

# Keep strong references to fire-and-forget TTS tasks. asyncio.create_task
# returns a Task that the event loop only holds via a WEAK reference — if
# garbage collected mid-execution the task silently vanishes. Storing the
# Task in this set + removing it via done_callback prevents GC.
_pending_tts_tasks: set = set()

# Node names that emit stage events (Phase 6.9: conversation node removed)
_STAGE_NODES = {
    "memory", "planner", "retriever_agent", "synthesizer",
    "grader", "error_handler",
}


def _get_redis() -> aioredis.Redis:
    global _redis
    if _redis is None:
        _redis = aioredis.from_url(
            "redis://localhost:6379/0",
            socket_timeout=5,
            socket_connect_timeout=5,
        )
    return _redis


def _get_graph():
    return _graph


@asynccontextmanager
async def lifespan(application: FastAPI):
    configure_root_logger(level=os.getenv("LOG_LEVEL", "INFO"))
    logger.info("startup", extra={"event": "lifespan_start"})

    # Before anything else: import the packages that are otherwise imported
    # lazily, so a missing one surfaces here instead of mid-incident. See
    # shared/preflight.py. Never raises — an operator who knowingly runs without
    # the Gemini fallback should still get a server.
    run_preflight()

    global _graph
    _graph = await build_graph_async()
    _get_redis()

    # Character personas live in the DB (characters.persona) but get_persona is
    # synchronous, so they are read once here rather than per request. Returns 0
    # and logs when the DB is unreachable; personas/*.md then serve every lookup.
    personas_loaded = await preload_personas_from_db()

    logger.info("startup_complete", extra={
        "event": "lifespan_complete",
        "graph_loaded": _graph is not None,
        "personas_loaded": personas_loaded,
        "config_source": env_source(),
    })
    yield
    if _redis is not None:
        await _redis.aclose()
    logger.info("shutdown", extra={"event": "lifespan_end"})


def create_app() -> FastAPI:
    application = FastAPI(title="VVA LangGraph v2.5", lifespan=lifespan)

    # CORS — frontend at port 3000 (or wherever) calls backend cross-origin.
    # Browser sends OPTIONS preflight before POST /chat; without this middleware
    # FastAPI returns 405 Method Not Allowed and the browser blocks the request.
    _ALLOWED_ORIGINS = [
        o.strip()
        for o in os.getenv(
            "ALLOWED_ORIGINS",
            "http://localhost:3000,http://localhost:8080,http://localhost:5173",
        ).split(",")
        if o.strip()
    ]
    application.add_middleware(
        CORSMiddleware,
        allow_origins=_ALLOWED_ORIGINS,
        allow_credentials=False,      # "*" + credentials disallowed by spec
        allow_methods=["*"],
        allow_headers=["*"],
    )
    application.include_router(billing_router)

    @application.get("/health")
    async def health():
        """Liveness — no dependency checks. Kubernetes/load balancer probe."""
        return {"status": "ok"}

    @application.get("/debug/pgstats")
    async def pgstats(reset: bool = False):
        """How many DB round-trips the last request(s) cost. Off unless VVA_PG_STATS=1.

        Exists because a local database hides this entirely: at <1 ms per query
        the count is invisible, and on a managed database the same code paid
        +28% per turn. Counting beats guessing — an estimate from total latency
        was wrong by ~10x.
        """
        from ..db.postgres import STATS, STATS_ENABLED

        if not STATS_ENABLED:
            return {"enabled": False, "hint": "start the server with VVA_PG_STATS=1"}
        payload = {
            "enabled": True,
            "queries": STATS.count,
            "db_seconds": round(STATS.seconds, 3),
            "by_kind": {k: {"n": n, "ms": round(t * 1000)} for k, (n, t) in STATS.by_kind.items()},
        }
        if reset:
            STATS.reset()
        return payload

    @application.get("/health/detailed")
    async def health_detailed():
        """Readiness — parallel checks with timeouts, breaker status, MCP."""
        graph = _get_graph()
        redis_client = _get_redis()
        result = await run_all_checks(graph, redis_client)
        # 503 only when a CRITICAL dep is down (see health.CRITICAL_CHECKS).
        # A failing optional dep reports status "degraded" on a 200 so the
        # load balancer keeps this instance in rotation.
        status_code = 200 if result["all_ok"] else 503
        return JSONResponse(
            content={
                "status": result["status"],
                "checks": result["checks"],
                "degraded": result["degraded"],
            },
            status_code=status_code,
        )

    @application.post("/chat")
    async def chat(req: ChatRequest, request: Request, background_tasks: BackgroundTasks):
        graph = _get_graph()
        if graph is None:
            raise HTTPException(503, "Graph not loaded yet")
        request_id = str(uuid.uuid4())

        uid = await resolve_user_id(request, req.user_id)

        # Lazy STM warm-up: if Redis STM is empty (new session or resumed from
        # history), backfill it from PostgreSQL before the graph runs. This is a
        # prerequisite of /chat — not a side effect of session resume — so it
        # lives here rather than in GET /sessions/{id}.
        try:
            if not await _get_redis().get(f"stm:{req.session_id}"):
                recent = await load_session_messages(
                    user_id=uid, session_id=req.session_id, limit=6,
                )
                if recent and recent["messages"]:
                    await populate_stm_from_messages(req.session_id, recent["messages"])
        except Exception as exc:
            logger.warning("stm_warmup_failed", extra={"error": str(exc)})

        state = {"messages": [], "errors": [], "retry_count": 0, "total_tokens": 0}
        config = {"configurable": {
            "user_id": uid,
            "session_id": req.session_id,
            "query": req.query,
            "persona_id": req.persona_id,
            "output_mode": req.output_mode,
            "request_id": request_id,
            "token_limit": req.token_limit,
            "web_search": req.web_search,
        }}

        async def event_generator():
            with with_request_id(request_id):
                async for sse_event in _stream_chat(req, request_id, config, state, background_tasks, request, uid):
                    yield sse_event

        return stream_response(event_generator())

    @application.post("/tts", response_model=TTSTaskResponse)
    async def tts_synthesize(body: TTSRequest):
        """Fire a TTS job for text the user asked to hear, return its task id.

        Returns immediately rather than blocking: VieNeu runs on CPU at roughly
        18ms per character, so a full clinical answer is 30-45s. The caller polls
        GET /tts/{task_id}/result, which already exists for the /chat path.
        """
        task_id = str(uuid.uuid4())
        persona = get_persona(body.persona_id)
        voice_path = persona.get("voice_identity", {}).get("voice_path")
        task = asyncio.create_task(synthesize_speech_async(
            text=body.text,
            task_id=task_id,
            voice_path=voice_path,
        ))
        # Same reason as the /chat path: without a strong reference the event
        # loop may garbage-collect a task nobody is awaiting.
        _pending_tts_tasks.add(task)
        task.add_done_callback(_pending_tts_tasks.discard)
        return TTSTaskResponse(task_id=task_id)

    @application.get("/tts/{task_id}/result")
    async def tts_result(task_id: str):
        raw = await _get_redis().get(f"task_result:{task_id}")
        if raw is None or not isinstance(raw, (bytes, str)):
            raise HTTPException(404, "Task not ready or expired")
        try:
            return json.loads(raw)
        except (json.JSONDecodeError, TypeError):
            raise HTTPException(500, "Corrupt task result in cache")

    @application.get("/sessions", response_model=SessionListResponse)
    async def list_sessions(request: Request, user_id: str = Query(...), limit: int = 50):
        uid = await resolve_user_id(request, user_id)
        rows = await list_user_sessions(user_id=uid, limit=limit)
        return SessionListResponse(
            sessions=[SessionListItem(**r) for r in rows],
            total=len(rows),
        )

    @application.get("/sessions/{session_id}", response_model=SessionResumeResponse)
    async def get_session(session_id: str, request: Request, user_id: str = Query(...), limit: int = 50):
        uid = await resolve_user_id(request, user_id)
        row = await load_session_messages(user_id=uid, session_id=session_id, limit=limit)
        if not row:
            raise HTTPException(404, "Session not found")
        return SessionResumeResponse(
            session_id=session_id,
            messages=row["messages"] or [],
            stm_populated=False,
            last_updated=row["updated_at"].isoformat(),
        )

    @application.delete("/sessions/{user_id}/{session_id}")
    async def delete_session(user_id: str, session_id: str, request: Request):
        """Delete a session row + clear its Redis STM."""
        uid = await resolve_user_id(request, user_id)
        pg = get_pg_client()
        await pg.connect()
        result = await pg.execute(
            "DELETE FROM conversations WHERE user_id = $1::uuid AND session_id = $2::uuid",
            uid, session_id,
        )
        # Clear Redis STM (best-effort)
        try:
            await _get_redis().delete(f"stm:{session_id}")
        except Exception:
            pass
        return {"deleted": session_id, "result": result}

    @application.post("/users/{user_id}/memory",
                       response_model=UserMemoryItem)
    async def create_user_memory(user_id: str, body: UserMemoryCreate, request: Request):
        """Create a user fact (Tier 1 always-on memory). D14 MVP: user self-reports."""
        uid = await resolve_user_id(request, user_id)
        pg = get_pg_client()
        await pg.connect()

        # Ensure user row exists
        await pg.execute(
            "INSERT INTO users (id) VALUES ($1::uuid) ON CONFLICT (id) DO NOTHING", uid,
        )

        row = await pg.fetchrow(
            """INSERT INTO user_memory (user_id, fact_text, category)
               VALUES ($1::uuid, $2, $3)
               RETURNING id, created_at""",
            uid, body.fact_text, body.category,
        )
        return UserMemoryItem(
            id=str(row["id"]),
            fact_text=body.fact_text,
            category=body.category,
            valid=True,
            created_at=row["created_at"].isoformat(),
        )

    @application.get("/users/{user_id}/memory",
                      response_model=UserMemoryListResponse)
    async def list_user_memory(user_id: str, request: Request):
        """List user facts (valid=true, newest first)."""
        uid = await resolve_user_id(request, user_id)
        pg = get_pg_client()
        await pg.connect()

        rows = await pg.fetch(
            """SELECT id, fact_text, category, valid, created_at
               FROM user_memory
               WHERE user_id = $1::uuid AND valid = true
               ORDER BY created_at DESC
               LIMIT 50""",
            uid,
        )
        return UserMemoryListResponse(facts=[
            UserMemoryItem(
                id=str(r["id"]),
                fact_text=r["fact_text"],
                category=r["category"],
                valid=r["valid"],
                created_at=r["created_at"].isoformat(),
            )
            for r in rows
        ])

    @application.delete("/users/{user_id}/memory/{fact_id}")
    async def delete_user_memory(user_id: str, fact_id: str, request: Request):
        """Hard-delete a user fact. Ownership verified: fact must belong to user."""
        uid = await resolve_user_id(request, user_id)
        pg = get_pg_client()
        await pg.connect()

        result = await pg.execute(
            """DELETE FROM user_memory
               WHERE id = $1::uuid AND user_id = $2::uuid""",
            fact_id, uid,
        )
        if result == "DELETE 0":
            raise HTTPException(404, "Fact not found or not owned by this user")
        return {"deleted": fact_id}

    @application.delete("/sessions/{session_id}/messages/{message_id}")
    async def delete_message(session_id: str, message_id: str, request: Request, user_id: str = Query(...)):
        """GDPR: delete a single message + mark-dirty summaries + fire re-summarize."""
        from langgraph_agents.db.gdpr import delete_message as gdpr_delete_message
        from langgraph_agents.db.gdpr import get_dirty_chunks
        from langgraph_agents.nodes.summarizer import rebuild_dirty_chunk, _pending_summarizer_tasks

        uid = await resolve_user_id(request, user_id)

        # Ownership verify: message must belong to user via session
        pg = get_pg_client()
        await pg.connect()
        owner = await pg.fetchrow(
            """SELECT 1 FROM messages m
               JOIN conversations c ON m.session_id = c.session_id
               WHERE m.id = $1::uuid AND c.user_id = $2::uuid""",
            message_id, uid,
        )
        if not owner:
            raise HTTPException(404, "Message not found or not owned by this user")

        result = await gdpr_delete_message(message_id, session_id)

        # Fire background re-summarize for dirty chunks
        for chunk in await get_dirty_chunks(session_id):
            task = asyncio.create_task(rebuild_dirty_chunk(session_id, chunk["id"]))
            _pending_summarizer_tasks.add(task)
            task.add_done_callback(_pending_summarizer_tasks.discard)

        return result

    @application.delete("/users/{user_id}")
    async def delete_user_endpoint(user_id: str, request: Request):
        """GDPR: hard-delete user + cascade all data + clear Redis STM."""
        from langgraph_agents.db.gdpr import delete_user as gdpr_delete_user

        uid = await resolve_user_id(request, user_id)

        # Collect session_ids for Redis cleanup
        pg = get_pg_client()
        await pg.connect()
        sessions = await pg.fetch(
            "SELECT session_id FROM conversations WHERE user_id = $1::uuid", uid,
        )
        session_ids = [str(r["session_id"]) for r in sessions]

        result = await gdpr_delete_user(uid)

        # Clear Redis STM for all user's sessions
        for sid in session_ids:
            try:
                await _get_redis().delete(f"stm:{sid}")
            except Exception:
                pass

        return result

    return application


# ── Chat stream helper ─────────────────────────────────────────────


async def _stream_chat(req, request_id, config, state, background_tasks, request=None, resolved_user_id=None):
    """Core SSE stream: graph execution + post-processing.

    Wrapped in with_request_id by the caller so every log line from this
    coroutine carries the correct request_id.
    """
    t0 = time.time()
    final_state: dict = {}
    speech_task_id: str | None = None
    conversation_stage_started = False

    graph = _get_graph()
    logger.info("chat_start", extra={"query": req.query[:120], "session_id": req.session_id})

    async for mode, payload in graph.astream(
        state, config, stream_mode=["updates", "custom"]
    ):
        if request is not None and await request.is_disconnected():
            logger.info("client_disconnected", extra={"request_id": request_id})
            return

        if mode == "updates":
            if not isinstance(payload, dict):
                continue
            for node_name, node_output in payload.items():
                if node_name not in _STAGE_NODES:
                    continue

                extra: dict = {}
                if node_name == "planner" and isinstance(node_output, dict):
                    extra["required_outputs"] = node_output.get("required_outputs")
                    extra["needs_clarification"] = node_output.get("needs_clarification", False)
                if node_name == "grader" and isinstance(node_output, dict):
                    extra["result"] = node_output.get("grader_result")

                yield encode_event(
                    "stage",
                    {"node": node_name, "status": "complete", **extra},
                )

                if isinstance(node_output, dict):
                    # Capture tracking fields from any node as they arrive
                    if "required_outputs" in node_output:
                        final_state["required_outputs"] = node_output["required_outputs"]
                    if "grader_result" in node_output:
                        final_state["grader_result"] = node_output["grader_result"]
                    # synthesizer / error_handler / grader can set final_answer
                    if node_output.get("final_answer"):
                        final_state.update(node_output)

        elif mode == "custom":
            if isinstance(payload, dict) and "content" in payload:
                if not conversation_stage_started:
                    yield encode_event(
                        "stage",
                        {"node": "synthesizer", "status": "started"},
                    )
                    conversation_stage_started = True
                yield encode_event("token", {"content": payload["content"]})

    final_answer = final_state.get("final_answer") or "Xin lỗi, tôi không thể xử lý yêu cầu này."

    # Eager session write
    if final_state.get("final_answer"):
        try:
            await write_session_turn(
                user_id=resolved_user_id or req.user_id,
                session_id=req.session_id,
                user_query=req.query,
                assistant_answer=final_answer,
                total_tokens=final_state.get("total_tokens", 0),
                grader_result=final_state.get("grader_result", "pass"),
            )
            yield encode_event("session_persisted", {"session_id": req.session_id})
        except Exception as exc:
            logger.warning("session_persist_failed", extra={"error": str(exc)})

    # Background summarizer M.5 — fire-and-forget after session write
    if final_state.get("final_answer"):
        try:
            await maybe_summarize(req.session_id)
        except Exception as exc:
            logger.warning("summarizer_check_failed", extra={"error": str(exc)})

    # Fire TTS if needed
    if req.output_mode in ("speech", "both") and final_state.get("final_answer"):
        speech_task_id = str(uuid.uuid4())
        persona = get_persona(req.persona_id)
        voice_path = persona.get("voice_identity", {}).get("voice_path")
        # asyncio.create_task fires immediately (parallel with the polling loop
        # below). FastAPI BackgroundTasks would run only AFTER the streaming
        # response generator returns — but we're still streaming, so the task
        # would never start until our 15s poll already timed out. Wrong order.
        # Strong reference via _pending_tts_tasks prevents GC of the task.
        _tts_task = asyncio.create_task(synthesize_speech_async(
            text=final_answer,
            task_id=speech_task_id,
            voice_path=voice_path,
        ))
        _pending_tts_tasks.add(_tts_task)
        _tts_task.add_done_callback(_pending_tts_tasks.discard)
        yield encode_event("speech_pending", {"task_id": speech_task_id})

        # Must outlast the TTS client's own timeout (services.vieneu_tts.timeout,
        # 120s). Give up sooner and we emit speech_failed while the task is still
        # running, then it writes speech_ready to Redis that nobody reads.
        async for sse_event in _poll_speech_result(speech_task_id, timeout=130):
            yield sse_event

    elapsed_ms = round((time.time() - t0) * 1000)
    logger.info("chat_complete", extra={
        "elapsed_ms": elapsed_ms,
        "total_tokens": final_state.get("total_tokens", 0),
        "required_outputs": final_state.get("required_outputs", []),
        "speech_task_id": speech_task_id,
    })

    yield encode_event("done", {
        "request_id": request_id,
        "total_tokens": final_state.get("total_tokens", 0),
        "required_outputs": final_state.get("required_outputs", []),
        "speech_task_id": speech_task_id,
    })


# ── Speech polling helper ──────────────────────────────────────────


async def _poll_speech_result(task_id: str, timeout: float = 15.0):
    """Poll Redis task_result:{task_id} every 250ms up to timeout.

    Yields:
        - encode_event("speech_ready", {...}) when payload event="speech_ready"
        - encode_event("speech_failed", {...}) when payload event="speech_failed"
        - encode_event("speech_failed", ...) on timeout
    """
    deadline = asyncio.get_event_loop().time() + timeout
    key = f"task_result:{task_id}"

    while asyncio.get_event_loop().time() < deadline:
        raw = await _get_redis().get(key)
        if raw is not None and isinstance(raw, (bytes, str)):
            try:
                payload = json.loads(raw)
                event_name = payload.get("event", "speech_failed")
                yield encode_event(event_name, payload)
                return
            except json.JSONDecodeError:
                pass
        await asyncio.sleep(0.25)

    yield encode_event("speech_failed", {
        "task_id": task_id,
        "error": f"TTS task timeout after {timeout}s",
    })
