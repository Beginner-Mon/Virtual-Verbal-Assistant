"""FastAPI layer for the LangGraph v2.5 pipeline.

POST /chat        → SSE stream (stage + tool + token + done events)
POST /tts         → fire a TTS job, returns a task id
GET  /tts/{task_id}/result   → poll Redis for TTS result (fallback)
GET  /health      → liveness (no dependency checks)
GET  /health/detailed → readiness (parallel DB/Redis/LLM checks, 3s timeout each)
DELETE /sessions/{sid}/messages/{mid}, DELETE /me → GDPR

Sessions and user memory are not here: they come from api/routes_crud.py, which
this app mounts and the CRUD Lambda (api/crud_app.py) serves on its own. What
stays in this file needs the graph, or fires background work after responding —
which a Lambda cannot do, since the sandbox freezes when the response returns.
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
from fastapi import FastAPI, HTTPException, BackgroundTasks, Depends, Request
from fastapi.responses import JSONResponse

from langgraph_agents.api.auth import current_user_id, verify_auth_config
from langgraph_agents.api.billing import router as billing_router
from langgraph_agents.api.crud_app import add_cors
from langgraph_agents.api.motion_status import motion_status
from langgraph_agents.api.routes_characters import router as characters_router
from langgraph_agents.api.routes_crud import router as crud_router
from langgraph_agents.api.schemas import (
    ChatRequest, TTSRequest, TTSTaskResponse,
)
from langgraph_agents.api.sse import encode_event, stream_response
from langgraph_agents.graph import build_graph_async
from langgraph_agents.nodes._persona_loader import (
    get_persona, get_ui_string, preload_personas_from_db,
)
from langgraph_agents.services.vieneu_tts.tasks import synthesize_speech_async
from langgraph_agents.services.vieneu_tts.voice import resolve_voice
from langgraph_agents.nodes.summarizer import maybe_summarize
from langgraph_agents.db.session_store import (
    load_session_messages, populate_stm_from_messages, write_session_turn,
)
from langgraph_agents.shared.logging import (
    configure_root_logger, get_logger, with_request_id,
)
from langgraph_agents.shared import get_pg_client
from langgraph_agents.shared.env import env_source
from langgraph_agents.shared.stm import get_stm
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


def tts_enabled() -> bool:
    """Whether a VieNeu-TTS service is configured for this deployment.

    Absence of VIENEU_TTS_URL is the switch, rather than a separate ENABLE_TTS
    flag, because the two can disagree: a flag saying "on" with no URL to call
    is a deployment that fails one request at a time instead of at startup.
    One value, and it is the one the code actually needs.

    TTS is unhosted as of 21-08 (Owner deferred it), so the deployed agent runs
    with this false. Local development is unaffected — `services/vieneu_tts`
    still defaults to localhost:5000 when the variable IS set.
    """
    return bool(os.getenv("VIENEU_TTS_URL", "").strip())


def _get_redis() -> aioredis.Redis:
    """Redis for TTS task results ONLY. Short-term memory goes through get_stm().

    The two used to share this client and a hardcoded localhost URL. They are
    separated because they have different requirements: STM is one small read
    and one write per turn, so it tolerates any key-value store and now runs on
    DynamoDB when deployed (shared/stm.py). `_poll_speech_result` polls every
    250ms for up to 130 seconds — 520 reads per answer — which is a genuine
    Redis-shaped workload and the one place where the latency argument holds.

    Nothing reaches this while TTS is off (VIENEU_TTS_URL unset), and the URL is
    read from the environment rather than hardcoded so that turning TTS back on
    does not require finding this line.
    """
    global _redis
    if _redis is None:
        _redis = aioredis.from_url(
            os.getenv("REDIS_URL", "redis://localhost:6379/0"),
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

    # Allowed to raise, and first: a process that cannot verify tokens should
    # fail its deploy rather than answer 401 to everyone and look like a user
    # problem. Unlike run_preflight() below, this one is not survivable.
    verify_auth_config()

    # Before anything else: import the packages that are otherwise imported
    # lazily, so a missing one surfaces here instead of mid-incident. See
    # shared/preflight.py. Never raises — an operator who knowingly runs without
    # the Gemini fallback should still get a server.
    run_preflight()

    global _graph
    _graph = await build_graph_async()

    # `_get_redis()` used to be called here. Removed 21-08: it only constructed a
    # lazy client (redis.asyncio does not dial until first command), so it proved
    # nothing about reachability while making every deployment look like it
    # needed a Redis. TTS opens its own on first use; short-term memory no longer
    # goes through Redis at all — see shared/stm.py.
    #
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
    # Shared with crud_app so the two deployments cannot disagree about origins.
    add_cors(application)
    application.include_router(billing_router)

    # The session and user-memory routes, defined once in routes_crud.py and
    # served here as well as by the CRUD Lambda. Mounting the same router is
    # what keeps local development exercising the code that gets deployed.
    application.include_router(characters_router)
    application.include_router(crud_router)

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
    async def chat(
        req: ChatRequest,
        request: Request,
        background_tasks: BackgroundTasks,
        uid: str = Depends(current_user_id),
    ):
        graph = _get_graph()
        if graph is None:
            raise HTTPException(503, "Graph not loaded yet")
        request_id = str(uuid.uuid4())

        # Lazy STM warm-up: if Redis STM is empty (new session or resumed from
        # history), backfill it from PostgreSQL before the graph runs. This is a
        # prerequisite of /chat — not a side effect of session resume — so it
        # lives here rather than in GET /sessions/{id}.
        #
        # The PostgreSQL read comes FIRST and is also the existence check.
        # DELETE /sessions/{id} runs on the CRUD Lambda, which has no Redis, so
        # it cannot clear stm:{session_id} — a deleted session can leave its
        # short-term memory behind. Trusting a cache whose row is gone would let
        # a deleted conversation keep talking, so a session with no rows gets its
        # STM key dropped rather than reused.
        try:
            recent = await load_session_messages(
                user_id=uid, session_id=req.session_id, limit=6,
            )
            if recent and recent["messages"]:
                if not await get_stm().get(req.session_id):
                    await populate_stm_from_messages(req.session_id, recent["messages"])
            else:
                # No rows: either a brand-new session (nothing to drop) or one
                # that was deleted while its STM survived.
                await get_stm().delete(req.session_id)
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

        503 rather than a task id when no TTS service is configured. Handing back
        an id would be worse than an error: the caller would poll a key that
        nothing will ever write, and give up only on its own timeout.
        """
        if not tts_enabled():
            raise HTTPException(
                503, "text-to-speech is not configured on this deployment",
            )

        task_id = str(uuid.uuid4())
        persona = get_persona(body.persona_id)
        # No query to fall back on here — /tts is given loose text, not a turn.
        # The character's own language is the last resort instead.
        voice_path, language = resolve_voice(
            body.persona_id,
            body.text,
            persona_lang=persona.get("voice_identity", {}).get("language", "vi"),
        )
        task = asyncio.create_task(synthesize_speech_async(
            text=body.text,
            task_id=task_id,
            voice_path=voice_path,
            language=language,
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

    @application.get("/motion/{job_id}")
    async def motion_status_endpoint(job_id: str, uid: str = Depends(current_user_id)):
        """Poll target for a motion job kimodo_node (nodes/kimodo.py) enqueued.

        `Depends(current_user_id)` here is a second, redundant-looking check —
        API Gateway's Cognito authorizer (rest_api_stack.py) already rejects an
        unauthenticated request before it reaches this Lambda. It stays anyway
        for the same reason /chat and the GDPR routes carry it: local
        development runs this app directly, with no API Gateway in front of it
        at all, so this is the only gate that exists outside AWS.
        job_id is content-addressed (an HMAC of prompt+params, not tied to a
        session — see vva_motion/jobs.py's compute_job_id), so there is no
        per-row ownership to check beyond "is this caller authenticated".

        motion_status() makes synchronous boto3 calls (DynamoDB, and SSM/
        CloudFront signing on the done path) — asyncio.to_thread keeps them
        off the event loop, the same pattern nodes/kimodo.py already uses for
        the same client.
        """
        result = await asyncio.to_thread(motion_status, job_id)
        if result["status"] == "not_found":
            raise HTTPException(404, "job not found")
        return result

    # Sessions and user memory are NOT defined here — they come from
    # crud_router (api/routes_crud.py), mounted above, and are also what the
    # CRUD Lambda serves. The two GDPR routes below stay put because both fire
    # background work after responding: on Lambda the sandbox freezes the moment
    # the response is returned, so the task would never run.
    #
    # ── Gated, and the gate is the point ─────────────────────────────────────
    #
    # These two routes DELETE USER DATA and default to OFF. Until 21-08 that was
    # academic: this module ran only on a developer machine. Deploying the agent
    # to Lambda changes it — an ungated create_app() would publish DELETE /me the
    # same day /chat ships.
    #
    # Owner deferred account deletion on 21-08 specifically to reconsider its
    # security, and the reconsideration is not cosmetic. What delete_user()
    # actually does today is delete PostgreSQL rows and nothing else: the Cognito
    # user survives, so the person can still sign in and routes_crud.py:113
    # recreates their `users` row on the next write. Calling that "delete my
    # account" in a UI would be telling users something untrue. It also leaves
    # the DynamoDB UserMappings row, so re-registering with the same email links
    # the new sign-up to the OLD identity.
    #
    # Default false rather than "remember to set it false in prod", because the
    # failure mode of forgetting is unrecoverable — there is no per-user backup.
    # See docs/tracking/tech-debt.md and docs/plans/langgraph-agent-hosting.md §7.
    if os.getenv("ENABLE_GDPR_ROUTES", "false").strip().lower() != "true":
        logger.info("gdpr_routes_disabled", extra={
            "reason": "ENABLE_GDPR_ROUTES is not 'true'",
            "routes": ["DELETE /me", "DELETE /sessions/{sid}/messages/{mid}"],
        })
        return application

    logger.warning("gdpr_routes_enabled", extra={
        "routes": ["DELETE /me", "DELETE /sessions/{sid}/messages/{mid}"],
        "note": "these routes delete user data and cannot be undone",
    })

    @application.delete("/sessions/{session_id}/messages/{message_id}")
    async def delete_message(
        session_id: str,
        message_id: str,
        uid: str = Depends(current_user_id),
    ):
        """GDPR: delete a single message + mark-dirty summaries + fire re-summarize."""
        from langgraph_agents.db.gdpr import delete_message as gdpr_delete_message
        from langgraph_agents.db.gdpr import get_dirty_chunks
        from langgraph_agents.nodes.summarizer import rebuild_dirty_chunk, _pending_summarizer_tasks

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

    @application.delete("/me")
    async def delete_user_endpoint(uid: str = Depends(current_user_id)):
        """GDPR: hard-delete the calling user + cascade all data + clear Redis STM.

        Deletes whoever the token says you are — there is no path parameter,
        because a route that takes a user id to delete is one authorization bug
        away from deleting somebody else's account.
        """
        from langgraph_agents.db.gdpr import delete_user as gdpr_delete_user

        # Collect session_ids for Redis cleanup
        pg = get_pg_client()
        await pg.connect()
        sessions = await pg.fetch(
            "SELECT session_id FROM conversations WHERE user_id = $1::uuid", uid,
        )
        session_ids = [str(r["session_id"]) for r in sessions]

        result = await gdpr_delete_user(uid)

        # Clear cached short-term memory for all the user's sessions. The store
        # swallows and reports its own failures, so a cache that is down cannot
        # turn a successful deletion into a 500 — the authoritative rows are
        # already gone by this point.
        for sid in session_ids:
            await get_stm().delete(sid)

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
                if node_name == "kimodo" and isinstance(node_output, dict):
                    # Capture only — kimodo is deliberately absent from
                    # _STAGE_NODES (R26): persisting motion_job_id to the DB
                    # must not change what goes out over SSE, so this branch
                    # emits no event and runs ahead of the _STAGE_NODES gate
                    # below. Do not "fix" the asymmetry by adding kimodo to
                    # _STAGE_NODES — that was considered and rejected because
                    # it would ship a client-visible stage event with no
                    # consumer (the frontend task that would use it is
                    # deferred out of this run).
                    #
                    # Defensive: a malformed/unexpected ToolMessage must not
                    # raise inside the streaming loop and kill the response —
                    # a missing motion id is a far smaller failure than a
                    # broken chat turn.
                    try:
                        for msg in node_output.get("messages", []):
                            content = getattr(msg, "content", None)
                            if not isinstance(content, str):
                                continue
                            job_payload = json.loads(content)
                            if job_payload.get("state") in ("queued", "cache_hit"):
                                job_id = job_payload.get("job_id")
                                if job_id:
                                    final_state["motion_job_id"] = job_id
                    except Exception as exc:
                        logger.warning("kimodo_job_id_capture_failed", extra={"error": str(exc)})

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

    final_answer = final_state.get("final_answer") or get_ui_string(
        req.persona_id or "eca_default", "error_unavailable"
    )

    # Eager session write
    if final_state.get("final_answer"):
        try:
            await write_session_turn(
                user_id=resolved_user_id,
                session_id=req.session_id,
                user_query=req.query,
                assistant_answer=final_answer,
                total_tokens=final_state.get("total_tokens", 0),
                grader_result=final_state.get("grader_result", "pass"),
                motion_job_id=final_state.get("motion_job_id"),
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
        if not tts_enabled():
            # Emit speech_disabled, and specifically NOT speech_pending.
            #
            # The difference matters to the client, not just to the log.
            # speech_pending is a promise that speech_ready or speech_failed
            # follows; the UI shows "generating audio" and waits for it. With no
            # TTS service configured, nothing would ever follow, so the promise
            # would be a permanent spinner.
            #
            # The alternative — letting the code below run anyway — is worse than
            # a spinner. _poll_speech_result waits 130 seconds, and on Lambda a
            # streaming invocation is billed for its FULL duration even after the
            # client disconnects. Every voice turn would cost two minutes of
            # memory to reach a failure that was knowable up front.
            logger.info("speech_disabled", extra={"reason": "VIENEU_TTS_URL not set"})
            yield encode_event("speech_disabled", {
                "reason": "text-to-speech is not configured on this deployment",
            })
        else:
            speech_task_id = str(uuid.uuid4())
            persona = get_persona(req.persona_id)
            # The query is the tie-breaker: a reply of "Có." or "OK" carries no
            # language signal, and the synthesizer was told to answer in the
            # query's language anyway.
            voice_path, speech_language = resolve_voice(
                req.persona_id,
                final_answer,
                query=req.query,
                persona_lang=persona.get("voice_identity", {}).get("language", "vi"),
            )
            # asyncio.create_task fires immediately (parallel with the polling
            # loop below). FastAPI BackgroundTasks would run only AFTER the
            # streaming response generator returns — but we're still streaming,
            # so the task would never start until our 15s poll already timed
            # out. Wrong order. Strong reference via _pending_tts_tasks
            # prevents GC of the task.
            _tts_task = asyncio.create_task(synthesize_speech_async(
                text=final_answer,
                task_id=speech_task_id,
                voice_path=voice_path,
                language=speech_language,
            ))
            _pending_tts_tasks.add(_tts_task)
            _tts_task.add_done_callback(_pending_tts_tasks.discard)
            yield encode_event("speech_pending", {"task_id": speech_task_id})

            # Must outlast the TTS client's own timeout
            # (services.vieneu_tts.timeout, 120s). Give up sooner and we emit
            # speech_failed while the task is still running, then it writes
            # speech_ready to Redis that nobody reads.
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
