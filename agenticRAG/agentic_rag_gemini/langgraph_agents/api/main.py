"""FastAPI layer for the LangGraph v2.5 pipeline.

POST /chat        → SSE stream (stage + tool + token + done events)
GET  /health      → liveness (no dependency checks)
GET  /health/detailed → readiness (parallel DB/Redis/LLM checks, 3s timeout each)
GET  /tts/{task_id}/result   → poll Redis for TTS result (fallback)
GET  /sessions    → list user sessions
POST /sessions/{session_id}/resume → load session + populate STM
"""

from __future__ import annotations

import asyncio
import json
import os
import time
import uuid
from contextlib import asynccontextmanager

import redis as sync_redis
from fastapi import FastAPI, HTTPException, BackgroundTasks, Query
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

from langgraph_agents.api.schemas import (
    ChatRequest, SessionListItem, SessionListResponse, SessionResumeResponse,
)
from langgraph_agents.api.sse import encode_event, stream_response
from langgraph_agents.graph import build_graph_async
from langgraph_agents.nodes._persona_loader import get_persona
from langgraph_agents.services.vieneu_tts.tasks import synthesize_speech_async
from langgraph_agents.db.session_store import (
    list_user_sessions, load_session_messages,
    populate_stm_from_messages, write_session_turn,
)
from langgraph_agents.shared.logging import (
    configure_root_logger, get_logger, with_request_id,
)
from langgraph_agents.api.health import run_all_checks

logger = get_logger("langgraph.api")

# ── App-scoped singletons (initialized in lifespan, shared across requests) ──
# Module globals are correct here: graph + redis client live for the FastAPI
# process lifetime, not per-request. ContextVar is wrong scope and breaks
# cross-task visibility (lifespan sets in its own context → request tasks
# spawned later get the default None).

_graph = None
_redis: sync_redis.Redis | None = None

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


def _get_redis() -> sync_redis.Redis:
    global _redis
    if _redis is None:
        # socket_timeout bounds thread occupancy when health check or any other
        # caller hangs on Redis. 5s is generous for normal ops (set/get/ping)
        # but ensures asyncio.to_thread workers don't pile up if Redis goes
        # unresponsive. socket_connect_timeout covers initial TCP handshake.
        _redis = sync_redis.Redis.from_url(
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
    global _graph
    _graph = await build_graph_async()
    _get_redis()
    logger.info("startup_complete", extra={
        "event": "lifespan_complete",
        "graph_loaded": _graph is not None,
    })
    yield
    logger.info("shutdown", extra={"event": "lifespan_end"})


def create_app() -> FastAPI:
    application = FastAPI(title="VVA LangGraph v2.5", lifespan=lifespan)

    # CORS — frontend at port 3000 (or wherever) calls backend cross-origin.
    # Browser sends OPTIONS preflight before POST /chat; without this middleware
    # FastAPI returns 405 Method Not Allowed and the browser blocks the request.
    application.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],          # dev: allow all. Phase 7: lock down to known origins
        allow_credentials=False,      # "*" + credentials disallowed by spec
        allow_methods=["*"],
        allow_headers=["*"],
    )

    @application.get("/health")
    async def health():
        """Liveness — no dependency checks. Kubernetes/load balancer probe."""
        return {"status": "ok"}

    @application.get("/health/detailed")
    async def health_detailed():
        """Readiness — parallel checks with timeouts, breaker status, MCP."""
        graph = _get_graph()
        redis_client = _get_redis()
        result = await run_all_checks(graph, redis_client)
        status_code = 200 if result["all_ok"] else 503
        return JSONResponse(
            content={"status": result["status"], "checks": result["checks"]},
            status_code=status_code,
        )

    @application.post("/chat")
    async def chat(req: ChatRequest, background_tasks: BackgroundTasks):
        graph = _get_graph()
        if graph is None:
            raise HTTPException(503, "Graph not loaded yet")
        request_id = str(uuid.uuid4())
        state = {"messages": [], "errors": [], "retry_count": 0, "total_tokens": 0}
        config = {"configurable": {
            "user_id": req.user_id,
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
                async for sse_event in _stream_chat(req, request_id, config, state, background_tasks):
                    yield sse_event

        return stream_response(event_generator())

    @application.get("/tts/{task_id}/result")
    async def tts_result(task_id: str):
        raw = _get_redis().get(f"task_result:{task_id}")
        if raw is None or not isinstance(raw, (bytes, str)):
            raise HTTPException(404, "Task not ready or expired")
        try:
            return json.loads(raw)
        except (json.JSONDecodeError, TypeError):
            raise HTTPException(500, "Corrupt task result in cache")

    @application.get("/sessions", response_model=SessionListResponse)
    async def list_sessions(user_id: str = Query(...), limit: int = 50):
        rows = await list_user_sessions(user_id=user_id, limit=limit)
        return SessionListResponse(
            sessions=[SessionListItem(**r) for r in rows],
            total=len(rows),
        )

    @application.post("/sessions/{session_id}/resume", response_model=SessionResumeResponse)
    async def resume_session(session_id: str, user_id: str = Query(...)):
        row = await load_session_messages(user_id=user_id, session_id=session_id)
        if not row:
            raise HTTPException(404, "Session not found")

        messages = row["messages"] or []
        await populate_stm_from_messages(session_id, messages)

        return SessionResumeResponse(
            session_id=session_id,
            messages=messages,
            stm_populated=True,
            last_updated=row["updated_at"].isoformat(),
        )

    @application.delete("/sessions/{user_id}/{session_id}")
    async def delete_session(user_id: str, session_id: str):
        """Delete a session row + clear its Redis STM."""
        from langgraph_agents.db.session_store import _to_uuid
        from langgraph_agents.shared import get_pg_client
        pg = get_pg_client()
        await pg.connect()
        result = await pg.execute(
            "DELETE FROM conversations WHERE user_id = $1::uuid AND session_id = $2::uuid",
            _to_uuid(user_id), session_id,
        )
        # Clear Redis STM (best-effort)
        try:
            import redis.asyncio as aioredis
            r = aioredis.from_url("redis://localhost:6379/0")
            await r.delete(f"stm:{session_id}")
            close_fn = getattr(r, "aclose", None) or r.close
            await close_fn()
        except Exception:
            pass
        return {"deleted": session_id, "result": result}

    return application


# ── Chat stream helper ─────────────────────────────────────────────


async def _stream_chat(req, request_id, config, state, background_tasks):
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
        if mode == "updates":
            if not isinstance(payload, dict):
                continue
            for node_name, node_output in payload.items():
                if node_name not in _STAGE_NODES:
                    continue

                extra: dict = {}
                if node_name == "planner" and isinstance(node_output, dict):
                    extra["intent"] = node_output.get("intent")
                    extra["needs_clarification"] = node_output.get("needs_clarification", False)
                if node_name == "grader" and isinstance(node_output, dict):
                    extra["result"] = node_output.get("grader_result")

                yield encode_event(
                    "stage",
                    {"node": node_name, "status": "complete", **extra},
                )

                # synthesizer / error_handler / grader can set final_answer
                if isinstance(node_output, dict) and node_output.get("final_answer"):
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
                user_id=req.user_id,
                session_id=req.session_id,
                user_query=req.query,
                assistant_answer=final_answer,
                intent=final_state.get("intent", ""),
                tokens=final_state.get("total_tokens", 0),
            )
            yield encode_event("session_persisted", {"session_id": req.session_id})
        except Exception as exc:
            logger.warning("session_persist_failed", extra={"error": str(exc)})

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

        async for sse_event in _poll_speech_result(speech_task_id, timeout=60):
            yield sse_event

    elapsed_ms = round((time.time() - t0) * 1000)
    logger.info("chat_complete", extra={
        "elapsed_ms": elapsed_ms,
        "total_tokens": final_state.get("total_tokens", 0),
        "intent": final_state.get("intent", ""),
        "speech_task_id": speech_task_id,
    })

    yield encode_event("done", {
        "request_id": request_id,
        "total_tokens": final_state.get("total_tokens", 0),
        "intent": final_state.get("intent", ""),
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
        raw = _get_redis().get(key)
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
