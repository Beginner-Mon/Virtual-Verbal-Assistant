"""FastAPI layer for the LangGraph v2.5 pipeline.

POST /chat        → SSE stream (stage + tool + token + done events)
GET  /health      → service health
GET  /tts/{task_id}/result   → poll Redis for TTS result (fallback)
GET  /sessions    → list user sessions
POST /sessions/{session_id}/resume → load session + populate STM
"""

from __future__ import annotations

import asyncio
import json
import logging
import uuid
from contextlib import asynccontextmanager

import redis as sync_redis
from fastapi import FastAPI, HTTPException, BackgroundTasks, Query

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

logger = logging.getLogger("langgraph.api")

_graph = None
_redis: sync_redis.Redis | None = None

# Node names that emit stage events
_STAGE_NODES = {
    "memory", "planner", "retriever_agent", "synthesizer",
    "grader", "conversation", "error_handler",
}


def _get_redis() -> sync_redis.Redis:
    global _redis
    if _redis is None:
        _redis = sync_redis.Redis.from_url("redis://localhost:6379/0")
    return _redis


@asynccontextmanager
async def lifespan(application: FastAPI):
    global _graph
    _graph = await build_graph_async()
    _get_redis()
    yield


def create_app() -> FastAPI:
    application = FastAPI(title="VVA LangGraph v2.5", lifespan=lifespan)

    @application.get("/health")
    async def health():
        return {"status": "ok", "graph_loaded": _graph is not None}

    @application.post("/chat")
    async def chat(req: ChatRequest, background_tasks: BackgroundTasks):
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
        }}

        async def event_generator():
            final_state = {}
            speech_task_id: str | None = None

            async for event in _graph.astream_events(state, config=config, version="v2"):
                ev_type = event["event"]
                name = event.get("name", "")
                meta = event.get("metadata", {})

                # 1. Node start/end → stage events
                if ev_type == "on_chain_start" and name in _STAGE_NODES:
                    yield encode_event("stage", {"node": name, "status": "started"})

                elif ev_type == "on_chain_end" and name in _STAGE_NODES:
                    output = event["data"].get("output", {})
                    extra = {}
                    if name == "planner" and isinstance(output, dict):
                        extra["intent"] = output.get("intent")
                        extra["needs_clarification"] = output.get("needs_clarification", False)
                    if name == "grader" and isinstance(output, dict):
                        extra["result"] = output.get("grader_result")
                    yield encode_event("stage", {"node": name, "status": "complete", **extra})

                # 2. Tool calling events
                elif ev_type == "on_tool_start":
                    yield encode_event("tool_calling", {"tool": name})

                elif ev_type == "on_tool_end":
                    output = event["data"].get("output")
                    count = len(output) if isinstance(output, list) else 1
                    yield encode_event("tool_complete", {"tool": name, "result_count": count})

                # 3. Token stream — only from conversation node
                elif ev_type == "on_chat_model_stream":
                    langgraph_node = meta.get("langgraph_node", "")
                    if langgraph_node == "conversation":
                        chunk = event["data"]["chunk"]
                        content = chunk.content if hasattr(chunk, "content") else str(chunk)
                        if content:
                            yield encode_event("token", {"content": content})

                # 4. Capture final state from root graph end event
                if ev_type == "on_chain_end" and name == "LangGraph":
                    output = event["data"].get("output", {})
                    if isinstance(output, dict):
                        final_state = output

            # ── Post-graph: final answer ──────────────────────────
            final_answer = final_state.get("final_answer") or "Xin lỗi, tôi không thể xử lý yêu cầu này."

            # 5. Eager session write
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
                    logger.warning("Session persist failed: %s", exc)

            # 6. Fire TTS BackgroundTask if needed
            if req.output_mode in ("speech", "both") and final_state.get("final_answer"):
                speech_task_id = str(uuid.uuid4())
                persona = get_persona(req.persona_id)
                voice_path = persona.get("voice_identity", {}).get("voice_path")
                background_tasks.add_task(
                    synthesize_speech_async,
                    text=final_answer,
                    task_id=speech_task_id,
                    voice_path=voice_path,
                )
                yield encode_event("speech_pending", {"task_id": speech_task_id})

                async for sse_event in _poll_speech_result(speech_task_id, timeout=15):
                    yield sse_event

            # 7. Done
            yield encode_event("done", {
                "request_id": request_id,
                "total_tokens": final_state.get("total_tokens", 0),
                "intent": final_state.get("intent", ""),
                "speech_task_id": speech_task_id,
            })

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

    return application


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
