"""FastAPI layer for the LangGraph v2.4.1 pipeline.

POST /chat        → run graph + fire async TTS via BackgroundTasks
GET  /health      → service health
GET  /tts/{task_id}/result   → poll Redis for TTS result
"""

import json
import logging
import uuid
from contextlib import asynccontextmanager

import redis as sync_redis
from fastapi import FastAPI, HTTPException, BackgroundTasks

from langgraph_agents.api.schemas import ChatRequest, ChatResponse
from langgraph_agents.graph import build_graph_async
from langgraph_agents.nodes._persona_loader import get_persona
from langgraph_agents.services.vieneu_tts.tasks import synthesize_speech_async

logger = logging.getLogger("langgraph.api")

_graph = None
_redis: sync_redis.Redis | None = None


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
    """Factory — use with `uvicorn langgraph_agents.api.main:create_app --factory`."""
    application = FastAPI(title="VVA LangGraph v2.4.1", lifespan=lifespan)

    @application.get("/health")
    async def health():
        return {"status": "ok", "graph_loaded": _graph is not None}

    @application.post("/chat", response_model=ChatResponse)
    async def chat(req: ChatRequest, background_tasks: BackgroundTasks):
        request_id = str(uuid.uuid4())
        state = {
            "messages": [], "errors": [], "retry_count": 0, "total_tokens": 0,
        }
        config = {
            "configurable": {
                "user_id": req.user_id,
                "session_id": req.session_id,
                "query": req.query,
                "persona_id": req.persona_id,
                "output_mode": req.output_mode,
                "request_id": request_id,
                "token_limit": req.token_limit,
            }
        }
        result = await _graph.ainvoke(state, config=config)

        speech_task_id: str | None = None
        if req.output_mode in ("speech", "both") and result.get("final_answer"):
            persona = get_persona(req.persona_id)
            voice_path = persona.get("voice_identity", {}).get("voice_path")
            speech_task_id = str(uuid.uuid4())
            background_tasks.add_task(
                synthesize_speech_async,
                text=result["final_answer"],
                task_id=speech_task_id,
                voice_path=voice_path,
            )

        final_answer = result.get("final_answer") or "Xin lỗi, tôi không thể xử lý yêu cầu này."

        return ChatResponse(
            request_id=request_id,
            final_answer=final_answer,
            intent=result.get("intent", ""),
            confidence=result.get("confidence", 0.0),
            needs_clarification=result.get("needs_clarification", False),
            speech_task_id=speech_task_id,
            total_tokens=result.get("total_tokens", 0),
            grader_result=result.get("grader_result"),
            grader_warning=result.get("grader_warning"),
            errors=result.get("errors", []),
        )

    @application.get("/tts/{task_id}/result")
    async def tts_result(task_id: str):
        raw = _get_redis().get(f"task_result:{task_id}")
        if raw is None or not isinstance(raw, (bytes, str)):
            raise HTTPException(404, "Task not ready or expired")
        try:
            return json.loads(raw)
        except (json.JSONDecodeError, TypeError):
            raise HTTPException(500, "Corrupt task result in cache")

    return application
