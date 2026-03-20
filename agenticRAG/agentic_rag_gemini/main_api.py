"""Unified main API endpoint for the complete three-service pipeline.

This is the single entry point for the frontend. It coordinates:
1. AgenticRAG (port 8000) — query processing and text generation
2. DART (port 5001, WSL/Linux) — text-to-motion generation

Frontend calls: POST /answer { query, user_id }
Response: { text_answer, motion, generation_time_ms, errors }

NOTE: Motion generation is now handled by AgenticRAG based on extracted exercise names.
"""

import time
import logging
from typing import Optional, List, Dict, Any, Literal

import httpx
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
import uvicorn

from utils.logger import get_logger

logger = get_logger(__name__)

# ===========================
# Service URLs
# ===========================

AGENTIC_RAG_URL = "http://localhost:8000"
DART_URL = "http://localhost:5001"

# HTTP timeout (seconds) for downstream service calls.
# AgenticRAG LLM calls can take 15-30s — must be well above that.
DOWNSTREAM_TIMEOUT = 90.0


# ===========================
# Request / Response Models
# ===========================


class ConversationTurn(BaseModel):
    """Single conversation turn."""

    role: str = Field(..., description="'user' or 'assistant'")
    content: str = Field(..., description="Message content")


class AnswerRequest(BaseModel):
    """Request for a complete answer."""

    query: str = Field(..., description="User query")
    user_id: str = Field(default="default", description="User identifier")
    motion_format: Literal["glb", "npz"] = Field(
        default="glb",
        description="Requested motion output format: 'glb' or 'npz'",
    )
    conversation_history: Optional[List[ConversationTurn]] = Field(
        None, description="Previous conversation turns"
    )


class MotionMetadata(BaseModel):
    """Motion output metadata returned from DART."""

    motion_file_url: str = Field(..., description="URL to download the generated motion file (.glb or .npz)")
    num_frames: int = Field(..., description="Total number of motion frames")
    fps: int = Field(..., description="Frames per second (always 30 for DART)")
    duration_seconds: float = Field(..., description="Total clip duration in seconds")
    text_prompt: str = Field(..., description="The prompt that was sent to DART")


class TTSMetadata(BaseModel):
    """TTS output metadata returned from SpeechLLM."""
    audio_file: str = Field(..., description="Filename of the generated audio file")
    audio_url: str = Field(..., description="URL to download the audio file")
    text: str = Field(..., description="Text that was synthesized")
    emotion: Optional[str] = Field(None, description="Emotion used for synthesis")


class AnswerResponse(BaseModel):
    """Combined response from AgenticRAG + DART + TTS."""

    text_answer: str = Field(..., description="Text response from AgenticRAG")
    exercises: List[Dict[str, str]] = Field(
        default_factory=list, description="List of recommended exercises from AgenticRAG"
    )
    motion: Optional[MotionMetadata] = Field(None, description="Motion output from DART")
    tts: Optional[TTSMetadata] = Field(None, description="Speech output from TTS/SpeechLLM")
    generation_time_ms: float = Field(..., description="Total wall-clock time in ms")
    errors: Optional[Dict[str, str]] = Field(None, description="Per-service errors if any")


# ===========================
# Downstream helpers
# ===========================


async def call_agenticrag(
    client: httpx.AsyncClient,
    query: str,
    user_id: str,
    conversation_history: Optional[List[Dict[str, str]]],
) -> Dict[str, Any]:
    """POST to AgenticRAG /query and return the parsed JSON body."""
    payload: Dict[str, Any] = {
        "query": query,
        "user_id": user_id,
    }
    if conversation_history:
        payload["conversation_history"] = conversation_history

    logger.info(f"[AgenticRAG] → POST {AGENTIC_RAG_URL}/query  query={query[:80]}...")
    resp = await client.post(f"{AGENTIC_RAG_URL}/query", json=payload)
    resp.raise_for_status()
    data = resp.json()
    logger.info(f"[AgenticRAG] ← {resp.status_code} OK")
    return data

# ===========================
# FastAPI Application
# ===========================

app = FastAPI(
    title="Unified Multi-Service Pipeline API",
    description="Single entry point that fans out to AgenticRAG (port 8000) and DART (port 5001)",
    version="2.0.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ===========================
# API Routes
# ===========================



@app.post("/answer", response_model=AnswerResponse, summary="Get text answer + motion + speech")
async def get_answer(request: AnswerRequest) -> AnswerResponse:
    """
    Passes the query to AgenticRAG, which orchestrates text + motion generation,
    then calls TTS (SpeechLLM) to synthesize speech, and returns the merged results.
    """
    t_start = time.perf_counter()

    # Convert history to the dict format expected by AgenticRAG
    history: Optional[List[Dict[str, str]]] = None
    if request.conversation_history:
        history = [{"role": t.role, "content": t.content} for t in request.conversation_history]

    errors: Dict[str, str] = {}
    rag_data: Optional[Dict[str, Any]] = None
    tts_data: Optional[Dict[str, Any]] = None

    import asyncio

    TTS_URL = "http://localhost:5000"  # SpeechLLM TTS API
    requested_motion_format = (request.motion_format or "glb").lower().strip()
    if requested_motion_format not in {"glb", "npz"}:
        raise HTTPException(status_code=400, detail="motion_format must be 'glb' or 'npz'")


    async with httpx.AsyncClient(timeout=DOWNSTREAM_TIMEOUT) as client:
        # 1. Call AgenticRAG
        try:
            rag_data = await call_agenticrag(client, request.query, request.user_id, history)
            text_answer = rag_data.get("text_answer", "")
            exercises   = rag_data.get("exercises", [])
        except Exception as e:
            logger.error(f"[AgenticRAG] failed: {e}")
            errors["agenticrag"] = str(e)
            text_answer = "[AgenticRAG unavailable — check that port 8000 is running]"
            exercises = []
            rag_data = None

        # 2. Run TTS and motion (DART) concurrently if AgenticRAG succeeded
        tts: Optional[TTSMetadata] = None
        motion: Optional[MotionMetadata] = None
        if rag_data:
            # Prepare DART motion prompt and params
            motion_prompt = rag_data.get("exercise_motion_prompt") or rag_data.get("motion_prompt") or "walk*5"
            # Use defaults if not present
            dart_body = {
                "text_prompt": motion_prompt,
                "guidance_scale": 5.0,
                "num_steps": 50,
                "gender": "female",
                "output_format": requested_motion_format,
            }
            # Optionally add respacing/seed if present in rag_data
            if "respacing" in rag_data:
                dart_body["respacing"] = rag_data["respacing"]
            if "seed" in rag_data:
                dart_body["seed"] = rag_data["seed"]

            async def get_motion():
                try:
                    resp = await client.post(f"{DART_URL}/generate", json=dart_body)
                    resp.raise_for_status()
                    dart_data = resp.json()
                    motion_file_url = f"{DART_URL}{dart_data['motion_file_url']}" if dart_data.get("motion_file_url") else ""
                    logger.info(f"[DART] output_format={requested_motion_format} motion_file_url={motion_file_url}")
                    return MotionMetadata(
                        motion_file_url=motion_file_url,
                        num_frames=dart_data.get("num_frames", 0),
                        fps=dart_data.get("fps", 30),
                        duration_seconds=dart_data.get("duration_seconds", 0.0),
                        text_prompt=dart_data.get("text_prompt", motion_prompt),
                    )
                except Exception as e:
                    logger.error(f"[DART] failed: {e}")
                    errors["dart"] = str(e)
                    return None

            async def get_tts():
                try:
                    tts_payload = {"text": text_answer, "user_id": request.user_id}
                    tts_resp = await client.post(f"{TTS_URL}/synthesize", json=tts_payload)
                    tts_resp.raise_for_status()
                    tts_data = tts_resp.json()
                    audio_file = tts_data.get("audio_file", "")
                    audio_url = f"{TTS_URL}/audio/{audio_file}" if audio_file else ""
                    return TTSMetadata(
                        audio_file=audio_file,
                        audio_url=audio_url,
                        text=tts_data.get("text", text_answer),
                        emotion=tts_data.get("emotion", None),
                    )
                except Exception as e:
                    logger.error(f"[TTS] failed: {e}")
                    errors["tts"] = str(e)
                    return None

            motion, tts = await asyncio.gather(get_motion(), get_tts())
        else:
            motion = None
            tts = None

    generation_time_ms = (time.perf_counter() - t_start) * 1000

    # Check if AgenticRAG reported internal errors for DART
    if rag_data and "dart" in rag_data.get("errors", {}):
        errors["dart"] = rag_data["errors"]["dart"]

    logger.info(
        f"[{request.user_id}] Completed in {generation_time_ms:.0f}ms  "
        f"rag={'ok' if rag_data else 'ERROR'}  dart={'ok' if motion else 'none/ERROR'}  tts={'ok' if tts else 'none/ERROR'}"
    )

    return AnswerResponse(
        text_answer=text_answer,
        exercises=exercises,
        motion=motion,
        tts=tts,
        generation_time_ms=round(generation_time_ms, 1),
        errors=errors if errors else None,
    )


@app.get("/health", summary="Health check")
async def health_check() -> Dict[str, Any]:
    """Ping both downstream services and report their status."""
    statuses: Dict[str, str] = {}

    async with httpx.AsyncClient(timeout=5.0) as client:
        for name, base_url in [("agenticrag", AGENTIC_RAG_URL), ("dart", DART_URL)]:
            try:
                r = await client.get(f"{base_url}/health")
                statuses[name] = "ok" if r.status_code == 200 else f"http_{r.status_code}"
            except Exception as exc:
                statuses[name] = f"unreachable ({type(exc).__name__})"

    overall = "healthy" if all(v == "ok" for v in statuses.values()) else "degraded"
    return {"status": overall, "services": statuses}


@app.get("/info", summary="Get service info")
async def get_info() -> Dict[str, Any]:
    """Return configuration and endpoint documentation."""
    return {
        "service": "Unified Multi-Service Pipeline",
        "version": "2.0.0",
        "upstream_services": {
            "agenticrag": f"{AGENTIC_RAG_URL}/query",
        },
        "endpoints": {
            "POST /answer": "Proxy to AgenticRAG and return merged text+motion response",
            "GET /health": "Ping downstream services",
            "GET /info": "This document",
        },
    }


# ===========================
# Main
# ===========================

if __name__ == "__main__":
    logger.info("Starting Unified Pipeline API on port 8080...")
    logger.info("")
    logger.info("Requires these services to be running:")
    logger.info(f"  AgenticRAG : {AGENTIC_RAG_URL}  (Windows, firstconda env)")
    logger.info("")
    logger.info("Frontend: POST http://localhost:8080/answer")
    logger.info("")

    uvicorn.run(
        app,
        host="0.0.0.0",
        port=8080,
        log_level="info",
    )
