"""Unified main API endpoint for the complete three-service pipeline.

This is the single entry point for the frontend. It coordinates:
1. AgenticRAG (port 8000) — query processing and text generation
2. DART (port 5001, WSL/Linux) — text-to-motion generation

Frontend calls: POST /answer { query, user_id }
Response: { text_answer, motion, generation_time_ms, errors }

NOTE: DART input is currently hardcoded to "jump*20" for integration testing.
"""

import time
import logging
from typing import Optional, List, Dict, Any

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

# Hardcoded DART motion prompt (for integration testing)
DART_HARDCODED_PROMPT = "jump*20"

# HTTP timeout (seconds) for downstream service calls
DOWNSTREAM_TIMEOUT = 60.0


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
    conversation_history: Optional[List[ConversationTurn]] = Field(
        None, description="Previous conversation turns"
    )


class MotionMetadata(BaseModel):
    """Motion output metadata returned from DART."""

    motion_file_url: str = Field(..., description="URL to download the .npz motion file")
    num_frames: int = Field(..., description="Total number of motion frames")
    fps: int = Field(..., description="Frames per second (always 30 for DART)")
    duration_seconds: float = Field(..., description="Total clip duration in seconds")
    text_prompt: str = Field(..., description="The prompt that was sent to DART")


class AnswerResponse(BaseModel):
    """Combined response from AgenticRAG + DART."""

    text_answer: str = Field(..., description="Text response from AgenticRAG")
    motion: Optional[MotionMetadata] = Field(None, description="Motion output from DART")
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


async def call_dart(
    client: httpx.AsyncClient,
    text_prompt: str,
) -> Dict[str, Any]:
    """POST to DART /generate and return the parsed JSON body."""
    payload = {
        "text_prompt": text_prompt,
        "guidance_scale": 5.0,
        "num_steps": 50,
        "respacing": "ddim50",
    }

    logger.info(f"[DART] → POST {DART_URL}/generate  prompt={text_prompt!r}")
    resp = await client.post(f"{DART_URL}/generate", json=payload)
    resp.raise_for_status()
    data = resp.json()
    logger.info(f"[DART] ← {resp.status_code} OK  frames={data.get('num_frames')}")
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


@app.post("/answer", response_model=AnswerResponse, summary="Get text answer + motion")
async def get_answer(request: AnswerRequest) -> AnswerResponse:
    """
    Fan out to AgenticRAG and DART **in parallel**, then merge results.

    Flow:
    ┌─ AgenticRAG POST /query  (uses user query) ─┐
    │                                              │  both start simultaneously
    └─ DART POST /generate    (hardcoded: jump*20)─┘
                       ↓
            merge → AnswerResponse
    """
    t_start = time.perf_counter()

    # Convert history to the dict format expected by AgenticRAG
    history: Optional[List[Dict[str, str]]] = None
    if request.conversation_history:
        history = [{"role": t.role, "content": t.content} for t in request.conversation_history]

    errors: Dict[str, str] = {}
    rag_data: Optional[Dict[str, Any]] = None
    dart_data: Optional[Dict[str, Any]] = None

    # ── Fan-out: call both services concurrently ─────────────────────────────
    import asyncio

    async with httpx.AsyncClient(timeout=DOWNSTREAM_TIMEOUT) as client:
        rag_task = asyncio.create_task(
            call_agenticrag(client, request.query, request.user_id, history)
        )
        dart_task = asyncio.create_task(
            call_dart(client, DART_HARDCODED_PROMPT)
        )

        # Gather — return_exceptions=True so one failure doesn't kill the other
        results = await asyncio.gather(rag_task, dart_task, return_exceptions=True)

    rag_result, dart_result = results

    # ── Unpack AgenticRAG result ─────────────────────────────────────────────
    if isinstance(rag_result, Exception):
        logger.error(f"[AgenticRAG] failed: {rag_result}")
        errors["agenticrag"] = str(rag_result)
        text_answer = "[AgenticRAG unavailable — check that port 8000 is running]"
    else:
        rag_data = rag_result
        text_answer = rag_data.get("text_answer", "")

    # ── Unpack DART result ────────────────────────────────────────────────────
    motion: Optional[MotionMetadata] = None
    if isinstance(dart_result, Exception):
        logger.error(f"[DART] failed: {dart_result}")
        errors["dart"] = str(dart_result)
    else:
        dart_data = dart_result
        motion = MotionMetadata(
            motion_file_url=dart_data.get("motion_file_url", ""),
            num_frames=dart_data.get("num_frames", 0),
            fps=dart_data.get("fps", 30),
            duration_seconds=dart_data.get("duration_seconds", 0.0),
            text_prompt=dart_data.get("text_prompt", DART_HARDCODED_PROMPT),
        )

    generation_time_ms = (time.perf_counter() - t_start) * 1000
    logger.info(
        f"[{request.user_id}] Completed in {generation_time_ms:.0f}ms  "
        f"rag={'ok' if rag_data else 'ERROR'}  dart={'ok' if dart_data else 'ERROR'}"
    )

    return AnswerResponse(
        text_answer=text_answer,
        motion=motion,
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
            "dart": f"{DART_URL}/generate  (hardcoded prompt: '{DART_HARDCODED_PROMPT}')",
        },
        "endpoints": {
            "POST /answer": "Fan-out to AgenticRAG + DART, merge and return",
            "GET /health": "Ping both downstream services",
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
    logger.info(f"  DART       : {DART_URL}  (WSL/Linux, DART env)")
    logger.info("")
    logger.info(f"DART motion prompt is hardcoded to: '{DART_HARDCODED_PROMPT}'")
    logger.info("")
    logger.info("Frontend: POST http://localhost:8080/answer")
    logger.info("")

    uvicorn.run(
        app,
        host="0.0.0.0",
        port=8080,
        log_level="info",
    )
