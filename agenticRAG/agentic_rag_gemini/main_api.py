"""Unified main API endpoint for the complete three-service pipeline.

This is the single entry point for the frontend. It coordinates:
1. AgenticRAG (query processing)
2. SpeechLLm (voice synthesis)
3. DART (motion generation)

Frontend calls: POST /answer { query, user_id }
Response: { text_answer, voice, motion, generation_time_ms }
"""

import logging
from typing import Optional, List, Dict, Any
from contextlib import asynccontextmanager

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
import uvicorn

from orchestration import PipelineOrchestrator, format_pipeline_result
from utils.logger import get_logger

logger = get_logger(__name__)

# ===========================
# Request/Response Models
# ===========================


class ConversationTurn(BaseModel):
    """Single conversation turn."""

    role: str = Field(..., description="'user' or 'assistant'")
    content: str = Field(...)


class AnswerRequest(BaseModel):
    """Request for a complete answer."""

    query: str = Field(..., description="User query")
    user_id: str = Field(default="default", description="User identifier")
    conversation_history: Optional[List[ConversationTurn]] = Field(
        None, description="Previous turns"
    )


class VoiceMetadata(BaseModel):
    """Voice output metadata."""

    file: str = Field(..., description="Path to audio file")
    duration_seconds: float = Field(..., description="Duration in seconds")


class MotionMetadata(BaseModel):
    """Motion output metadata."""

    file: str = Field(..., description="Path to motion file")
    num_frames: int = Field(..., description="Number of frames")
    fps: int = Field(..., description="Frames per second")


class AnswerResponse(BaseModel):
    """Complete answer response from all services."""

    text_answer: str = Field(..., description="Text response")
    voice: Optional[VoiceMetadata] = Field(None, description="Voice output")
    motion: Optional[MotionMetadata] = Field(None, description="Motion output")
    generation_time_ms: float = Field(..., description="Total generation time")
    errors: Optional[Dict[str, str]] = Field(None, description="Service errors if any")


# ===========================
# Global State
# ===========================

pipeline_orchestrator: Optional[PipelineOrchestrator] = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    """FastAPI lifespan for startup/shutdown."""
    global pipeline_orchestrator
    logger.info("Starting unified API server...")
    pipeline_orchestrator = PipelineOrchestrator()
    yield
    logger.info("Shutting down unified API server...")


# ===========================
# FastAPI Application
# ===========================

app = FastAPI(
    title="Unified Multi-Service Pipeline API",
    description="Single entry point for AgenticRAG, SpeechLLm, and DART",
    version="1.0.0",
    lifespan=lifespan,
)

# Enable CORS for test UI
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


@app.post("/answer", response_model=AnswerResponse, summary="Get complete answer with voice and motion")
async def get_answer(request: AnswerRequest) -> AnswerResponse:
    """Get a complete answer with text, voice, and motion.

    This endpoint coordinates:
    1. AgenticRAG for query understanding and response generation
    2. SpeechLLm for voice synthesis
    3. DART for motion generation

    Args:
        request: AnswerRequest with query and optional history

    Returns:
        AnswerResponse with text, voice, motion, and generation time

    Example:
        POST /answer
        {
            "query": "How do I walk forward?",
            "user_id": "user123",
            "conversation_history": []
        }

        Response:
        {
            "text_answer": "To walk forward, move one foot in front of the other...",
            "voice": {
                "file": "/path/to/audio.wav",
                "duration_seconds": 5.2
            },
            "motion": {
                "file": "/path/to/motion.npy",
                "num_frames": 160,
                "fps": 30
            },
            "generation_time_ms": 3500.0,
            "errors": null
        }
    """
    global pipeline_orchestrator

    if pipeline_orchestrator is None:
        raise HTTPException(status_code=500, detail="Pipeline not initialized")

    try:
        logger.info(f"[{request.user_id}] Processing answer request: {request.query[:100]}...")

        # Convert history to dict format if provided
        history = None
        if request.conversation_history:
            history = [turn.dict() for turn in request.conversation_history]

        # Process through pipeline
        result = pipeline_orchestrator.process_query_sync(
            query=request.query,
            user_id=request.user_id,
            conversation_history=history,
        )

        # Format response
        formatted = format_pipeline_result(result)

        response = AnswerResponse(
            text_answer=formatted["text_answer"],
            voice=VoiceMetadata(**formatted["voice"]) if formatted["voice"] else None,
            motion=MotionMetadata(**formatted["motion"]) if formatted["motion"] else None,
            generation_time_ms=formatted["generation_time_ms"],
            errors=formatted["errors"],
        )

        logger.info(
            f"[{request.user_id}] Answer request complete in {response.generation_time_ms:.1f}ms"
        )

        return response

    except Exception as e:
        logger.error(f"[{request.user_id}] Error processing answer request: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/health", summary="Health check")
async def health_check() -> Dict[str, str]:
    """Health check endpoint."""
    return {
        "status": "healthy",
        "service": "unified-pipeline",
        "orchestrator": "ready" if pipeline_orchestrator else "not-initialized",
    }


@app.get("/info", summary="Get service info")
async def get_info() -> Dict[str, Any]:
    """Get service information."""
    return {
        "service": "Unified Multi-Service Pipeline",
        "version": "1.0.0",
        "description": "Coordinates AgenticRAG, SpeechLLm, and DART",
        "components": {
            "agenticrag": "Query processing and decision making",
            "speechllm": "Voice synthesis",
            "dart": "Motion generation",
        },
        "endpoints": {
            "POST /answer": "Get complete answer with text, voice, and motion",
            "GET /health": "Health check",
            "GET /info": "Service information",
        },
        "latency_target_ms": 5000,
        "orchestrator": "ready" if pipeline_orchestrator else "not-initialized",
    }


# ===========================
# Main
# ===========================

if __name__ == "__main__":
    logger.info("Starting Unified Multi-Service Pipeline API on port 8080...")
    logger.info("")
    logger.info("Make sure you have these services running:")
    logger.info("  1. SpeechLLm:      python SpeechLLm/api_server.py         (port 5000)")
    logger.info("  2. DART:           python text-to-motion/DART/api_server.py (port 5001, Linux)")
    logger.info("  3. AgenticRAG:     python agenticRAG/agentic_rag_gemini/api_server.py (port 8000)")
    logger.info("")
    logger.info("Frontend can call: POST http://localhost:8080/answer")
    logger.info("")

    uvicorn.run(
        app,
        host="0.0.0.0",
        port=8080,
        log_level="info",
    )
