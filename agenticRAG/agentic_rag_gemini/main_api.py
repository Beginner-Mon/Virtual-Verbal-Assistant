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
import os
import asyncio
import uuid
import re
from typing import Optional, List, Dict, Any, Literal

import httpx
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
import uvicorn

from utils.logger import get_logger

logger = get_logger(__name__)


def _env_int(name: str, default: int) -> int:
    value = os.getenv(name)
    if value is None:
        return default
    try:
        return int(value)
    except ValueError:
        logger.warning(f"Invalid integer for {name}={value!r}; using default {default}")
        return default


def _env_float(name: str, default: float) -> float:
    value = os.getenv(name)
    if value is None:
        return default
    try:
        return float(value)
    except ValueError:
        logger.warning(f"Invalid float for {name}={value!r}; using default {default}")
        return default

# ===========================
# Service URLs
# ===========================

AGENTIC_RAG_HOST = os.getenv("AGENTIC_RAG_HOST", "localhost")
AGENTIC_RAG_PORT = _env_int("AGENTIC_RAG_PORT", 8000)
DART_HOST = os.getenv("DART_HOST", "localhost")
DART_PORT = _env_int("DART_PORT", 5001)
TTS_HOST = os.getenv("TTS_HOST", "localhost")
TTS_PORT = _env_int("TTS_PORT", 5000)

AGENTIC_RAG_URL = os.getenv("AGENTIC_RAG_URL", f"http://{AGENTIC_RAG_HOST}:{AGENTIC_RAG_PORT}")
DART_URL = os.getenv("DART_URL", f"http://{DART_HOST}:{DART_PORT}")
TTS_URL = os.getenv("TTS_URL", f"http://{TTS_HOST}:{TTS_PORT}")

MAIN_API_HOST = os.getenv("MAIN_API_HOST", "0.0.0.0")
MAIN_API_PORT = _env_int("MAIN_API_PORT", 8080)
MAIN_API_ASYNC_ENRICHMENT = os.getenv("MAIN_API_ASYNC_ENRICHMENT", "true").lower() in {"1", "true", "yes", "on"}
MOTION_DEFAULT_DURATION_SECONDS = _env_float("MOTION_DEFAULT_DURATION_SECONDS", 12.0)


def _normalize_motion_description(prompt: str) -> str:
    normalized = (prompt or "").strip()
    if not normalized:
        return normalized

    # Legacy cleanup: convert "squat*12" or multi-action "a*5,b*3" into plain text.
    cleaned = re.sub(r"\*\s*\d+", "", normalized)
    cleaned = re.sub(r"\s*,\s*", ", ", cleaned)
    cleaned = re.sub(r"\s+", " ", cleaned).strip(" ,")
    return cleaned


def _resolve_motion_duration_seconds(rag_data: Dict[str, Any]) -> float:
    raw = rag_data.get("duration_seconds")
    if raw is None and isinstance(rag_data.get("motion_prompt"), dict):
        mp = rag_data.get("motion_prompt") or {}
        raw = mp.get("duration_seconds") or mp.get("duration_estimate_seconds")
    try:
        value = float(raw) if raw is not None else MOTION_DEFAULT_DURATION_SECONDS
    except (TypeError, ValueError):
        value = MOTION_DEFAULT_DURATION_SECONDS
    return max(1.0, min(value, 120.0))


def _detect_query_language(query: str) -> str:
    """Detect query language and map to en | vi | jp | other."""
    text = (query or "").strip()
    if not text:
        return "other"

    if re.search(r"[\u3040-\u30ff\u31f0-\u31ff\u4e00-\u9fff]", text):
        return "jp"

    vi_chars_pattern = r"[ăâđêôơưĂÂĐÊÔƠƯáàảãạắằẳẵặấầẩẫậéèẻẽẹếềểễệíìỉĩịóòỏõọốồổỗộớờởỡợúùủũụứừửữựýỳỷỹỵ]"
    if re.search(vi_chars_pattern, text):
        return "vi"

    lowered = f" {text.lower()} "
    vi_keywords = (
        " bạn ", " tôi ", " chúng tôi ", " xin chào ", " cảm ơn ", " bài tập ",
        " đau ", " không ", " giúp ", " như thế nào ", " thế nào ",
    )
    if any(k in lowered for k in vi_keywords):
        return "vi"

    if re.search(r"[a-zA-Z]", text):
        if not re.search(r"[\u0400-\u04FF\u0600-\u06FF\u0590-\u05FF\u0900-\u097F\u0E00-\u0E7F]", text):
            return "en"

    return "other"

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


class QueryRequestCompat(BaseModel):
    """Compatibility request so 8080 can accept 8000-style payloads."""

    query: str = Field(..., description="User query")
    user_id: str = Field(default="default", description="User identifier")
    conversation_history: Optional[List[ConversationTurn]] = Field(
        None, description="Previous conversation turns"
    )
    motion_duration_seconds: Optional[float] = Field(
        None,
        description="Compatibility field. Main API infers duration from downstream metadata.",
    )
    stream: bool = Field(False, description="Compatibility field. Streaming is not supported on 8080.")


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

    request_id: str = Field(..., description="Request identifier for status polling")
    status: str = Field(..., description="processing or completed")
    pending_services: List[str] = Field(default_factory=list, description="Background services still running")
    language: str = Field("other", description="Detected language code: en | vi | jp | other")
    selected_strategy: str = Field("unknown", description="The routing strategy, e.g., visualize_motion")
    progress_stage: str = Field("completed", description="Current execution stage for polling visibility")

    text_answer: str = Field(..., description="Text response from AgenticRAG")
    exercises: List[Dict[str, str]] = Field(
        default_factory=list, description="List of recommended exercises from AgenticRAG"
    )
    motion: Optional[MotionMetadata] = Field(None, description="Motion output from DART")
    tts: Optional[TTSMetadata] = Field(None, description="Speech output from TTS/SpeechLLM")
    generation_time_ms: float = Field(..., description="Total wall-clock time in ms")
    errors: Optional[Dict[str, str]] = Field(None, description="Per-service errors if any")


class UnifiedTaskResponseCompat(BaseModel):
    """Task envelope compatible with AgenticRAG /process_query polling contract."""

    task_id: str = Field(..., description="Unique task identifier")
    status: str = Field(..., description="processing | completed | failed")
    progress_stage: str = Field(..., description="queued | text_ready | motion_generation | completed | failed")
    result: Optional[Dict[str, Any]] = Field(None, description="Normalized result payload")
    error: Optional[str] = Field(None, description="Error text when task fails")


TASK_CONTEXT: Dict[str, Dict[str, Any]] = {}
TASK_CONTEXT_LOCK = asyncio.Lock()


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


def _build_motion_from_agenticrag(rag_data: Dict[str, Any]) -> Optional[MotionMetadata]:
    """Map AgenticRAG motion payload to unified MotionMetadata when available."""
    motion = rag_data.get("motion")
    if not isinstance(motion, dict):
        return None

    motion_file = motion.get("motion_file")
    if not motion_file:
        return None

    frames = int(motion.get("frames", 0) or 0)
    fps = int(motion.get("fps", 30) or 30)
    duration_seconds = round(frames / fps, 2) if frames > 0 and fps > 0 else 0.0
    text_prompt = rag_data.get("exercise_motion_prompt") or rag_data.get("query") or ""

    return MotionMetadata(
        motion_file_url=f"{DART_URL}/download/{motion_file}",
        num_frames=frames,
        fps=fps,
        duration_seconds=duration_seconds,
        text_prompt=text_prompt,
    )


async def _generate_motion_from_dart(
    client: httpx.AsyncClient,
    motion_prompt: str,
    duration_seconds: float,
    rag_data: Dict[str, Any],
) -> MotionMetadata:
    """Generate motion by calling DART API directly."""
    normalized_prompt = _normalize_motion_description(motion_prompt)
    dart_body: Dict[str, Any] = {
        "text_prompt": normalized_prompt,
        "duration_seconds": duration_seconds,
        "guidance_scale": 5.0,
        "num_steps": 50,
        "gender": "female",
        "output_format": rag_data.get("motion_format", "glb"),
    }
    if "respacing" in rag_data:
        dart_body["respacing"] = rag_data["respacing"]
    if "seed" in rag_data:
        dart_body["seed"] = rag_data["seed"]

    resp = await client.post(f"{DART_URL}/generate", json=dart_body)
    resp.raise_for_status()
    dart_data = resp.json()
    motion_file_url = f"{DART_URL}{dart_data['motion_file_url']}" if dart_data.get("motion_file_url") else ""
    return MotionMetadata(
        motion_file_url=motion_file_url,
        num_frames=dart_data.get("num_frames", 0),
        fps=dart_data.get("fps", 30),
        duration_seconds=dart_data.get("duration_seconds", 0.0),
        text_prompt=dart_data.get("text_prompt", normalized_prompt),
    )


async def _generate_tts(
    client: httpx.AsyncClient,
    text_answer: str,
    user_id: str,
) -> TTSMetadata:
    """Generate TTS from SpeechLLM."""
    tts_payload = {"text": text_answer, "user_id": user_id}
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


def _model_to_dict(value: Any) -> Dict[str, Any]:
    if value is None:
        return {}
    if isinstance(value, dict):
        return value
    if hasattr(value, "model_dump"):
        return value.model_dump()
    if hasattr(value, "dict"):
        return value.dict()
    return dict(value)


def _extract_motion_file_name(motion_file_url: Optional[str]) -> Optional[str]:
    if not motion_file_url:
        return None
    base = motion_file_url.split("?", 1)[0].rstrip("/")
    if not base:
        return None
    return base.split("/")[-1] or None


def _to_progress_stage(stage: Optional[str], status: str) -> str:
    stage_value = (stage or "").strip().lower()
    status_value = (status or "").strip().lower()
    if status_value == "failed":
        return "failed"
    if status_value == "completed":
        return "completed"
    if stage_value in {"queued", "text_ready", "motion_generation", "completed", "failed"}:
        return stage_value
    if stage_value in {"rag_processing", "voice_synthesis"}:
        return "text_ready"
    return "queued"


def _answer_to_query_payload(answer: AnswerResponse, query: str, user_id: str) -> Dict[str, Any]:
    motion_payload = None
    if answer.motion:
        motion_file_url = answer.motion.motion_file_url
        motion_payload = {
            "motion_file": _extract_motion_file_name(motion_file_url),
            "motion_file_url": motion_file_url,
            "frames": answer.motion.num_frames,
            "fps": answer.motion.fps,
            "duration_seconds": answer.motion.duration_seconds,
            "text_prompt": answer.motion.text_prompt,
        }

    motion_job_payload = None
    if answer.status == "processing":
        motion_job_payload = {
            "job_id": answer.request_id,
            "status": "queued" if answer.progress_stage != "completed" else "completed",
            "motion_file_url": motion_payload.get("motion_file_url") if motion_payload else None,
            "video_url": None,
            "error": None,
        }

    error_text = None
    if answer.errors:
        error_text = "; ".join(f"{k}: {v}" for k, v in answer.errors.items())

    return {
        "query": query,
        "user_id": user_id,
        "language": answer.language,
        "text_answer": answer.text_answer,
        "exercises": answer.exercises,
        "exercise_motion_prompt": answer.motion.text_prompt if answer.motion else None,
        "motion": motion_payload,
        "motion_job": motion_job_payload,
        "orchestrator_decision": {
            "action": "full_pipeline",
            "intent": answer.selected_strategy,
            "confidence": 1.0,
            "language": answer.language,
            "reasoning": "Main API orchestrated downstream services",
            "parameters": {"progress_stage": answer.progress_stage, "request_id": answer.request_id},
        },
        "motion_prompt": {
            "description": answer.motion.text_prompt,
            "duration_seconds": answer.motion.duration_seconds,
        } if answer.motion else None,
        "voice_prompt": None,
        "metadata": {
            "request_id": answer.request_id,
            "status": answer.status,
            "pending_services": answer.pending_services,
            "progress_stage": answer.progress_stage,
            "tts": _model_to_dict(answer.tts) if answer.tts else None,
        },
        "pipeline_trace": {"path": "main_api_full_pipeline", "errors": answer.errors or {}},
        "performance": {"total_ms": answer.generation_time_ms},
        "errors": answer.errors,
        "error": error_text,
    }


def _query_to_task_payload(task_id: str, answer: AnswerResponse, query: str, user_id: str) -> UnifiedTaskResponseCompat:
    status_value = "failed" if answer.errors and not answer.text_answer else answer.status
    progress_stage = _to_progress_stage(answer.progress_stage, status_value)
    result = _answer_to_query_payload(answer, query, user_id)
    error_text = result.get("error")
    return UnifiedTaskResponseCompat(
        task_id=task_id,
        status=status_value,
        progress_stage=progress_stage,
        result=result,
        error=error_text,
    )


ANSWER_JOBS: Dict[str, Dict[str, Any]] = {}
ANSWER_JOBS_LOCK = asyncio.Lock()


async def _run_async_enrichment(
    request_id: str,
    text_answer: str,
    user_id: str,
    motion_prompt: Optional[str],
    motion_duration_seconds: float,
    rag_data: Dict[str, Any],
) -> None:
    """Run motion/TTS asynchronously and persist results in in-memory job store."""
    errors: Dict[str, str] = {}
    motion: Optional[MotionMetadata] = None
    tts: Optional[TTSMetadata] = None

    async with httpx.AsyncClient(timeout=DOWNSTREAM_TIMEOUT) as client:
        async def maybe_motion() -> Optional[MotionMetadata]:
            if not motion_prompt:
                return None
            try:
                return await _generate_motion_from_dart(client, motion_prompt, motion_duration_seconds, rag_data)
            except Exception as exc:
                logger.error(f"[DART] async failed: {exc}")
                errors["dart"] = str(exc)
                return None

        async def maybe_tts() -> Optional[TTSMetadata]:
            try:
                return await _generate_tts(client, text_answer, user_id)
            except Exception as exc:
                logger.error(f"[TTS] async failed: {exc}")
                errors["tts"] = str(exc)
                return None

        motion, tts = await asyncio.gather(maybe_motion(), maybe_tts())

    async with ANSWER_JOBS_LOCK:
        job = ANSWER_JOBS.get(request_id)
        if not job:
            return
        job["motion"] = motion
        job["tts"] = tts
        job["errors"] = errors if errors else None
        job["pending_services"] = []
        job["status"] = "completed"

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

    request_id = str(uuid.uuid4())[:12]
    errors: Dict[str, str] = {}
    rag_data: Optional[Dict[str, Any]] = None
    language = _detect_query_language(request.query)
    text_answer = ""
    exercises: List[Dict[str, str]] = []
    motion: Optional[MotionMetadata] = None
    tts: Optional[TTSMetadata] = None

    async with httpx.AsyncClient(timeout=DOWNSTREAM_TIMEOUT) as client:
        # 1. Call AgenticRAG
        try:
            rag_data = await call_agenticrag(client, request.query, request.user_id, history)
            language = rag_data.get("language", language)
            text_answer = rag_data.get("text_answer", "")
            exercises = rag_data.get("exercises", [])
            motion = _build_motion_from_agenticrag(rag_data)
        except Exception as e:
            logger.error(f"[AgenticRAG] failed: {e}")
            errors["agenticrag"] = str(e)
            text_answer = f"[AgenticRAG unavailable — check that {AGENTIC_RAG_URL} is running]"
            exercises = []
            rag_data = None

        # 2. Resolve motion prompt from AgenticRAG response.
        motion_prompt: Optional[str] = None
        if rag_data:
            motion_prompt = rag_data.get("exercise_motion_prompt")
            if not motion_prompt:
                motion_prompt_obj = rag_data.get("motion_prompt")
                if isinstance(motion_prompt_obj, dict):
                    motion_prompt = motion_prompt_obj.get("description") or motion_prompt_obj.get("primitive_sequence")
        motion_duration_seconds = _resolve_motion_duration_seconds(rag_data or {})

        # 3. If no motion came from AgenticRAG and async mode is disabled, run sync enrichment.
        if rag_data and not MAIN_API_ASYNC_ENRICHMENT:
            if motion is None and motion_prompt:
                try:
                    motion = await _generate_motion_from_dart(client, motion_prompt, motion_duration_seconds, rag_data)
                except Exception as exc:
                    logger.error(f"[DART] sync failed: {exc}")
                    errors["dart"] = str(exc)
            try:
                tts = await _generate_tts(client, text_answer, request.user_id)
            except Exception as exc:
                logger.error(f"[TTS] sync failed: {exc}")
                errors["tts"] = str(exc)

        selected_strategy = "unknown"
        if rag_data:
            selected_strategy = rag_data.get("orchestrator_decision", {}).get("intent", "unknown")

        # 4. Async mode: return text quickly and finish services in background.
        pending_services: List[str] = []
        status = "completed"
        progress_stage = "completed"
        if rag_data and MAIN_API_ASYNC_ENRICHMENT:
            if motion is None and motion_prompt:
                pending_services.append("dart")
                progress_stage = "motion_generation"
            pending_services.append("tts")
            if pending_services:
                status = "processing"
                if progress_stage == "completed":
                    progress_stage = "voice_synthesis"
                async with ANSWER_JOBS_LOCK:
                    ANSWER_JOBS[request_id] = {
                        "request_id": request_id,
                        "status": status,
                        "pending_services": pending_services.copy(),
                        "language": language,
                        "selected_strategy": selected_strategy,
                        "progress_stage": progress_stage,
                        "text_answer": text_answer,
                        "exercises": exercises,
                        "motion": motion,
                        "tts": tts,
                        "errors": errors if errors else None,
                        "started_at": time.time(),
                    }
                asyncio.create_task(
                    _run_async_enrichment(
                        request_id=request_id,
                        text_answer=text_answer,
                        user_id=request.user_id,
                        motion_prompt=motion_prompt if motion is None else None,
                        motion_duration_seconds=motion_duration_seconds,
                        rag_data=rag_data,
                    )
                )

    generation_time_ms = (time.perf_counter() - t_start) * 1000

    logger.info(
        f"[{request.user_id}] Completed in {generation_time_ms:.0f}ms  "
        f"rag={'ok' if rag_data else 'ERROR'}  dart={'ok' if motion else 'pending/none'}  "
        f"tts={'ok' if tts else ('pending' if MAIN_API_ASYNC_ENRICHMENT and rag_data else 'none/ERROR')} "
        f"status={status if rag_data else 'completed'}"
    )

    return AnswerResponse(
        request_id=request_id,
        status=status if rag_data else "completed",
        pending_services=pending_services if rag_data else [],
        language=language,
        selected_strategy=selected_strategy if rag_data else "unknown",
        progress_stage=progress_stage if rag_data else "completed",
        text_answer=text_answer,
        exercises=exercises,
        motion=motion,
        tts=tts,
        generation_time_ms=round(generation_time_ms, 1),
        errors=errors if errors else None,
    )


@app.get("/answer/status/{request_id}", response_model=AnswerResponse, summary="Get async enrichment status")
async def get_answer_status(request_id: str) -> AnswerResponse:
    """Fetch latest state for an async /answer request."""
    async with ANSWER_JOBS_LOCK:
        job = ANSWER_JOBS.get(request_id)
    if not job:
        raise HTTPException(status_code=404, detail=f"Unknown request_id: {request_id}")

    return AnswerResponse(
        request_id=job["request_id"],
        status=job.get("status", "processing"),
        pending_services=job.get("pending_services", []),
        language=job.get("language", "other"),
        selected_strategy=job.get("selected_strategy", "unknown"),
        progress_stage=job.get("progress_stage", "completed") if job.get("status") == "processing" else "completed",
        text_answer=job.get("text_answer", ""),
        exercises=job.get("exercises", []),
        motion=job.get("motion"),
        tts=job.get("tts"),
        generation_time_ms=round((time.time() - job.get("started_at", time.time())) * 1000, 1),
        errors=job.get("errors"),
    )


@app.post("/query", summary="Compatibility endpoint for AgenticRAG-style clients")
async def query_compat(request: QueryRequestCompat) -> Dict[str, Any]:
    """Expose /query on 8080 by routing through the full /answer pipeline."""
    answer = await get_answer(
        AnswerRequest(
            query=request.query,
            user_id=request.user_id,
            conversation_history=request.conversation_history,
            motion_format="glb",
        )
    )
    return _answer_to_query_payload(answer, request.query, request.user_id)


@app.post("/process_query", response_model=UnifiedTaskResponseCompat, summary="Submit async query task")
async def process_query_compat(request: QueryRequestCompat) -> UnifiedTaskResponseCompat:
    """Expose /process_query on 8080 with task envelope compatible with port 8000."""
    answer = await get_answer(
        AnswerRequest(
            query=request.query,
            user_id=request.user_id,
            conversation_history=request.conversation_history,
            motion_format="glb",
        )
    )
    task_id = answer.request_id
    async with TASK_CONTEXT_LOCK:
        TASK_CONTEXT[task_id] = {
            "query": request.query,
            "user_id": request.user_id,
            "final_answer": _model_to_dict(answer) if answer.status == "completed" else None,
        }
    return _query_to_task_payload(task_id, answer, request.query, request.user_id)


@app.get("/tasks/{task_id}", response_model=UnifiedTaskResponseCompat, summary="Get async query task status")
async def get_task_compat(task_id: str) -> UnifiedTaskResponseCompat:
    """Expose /tasks polling contract on 8080 for clients expecting AgenticRAG style."""
    async with TASK_CONTEXT_LOCK:
        context = TASK_CONTEXT.get(task_id)

    if context is None:
        raise HTTPException(status_code=404, detail=f"Task not found: {task_id}")

    query = context.get("query", "")
    user_id = context.get("user_id", "default")

    final_answer = context.get("final_answer")
    if final_answer:
        answer_obj = AnswerResponse(**final_answer)
        return _query_to_task_payload(task_id, answer_obj, query, user_id)

    try:
        answer_obj = await get_answer_status(task_id)
    except HTTPException as exc:
        if exc.status_code == 404:
            raise HTTPException(status_code=404, detail=f"Task not found: {task_id}")
        raise

    payload = _query_to_task_payload(task_id, answer_obj, query, user_id)

    if payload.status == "completed":
        async with TASK_CONTEXT_LOCK:
            if task_id in TASK_CONTEXT:
                TASK_CONTEXT[task_id]["final_answer"] = _model_to_dict(answer_obj)

    return payload


@app.get("/health", summary="Health check")
async def health_check() -> Dict[str, Any]:
    """Ping both downstream services and report their status concurrently."""
    statuses: Dict[str, str] = {}

    async def check_service(name: str, base_url: str):
        async with httpx.AsyncClient(timeout=5.0) as client:
            try:
                r = await client.get(f"{base_url}/health")
                statuses[name] = "ok" if r.status_code == 200 else f"http_{r.status_code}"
            except Exception as exc:
                statuses[name] = f"unreachable ({type(exc).__name__})"

    await asyncio.gather(
        check_service("agenticrag", AGENTIC_RAG_URL),
        check_service("dart", DART_URL)
    )

    overall = "healthy" if all(v == "ok" for v in statuses.values()) else "degraded"
    return {"status": overall, "services": statuses}


@app.get("/info", summary="Get service info")
async def get_info() -> Dict[str, Any]:
    """Return configuration and endpoint documentation."""
    return {
        "service": "Unified Multi-Service Pipeline",
        "version": "2.0.0",
        "async_enrichment": MAIN_API_ASYNC_ENRICHMENT,
        "upstream_services": {
            "agenticrag": f"{AGENTIC_RAG_URL}/query",
            "dart": f"{DART_URL}/generate",
            "tts": f"{TTS_URL}/synthesize",
        },
        "endpoints": {
            "POST /answer": "Fast text-first response, optional async motion/TTS enrichment",
            "GET /answer/status/{request_id}": "Poll async enrichment status/results",
            "POST /query": "Compatibility alias for 8000-style query clients",
            "POST /process_query": "Compatibility task submission endpoint",
            "GET /tasks/{task_id}": "Compatibility task polling endpoint",
            "GET /health": "Ping downstream services",
            "GET /info": "This document",
        },
    }


# ===========================
# Main
# ===========================

if __name__ == "__main__":
    logger.info(f"Starting Unified Pipeline API on {MAIN_API_HOST}:{MAIN_API_PORT}...")
    logger.info("")
    logger.info("Requires these services to be running:")
    logger.info(f"  AgenticRAG : {AGENTIC_RAG_URL}  (Windows, firstconda env)")
    logger.info(f"  DART       : {DART_URL}  (WSL/Linux, DART env)")
    logger.info(f"  SpeechLLM  : {TTS_URL}")
    logger.info("")
    logger.info(f"Frontend: POST http://localhost:{MAIN_API_PORT}/answer")
    logger.info("")

    uvicorn.run(
        app,
        host=MAIN_API_HOST,
        port=MAIN_API_PORT,
        log_level="info",
    )
