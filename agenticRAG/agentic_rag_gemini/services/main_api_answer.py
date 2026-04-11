import asyncio
import time
import uuid
from typing import Any, Dict, List, Optional

import httpx
from fastapi import HTTPException

from core.config.settings import get_main_api_settings
from schemas.main_api import AnswerRequest, AnswerResponse, MotionJobStatus, MotionMetadata, TTSMetadata
from stores.main_api_stores import answer_job_store
from utils.logger import get_logger

from .main_api_downstream import (
    build_motion_from_agenticrag,
    call_agenticrag,
    detect_query_language,
    generate_motion_from_dart,
    generate_tts,
    resolve_motion_duration_seconds,
)
from .main_api_enrichment import run_async_enrichment

logger = get_logger(__name__)
SETTINGS = get_main_api_settings()


async def get_answer_impl(request: AnswerRequest) -> AnswerResponse:
    """Main /answer orchestration logic."""
    t_start = time.perf_counter()
    debug_payload: Dict[str, Any] = {
        "request": {
            "user_id": request.user_id,
            "motion_format": request.motion_format,
            "conversation_turns": len(request.conversation_history or []),
            "query_length": len(request.query or ""),
        },
        "config": {
            "async_enrichment": SETTINGS.async_enrichment,
            "downstream_timeout_sec": SETTINGS.downstream_timeout,
            "agentic_rag_url": SETTINGS.agentic_rag_url,
            "dart_url": SETTINGS.dart_url,
            "tts_url": SETTINGS.tts_url,
        },
        "timings_ms": {},
        "services": {
            "agenticrag": {"status": "pending", "mode": "sync"},
            "dart": {"status": "pending", "mode": "agenticrag_or_sync"},
            "tts": {"status": "pending", "mode": "sync_or_async"},
        },
    }

    session_id = request.session_id
    history: Optional[List[Dict[str, str]]] = None
    if not session_id and request.conversation_history:
        history = [{"role": t.role, "content": t.content} for t in request.conversation_history]

    request_id = str(uuid.uuid4())[:12]
    errors: Dict[str, str] = {}
    rag_data: Optional[Dict[str, Any]] = None
    language = detect_query_language(request.query)
    text_answer = ""
    exercises: List[Dict[str, str]] = []
    motion: Optional[MotionMetadata] = None
    motion_job: Optional[MotionJobStatus] = None
    tts: Optional[TTSMetadata] = None


    async with httpx.AsyncClient(timeout=SETTINGS.downstream_timeout) as client:
        rag_t0 = time.perf_counter()
        try:
            rag_data = await call_agenticrag(
                client=client,
                base_url=SETTINGS.agentic_rag_url,
                query=request.query,
                user_id=request.user_id,
                conversation_history=history,
                session_id=session_id,
            )
            debug_payload["timings_ms"]["agenticrag"] = round((time.perf_counter() - rag_t0) * 1000, 1)
            debug_payload["services"]["agenticrag"] = {
                "status": "ok",
                "mode": "sync",
                "elapsed_ms": debug_payload["timings_ms"]["agenticrag"],
            }
            language = rag_data.get("language", language)
            text_answer = rag_data.get("text_answer", "")
            exercises = rag_data.get("exercises", [])
            motion = build_motion_from_agenticrag(rag_data, SETTINGS.dart_url)
            motion_job_raw = rag_data.get("motion_job")
            if isinstance(motion_job_raw, dict):
                motion_job = MotionJobStatus(**motion_job_raw)
                logger.info("Adopting existing motion_job from AgenticRAG: %s", motion_job.job_id)

            debug_payload["rag_summary"] = {
                "text_answer_length": len(text_answer or ""),
                "exercise_count": len(exercises or []),
                "has_motion_from_rag": motion is not None,
                "has_motion_job_from_rag": motion_job is not None,
                "orchestrator_decision": rag_data.get("orchestrator_decision"),
            }

        except Exception as exc:
            logger.error(f"[AgenticRAG] failed: {exc}")
            errors["agenticrag"] = str(exc)
            debug_payload["timings_ms"]["agenticrag"] = round((time.perf_counter() - rag_t0) * 1000, 1)
            debug_payload["services"]["agenticrag"] = {
                "status": "failed",
                "mode": "sync",
                "elapsed_ms": debug_payload["timings_ms"]["agenticrag"],
                "error": str(exc),
            }
            text_answer = f"[AgenticRAG unavailable - check that {SETTINGS.agentic_rag_url} is running]"
            exercises = []
            rag_data = None

        motion_prompt: Optional[str] = None
        semantic_bridge_prompt: Optional[str] = None
        if rag_data:
            motion_prompt = rag_data.get("exercise_motion_prompt")
            if not motion_prompt:
                motion_prompt_obj = rag_data.get("motion_prompt")
                if isinstance(motion_prompt_obj, dict):
                    motion_prompt = motion_prompt_obj.get("description") or motion_prompt_obj.get("primitive_sequence")
            # The exercise_motion_prompt may already carry the semantic bridge
            # result (set by the parallel Task B in app.py). We also check for
            # an explicit field in case the AgenticRAG response includes it.
            semantic_bridge_prompt = rag_data.get("semantic_bridge_prompt")

        motion_duration_seconds = resolve_motion_duration_seconds(
            rag_data or {},
            SETTINGS.motion_default_duration_seconds,
        )
        debug_payload["rag_summary"] = {
            **(debug_payload.get("rag_summary") or {}),
            "motion_prompt_present": bool(motion_prompt),
            "semantic_bridge_prompt_present": bool(semantic_bridge_prompt),
            "resolved_motion_duration_seconds": motion_duration_seconds,
        }

        if rag_data and not SETTINGS.async_enrichment:
            if motion is None and motion_prompt:
                dart_t0 = time.perf_counter()
                try:
                    motion = await generate_motion_from_dart(
                        client=client,
                        dart_url=SETTINGS.dart_url,
                        motion_prompt=motion_prompt,
                        duration_seconds=motion_duration_seconds,
                        motion_format=request.motion_format,
                        rag_data=rag_data,
                        semantic_bridge_prompt=semantic_bridge_prompt,
                    )
                    debug_payload["timings_ms"]["dart_sync"] = round((time.perf_counter() - dart_t0) * 1000, 1)
                    debug_payload["services"]["dart"] = {
                        "status": "ok",
                        "mode": "sync",
                        "elapsed_ms": debug_payload["timings_ms"]["dart_sync"],
                        "motion_file_url": motion.motion_file_url,
                    }
                except Exception as exc:
                    logger.error(f"[DART] sync failed: {exc}")
                    errors["dart"] = str(exc)
                    debug_payload["timings_ms"]["dart_sync"] = round((time.perf_counter() - dart_t0) * 1000, 1)
                    debug_payload["services"]["dart"] = {
                        "status": "failed",
                        "mode": "sync",
                        "elapsed_ms": debug_payload["timings_ms"]["dart_sync"],
                        "error": str(exc),
                    }
            elif motion is not None:
                debug_payload["services"]["dart"] = {
                    "status": "ok",
                    "mode": "agenticrag",
                    "elapsed_ms": 0.0,
                    "motion_file_url": motion.motion_file_url,
                }
            else:
                debug_payload["services"]["dart"] = {
                    "status": "skipped",
                    "mode": "sync",
                    "reason": "No motion prompt",
                }

            tts_t0 = time.perf_counter()
            try:
                tts = await generate_tts(
                    client=client,
                    tts_url=SETTINGS.tts_url,
                    text_answer=text_answer,
                    user_id=request.user_id,
                )
                debug_payload["timings_ms"]["tts_sync"] = round((time.perf_counter() - tts_t0) * 1000, 1)
                debug_payload["services"]["tts"] = {
                    "status": "ok",
                    "mode": "sync",
                    "elapsed_ms": debug_payload["timings_ms"]["tts_sync"],
                    "audio_file": tts.audio_file,
                }
            except Exception as exc:
                logger.error(f"[TTS] sync failed: {exc}")
                errors["tts"] = str(exc)
                debug_payload["timings_ms"]["tts_sync"] = round((time.perf_counter() - tts_t0) * 1000, 1)
                debug_payload["services"]["tts"] = {
                    "status": "failed",
                    "mode": "sync",
                    "elapsed_ms": debug_payload["timings_ms"]["tts_sync"],
                    "error": str(exc),
                }

        selected_strategy = "unknown"
        if rag_data:
            selected_strategy = rag_data.get("orchestrator_decision", {}).get("intent", "unknown")

        pending_services: List[str] = []
        status = "completed"
        progress_stage = "completed"
        if rag_data and SETTINGS.async_enrichment:
            if motion is None and motion_job is not None:
                # Early Warmup already handling it
                debug_payload["services"]["dart"] = {
                    "status": "pending",
                    "mode": "async",
                    "reason": f"Adopted from AgenticRAG warmup: {motion_job.job_id}",
                }
            elif motion is None and motion_prompt:
                pending_services.append("dart")
                progress_stage = "motion_generation"
                debug_payload["services"]["dart"] = {
                    "status": "pending",
                    "mode": "async",
                    "reason": "Queued for async enrichment",
                }

            elif motion is not None:
                debug_payload["services"]["dart"] = {
                    "status": "ok",
                    "mode": "agenticrag",
                    "elapsed_ms": 0.0,
                    "motion_file_url": motion.motion_file_url,
                }
            else:
                debug_payload["services"]["dart"] = {
                    "status": "skipped",
                    "mode": "async",
                    "reason": "No motion prompt",
                }

            pending_services.append("tts")
            debug_payload["services"]["tts"] = {
                "status": "pending",
                "mode": "async",
                "reason": "Queued for async enrichment",
            }

            if pending_services:
                status = "processing"
                if progress_stage == "completed":
                    progress_stage = "voice_synthesis"
                await answer_job_store.set(
                    request_id,
                    {
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
                        "debug": debug_payload if SETTINGS.include_debug else None,
                    },
                )
                asyncio.create_task(
                    run_async_enrichment(
                        request_id=request_id,
                        text_answer=text_answer,
                        user_id=request.user_id,
                        motion_prompt=motion_prompt if motion is None else None,
                        motion_duration_seconds=motion_duration_seconds,
                        motion_format=request.motion_format,
                        rag_data=rag_data,
                        answer_store=answer_job_store,
                        downstream_timeout=SETTINGS.downstream_timeout,
                        dart_url=SETTINGS.dart_url,
                        tts_url=SETTINGS.tts_url,
                        semantic_bridge_prompt=semantic_bridge_prompt if motion is None else None,
                    )
                )

    generation_time_ms = (time.perf_counter() - t_start) * 1000
    debug_payload["timings_ms"]["total"] = round(generation_time_ms, 1)
    debug_payload["final_state"] = {
        "status": status if rag_data else "completed",
        "progress_stage": progress_stage if rag_data else "completed",
        "pending_services": pending_services if rag_data else [],
    }
    debug_payload["errors"] = errors if errors else None

    logger.info(
        f"[{request.user_id}] Completed in {generation_time_ms:.0f}ms  "
        f"rag={'ok' if rag_data else 'ERROR'}  dart={'ok' if motion or motion_job else 'pending/none'}  "
        f"tts={'ok' if tts else ('pending' if SETTINGS.async_enrichment and rag_data else 'none/ERROR')} "
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
        motion_job=motion_job,
        tts=tts,
        generation_time_ms=round(generation_time_ms, 1),
        errors=errors if errors else None,
        debug=debug_payload if SETTINGS.include_debug else None,
    )


async def get_answer_status_impl(request_id: str) -> AnswerResponse:
    job = await answer_job_store.get(request_id)
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
        debug=job.get("debug") if SETTINGS.include_debug else None,
    )
