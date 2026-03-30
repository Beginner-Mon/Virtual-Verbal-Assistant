import time
import asyncio
from typing import Any, Dict, Literal, Optional

import httpx

from schemas.main_api import MotionMetadata, TTSMetadata
from stores.main_api_stores import InMemoryAnswerJobStore
from utils.logger import get_logger

from .main_api_downstream import generate_motion_from_dart, generate_tts

logger = get_logger(__name__)


async def run_async_enrichment(
    request_id: str,
    text_answer: str,
    user_id: str,
    motion_prompt: Optional[str],
    motion_duration_seconds: float,
    motion_format: Literal["glb", "npz"],
    rag_data: Dict[str, Any],
    answer_store: InMemoryAnswerJobStore,
    downstream_timeout: float,
    dart_url: str,
    tts_url: str,
) -> None:
    """Run motion/TTS asynchronously and persist results in in-memory job store."""
    errors: Dict[str, str] = {}
    motion: Optional[MotionMetadata] = None
    tts: Optional[TTSMetadata] = None
    async_timings_ms: Dict[str, float] = {}
    async_services: Dict[str, Any] = {
        "dart": {"mode": "async", "status": "skipped" if not motion_prompt else "pending"},
        "tts": {"mode": "async", "status": "pending"},
    }

    async with httpx.AsyncClient(timeout=downstream_timeout) as client:
        async def maybe_motion() -> Optional[MotionMetadata]:
            if not motion_prompt:
                return None
            t0 = time.perf_counter()
            try:
                result = await generate_motion_from_dart(
                    client=client,
                    dart_url=dart_url,
                    motion_prompt=motion_prompt,
                    duration_seconds=motion_duration_seconds,
                    motion_format=motion_format,
                    rag_data=rag_data,
                )
                async_timings_ms["dart_async"] = round((time.perf_counter() - t0) * 1000, 1)
                async_services["dart"] = {
                    "mode": "async",
                    "status": "ok",
                    "elapsed_ms": async_timings_ms["dart_async"],
                    "motion_file_url": result.motion_file_url,
                }
                return result
            except Exception as exc:
                logger.error(f"[DART] async failed: {exc}")
                errors["dart"] = str(exc)
                async_timings_ms["dart_async"] = round((time.perf_counter() - t0) * 1000, 1)
                async_services["dart"] = {
                    "mode": "async",
                    "status": "failed",
                    "elapsed_ms": async_timings_ms["dart_async"],
                    "error": str(exc),
                }
                return None

        async def maybe_tts() -> Optional[TTSMetadata]:
            t0 = time.perf_counter()
            try:
                result = await generate_tts(
                    client=client,
                    tts_url=tts_url,
                    text_answer=text_answer,
                    user_id=user_id,
                )
                async_timings_ms["tts_async"] = round((time.perf_counter() - t0) * 1000, 1)
                async_services["tts"] = {
                    "mode": "async",
                    "status": "ok",
                    "elapsed_ms": async_timings_ms["tts_async"],
                    "audio_file": result.audio_file,
                }
                return result
            except Exception as exc:
                logger.error(f"[TTS] async failed: {exc}")
                errors["tts"] = str(exc)
                async_timings_ms["tts_async"] = round((time.perf_counter() - t0) * 1000, 1)
                async_services["tts"] = {
                    "mode": "async",
                    "status": "failed",
                    "elapsed_ms": async_timings_ms["tts_async"],
                    "error": str(exc),
                }
                return None

        motion, tts = await asyncio.gather(maybe_motion(), maybe_tts())

    job = await answer_store.get(request_id)
    if not job:
        return
    await answer_store.update(
        request_id,
        {
            "motion": motion,
            "tts": tts,
            "errors": errors if errors else None,
            "pending_services": [],
            "status": "completed",
        },
    )
    job = await answer_store.get(request_id)
    if not job:
        return

    if isinstance(job.get("debug"), dict):
        debug = job["debug"]
        debug.setdefault("timings_ms", {})
        debug.setdefault("services", {})
        debug["timings_ms"].update(async_timings_ms)
        debug["services"].update(async_services)
        debug["async_enrichment_completed"] = True
        debug["async_enrichment_errors"] = errors if errors else None
        await answer_store.update(request_id, {"debug": debug})
