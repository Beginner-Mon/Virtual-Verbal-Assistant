import time
import asyncio
from typing import Any, Dict, Literal, Optional

import httpx

from schemas.main_api import MotionMetadata, TTSMetadata
from stores.main_api_stores import InMemoryAnswerJobStore
from utils.logger import get_logger

from .main_api_downstream import generate_motion_from_dart, generate_tts

logger = get_logger(__name__)
MOTION_JOB_POLL_INTERVAL_SECONDS = 0.8


def _motion_from_job_state(
    job_state: Dict[str, Any],
    fallback_prompt: Optional[str],
    fallback_duration_seconds: float,
) -> Optional[MotionMetadata]:
    motion_file_url = job_state.get("motion_file_url")
    if not motion_file_url:
        return None

    try:
        num_frames = int(job_state.get("frames") or 0)
    except (TypeError, ValueError):
        num_frames = 0

    try:
        fps = int(job_state.get("fps") or 30)
    except (TypeError, ValueError):
        fps = 30

    try:
        duration_seconds = float(job_state.get("duration_seconds") or fallback_duration_seconds)
    except (TypeError, ValueError):
        duration_seconds = fallback_duration_seconds

    if num_frames <= 0 and duration_seconds > 0 and fps > 0:
        num_frames = max(1, int(round(duration_seconds * fps)))
    if duration_seconds <= 0 and num_frames > 0 and fps > 0:
        duration_seconds = round(float(num_frames) / float(fps), 3)

    prompt_text = (fallback_prompt or "").strip()
    if not prompt_text:
        selected_candidate = job_state.get("selected_candidate") or {}
        if isinstance(selected_candidate, dict):
            prompt_text = (
                str(selected_candidate.get("rewritten_prompt") or "").strip()
                or str(selected_candidate.get("text_description") or "").strip()
            )

    return MotionMetadata(
        motion_file_url=str(motion_file_url),
        num_frames=max(0, num_frames),
        fps=max(1, fps),
        duration_seconds=max(0.0, duration_seconds),
        text_prompt=prompt_text,
    )


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
    semantic_bridge_prompt: Optional[str] = None,
    agentic_rag_url: Optional[str] = None,
    adopted_motion_job: Optional[Dict[str, Any]] = None,
) -> None:
    """Run motion/TTS asynchronously and persist results in in-memory job store."""
    errors: Dict[str, str] = {}
    motion: Optional[MotionMetadata] = None
    tts: Optional[TTSMetadata] = None
    motion_job_state: Optional[Dict[str, Any]] = (
        dict(adopted_motion_job) if isinstance(adopted_motion_job, dict) else None
    )
    adopted_motion_job_id = (
        str((motion_job_state or {}).get("job_id") or "").strip() if motion_job_state else ""
    )
    async_timings_ms: Dict[str, float] = {}
    async_services: Dict[str, Any] = {
        "dart": {
            "mode": "async",
            "status": (
                "pending" if adopted_motion_job_id or motion_prompt else "skipped"
            ),
        },
        "tts": {"mode": "async", "status": "pending"},
    }

    if adopted_motion_job_id:
        async_services["dart"].update(
            {
                "source": "motion_job",
                "job_id": adopted_motion_job_id,
                "reason": "Polling adopted AgenticRAG motion job",
            }
        )

    async with httpx.AsyncClient(timeout=downstream_timeout) as client:
        async def wait_for_adopted_motion_job(job_id: str) -> MotionMetadata:
            nonlocal motion_job_state

            if not agentic_rag_url:
                raise RuntimeError("agentic_rag_url is required to poll adopted motion jobs")

            base_url = str(agentic_rag_url).rstrip("/")
            fallback_prompt = semantic_bridge_prompt or motion_prompt
            deadline = time.perf_counter() + max(5.0, float(downstream_timeout))

            while True:
                resp = await client.get(f"{base_url}/job-status/{job_id}")
                resp.raise_for_status()

                payload = resp.json()
                if isinstance(payload, dict):
                    motion_job_state = payload
                else:
                    motion_job_state = {"job_id": job_id, "status": "queued"}

                status = str((motion_job_state or {}).get("status") or "").strip().lower()
                if status == "completed":
                    adopted_motion = _motion_from_job_state(
                        motion_job_state,
                        fallback_prompt=fallback_prompt,
                        fallback_duration_seconds=motion_duration_seconds,
                    )
                    if adopted_motion is None:
                        raise RuntimeError("Adopted motion job completed without motion_file_url")
                    return adopted_motion

                if status == "failed":
                    error_text = (motion_job_state or {}).get("error") or "Adopted motion job failed"
                    raise RuntimeError(str(error_text))

                if time.perf_counter() >= deadline:
                    raise RuntimeError(
                        f"Adopted motion job timed out after {max(5.0, float(downstream_timeout)):.1f}s"
                    )

                await asyncio.sleep(MOTION_JOB_POLL_INTERVAL_SECONDS)

        async def maybe_motion() -> Optional[MotionMetadata]:
            if not motion_prompt and not adopted_motion_job_id:
                return None
            t0 = time.perf_counter()
            try:
                if adopted_motion_job_id:
                    result = await wait_for_adopted_motion_job(adopted_motion_job_id)
                else:
                    result = await generate_motion_from_dart(
                        client=client,
                        dart_url=dart_url,
                        motion_prompt=motion_prompt,
                        duration_seconds=motion_duration_seconds,
                        motion_format=motion_format,
                        rag_data=rag_data,
                        semantic_bridge_prompt=semantic_bridge_prompt,
                    )
                async_timings_ms["dart_async"] = round((time.perf_counter() - t0) * 1000, 1)
                async_services["dart"] = {
                    "mode": "async",
                    "status": "ok",
                    "elapsed_ms": async_timings_ms["dart_async"],
                    "motion_file_url": result.motion_file_url,
                }
                if adopted_motion_job_id:
                    async_services["dart"].update({
                        "source": "motion_job",
                        "job_id": adopted_motion_job_id,
                    })
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
                if adopted_motion_job_id:
                    failed_state = dict(motion_job_state or {})
                    failed_state.setdefault("job_id", adopted_motion_job_id)
                    failed_state["status"] = "failed"
                    failed_state["error"] = str(exc)
                    motion_job_state = failed_state
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
    updates: Dict[str, Any] = {
        "motion": motion,
        "tts": tts,
        "errors": errors if errors else None,
        "pending_services": [],
        "status": "completed",
    }
    if motion_job_state is not None:
        updates["motion_job"] = motion_job_state

    await answer_store.update(request_id, updates)
    job = await answer_store.get(request_id)
    if not job:
        return

    if isinstance(job.get("debug"), dict):
        debug = job["debug"]
        debug.setdefault("timings_ms", {})
        debug.setdefault("services", {})
        debug["timings_ms"].update(async_timings_ms)
        debug["services"].update(async_services)
        if motion_job_state is not None:
            debug["motion_job"] = motion_job_state
        debug["async_enrichment_completed"] = True
        debug["async_enrichment_errors"] = errors if errors else None
        await answer_store.update(request_id, {"debug": debug})
