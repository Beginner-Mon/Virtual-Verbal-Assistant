"""TTS task — async function fired by FastAPI BackgroundTasks (v2.4.1).

No Celery. Result persisted to Redis task_result:{task_id} with 1h TTL.
GET /tts/{task_id}/result polls this until populated (Phase 5: SSE push).
"""

from __future__ import annotations

import json
import logging

import redis.asyncio as aioredis

from langgraph_agents.services.exceptions import ServiceUnavailableError
from langgraph_agents.services.vieneu_tts.client import get_vieneu_tts_client

logger = logging.getLogger("langgraph.tasks.vieneu_tts")

_REDIS_URL = "redis://localhost:6379/0"   # v2.4.1: DB 0 (bo DB 1 broker)
_VIENEU_BASE = "http://localhost:5000"     # also used to build audio URL


async def synthesize_speech_async(
    text: str,
    task_id: str,
    voice_path: str | None = None,
) -> None:
    """Run VieNeu-TTS synthesize, persist result to Redis.

    Designed for FastAPI BackgroundTasks. Never raises (errors persisted as
    speech_failed payload). Caller does not await result — fire-and-forget.

    VieNeu /synthesize returns {"audio_file": "<filename>", ...}. We build the
    full audio URL via {base}/audio/{filename} for the UI to play.
    """
    client = get_vieneu_tts_client()
    logger.info("tts_start task_id=%s text_len=%d voice_path=%s", task_id, len(text), voice_path)

    try:
        result = await client.synthesize(text=text, voice_path=voice_path)
        logger.info("tts_synthesize_done task_id=%s result_keys=%s", task_id, list(result.keys()))
        audio_file = result.get("audio_file") or result.get("audio_url", "")
        # If service already returned full URL, use as-is; else build from filename.
        if audio_file.startswith("http"):
            audio_url = audio_file
        elif audio_file:
            audio_url = f"{_VIENEU_BASE}/audio/{audio_file}"
        else:
            audio_url = ""
        payload = {
            "event": "speech_ready",
            "task_id": task_id,
            "url": audio_url,
        }
        logger.info("tts_payload_built task_id=%s url=%s", task_id, audio_url)
    except ServiceUnavailableError as exc:
        logger.error("VieNeu-TTS unavailable: %s", exc)
        payload = {"event": "speech_failed", "task_id": task_id, "error": str(exc)}
    except Exception as exc:
        logger.exception("Unexpected TTS error")
        payload = {"event": "speech_failed", "task_id": task_id, "error": str(exc)}

    logger.info("tts_persist_start task_id=%s payload_event=%s", task_id, payload.get("event"))
    await _persist(task_id, payload)
    logger.info("tts_persist_done task_id=%s", task_id)


async def _persist(task_id: str, payload: dict) -> None:
    """Best-effort Redis write — never raise back to caller."""
    r = aioredis.from_url(_REDIS_URL, socket_connect_timeout=2)
    try:
        await r.setex(f"task_result:{task_id}", 3600, json.dumps(payload))
    except Exception:
        logger.warning("Failed to persist TTS result to Redis (task_id=%s)", task_id)
    finally:
        close_fn = getattr(r, "aclose", None) or r.close
        try:
            await close_fn()
        except Exception:
            pass
