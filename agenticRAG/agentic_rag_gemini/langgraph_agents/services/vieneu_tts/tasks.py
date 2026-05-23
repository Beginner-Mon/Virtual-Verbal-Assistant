"""Celery task for VieNeu-TTS speech synthesis."""

from __future__ import annotations

import json
import logging

from langgraph_agents.celery_app import celery_app
from langgraph_agents.services.exceptions import ServiceUnavailableError

logger = logging.getLogger("langgraph.tasks.vieneu_tts")


def _load_config() -> tuple[str, str]:
    from pathlib import Path
    import yaml
    config_path = Path(__file__).resolve().parents[5] / "config" / "langgraph.yaml"
    if config_path.exists():
        with open(config_path, "r", encoding="utf-8") as f:
            cfg = yaml.safe_load(f).get("langgraph", {})
            redis_url = cfg.get("memory", {}).get("redis_url", "redis://localhost:6379")
            vieneu_url = cfg.get("services", {}).get("vieneu_tts", {}).get("url", "http://localhost:5000")
            return redis_url, vieneu_url
    return "redis://localhost:6379", "http://localhost:5000"


_REDIS_URL, _VIENEU_BASE_URL = _load_config()


def _get_redis():
    import redis
    return redis.Redis.from_url(_REDIS_URL, decode_responses=True)


if celery_app is not None:

    @celery_app.task(
        name="langgraph.synthesize_speech",
        bind=True,
        acks_late=True,
        max_retries=1,
        default_retry_delay=3,
    )
    def synthesize_speech(self, text, request_id, session_id, voice_path=None):
        """Synthesize speech via VieNeu-TTS, persist result to Redis, publish event."""
        self.update_state(state="STARTED", meta={"stage": "tts_call"})

        try:
            from langgraph_agents.services.vieneu_tts.client import get_vieneu_tts_client

            client = get_vieneu_tts_client()
            result = client.synthesize_sync(text=text, voice_path=voice_path)

            # Build audio URL from filename returned by VieNeu-TTS server
            audio_file = result.get("audio_file", "")
            audio_url = result.get("audio_url") or (
                f"{_VIENEU_BASE_URL}/audio/{audio_file}" if audio_file else ""
            )

            # Write success to Redis
            r = _get_redis()
            result_payload = json.dumps({
                "event": "speech_ready",
                "task_id": self.request.id,
                "audio_url": audio_url,
                "tts_time_sec": result.get("tts_time_sec", 0),
            })
            r.setex(f"task_result:{self.request.id}", 3600, result_payload)
            r.publish(f"task_events:{session_id}", result_payload)
            r.close()

            return result

        except ServiceUnavailableError as exc:
            logger.error("VieNeu-TTS unavailable — not retrying: %s", exc)
            r = _get_redis()
            fail_payload = json.dumps({
                "event": "speech_failed",
                "task_id": self.request.id,
                "error": str(exc),
            })
            r.setex(f"task_result:{self.request.id}", 3600, fail_payload)
            r.publish(f"task_events:{session_id}", fail_payload)
            r.close()
            raise

        except Exception as exc:
            logger.warning("Speech task failed, retrying: %s", exc)
            try:
                self.retry(exc=exc)
            except self.MaxRetriesExceededError:
                logger.error("Speech task retries exhausted")
                r = _get_redis()
                fail_payload = json.dumps({
                    "event": "speech_failed",
                    "task_id": self.request.id,
                    "error": str(exc),
                })
                r.setex(f"task_result:{self.request.id}", 3600, fail_payload)
                r.publish(f"task_events:{session_id}", fail_payload)
                r.close()
                raise

else:
    logger.warning("Celery not available — synthesize_speech task not registered")
