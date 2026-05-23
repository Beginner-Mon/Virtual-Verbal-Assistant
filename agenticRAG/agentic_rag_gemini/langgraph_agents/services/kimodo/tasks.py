"""Celery task for Kimodo motion generation."""

from __future__ import annotations

import json
import logging

from langgraph_agents.celery_app import celery_app
from langgraph_agents.services.exceptions import ServiceUnavailableError

logger = logging.getLogger("langgraph.tasks.kimodo")


def _load_redis_url() -> str:
    from pathlib import Path
    import yaml
    config_path = Path(__file__).resolve().parents[5] / "config" / "langgraph.yaml"
    if config_path.exists():
        with open(config_path, "r", encoding="utf-8") as f:
            return yaml.safe_load(f).get("langgraph", {}).get("memory", {}).get("redis_url", "redis://localhost:6379")
    return "redis://localhost:6379"


_REDIS_URL = _load_redis_url()


def _get_redis():
    import redis
    return redis.Redis.from_url(_REDIS_URL, decode_responses=True)


if celery_app is not None:

    @celery_app.task(
        name="langgraph.generate_motion",
        bind=True,
        acks_late=True,
        max_retries=1,
        default_retry_delay=5,
    )
    def generate_motion(self, prompt, constraints, duration_seconds, request_id, session_id):
        """Generate 3D motion via Kimodo, persist result to Redis, publish event."""
        self.update_state(state="STARTED", meta={"stage": "kimodo_call"})

        try:
            from langgraph_agents.services.kimodo.client import get_kimodo_client

            client = get_kimodo_client()
            result = client.generate_sync(
                prompt=prompt,
                constraints=constraints or [],
                duration_seconds=duration_seconds,
            )

            # Write success to Redis
            r = _get_redis()
            result_payload = json.dumps({
                "event": "motion_ready",
                "task_id": self.request.id,
                "url": result.get("motion_file_url", ""),
                "duration_seconds": result.get("duration_seconds", 0),
                "frames": result.get("frames", 0),
            })
            r.setex(f"task_result:{self.request.id}", 3600, result_payload)
            r.publish(f"task_events:{session_id}", result_payload)
            r.close()

            return result

        except ServiceUnavailableError as exc:
            logger.error("Kimodo unavailable — not retrying: %s", exc)
            r = _get_redis()
            fail_payload = json.dumps({
                "event": "motion_failed",
                "task_id": self.request.id,
                "error": str(exc),
            })
            r.setex(f"task_result:{self.request.id}", 3600, fail_payload)
            r.publish(f"task_events:{session_id}", fail_payload)
            r.close()
            raise

        except Exception as exc:
            logger.warning("Motion task failed, retrying: %s", exc)
            try:
                self.retry(exc=exc)
            except self.MaxRetriesExceededError:
                logger.error("Motion task retries exhausted")
                r = _get_redis()
                fail_payload = json.dumps({
                    "event": "motion_failed",
                    "task_id": self.request.id,
                    "error": str(exc),
                })
                r.setex(f"task_result:{self.request.id}", 3600, fail_payload)
                r.publish(f"task_events:{session_id}", fail_payload)
                r.close()
                raise

else:
    logger.warning("Celery not available — generate_motion task not registered")
