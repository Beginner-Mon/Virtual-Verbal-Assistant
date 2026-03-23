"""Async motion job producer/status helper for API layer."""

from importlib import import_module
import os
from typing import Any, Dict, Optional
from urllib.parse import urljoin

try:
    AsyncResult = import_module("celery.result").AsyncResult
except Exception:  # pragma: no cover
    AsyncResult = None

from celery_app import celery_app

API_PUBLIC_BASE_URL = os.getenv("API_PUBLIC_BASE_URL", "http://localhost:8000")


def _build_absolute_video_url(video_url: Optional[str]) -> Optional[str]:
    if not video_url:
        return None
    if video_url.startswith("http://") or video_url.startswith("https://"):
        return video_url
    return urljoin(API_PUBLIC_BASE_URL.rstrip("/") + "/", video_url.lstrip("/"))


class MotionJobManager:
    """Small wrapper around Celery AsyncResult for motion jobs."""

    TASK_NAME = "tasks.motion_tasks.render_motion_job"

    def enqueue(self, user_query: str, motion_prompt: str, user_id: str, duration_seconds: float) -> str:
        if celery_app is None:
            raise RuntimeError("Celery is not available. Install celery/redis dependencies first.")
        task = celery_app.send_task(
            self.TASK_NAME,
            kwargs={
                "user_query": user_query,
                "motion_prompt": motion_prompt,
                "user_id": user_id,
                "duration_seconds": duration_seconds,
            },
        )
        return task.id

    def get_status(self, job_id: str) -> Dict[str, Any]:
        if celery_app is None or AsyncResult is None:
            return {
                "job_id": job_id,
                "status": "failed",
                "video_url": None,
                "error": "Celery is not available in this runtime.",
            }

        result = AsyncResult(job_id, app=celery_app)
        status_map = {
            "PENDING": "queued",
            "RECEIVED": "queued",
            "STARTED": "processing",
            "RETRY": "processing",
            "SUCCESS": "completed",
            "FAILURE": "failed",
            "REVOKED": "failed",
        }
        mapped_status = status_map.get(result.status, "queued")

        payload: Dict[str, Any] = {
            "job_id": job_id,
            "status": mapped_status,
            "video_url": None,
            "error": None,
            "stage": None,
        }

        if isinstance(result.info, dict):
            payload["stage"] = result.info.get("stage")

        if mapped_status == "completed" and isinstance(result.result, dict):
            raw_url = result.result.get("video_url")
            payload["video_url"] = _build_absolute_video_url(raw_url)
            payload["status"] = result.result.get("status", "completed")
            payload["error"] = result.result.get("error")
            payload["selected_strategy"] = result.result.get("selected_strategy")
            payload["selected_candidate"] = result.result.get("selected_candidate")
            if payload["status"] == "failed" and not payload["error"]:
                payload["error"] = "Motion job returned failed status"
        elif mapped_status == "failed":
            if isinstance(result.result, dict):
                payload["error"] = result.result.get("error")
            else:
                payload["error"] = str(result.result)

            if not payload["error"] and isinstance(result.info, dict):
                payload["error"] = result.info.get("error")
            if not payload["error"]:
                payload["error"] = "Motion job failed without explicit error message"

        elif mapped_status in {"queued", "processing"} and isinstance(result.info, dict):
            payload["error"] = result.info.get("error")

        return payload
