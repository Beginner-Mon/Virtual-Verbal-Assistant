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


def _build_absolute_url(url_or_path: Optional[str], base_url: Optional[str] = None) -> Optional[str]:
    if not url_or_path:
        return None
    if url_or_path.startswith("http://") or url_or_path.startswith("https://"):
        return url_or_path
    base = (base_url or API_PUBLIC_BASE_URL).rstrip("/") + "/"
    return urljoin(base, url_or_path.lstrip("/"))


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

    def get_status(self, job_id: str, request_base_url: Optional[str] = None) -> Dict[str, Any]:
        if celery_app is None or AsyncResult is None:
            return {
                "job_id": job_id,
                "status": "failed",
                "motion_file_url": None,
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
            "motion_file_url": None,
            "video_url": None,
            "frames": None,
            "fps": None,
            "duration_seconds": None,
            "error": None,
            "stage": None,
        }

        if isinstance(result.info, dict):
            payload["stage"] = result.info.get("stage")

        if mapped_status == "completed" and isinstance(result.result, dict):
            raw_motion_url = result.result.get("motion_file_url")
            raw_video_url = result.result.get("video_url")
            payload["motion_file_url"] = _build_absolute_url(raw_motion_url, request_base_url)
            payload["video_url"] = _build_absolute_url(raw_video_url, request_base_url)
            payload["status"] = result.result.get("status", "completed")
            payload["error"] = result.result.get("error")
            payload["frames"] = result.result.get("frames")
            payload["fps"] = result.result.get("fps")
            payload["duration_seconds"] = result.result.get("duration_seconds")
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
