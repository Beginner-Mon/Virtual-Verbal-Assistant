"""Celery application configuration for async motion jobs."""

import os
from importlib import import_module

from utils.logger import get_logger

logger = get_logger(__name__)

try:
    Celery = import_module("celery").Celery
except Exception:  # pragma: no cover - graceful fallback when celery is absent
    Celery = None

REDIS_URL = os.getenv("REDIS_URL", "redis://localhost:6379/0")
CELERY_BROKER_URL = os.getenv("CELERY_BROKER_URL", REDIS_URL)
CELERY_RESULT_BACKEND = os.getenv("CELERY_RESULT_BACKEND", REDIS_URL)
CLEANUP_INTERVAL_MINUTES = int(os.getenv("MOTION_CLEANUP_INTERVAL_MINUTES", "30"))

if Celery is None:
    celery_app = None
    logger.warning("Celery is not installed; async motion queue features are disabled.")
else:
    celery_app = Celery(
        "agentic_rag_motion",
        broker=CELERY_BROKER_URL,
        backend=CELERY_RESULT_BACKEND,
        include=[
            "tasks.motion_tasks",
            "tasks.cleanup_tasks",
        ],
    )

    celery_app.conf.update(
        task_serializer="json",
        accept_content=["json"],
        result_serializer="json",
        timezone="UTC",
        enable_utc=True,
        task_track_started=True,
        task_time_limit=int(os.getenv("MOTION_TASK_TIME_LIMIT_SECONDS", "300")),
        task_soft_time_limit=int(os.getenv("MOTION_TASK_SOFT_TIME_LIMIT_SECONDS", "240")),
        beat_schedule={
            "cleanup-old-motion-videos": {
                "task": "tasks.cleanup_tasks.cleanup_old_motion_videos",
                "schedule": CLEANUP_INTERVAL_MINUTES * 60,
            }
        },
    )

    celery_app.autodiscover_tasks(["tasks"], related_name=None)
