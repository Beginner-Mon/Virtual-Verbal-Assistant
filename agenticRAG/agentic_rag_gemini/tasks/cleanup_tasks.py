"""Celery Beat scheduled tasks for resource cleanup."""

import os
import time
from pathlib import Path

from celery_app import celery_app
from utils.logger import get_logger

logger = get_logger(__name__)

VIDEO_ROOT = Path(os.getenv("MOTION_VIDEO_ROOT", "./static/videos"))
RETENTION_HOURS = int(os.getenv("MOTION_VIDEO_RETENTION_HOURS", "6"))


def _celery_task(*args, **kwargs):
    if celery_app is None:
        def _wrap(fn):
            return fn
        return _wrap
    return celery_app.task(*args, **kwargs)


@_celery_task(name="tasks.cleanup_tasks.cleanup_old_motion_videos")
def cleanup_old_motion_videos() -> dict:
    """Delete local motion video artifacts older than retention threshold."""
    VIDEO_ROOT.mkdir(parents=True, exist_ok=True)

    now = time.time()
    threshold = now - RETENTION_HOURS * 3600
    deleted = 0

    artifact_patterns = ["job_*.mp4", "motion_*.mp4", "*.npz"]
    for pattern in artifact_patterns:
        for path in VIDEO_ROOT.glob(pattern):
            if not path.is_file() or path.is_symlink():
                continue
            try:
                if path.stat().st_mtime < threshold:
                    path.unlink(missing_ok=True)
                    deleted += 1
            except Exception as exc:
                logger.warning("Failed deleting old motion file %s: %s", path, exc)

    logger.info("Cleanup finished: deleted=%d retention_hours=%d", deleted, RETENTION_HOURS)
    return {"deleted": deleted, "retention_hours": RETENTION_HOURS}
