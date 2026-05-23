"""Approval gate for motion rendering.

Mounted at: POST /render/approve/{request_id}
Called by: frontend "Mô phỏng 3D" button click
Flow: read motion_pending from Redis -> fire Celery task -> delete pending key
"""

from __future__ import annotations

import json
import logging
from pathlib import Path

from fastapi import APIRouter, HTTPException

logger = logging.getLogger("langgraph.streaming.approval")

router = APIRouter(tags=["render"])


def _load_redis_url() -> str:
    import yaml
    config_path = Path(__file__).resolve().parents[4] / "config" / "langgraph.yaml"
    if config_path.exists():
        with open(config_path, "r", encoding="utf-8") as f:
            return (
                yaml.safe_load(f)
                .get("langgraph", {})
                .get("memory", {})
                .get("redis_url", "redis://localhost:6379")
            )
    return "redis://localhost:6379"


_REDIS_URL = _load_redis_url()


@router.post("/render/approve/{request_id}", status_code=202)
async def approve_motion_render(request_id: str):
    """Approve a pending motion render and enqueue the Celery task.

    Returns 202 with task_id on success.
    Returns 404 if no pending motion payload exists.
    Returns 400 if the stored payload is malformed.
    Returns 503 if Celery is unavailable.
    Returns 500 on Redis errors.
    """
    import redis.asyncio as aioredis

    # 1. Read pending payload from Redis
    try:
        r = aioredis.from_url(_REDIS_URL, socket_connect_timeout=2)
        try:
            raw = await r.get(f"motion_pending:{request_id}")
        finally:
            await r.close()
    except Exception as exc:
        logger.error("Redis read error for %s: %s", request_id, exc)
        raise HTTPException(status_code=500, detail="Redis unavailable")

    if raw is None:
        raise HTTPException(
            status_code=404,
            detail="No pending motion for this request",
        )

    # 2. Parse payload
    try:
        payload = json.loads(raw)
    except json.JSONDecodeError:
        logger.error("Malformed motion pending payload for %s", request_id)
        raise HTTPException(status_code=400, detail="Malformed pending motion payload")

    prompt = payload.get("prompt", "")
    constraints = payload.get("constraints", [])
    session_id = payload.get("session_id", "")

    # 3. Check Celery availability
    try:
        from langgraph_agents.celery_app import celery_app
        if celery_app is None:
            raise HTTPException(status_code=503, detail="Task queue unavailable")
    except ImportError:
        raise HTTPException(status_code=503, detail="Task queue unavailable")

    # 4. Fire Celery task
    try:
        task = celery_app.send_task(
            "langgraph.generate_motion",
            args=(prompt, constraints, 3.0, request_id, session_id),
        )
    except Exception as exc:
        logger.error("Failed to enqueue motion task for %s: %s", request_id, exc)
        raise HTTPException(status_code=503, detail=f"Failed to enqueue task: {exc}")

    # 5. Delete pending key + store tracking keys
    try:
        r = aioredis.from_url(_REDIS_URL, socket_connect_timeout=2)
        try:
            await r.delete(f"motion_pending:{request_id}")
            await r.setex(
                f"motion_task:{request_id}",
                3600,
                json.dumps({"task_id": task.id, "status": "queued"}),
            )
            await r.sadd(f"pending_tasks:{session_id}", task.id)
            await r.expire(f"pending_tasks:{session_id}", 3600)
        finally:
            await r.close()
    except Exception as exc:
        logger.error("Redis write error for %s: %s", request_id, exc)
        raise HTTPException(status_code=500, detail="Failed to update task state")

    logger.info("Motion render approved: request=%s task=%s", request_id, task.id)

    return {"task_id": task.id, "status": "queued"}
