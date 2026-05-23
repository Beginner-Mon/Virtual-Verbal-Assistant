"""Celery app for LangGraph async tasks (Kimodo + VieNeu-TTS).

Separate from the existing celery_app.py to avoid modifying legacy code.
Uses Redis DB 1 (/1) to isolate from existing Celery broker on /0.

Worker command:
  celery -A langgraph_agents.celery_app worker -l info -Q langgraph
"""

from __future__ import annotations

import logging
from pathlib import Path

logger = logging.getLogger("langgraph.celery")

celery_app = None

try:
    import yaml
    from celery import Celery

    def _load_celery_config() -> dict:
        config_path = Path(__file__).resolve().parents[3] / "config" / "langgraph.yaml"
        if config_path.exists():
            with open(config_path, "r", encoding="utf-8") as f:
                return yaml.safe_load(f).get("langgraph", {}).get("celery", {})
        return {}

    cfg = _load_celery_config()

    celery_app = Celery("langgraph_agents")
    celery_app.conf.update(
        broker_url=cfg.get("broker_url", "redis://localhost:6379/1"),
        result_backend=cfg.get("result_backend", "redis://localhost:6379/1"),
        task_time_limit=cfg.get("task_time_limit", 300),
        task_soft_time_limit=cfg.get("task_soft_time_limit", 240),
        task_serializer="json",
        accept_content=["json"],
        result_serializer="json",
        task_track_started=True,
        task_acks_late=True,
        task_default_queue="langgraph",
        task_routes={
            "langgraph.generate_motion": {"queue": "langgraph"},
            "langgraph.synthesize_speech": {"queue": "langgraph"},
        },
        imports=(
            "langgraph_agents.services.kimodo.tasks",
            "langgraph_agents.services.vieneu_tts.tasks",
        ),
    )

    logger.info("LangGraph Celery app initialized (broker=%s)", cfg.get("broker_url"))

except ImportError:
    logger.warning("Celery not installed — async tasks disabled")
