"""Celery app — DISABLED in v2.4.1 (reserved for Phase 7 hybrid cloud).

v2.4 used Celery for TTS background tasks. v2.4.1 switched to FastAPI
BackgroundTasks (in-process async) because:
  - VieNeu-TTS latency < 10s, no need for distributed worker
  - Removes 1 process + 1 Redis DB + 1 dependency
  - Simpler test mocking

This module is kept as a skeleton for Phase 7 (hybrid cloud) when:
  - TTS scale > 100 req/min → need queue + multiple workers
  - Add heavy async jobs (S3 batch upload, doc ingestion, scheduled tasks)
  - Edge worker (HP ProDesk) needs to consume jobs from cloud queue

To reactivate:
  1. Add celery>=5.4.0 to requirements-langgraph.txt
  2. Add langgraph.celery block to config/langgraph.yaml
  3. Convert TTS task in services/vieneu_tts/tasks.py back to @celery_app.task
  4. Wire synthesize_speech.delay(...) in api/main.py
"""

from __future__ import annotations

import logging

logger = logging.getLogger("langgraph.celery")

celery_app = None   # disabled

# Phase 7 reactivation snippet (commented):
#
# try:
#     import yaml
#     from celery import Celery
#     from pathlib import Path
#
#     def _load_cfg() -> dict:
#         path = Path(__file__).resolve().parents[3] / "config" / "langgraph.yaml"
#         if not path.exists():
#             return {}
#         with open(path, "r", encoding="utf-8") as f:
#             return yaml.safe_load(f).get("langgraph", {}).get("celery", {})
#
#     cfg = _load_cfg()
#     celery_app = Celery("langgraph_agents")
#     celery_app.conf.update(
#         broker_url=cfg.get("broker_url", "redis://localhost:6379/1"),
#         result_backend=cfg.get("result_backend", "redis://localhost:6379/1"),
#         task_default_queue="langgraph",
#         imports=("langgraph_agents.services.vieneu_tts.tasks",),
#     )
# except ImportError:
#     pass
