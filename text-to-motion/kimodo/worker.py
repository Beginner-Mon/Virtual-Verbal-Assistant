"""Kimodo worker — pulls work from DynamoDB instead of waiting for HTTP.

Two threads:
  main       poll -> generate motion (busy ~5s) -> poll -> ...
  heartbeat  writes last_seen every 30s, independently

Heartbeat MUST be its own thread: without it, the ~5s spent generating a motion
produces no heartbeat write and the Lambda concludes the worker is dead. This works
because PyTorch releases the GIL on CUDA calls and boto3 releases it during I/O.

Do NOT add a third thread to separate polling from generation: the bottleneck is
VRAM (16.36/24 GB), not vCPU, so pre-fetching a second job would not let it be
processed in parallel anyway.
"""
from __future__ import annotations

import logging
import os
import threading
import time

import boto3

from motion_engine import MotionEngine, build_base_name
from vva_motion.jobs import (
    HEARTBEAT_SECONDS, claim_next_job, complete_job, fail_job,
    recover_abandoned_jobs, write_heartbeat,
)

logger = logging.getLogger("kimodo.worker")

IDLE_SLEEP = 2
DEFAULT_OUT_DIR = "/workspace/outputs"


def run_once(table, engine, bucket: str, s3, out_dir: str = DEFAULT_OUT_DIR) -> bool:
    """Process exactly one job. Returns False if the queue is empty."""
    job = claim_next_job(table)
    if job is None:
        return False

    job_id = job["job_id"]
    try:
        output = engine.generate(
            job["prompt"], float(job.get("duration", 3.0)), int(job.get("steps", 100)),
        )
        base = build_base_name(job_id)
        npz_path, bvh_path = engine.save_outputs(output, out_dir, base)
        for path, ext in ((npz_path, "npz"), (bvh_path, "bvh")):
            s3.upload_file(path, bucket, f"motions/{base}.{ext}")
        complete_job(table, job_id, f"motions/{base}.bvh")
        logger.info("job done: %s", job_id)
    except Exception as exc:                      # noqa: BLE001 - a broken job must not kill the worker
        logger.exception("job failed: %s", job_id)
        fail_job(table, job_id, str(exc))
    return True


def heartbeat_loop(table) -> None:
    while True:
        try:
            write_heartbeat(table)
        except Exception:                          # noqa: BLE001
            logger.exception("heartbeat write failed")
        time.sleep(HEARTBEAT_SECONDS)


def main() -> None:
    logging.basicConfig(level=logging.INFO)

    table_name = os.environ["MOTION_TABLE"]
    bucket = os.environ["MOTION_BUCKET"]
    out_dir = os.environ.get("MCP_OUTPUT_DIR", DEFAULT_OUT_DIR)

    table = boto3.resource("dynamodb").Table(table_name)
    s3 = boto3.client("s3")

    engine = MotionEngine()
    engine.load()                                  # ~38s
    recovered = recover_abandoned_jobs(table)      # clean up rows left by a prior worker
    logger.info("model loaded, recovered %d abandoned job(s)", recovered)

    # Only start heartbeating AFTER the model is loaded: 'alive' must mean 'ready
    # to take work', not 'container started'. ECS reports RUNNING well before this.
    threading.Thread(target=heartbeat_loop, args=(table,), daemon=True).start()

    while True:
        if not run_once(table, engine, bucket, s3, out_dir):
            time.sleep(IDLE_SLEEP)                 # sleep ONLY when the queue is empty


if __name__ == "__main__":
    main()
