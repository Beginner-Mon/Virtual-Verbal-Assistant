"""Kimodo node — writes a queue row instead of calling MCP (D26 + Task 8).

Decisions encoded:
  D3:   needs_motion = HARD GATE (edge cứng, not LLM tool-choice)
  D26:  Motion = node riêng, parallel to retriever, NOT in retriever bind_tools
  Task 8: MCP dropped for this path entirely. D26's "MCP = transport, edge =
          control" reasoning assumed MCP's tool-discovery bought something
          here — it never did, because needs_motion is a hard edge and the
          LLM never chooses the motion tool. MCP stays for web search, where
          the LLM does choose.

The Kimodo node now:
  - Only runs when needs_motion=true (hard edge from planner)
  - Checks the worker heartbeat FIRST. A stale heartbeat means the worker is
    off, and the node returns `unavailable` WITHOUT writing a row — the
    queue must never be loaded with work nobody will pick up.
  - Writes a job row to DynamoDB (via vva_motion.jobs.enqueue) and returns
    immediately. It does not wait for the GPU worker.
  - A GPU worker (text-to-motion/kimodo/worker.py) picks the row up
    independently; this node never talks to it directly.

Job states returned: cache_hit, queued, busy, unavailable.
"""

from __future__ import annotations

import asyncio
import json
import os
import time
from functools import lru_cache

import boto3
from langchain_core.messages import ToolMessage
from langchain_core.runnables import RunnableConfig

from langgraph_agents.state import AgentState
from langgraph_agents.shared.logging import get_logger
from vva_motion.jobs import (
    MAX_QUEUE_DEPTH,
    SECONDS_PER_JOB,
    compute_job_id,
    enqueue,
    queue_depth,
    read_status,
    worker_alive,
)

logger = get_logger("langgraph.kimodo")

DEFAULT_DURATION = 3.0
DEFAULT_STEPS = 100

# THIRD copy of this string, after text-to-motion/kimodo/motion_engine.py and
# .../mcp_server.py (both `DEFAULT_MODEL_NAME`). This one is different from the
# other two: it feeds compute_job_id(), so it is part of the HMAC that IS the
# DynamoDB key.
#
# If it ever drifts from what the worker renders with, nothing breaks loudly.
# Every cache lookup misses, every request re-renders on the GPU, and the only
# symptom is a bill. DEFAULT_DURATION and DEFAULT_STEPS are in the same hash and
# carry the same hazard.
#
# It cannot simply be imported from motion_engine.py: that module lives in the
# GPU image and pulls torch, which the Lambda image deliberately does not have
# (vva_motion/ is the only thing COPY'd across, and it is boto3+stdlib only).
# Making it a shared constant would mean moving it into vva_motion/jobs.py —
# worth doing, but it changes what the worker image ships, so it is its own
# change. Until then: change one, change all three.
MODEL = "Kimodo-SMPLX-RP-v1"

_TABLE = None


def _table():
    """Lazily-initialised DynamoDB table handle — see shared/stm.py:199 for
    the same pattern. Built on first use rather than at import time, so
    importing this module never requires AWS credentials or MOTION_TABLE."""
    global _TABLE
    if _TABLE is None:
        _TABLE = boto3.resource("dynamodb").Table(os.environ["MOTION_TABLE"])
    return _TABLE


def _hash_secret_from_ssm(param: str) -> str | None:
    """Read an SSM SecureString. Cached per parameter name — same shape as
    llm.py's `_secret_from_ssm`, duplicated locally rather than imported:
    llm.py pulls in `langchain_openai` at module scope, and this module has no
    other reason to carry that weight.

    Not called at all in production's common case if MOTION_HASH_SECRET (the
    raw value) is set directly — see `_resolve_hash_secret` below. Reachable
    only when a real deployment sets MOTION_HASH_SECRET_PARAM instead
    (ruling R24: agent_stack.py never bakes this secret into the Lambda's
    environment / the CloudFormation template — only the SSM parameter NAME
    does).
    """
    try:
        value = boto3.client("ssm").get_parameter(
            Name=param, WithDecryption=True,
        )["Parameter"]["Value"]
        logger.info("kimodo_hash_secret_loaded_from_ssm", extra={"param": param})
        return value
    except Exception as exc:                                    # noqa: BLE001
        logger.warning("kimodo_hash_secret_ssm_failed", extra={
            "param": param, "error": str(exc),
        })
        return None


_hash_secret_from_ssm_cached = lru_cache(maxsize=1)(_hash_secret_from_ssm)


def _resolve_hash_secret() -> str | None:
    """MOTION_HASH_SECRET (the raw value) wins when set — the same
    env-wins-over-SSM precedence llm.py's `_resolve_api_key` uses, so a
    developer or a test can set it directly with no AWS involved. This is
    NOT cached: it must see monkeypatch.setenv() changes across test runs,
    unlike the SSM branch below (a real network call, cached since the
    secret cannot rotate mid-invocation anyway).

    Production sets only MOTION_HASH_SECRET_PARAM (ruling R24) — the raw
    value never appears in this Lambda's environment or the CloudFormation
    template, only the SSM parameter name does.
    """
    direct = os.environ.get("MOTION_HASH_SECRET")
    if direct:
        return direct
    param = os.environ.get("MOTION_HASH_SECRET_PARAM")
    return _hash_secret_from_ssm_cached(param) if param else None


def _msg(payload: dict) -> dict:
    return {
        "messages": [ToolMessage(
            content=json.dumps(payload),
            tool_call_id="kimodo_motion",
            name="generate_motion",
        )],
    }


async def kimodo_node(state: AgentState, config: RunnableConfig) -> dict:
    """Kimodo motion job enqueue node.

    Called via hard edge when planner sets needs_motion=true. Never calls the
    GPU directly: it checks the worker is alive, computes the job id, checks
    for an existing row, checks the queue depth, and writes the row. A GPU
    worker polls the table independently and picks the job up.
    """
    t0 = time.perf_counter()
    request_id = config["configurable"].get("request_id", "-")
    resolved_query = state.get("resolved_query") or config["configurable"]["query"]
    table = _table()

    logger.info("node_start", extra={
        "node": "kimodo", "request_id": request_id,
        "query_preview": resolved_query[:80],
    })

    # Đọc heartbeat TRƯỚC khi enqueue: worker tắt thì đừng hứa thứ không ai nhặt.
    #
    # worker_alive/read_status/queue_depth/enqueue are plain sync boto3 calls
    # (vva_motion/jobs.py is shared with the GPU worker's synchronous poll
    # loop, so it stays sync — see the module's own docstring). Each is a
    # blocking network round-trip; run it off the event loop via
    # asyncio.to_thread, same pattern as shared/stm.py:200-203's DynamoStore.
    if not await asyncio.to_thread(worker_alive, table):
        logger.warning("kimodo_worker_unavailable", extra={"request_id": request_id})
        return _msg({"state": "unavailable"})

    # asyncio.to_thread: on a cold Lambda this may be a real SSM network call
    # (_hash_secret_from_ssm_cached), not just an env read — see
    # _resolve_hash_secret's docstring.
    hash_secret = await asyncio.to_thread(_resolve_hash_secret)
    if not hash_secret:
        # Misconfigured deployment (neither MOTION_HASH_SECRET nor a working
        # MOTION_HASH_SECRET_PARAM) — degrade the same way a dead worker does
        # rather than raise. Motion is a non-critical feature of the turn.
        logger.error("kimodo_hash_secret_unavailable", extra={"request_id": request_id})
        return _msg({"state": "unavailable"})

    job_id = compute_job_id(
        hash_secret, resolved_query,
        DEFAULT_DURATION, DEFAULT_STEPS, MODEL,
    )

    existing = await asyncio.to_thread(read_status, table, job_id)
    if existing and existing["status"] == "done":
        return _msg({"state": "cache_hit", "job_id": job_id})   # GPU không chạy
    if existing and existing["status"] in ("queued", "processing"):
        return _msg({"state": "queued", "job_id": job_id,
                     "queue_position": 1, "eta_seconds": SECONDS_PER_JOB})

    depth = await asyncio.to_thread(queue_depth, table)
    if depth >= MAX_QUEUE_DEPTH:
        # Nhận job của người xếp thứ 200 tệ hơn từ chối: đã hứa một thứ họ sẽ không đợi.
        return _msg({"state": "busy", "retry_after_seconds": depth * SECONDS_PER_JOB})

    await asyncio.to_thread(
        enqueue, table, job_id, prompt=resolved_query, duration=DEFAULT_DURATION,
        steps=DEFAULT_STEPS, session_id=config["configurable"].get("session_id"),
    )

    elapsed_ms = round((time.perf_counter() - t0) * 1000)
    logger.info("node_complete", extra={
        "node": "kimodo", "request_id": request_id,
        "elapsed_ms": elapsed_ms, "job_id": job_id, "queue_position": depth + 1,
    })

    return _msg({"state": "queued", "job_id": job_id,
                 "queue_position": depth + 1,
                 "eta_seconds": (depth + 1) * SECONDS_PER_JOB})
