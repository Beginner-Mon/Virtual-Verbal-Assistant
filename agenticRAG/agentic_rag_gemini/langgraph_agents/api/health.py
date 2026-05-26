"""Dependency health checks for /health/detailed.

Each check returns CheckResult with ok, latency_ms, detail.
run_all_checks() runs them in parallel and aggregates.
"""

import asyncio
import logging
import os
import time
from dataclasses import dataclass

from langgraph_agents.shared.logging import get_logger

logger = get_logger("langgraph.health")


@dataclass
class CheckResult:
    name: str
    ok: bool
    latency_ms: float
    detail: str | None = None


async def check_redis(redis_client, timeout: float = 2.0) -> CheckResult:
    # asyncio.wait_for cancels the awaiting coroutine but NOT the underlying
    # thread spawned by asyncio.to_thread — that thread stays blocked in
    # socket.recv until the redis client's own socket_timeout fires (set at
    # client construction time in api/main.py:_get_redis). Together they
    # bound:
    #   • health response: 2s (wait_for) → returns 503 quickly
    #   • thread occupancy: 5s (socket_timeout) → thread releases shortly after
    # Phase 7 migration to redis.asyncio for true cancellation.
    t0 = time.perf_counter()
    try:
        await asyncio.wait_for(
            asyncio.to_thread(redis_client.ping),
            timeout=timeout,
        )
        elapsed_ms = round((time.perf_counter() - t0) * 1000, 1)
        return CheckResult(name="redis", ok=True, latency_ms=elapsed_ms)
    except asyncio.TimeoutError:
        elapsed_ms = round((time.perf_counter() - t0) * 1000, 1)
        return CheckResult(name="redis", ok=False, latency_ms=elapsed_ms,
                           detail=f"timeout after {timeout}s")
    except Exception as exc:
        elapsed_ms = round((time.perf_counter() - t0) * 1000, 1)
        return CheckResult(name="redis", ok=False, latency_ms=elapsed_ms, detail=str(exc))


async def check_postgres(timeout: float = 2.0) -> CheckResult:
    t0 = time.perf_counter()
    try:
        from langgraph_agents.db.postgres import PostgresClient
        pg = PostgresClient()
        await asyncio.wait_for(pg.connect(), timeout=timeout)
        elapsed_ms = round((time.perf_counter() - t0) * 1000, 1)
        return CheckResult(name="postgres", ok=True, latency_ms=elapsed_ms)
    except Exception as exc:
        elapsed_ms = round((time.perf_counter() - t0) * 1000, 1)
        return CheckResult(name="postgres", ok=False, latency_ms=elapsed_ms, detail=str(exc))


async def check_graph(graph) -> CheckResult:
    ok = graph is not None
    return CheckResult(name="graph", ok=ok, latency_ms=0.0,
                       detail=None if ok else "not_loaded")


async def check_llm(timeout: float = 3.0) -> CheckResult:
    if os.getenv("LLM_HEALTHCHECK", "0") != "1":
        return CheckResult(name="llm", ok=True, latency_ms=0.0, detail="skipped")
    t0 = time.perf_counter()
    try:
        from langgraph_agents.llm import get_chat_model
        llm = get_chat_model("health_check", temperature=0)
        await asyncio.wait_for(llm.ainvoke("ping"), timeout=timeout)
        elapsed_ms = round((time.perf_counter() - t0) * 1000, 1)
        return CheckResult(name="llm", ok=True, latency_ms=elapsed_ms)
    except Exception as exc:
        elapsed_ms = round((time.perf_counter() - t0) * 1000, 1)
        return CheckResult(name="llm", ok=False, latency_ms=elapsed_ms, detail=str(exc))


async def check_mcp(timeout: float = 3.0) -> CheckResult:
    t0 = time.perf_counter()
    try:
        from langgraph_agents.mcp.client import get_mcp_tools
        tools = await asyncio.wait_for(get_mcp_tools(), timeout=timeout)
        elapsed_ms = round((time.perf_counter() - t0) * 1000, 1)
        if tools:
            return CheckResult(name="mcp", ok=True, latency_ms=elapsed_ms,
                               detail=f"{len(tools)} tool(s)")
        return CheckResult(name="mcp", ok=True, latency_ms=elapsed_ms,
                           detail="degraded: 0 tools (graceful)")
    except Exception as exc:
        elapsed_ms = round((time.perf_counter() - t0) * 1000, 1)
        return CheckResult(name="mcp", ok=False, latency_ms=elapsed_ms, detail=str(exc))


async def run_all_checks(graph, redis_client) -> dict:
    """Run all dep checks in parallel. Returns {status, checks} dict for JSON response."""
    results: list[CheckResult] = await asyncio.gather(
        check_redis(redis_client),
        check_postgres(),
        check_graph(graph),
        check_llm(),
        check_mcp(),
        return_exceptions=True,
    )

    checks = {}
    for r in results:
        if isinstance(r, CheckResult):
            checks[r.name] = {
                "ok": r.ok,
                "latency_ms": r.latency_ms,
            }
            if r.detail:
                checks[r.name]["detail"] = r.detail

    # Circuit breaker status for each role
    try:
        from langgraph_agents.llm import get_breaker_snapshot
        for role in ["planner", "synthesizer", "conversation", "retriever"]:
            snap = get_breaker_snapshot(role)
            if snap:
                checks[f"breaker:{role}"] = snap["state"]
    except Exception:
        pass

    all_ok = all(
        v.get("ok", False) if isinstance(v, dict) else v in ("ok", "closed")
        for v in checks.values()
    )
    return {
        "status": "ready" if all_ok else "degraded",
        "checks": checks,
        "all_ok": all_ok,
    }
