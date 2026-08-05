"""Dependency health checks for /health/detailed.

Each check returns CheckResult with ok, latency_ms, detail.
run_all_checks() runs them in parallel and aggregates.
"""

import asyncio
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
    t0 = time.perf_counter()
    try:
        await asyncio.wait_for(redis_client.ping(), timeout=timeout)
        elapsed_ms = round((time.perf_counter() - t0) * 1000, 1)
        return CheckResult(name="redis", ok=True, latency_ms=elapsed_ms)
    except asyncio.TimeoutError:
        elapsed_ms = round((time.perf_counter() - t0) * 1000, 1)
        return CheckResult(name="redis", ok=False, latency_ms=elapsed_ms,
                           detail=f"timeout after {timeout}s")
    except Exception as exc:
        elapsed_ms = round((time.perf_counter() - t0) * 1000, 1)
        return CheckResult(name="redis", ok=False, latency_ms=elapsed_ms, detail=str(exc))


async def check_postgres(timeout: float = 5.0) -> CheckResult:
    """Ping Postgres over the pool the application actually uses.

    This used to do `PostgresClient()` — a brand-new client, and therefore a
    brand-new pool — on every probe, and never closed it. Measured against Neon:
    8 connections before, 18 after five probes, still 18 after idling. Two leaked
    per call, never reclaimed; at a 10s load-balancer interval that walks into
    Neon's 901-connection ceiling in about an hour.

    It also measured the wrong thing. `connect()` on a fresh client builds a
    pool, which on a managed database costs 1.3-1.5s — against a 2s budget, so
    the probe reported `ok: false` on the first call after idle while the
    database was perfectly healthy. Locally it was sub-millisecond and nobody
    noticed.

    Now: reuse the shared client (`connect()` is a no-op once the pool is warm)
    and issue a real `SELECT 1`, which is what "is the database answering?"
    actually means. The wider budget covers the one cold pool build after boot.
    """
    t0 = time.perf_counter()
    try:
        from langgraph_agents.shared import get_pg_client
        pg = get_pg_client()
        await asyncio.wait_for(pg.connect(), timeout=timeout)
        await asyncio.wait_for(pg.fetchval("SELECT 1"), timeout=timeout)
        elapsed_ms = round((time.perf_counter() - t0) * 1000, 1)
        return CheckResult(name="postgres", ok=True, latency_ms=elapsed_ms)
    except asyncio.TimeoutError:
        elapsed_ms = round((time.perf_counter() - t0) * 1000, 1)
        return CheckResult(name="postgres", ok=False, latency_ms=elapsed_ms,
                           detail=f"timeout after {timeout}s")
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


async def check_speechllm(timeout: float = 2.0) -> CheckResult:
    """Check VieNeu TTS service at VIENEU_URL/health."""
    base = os.getenv("VIENEU_URL", "http://localhost:5000").rstrip("/")
    t0 = time.perf_counter()
    try:
        import httpx
        async with httpx.AsyncClient() as client:
            res = await asyncio.wait_for(
                client.get(f"{base}/health"),
                timeout=timeout,
            )
        elapsed_ms = round((time.perf_counter() - t0) * 1000, 1)
        ok = res.status_code == 200
        return CheckResult(name="speechllm", ok=ok, latency_ms=elapsed_ms,
                           detail=None if ok else f"HTTP {res.status_code}")
    except Exception as exc:
        elapsed_ms = round((time.perf_counter() - t0) * 1000, 1)
        return CheckResult(name="speechllm", ok=False, latency_ms=elapsed_ms, detail=str(exc))


async def check_searxng(timeout: float = 2.0) -> CheckResult:
    """Check SearXNG at SEARXNG_URL/healthz."""
    base = os.getenv("SEARXNG_URL", "http://localhost:6666").rstrip("/")
    t0 = time.perf_counter()
    try:
        import httpx
        async with httpx.AsyncClient() as client:
            res = await asyncio.wait_for(
                client.get(f"{base}/healthz"),
                timeout=timeout,
            )
        elapsed_ms = round((time.perf_counter() - t0) * 1000, 1)
        ok = res.status_code == 200
        return CheckResult(name="searxng", ok=ok, latency_ms=elapsed_ms,
                           detail=None if ok else f"HTTP {res.status_code}")
    except Exception as exc:
        elapsed_ms = round((time.perf_counter() - t0) * 1000, 1)
        return CheckResult(name="searxng", ok=False, latency_ms=elapsed_ms, detail=str(exc))


async def run_all_checks(graph, redis_client) -> dict:
    """Run all dep checks in parallel. Returns {status, checks} dict for JSON response."""
    results: list[CheckResult] = await asyncio.gather(
        check_redis(redis_client),
        check_postgres(),
        check_graph(graph),
        check_llm(),
        check_mcp(),
        check_speechllm(),
        check_searxng(),
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

    # Circuit breaker status for each role (informational — see all_ok below).
    # "conversation" role removed in Phase 6.9; snapshot returns None → skipped.
    try:
        from langgraph_agents.llm import get_breaker_snapshot
        for role in ["planner", "synthesizer", "retriever"]:
            snap = get_breaker_snapshot(role)
            if snap:
                checks[f"breaker:{role}"] = snap["state"]
    except Exception:
        pass

    # Health (ok → HTTP 200, fail → 503) is computed ONLY over dependency checks
    # (dict entries with an "ok" field). Circuit-breaker states are reported for
    # visibility but do NOT flip the probe to 503: a single role's breaker
    # opening (e.g. a transient LLM-provider error) must not pull every instance
    # out of the load balancer — that would cascade across the fleet. An open
    # breaker surfaces as status "degraded" while the instance stays live (200).
    all_ok = all(
        v.get("ok", False)
        for v in checks.values()
        if isinstance(v, dict)
    )
    breaker_open = any(
        isinstance(v, str) and v in ("open", "half-open")
        for v in checks.values()
    )
    status = "ready" if (all_ok and not breaker_open) else "degraded"
    return {
        "status": status,
        "checks": checks,
        "all_ok": all_ok,
    }
