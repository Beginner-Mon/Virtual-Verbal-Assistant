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


# Dependencies without which this instance cannot answer a single request.
# Only these decide the HTTP code, i.e. whether the load balancer keeps sending
# traffic here.
CRITICAL_CHECKS = frozenset({"redis", "postgres", "graph", "llm"})

# Dependencies whose absence removes a feature but leaves the assistant usable:
# no MCP tools, no spoken reply, no web fallback — the chat itself still works,
# and each of these already degrades gracefully at the call site. Failing the
# readiness probe on them would pull healthy instances out of rotation over a
# side feature, and because all instances share these services it would pull out
# *every* instance at once.
OPTIONAL_CHECKS = frozenset({"mcp", "speechllm", "searxng"})


def _is_critical(name: str) -> bool:
    """Unlisted checks count as critical.

    A new check added to `run_all_checks` without a decision here would
    otherwise be silently exempt from the readiness probe — the failure mode
    that hides outages. Defaulting the other way makes the omission loud.
    """
    if name in OPTIONAL_CHECKS:
        return False
    if name not in CRITICAL_CHECKS:
        logger.warning(
            "health check %r is in neither CRITICAL_CHECKS nor OPTIONAL_CHECKS; "
            "treating it as critical",
            name,
        )
    return True


async def run_all_checks(graph, redis_client) -> dict:
    """Run all dep checks in parallel. Returns {status, checks} dict for JSON response.

    `status` is for humans, `all_ok` is for the load balancer:

    | status      | HTTP | meaning                                          |
    |-------------|------|--------------------------------------------------|
    | `ready`     | 200  | everything answering                             |
    | `degraded`  | 200  | an optional dep or a breaker is down; still serve |
    | `unhealthy` | 503  | a critical dep is down; take this instance out    |
    """
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
                # Stated per check so an on-call reader can tell at a glance
                # which red entry is the one that took the instance down.
                "critical": _is_critical(r.name),
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

    # The HTTP code (ok → 200, fail → 503) is computed ONLY over CRITICAL
    # dependency checks. Two categories are deliberately excluded:
    #
    #   - Optional deps (OPTIONAL_CHECKS). TTS being down costs the spoken
    #     reply, not the conversation.
    #   - Circuit-breaker states. A single role's breaker opening (e.g. a
    #     transient LLM-provider error) must not pull every instance out of the
    #     load balancer — every instance would trip at once and the fleet would
    #     empty.
    #
    # Both still surface, as status "degraded" on a live instance (200).
    dep_checks = {k: v for k, v in checks.items() if isinstance(v, dict)}

    all_ok = all(v.get("ok", False) for v in dep_checks.values() if v["critical"])
    optional_down = [k for k, v in dep_checks.items()
                     if not v["critical"] and not v.get("ok", False)]
    breaker_open = any(
        isinstance(v, str) and v in ("open", "half-open")
        for v in checks.values()
    )

    if not all_ok:
        status = "unhealthy"
    elif optional_down or breaker_open:
        status = "degraded"
    else:
        status = "ready"

    if optional_down:
        logger.warning("health: optional dependencies down: %s", ", ".join(optional_down))

    return {
        "status": status,
        "checks": checks,
        "all_ok": all_ok,
        "degraded": optional_down,
    }
