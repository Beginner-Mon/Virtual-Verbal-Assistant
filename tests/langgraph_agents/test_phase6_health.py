"""Tests for /health and /health/detailed endpoints (Phase 6 P0.2)."""

import os
from unittest.mock import AsyncMock, MagicMock, patch

import pytest


@pytest.fixture
def api_client(monkeypatch):
    mock_redis = MagicMock()
    mock_redis.ping.return_value = True

    mock_graph = MagicMock()
    mock_graph.astream = AsyncMock()

    # Reset all LLM circuit breakers to closed (other tests may have opened them)
    from langgraph_agents.llm import _cb_map
    for breaker in _cb_map.values():
        breaker._state.failures = 0
        breaker._state.state = "closed"
        breaker._state.opened_at = 0.0

    # Reset MCP breaker
    import langgraph_agents.mcp.client as mcp_mod
    mcp_mod._mcp_breaker._state.failures = 0
    mcp_mod._mcp_breaker._state.state = "closed"
    mcp_mod._mcp_breaker._state.opened_at = 0.0

    import langgraph_agents.api.main as api_module
    api_module._graph = mock_graph
    api_module._redis = mock_redis

    from langgraph_agents.api.main import create_app
    from fastapi.testclient import TestClient
    app = create_app()
    client = TestClient(app)

    yield client, mock_redis, mock_graph

    api_module._graph = None
    api_module._redis = None


# ── Liveness ─────────────────────────────────────────────────────────


@pytest.mark.unit
def test_health_liveness_always_200(api_client):
    """Liveness returns 200 even if all deps are down."""
    client, _, _ = api_client
    resp = client.get("/health")
    assert resp.status_code == 200
    assert resp.json()["status"] == "ok"


@pytest.mark.unit
def test_health_liveness_is_fast(api_client):
    """Liveness must respond in under 50ms (no dep checks)."""
    import time
    client, _, _ = api_client
    t0 = time.perf_counter()
    resp = client.get("/health")
    elapsed_ms = (time.perf_counter() - t0) * 1000
    assert resp.status_code == 200
    assert elapsed_ms < 100, f"Liveness took {elapsed_ms:.0f}ms, expected <100ms"


# ── Readiness: all OK ────────────────────────────────────────────────


@pytest.mark.unit
def test_health_detailed_all_ok(api_client):
    """When all deps are healthy, returns 200 with all ok."""
    client, _, _ = api_client

    with patch("langgraph_agents.api.health.check_postgres",
               new_callable=AsyncMock) as mock_pg, \
         patch("langgraph_agents.api.health.check_redis",
               new_callable=AsyncMock) as mock_redis, \
         patch("langgraph_agents.api.health.check_llm",
               new_callable=AsyncMock) as mock_llm, \
         patch("langgraph_agents.api.health.check_mcp",
               new_callable=AsyncMock) as mock_mcp, \
         patch("langgraph_agents.api.health.check_graph",
               new_callable=AsyncMock) as mock_graph, \
         patch("langgraph_agents.api.health.check_speechllm",
               new_callable=AsyncMock) as mock_speech, \
         patch("langgraph_agents.api.health.check_searxng",
               new_callable=AsyncMock) as mock_searxng:

        from langgraph_agents.api.health import CheckResult
        ok_pg = CheckResult(name="postgres", ok=True, latency_ms=1.0)
        ok_redis = CheckResult(name="redis", ok=True, latency_ms=0.5)
        ok_llm = CheckResult(name="llm", ok=True, latency_ms=0.0, detail="skipped")
        ok_mcp = CheckResult(name="mcp", ok=True, latency_ms=2.0, detail="2 tool(s)")
        ok_graph = CheckResult(name="graph", ok=True, latency_ms=0.0)
        ok_speech = CheckResult(name="speechllm", ok=True, latency_ms=0.0, detail="skipped")
        ok_searxng = CheckResult(name="searxng", ok=True, latency_ms=0.0, detail="skipped")

        mock_pg.return_value = ok_pg
        mock_redis.return_value = ok_redis
        mock_llm.return_value = ok_llm
        mock_mcp.return_value = ok_mcp
        mock_graph.return_value = ok_graph
        mock_speech.return_value = ok_speech
        mock_searxng.return_value = ok_searxng

        resp = client.get("/health/detailed")
        assert resp.status_code == 200
        body = resp.json()
        if isinstance(body, list):
            body = body[0]
        assert body["status"] == "ready"
        assert body["checks"]["postgres"]["ok"] is True
        assert body["checks"]["redis"]["ok"] is True


# ── Readiness: degraded ──────────────────────────────────────────────


@pytest.mark.unit
def test_health_detailed_degraded_when_redis_down(api_client):
    """When a check fails, returns 503 with degraded status."""
    client, _, _ = api_client

    with patch("langgraph_agents.api.health.check_redis",
               new_callable=AsyncMock) as mock_redis:
        from langgraph_agents.api.health import CheckResult
        mock_redis.return_value = CheckResult(
            name="redis", ok=False, latency_ms=2000.0, detail="connection refused",
        )

        # Also mock the others to pass
        with patch("langgraph_agents.api.health.check_postgres",
                   new_callable=AsyncMock) as mock_pg, \
             patch("langgraph_agents.api.health.check_llm",
                   new_callable=AsyncMock) as mock_llm, \
             patch("langgraph_agents.api.health.check_mcp",
                   new_callable=AsyncMock) as mock_mcp, \
             patch("langgraph_agents.api.health.check_speechllm",
                   new_callable=AsyncMock) as mock_speech, \
             patch("langgraph_agents.api.health.check_searxng",
                   new_callable=AsyncMock) as mock_searxng:
            ok_result = CheckResult(name="x", ok=True, latency_ms=0.0, detail="skipped")
            mock_pg.return_value = CheckResult(name="postgres", ok=True, latency_ms=1.0)
            mock_llm.return_value = ok_result
            mock_mcp.return_value = ok_result
            mock_speech.return_value = ok_result
            mock_searxng.return_value = ok_result

            resp = client.get("/health/detailed")
            assert resp.status_code == 503
            body = resp.json()
            if isinstance(body, list):
                body = body[0]
            assert body["status"] == "degraded"


# ── LLM_HEALTHCHECK toggle ───────────────────────────────────────────


@pytest.mark.unit
@pytest.mark.asyncio
async def test_llm_health_skipped_when_env_zero():
    """When LLM_HEALTHCHECK=0 (default), check_llm returns ok with detail=skipped."""
    with patch.dict(os.environ, {"LLM_HEALTHCHECK": "0"}, clear=False):
        from langgraph_agents.api.health import check_llm
        result = await check_llm(timeout=1.0)
        assert result.ok is True
        assert result.detail == "skipped"


@pytest.mark.unit
@pytest.mark.asyncio
async def test_llm_health_runs_when_env_one():
    """When LLM_HEALTHCHECK=1, check_llm actually calls the LLM."""
    with patch.dict(os.environ, {"LLM_HEALTHCHECK": "1"}, clear=False):
        with patch("langgraph_agents.llm.get_chat_model") as mock_gcm:
            mock_llm = MagicMock()
            mock_llm.ainvoke = AsyncMock(return_value="pong")
            mock_gcm.return_value = mock_llm

            from langgraph_agents.api.health import check_llm
            result = await check_llm(timeout=5.0)
            assert result.ok is True
            assert result.detail != "skipped"
            mock_llm.ainvoke.assert_called_once()


# ── Parallel execution ───────────────────────────────────────────────


@pytest.mark.unit
@pytest.mark.asyncio
async def test_run_all_checks_runs_in_parallel():
    """All checks must run concurrently — total elapsed < sum of individual latencies."""
    import time
    import asyncio

    async def slow_ok(name, delay):
        await asyncio.sleep(delay)
        from langgraph_agents.api.health import CheckResult
        return CheckResult(name=name, ok=True, latency_ms=delay * 1000)

    mock_graph = MagicMock()
    mock_redis = MagicMock()

    with patch("langgraph_agents.api.health.check_redis",
               new_callable=AsyncMock) as mock_redis_chk, \
         patch("langgraph_agents.api.health.check_postgres",
               new_callable=AsyncMock) as mock_pg, \
         patch("langgraph_agents.api.health.check_llm",
               new_callable=AsyncMock) as mock_llm, \
         patch("langgraph_agents.api.health.check_mcp",
               new_callable=AsyncMock) as mock_mcp, \
         patch("langgraph_agents.api.health.check_graph",
               new_callable=AsyncMock) as mock_graph_chk, \
         patch("langgraph_agents.api.health.check_speechllm",
               new_callable=AsyncMock) as mock_speech, \
         patch("langgraph_agents.api.health.check_searxng",
               new_callable=AsyncMock) as mock_searxng:

        mock_redis_chk.return_value = slow_ok("redis", 0.05)
        mock_pg.return_value = slow_ok("postgres", 0.05)
        mock_llm.return_value = slow_ok("llm", 0.05)
        mock_mcp.return_value = slow_ok("mcp", 0.05)
        mock_graph_chk.return_value = slow_ok("graph", 0.05)
        mock_speech.return_value = slow_ok("speechllm", 0.05)
        mock_searxng.return_value = slow_ok("searxng", 0.05)

        from langgraph_agents.api.health import run_all_checks
        t0 = time.perf_counter()
        result = await run_all_checks(mock_graph, mock_redis)
        elapsed_s = time.perf_counter() - t0

        # 7 checks each sleep 50ms → sequential = 350ms, parallel < 150ms
        assert elapsed_s < 0.20, f"Took {elapsed_s:.2f}s, expected <0.20s (parallel)"
        assert result["all_ok"] is True


# ── check_postgres must not build its own pool ──────────────────────────────
#
# Regression guard for the connection leak found 05/08: the probe used to do
# `PostgresClient()` per call and never close it — 2 connections leaked per
# probe against Neon, never reclaimed. These tests pin the two properties that
# fix depends on, because every other test in this file mocks check_postgres
# away and so covers none of its internals.


@pytest.mark.unit
@pytest.mark.asyncio
async def test_check_postgres_reuses_shared_client():
    """It must go through get_pg_client(), never construct a PostgresClient."""
    shared = MagicMock()
    shared.connect = AsyncMock()
    shared.fetchval = AsyncMock(return_value=1)

    from langgraph_agents.api import health

    with patch("langgraph_agents.shared.get_pg_client", return_value=shared), \
         patch("langgraph_agents.db.postgres.PostgresClient") as ctor:
        result = await health.check_postgres()

    assert result.ok is True
    assert ctor.call_count == 0, "check_postgres built its own pool again"
    shared.connect.assert_awaited_once()
    shared.fetchval.assert_awaited_once()


@pytest.mark.unit
@pytest.mark.asyncio
async def test_check_postgres_actually_queries_not_just_connects():
    """A reachable pool with a dead database must report not-ok.

    `connect()` succeeding only proves a pool exists. The probe has to ask the
    database something.
    """
    shared = MagicMock()
    shared.connect = AsyncMock()
    shared.fetchval = AsyncMock(side_effect=RuntimeError("server closed connection"))

    from langgraph_agents.api import health

    with patch("langgraph_agents.shared.get_pg_client", return_value=shared):
        result = await health.check_postgres()

    assert result.ok is False
    assert "server closed connection" in (result.detail or "")
