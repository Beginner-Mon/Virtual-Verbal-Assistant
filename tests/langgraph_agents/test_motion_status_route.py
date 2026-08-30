"""GET /motion/{job_id} as an actual FastAPI route, not just the pure function.

Task 9's brief only specified `api/motion_status.py` (the pure `motion_status()`
function) plus the CDK wiring in rest_api_stack.py — but the agent Lambda runs
FastAPI behind the Lambda Web Adapter (see agenticRAG/Dockerfile's
AWS_LWA_INVOKE_MODE), so the API Gateway route means nothing unless api/main.py
has a matching endpoint. Without it, GET /motion/{job_id} would 404 in
production despite `cdk synth` succeeding. These tests cover that endpoint.
"""
from __future__ import annotations

import pytest
from fastapi.testclient import TestClient

from langgraph_agents.api.auth import current_user_id
from langgraph_agents.api.main import create_app


def _route_methods(app, path: str) -> set[str]:
    for route in app.routes:
        if getattr(route, "path", None) == path:
            return set(getattr(route, "methods", None) or set())
    return set()


@pytest.mark.unit
def test_motion_route_exists_and_is_a_get():
    app = create_app()
    assert "GET" in _route_methods(app, "/motion/{job_id}")


@pytest.mark.unit
def test_motion_route_returns_404_for_unknown_job(monkeypatch):
    app = create_app()
    app.dependency_overrides[current_user_id] = lambda: "u1"
    try:
        monkeypatch.setattr(
            "langgraph_agents.api.main.motion_status", lambda job_id: {"status": "not_found"}
        )
        client = TestClient(app)
        resp = client.get("/motion/nope")
        assert resp.status_code == 404
    finally:
        app.dependency_overrides.clear()


@pytest.mark.unit
def test_motion_route_returns_signed_url_when_done(monkeypatch):
    app = create_app()
    app.dependency_overrides[current_user_id] = lambda: "u1"
    try:
        monkeypatch.setattr(
            "langgraph_agents.api.main.motion_status",
            lambda job_id: {"status": "done", "url": f"https://cdn/{job_id}?Signature=x"},
        )
        client = TestClient(app)
        resp = client.get("/motion/d1")
        assert resp.status_code == 200
        body = resp.json()
        assert body["status"] == "done"
        assert "Signature=" in body["url"]
    finally:
        app.dependency_overrides.clear()


@pytest.mark.unit
def test_motion_route_passes_through_queued_without_a_url(monkeypatch):
    app = create_app()
    app.dependency_overrides[current_user_id] = lambda: "u1"
    try:
        monkeypatch.setattr(
            "langgraph_agents.api.main.motion_status",
            lambda job_id: {"status": "queued"},
        )
        client = TestClient(app)
        resp = client.get("/motion/q1")
        assert resp.status_code == 200
        assert resp.json() == {"status": "queued"}
    finally:
        app.dependency_overrides.clear()
