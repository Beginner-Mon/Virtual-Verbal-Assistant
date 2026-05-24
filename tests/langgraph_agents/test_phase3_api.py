"""Unit tests for FastAPI /chat endpoint (v2.4.1: BackgroundTasks)."""

import json
from unittest.mock import AsyncMock, MagicMock

import pytest


@pytest.fixture
def api_client(monkeypatch):
    mock_graph = AsyncMock()
    mock_graph.ainvoke = AsyncMock(return_value={
        "final_answer": "Xin chào! Tôi là ECA.",
        "intent": "conversation",
        "confidence": 0.95,
        "needs_clarification": False,
        "total_tokens": 42,
        "grader_result": "pass",
        "grader_warning": None,
        "errors": [],
    })

    mock_redis = MagicMock()
    mock_redis.get.return_value = None

    captured_tasks = []

    async def fake_synth(**kwargs):
        captured_tasks.append(("synthesize_speech_async", kwargs))

    monkeypatch.setattr(
        "langgraph_agents.api.main.synthesize_speech_async",
        fake_synth,
    )

    # Set module-level globals directly — lifespan doesn't run in TestClient (no lifespan events)
    import langgraph_agents.api.main as api_module
    api_module._graph = mock_graph
    api_module._redis = mock_redis

    from langgraph_agents.api.main import create_app
    from fastapi.testclient import TestClient
    app = create_app()
    client = TestClient(app)

    yield client, mock_graph, captured_tasks, mock_redis


class TestSchemas:
    @pytest.mark.unit
    def test_chat_request_defaults(self):
        from langgraph_agents.api.schemas import ChatRequest
        req = ChatRequest(query="Hello")
        assert req.user_id == "anonymous"
        assert req.output_mode == "text"
        assert req.persona_id == "eca_default"

    @pytest.mark.unit
    def test_chat_response_serialization(self):
        from langgraph_agents.api.schemas import ChatResponse
        resp = ChatResponse(
            request_id="r1",
            final_answer="Hi",
            intent="conversation",
            confidence=0.9,
            speech_task_id="t1",
        )
        data = resp.model_dump()
        assert data["request_id"] == "r1"
        assert data["speech_task_id"] == "t1"


@pytest.mark.unit
def test_health_returns_ok(api_client):
    client, _, _, _ = api_client
    resp = client.get("/health")
    assert resp.status_code == 200
    assert resp.json()["status"] == "ok"


@pytest.mark.unit
def test_chat_text_mode_no_tts_task(api_client):
    client, _, captured, _ = api_client
    resp = client.post("/chat", json={"query": "Xin chào", "output_mode": "text"})
    assert resp.status_code == 200
    data = resp.json()
    assert data["final_answer"] == "Xin chào! Tôi là ECA."
    assert data["speech_task_id"] is None
    assert captured == []


@pytest.mark.unit
def test_chat_speech_mode_fires_background_task(api_client):
    client, _, captured, _ = api_client
    resp = client.post("/chat", json={"query": "Hãy đọc câu này", "output_mode": "speech"})
    assert resp.status_code == 200
    data = resp.json()
    assert data["speech_task_id"] is not None
    assert len(captured) == 1
    assert captured[0][0] == "synthesize_speech_async"


@pytest.mark.unit
def test_tts_result_404_when_missing(api_client):
    client, _, _, mock_redis = api_client
    mock_redis.get.return_value = None
    resp = client.get("/tts/nonexistent/result")
    assert resp.status_code == 404


@pytest.mark.unit
def test_tts_result_200_when_present(api_client):
    client, _, _, mock_redis = api_client
    mock_redis.get.return_value = b'{"event":"speech_ready","url":"http://x.wav"}'
    resp = client.get("/tts/abc/result")
    assert resp.status_code == 200
    assert resp.json()["event"] == "speech_ready"


@pytest.mark.unit
def test_tts_result_500_on_corrupt(api_client):
    client, _, _, mock_redis = api_client
    mock_redis.get.return_value = b'not json'
    resp = client.get("/tts/abc/result")
    assert resp.status_code == 500
