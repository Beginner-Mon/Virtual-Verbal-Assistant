"""Unit tests for FastAPI endpoints (v2.5: SSE /chat)."""

import json
from unittest.mock import AsyncMock, MagicMock

import pytest


def _set_redis(mock_redis):
    import langgraph_agents.api.main as api_module
    api_module._redis = mock_redis


@pytest.fixture
def api_client(monkeypatch):
    mock_redis = MagicMock()
    mock_redis.get.return_value = None
    _set_redis(mock_redis)

    from langgraph_agents.api.main import create_app
    from fastapi.testclient import TestClient
    app = create_app()
    client = TestClient(app)

    yield client, mock_redis

    import langgraph_agents.api.main as api_module
    api_module._redis = None


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
    client, _ = api_client
    resp = client.get("/health")
    assert resp.status_code == 200
    assert resp.json()["status"] == "ok"


@pytest.mark.unit
def test_tts_result_404_when_missing(api_client):
    client, mock_redis = api_client
    mock_redis.get.return_value = None
    _set_redis(mock_redis)
    resp = client.get("/tts/nonexistent/result")
    assert resp.status_code == 404


@pytest.mark.unit
def test_tts_result_200_when_present(api_client):
    client, mock_redis = api_client
    mock_redis.get.return_value = b'{"event":"speech_ready","url":"http://x.wav"}'
    _set_redis(mock_redis)
    resp = client.get("/tts/abc/result")
    assert resp.status_code == 200
    assert resp.json()["event"] == "speech_ready"


@pytest.mark.unit
def test_tts_result_500_on_corrupt(api_client):
    client, mock_redis = api_client
    mock_redis.get.return_value = b'not json'
    _set_redis(mock_redis)
    resp = client.get("/tts/abc/result")
    assert resp.status_code == 500
