"""Tests for SSE /chat endpoint + session endpoints (Phase 5 + P0.1/P0.2)."""

import json
from unittest.mock import AsyncMock, MagicMock, patch

import pytest


def _parse_sse_stream(raw: bytes) -> list[dict]:
    """Parse SSE text into [{event, data}] list."""
    events = []
    current_event = None
    for line in raw.decode("utf-8").splitlines():
        if line.startswith("event:"):
            current_event = line[len("event:"):].strip()
        elif line.startswith("data:"):
            data_str = line[len("data:"):].strip()
            try:
                events.append({"event": current_event, "data": json.loads(data_str)})
            except json.JSONDecodeError:
                events.append({"event": current_event, "data": data_str})
            current_event = None
    return events


def _make_fake_astream_stage_only():
    """Minimal fake astream — updates (no custom token stream).

    Phase 6.9: conversation node deleted, synthesizer produces final_answer
    for all intents (including chat/clarify).
    """

    async def fake_stream(state, config, stream_mode=None):
        yield ("updates", {"memory": {}})
        yield ("updates", {"planner": {}})
        yield ("updates", {"synthesizer": {
            "final_answer": "Xin chao!",
            "intent": "conversation",
            "total_tokens": 42,
        }})

    return fake_stream


def _make_fake_astream_with_tokens():
    """Astream that emits real token events via the custom stream channel."""

    async def fake_stream(state, config, stream_mode=None):
        yield ("updates", {"memory": {}})
        yield ("updates", {"planner": {"intent": "conversation"}})
        for tok in ["Xin ", "chào ", "bạn", "!"]:
            yield ("custom", {"content": tok})
        yield ("updates", {"synthesizer": {
            "final_answer": "Xin chào bạn!",
            "intent": "conversation",
            "total_tokens": 4,
        }})

    return fake_stream


def _make_fake_astream_with_tools():
    """Astream with retriever_agent node for exercise queries."""

    async def fake_stream(state, config, stream_mode=None):
        yield ("updates", {"planner": {"intent": "exercise_recommendation"}})
        yield ("updates", {"retriever_agent": {}})
        yield ("updates", {"synthesizer": {
            "final_answer": "Bai tap cho lung: cat-cow, child pose...",
            "intent": "exercise_recommendation",
            "total_tokens": 120,
        }})

    return fake_stream


def _set_graph(mock_graph):
    """Set the module-global graph for a test. Returns the mock."""
    import langgraph_agents.api.main as api_module
    api_module._graph = mock_graph
    return mock_graph


def _set_redis(mock_redis):
    """Set the module-global redis client for a test. Returns the mock."""
    import langgraph_agents.api.main as api_module
    api_module._redis = mock_redis
    return mock_redis


@pytest.fixture
def api_client():
    """Fixture: mock graph + redis. Creates fresh app per test."""
    mock_redis = MagicMock()
    mock_redis.get.return_value = None

    mock_graph = MagicMock()
    mock_graph.astream = _make_fake_astream_stage_only()

    _set_graph(mock_graph)
    _set_redis(mock_redis)

    from langgraph_agents.api.main import create_app
    from fastapi.testclient import TestClient
    app = create_app()
    client = TestClient(app)

    yield client, mock_redis, mock_graph

    # Reset module globals after test
    import langgraph_agents.api.main as api_module
    api_module._graph = None
    api_module._redis = None


# ── Health ─────────────────────────────────────────────────────────


@pytest.mark.unit
def test_health_returns_ok(api_client):
    client, _, _ = api_client
    resp = client.get("/health")
    assert resp.status_code == 200
    assert resp.json()["status"] == "ok"


@pytest.mark.unit
def test_health_detailed_returns_checks(api_client):
    client, _, _ = api_client
    resp = client.get("/health/detailed")
    assert resp.status_code in (200, 503)
    body = resp.json()
    # FastAPI can return (body, status_code) tuple directly;
    # if the JSON body is a list, unwrap it.
    if isinstance(body, list):
        body = body[0] if len(body) > 0 else {}
    assert "checks" in body
    assert "status" in body


# ── SSE stage events ───────────────────────────────────────────────


@pytest.mark.unit
def test_sse_chat_emits_stage_events(api_client, monkeypatch):
    client, _, mock_graph = api_client
    mock_graph.astream = _make_fake_astream_stage_only()
    _set_graph(mock_graph)

    resp = client.post("/chat", json={"query": "Xin chao"})
    assert resp.status_code == 200
    events = _parse_sse_stream(resp.content)
    stage_events = [e for e in events if e["event"] == "stage"]
    nodes_seen = {e["data"]["node"] for e in stage_events}
    assert "memory" in nodes_seen
    assert "planner" in nodes_seen
    assert "synthesizer" in nodes_seen


@pytest.mark.unit
def test_sse_chat_emits_done_event_last(api_client, monkeypatch):
    client, _, mock_graph = api_client
    mock_graph.astream = _make_fake_astream_stage_only()
    _set_graph(mock_graph)

    resp = client.post("/chat", json={"query": "Xin chào"})
    events = _parse_sse_stream(resp.content)
    assert events[-1]["event"] == "done"
    assert events[-1]["data"]["total_tokens"] == 42


# ── Token streaming ────────────────────────────────────────────────


@pytest.mark.unit
def test_sse_chat_emits_token_events(api_client, monkeypatch):
    client, _, mock_graph = api_client
    mock_graph.astream = _make_fake_astream_with_tokens()
    _set_graph(mock_graph)

    resp = client.post("/chat", json={"query": "Xin chào"})
    assert resp.status_code == 200
    events = _parse_sse_stream(resp.content)
    token_events = [e for e in events if e["event"] == "token"]
    assert len(token_events) == 4
    contents = "".join(e["data"]["content"] for e in token_events)
    assert contents == "Xin chào bạn!"


@pytest.mark.unit
def test_sse_chat_no_tokens_when_no_custom_events(api_client):
    client, _, mock_graph = api_client
    mock_graph.astream = _make_fake_astream_stage_only()
    _set_graph(mock_graph)

    resp = client.post("/chat", json={"query": "Xin chào"})
    events = _parse_sse_stream(resp.content)
    token_events = [e for e in events if e["event"] == "token"]
    assert len(token_events) == 0


@pytest.mark.unit
def test_sse_chat_stage_started_before_tokens(api_client, monkeypatch):
    client, _, mock_graph = api_client
    mock_graph.astream = _make_fake_astream_with_tokens()
    _set_graph(mock_graph)

    resp = client.post("/chat", json={"query": "Xin chào"})
    events = _parse_sse_stream(resp.content)

    first_token_idx = next((i for i, e in enumerate(events) if e["event"] == "token"), -1)
    assert first_token_idx > 0, "Expected at least 1 token event"

    prior_stage_started = [
        e for e in events[:first_token_idx]
        if e["event"] == "stage"
        and e["data"].get("node") == "synthesizer"
        and e["data"].get("status") == "started"
    ]
    assert len(prior_stage_started) == 1


# ── Retriever stage ────────────────────────────────────────────────


@pytest.mark.unit
def test_sse_chat_emits_retriever_stage(api_client, monkeypatch):
    client, _, mock_graph = api_client
    mock_graph.astream = _make_fake_astream_with_tools()
    _set_graph(mock_graph)

    resp = client.post("/chat", json={"query": "Bài tập cho đau lưng"})
    events = _parse_sse_stream(resp.content)
    stage_events = [e for e in events if e["event"] == "stage"]
    nodes_seen = {e["data"]["node"] for e in stage_events}
    assert "retriever_agent" in nodes_seen
    assert "planner" in nodes_seen


# ── Speech mode ────────────────────────────────────────────────────


@pytest.mark.unit
def test_sse_chat_speech_mode_emits_speech_pending(api_client, monkeypatch):
    client, _, mock_graph = api_client
    mock_graph.astream = _make_fake_astream_stage_only()
    _set_graph(mock_graph)

    import langgraph_agents.api.main as api_module
    fake_synth = AsyncMock()
    monkeypatch.setattr(api_module, "synthesize_speech_async", fake_synth)
    monkeypatch.setattr(api_module, "get_persona", lambda pid: {})

    async def fake_poll(task_id, timeout=15):
        yield api_module.encode_event("speech_ready", {"task_id": task_id, "url": "http://x.wav"})

    monkeypatch.setattr(api_module, "_poll_speech_result", fake_poll)

    resp = client.post("/chat", json={"query": "Hãy đọc", "output_mode": "speech"})
    events = _parse_sse_stream(resp.content)
    speech_pending = [e for e in events if e["event"] == "speech_pending"]
    assert len(speech_pending) == 1
    assert "task_id" in speech_pending[0]["data"]
    speech_ready = [e for e in events if e["event"] == "speech_ready"]
    assert len(speech_ready) == 1


# ── Session persisted ──────────────────────────────────────────────


@pytest.mark.unit
def test_sse_chat_session_persisted_before_done(api_client, monkeypatch):
    client, _, mock_graph = api_client
    mock_graph.astream = _make_fake_astream_stage_only()
    _set_graph(mock_graph)

    import langgraph_agents.api.main as api_module
    async def fake_write(*args, **kwargs):
        pass

    monkeypatch.setattr(api_module, "write_session_turn", fake_write)

    resp = client.post("/chat", json={"query": "Xin chào"})
    events = _parse_sse_stream(resp.content)
    non_done_indices = [i for i, e in enumerate(events) if e["event"] != "done"]
    done_index = next((i for i, e in enumerate(events) if e["event"] == "done"), -1)
    if non_done_indices:
        assert max(non_done_indices) < done_index


# ── Session persist failure does not break stream ──────────────────


@pytest.mark.unit
def test_sse_chat_persist_failure_does_not_block(api_client, monkeypatch):
    client, _, mock_graph = api_client
    mock_graph.astream = _make_fake_astream_stage_only()
    _set_graph(mock_graph)

    import langgraph_agents.api.main as api_module
    async def fake_write_fail(*args, **kwargs):
        raise RuntimeError("DB down")

    monkeypatch.setattr(api_module, "write_session_turn", fake_write_fail)

    resp = client.post("/chat", json={"query": "Xin chào"})
    events = _parse_sse_stream(resp.content)
    assert events[-1]["event"] == "done"


# ── TTS result (fallback) ──────────────────────────────────────────


@pytest.mark.unit
def test_tts_result_404_when_missing(api_client):
    client, mock_redis, _ = api_client
    mock_redis.get.return_value = None
    _set_redis(mock_redis)
    resp = client.get("/tts/nonexistent/result")
    assert resp.status_code == 404


@pytest.mark.unit
def test_tts_result_200_when_present(api_client):
    client, mock_redis, _ = api_client
    mock_redis.get.return_value = b'{"event":"speech_ready","url":"http://x.wav"}'
    _set_redis(mock_redis)
    resp = client.get("/tts/abc/result")
    assert resp.status_code == 200
    assert resp.json()["event"] == "speech_ready"


@pytest.mark.unit
def test_tts_result_500_on_corrupt(api_client):
    client, mock_redis, _ = api_client
    mock_redis.get.return_value = b'not json'
    _set_redis(mock_redis)
    resp = client.get("/tts/abc/result")
    assert resp.status_code == 500


# ── Schema tests ───────────────────────────────────────────────────


class TestSchemas:
    @pytest.mark.unit
    def test_chat_request_defaults(self):
        from langgraph_agents.api.schemas import ChatRequest
        req = ChatRequest(query="Hello")
        assert req.user_id == "anonymous"
        assert req.output_mode == "text"

    @pytest.mark.unit
    def test_session_list_item(self):
        from langgraph_agents.api.schemas import SessionListItem
        item = SessionListItem(
            session_id="s1", created_at="2026-01-01T00:00:00",
            updated_at="2026-01-02T00:00:00",
            first_user_message_preview="Hello...", message_count=5,
        )
        assert item.session_id == "s1"
        assert item.message_count == 5

    @pytest.mark.unit
    def test_session_resume_response(self):
        from langgraph_agents.api.schemas import SessionResumeResponse
        resp = SessionResumeResponse(
            session_id="s1", messages=[],
            stm_populated=True, last_updated="2026-01-02T00:00:00",
        )
        assert resp.stm_populated is True
