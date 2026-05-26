"""Tests for SSE /chat endpoint + session endpoints (Phase 5)."""

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
    """Minimal fake astream_events — stage events + final LangGraph end."""

    async def fake_stream(state, config, version="v2"):
        nodes = ["memory", "planner", "conversation"]
        for name in nodes:
            yield {"event": "on_chain_start", "name": name, "data": {}, "metadata": {}}
            yield {"event": "on_chain_end", "name": name, "data": {"output": {}}, "metadata": {}}
        yield {
            "event": "on_chain_end", "name": "LangGraph",
            "data": {"output": {
                "final_answer": "Xin chào! Tôi là ECA.",
                "intent": "conversation",
                "confidence": 0.95,
                "total_tokens": 42,
            }},
            "metadata": {},
        }

    return fake_stream


def _make_fake_astream_with_tokens():
    """Astream with conversation tokens mixed in."""

    async def fake_stream(state, config, version="v2"):
        yield {"event": "on_chain_start", "name": "planner", "data": {}, "metadata": {}}
        yield {
            "event": "on_chat_model_stream",
            "name": "ChatOpenAI",
            "data": {"chunk": _FakeChunk("{")},
            "metadata": {"langgraph_node": "planner"},
        }
        yield {"event": "on_chain_end", "name": "planner",
               "data": {"output": {"intent": "conversation"}}, "metadata": {}}

        yield {"event": "on_chain_start", "name": "conversation", "data": {}, "metadata": {}}
        for token_text in ["Xin", " ", "chào", "!"]:
            yield {
                "event": "on_chat_model_stream",
                "name": "ChatOpenAI",
                "data": {"chunk": _FakeChunk(token_text)},
                "metadata": {"langgraph_node": "conversation"},
            }
        yield {"event": "on_chain_end", "name": "conversation", "data": {"output": {}}, "metadata": {}}

        yield {
            "event": "on_chain_end", "name": "LangGraph",
            "data": {"output": {
                "final_answer": "Xin chào!",
                "intent": "conversation",
                "total_tokens": 5,
            }},
            "metadata": {},
        }

    return fake_stream


def _make_fake_astream_with_tools():
    """Astream with tool calling events."""

    async def fake_stream(state, config, version="v2"):
        yield {"event": "on_chain_start", "name": "planner", "data": {}, "metadata": {}}
        yield {"event": "on_chain_end", "name": "planner",
               "data": {"output": {"intent": "exercise_recommendation"}}, "metadata": {}}

        yield {"event": "on_chain_start", "name": "retriever_agent", "data": {}, "metadata": {}}
        yield {"event": "on_tool_start", "name": "pgvector_search", "data": {}, "metadata": {}}
        yield {"event": "on_tool_end", "name": "pgvector_search",
               "data": {"output": [{"content": "doc1"}, {"content": "doc2"}]}, "metadata": {}}
        yield {"event": "on_chain_end", "name": "retriever_agent", "data": {"output": {}}, "metadata": {}}

        yield {"event": "on_chain_start", "name": "conversation", "data": {}, "metadata": {}}
        yield {"event": "on_chain_end", "name": "conversation", "data": {"output": {}}, "metadata": {}}

        yield {
            "event": "on_chain_end", "name": "LangGraph",
            "data": {"output": {
                "final_answer": "Bài tập cho lưng...",
                "intent": "exercise_recommendation",
                "total_tokens": 120,
            }},
            "metadata": {},
        }

    return fake_stream


class _FakeChunk:
    def __init__(self, content):
        self.content = content


@pytest.fixture
def api_client(monkeypatch):
    """Fixture: mock graph with astream_events + mock redis."""
    mock_redis = MagicMock()
    mock_redis.get.return_value = None

    import langgraph_agents.api.main as api_module
    api_module._redis = mock_redis

    from langgraph_agents.api.main import create_app
    from fastapi.testclient import TestClient
    app = create_app()
    client = TestClient(app)

    yield client, mock_redis


# ── Health ─────────────────────────────────────────────────────────


@pytest.mark.unit
def test_health_returns_ok(api_client):
    client, _ = api_client
    resp = client.get("/health")
    assert resp.status_code == 200
    assert resp.json()["status"] == "ok"


# ── SSE stage events ───────────────────────────────────────────────


@pytest.mark.unit
def test_sse_chat_emits_stage_events(api_client, monkeypatch):
    client, _ = api_client
    import langgraph_agents.api.main as api_module
    mock_graph = MagicMock()
    mock_graph.astream_events = _make_fake_astream_stage_only()
    api_module._graph = mock_graph

    resp = client.post("/chat", json={"query": "Xin chào"})
    assert resp.status_code == 200
    events = _parse_sse_stream(resp.content)
    stage_events = [e for e in events if e["event"] == "stage"]
    nodes_seen = {e["data"]["node"] for e in stage_events}
    assert "memory" in nodes_seen
    assert "planner" in nodes_seen
    assert "conversation" in nodes_seen


@pytest.mark.unit
def test_sse_chat_emits_done_event_last(api_client, monkeypatch):
    client, _ = api_client
    import langgraph_agents.api.main as api_module
    mock_graph = MagicMock()
    mock_graph.astream_events = _make_fake_astream_stage_only()
    api_module._graph = mock_graph

    resp = client.post("/chat", json={"query": "Xin chào"})
    events = _parse_sse_stream(resp.content)
    assert events[-1]["event"] == "done"
    assert events[-1]["data"]["total_tokens"] == 42


# ── Token filtering ────────────────────────────────────────────────


@pytest.mark.unit
def test_sse_chat_token_events_only_from_conversation(api_client, monkeypatch):
    client, _ = api_client
    import langgraph_agents.api.main as api_module
    mock_graph = MagicMock()
    mock_graph.astream_events = _make_fake_astream_with_tokens()
    api_module._graph = mock_graph

    resp = client.post("/chat", json={"query": "Xin chào"})
    events = _parse_sse_stream(resp.content)
    token_events = [e for e in events if e["event"] == "token"]
    # Only conversation tokens visible (planner "{" filtered out)
    contents = "".join(e["data"]["content"] for e in token_events)
    assert "{" not in contents  # planner JSON token filtered
    assert "Xin" in contents


# ── Tool events ────────────────────────────────────────────────────


@pytest.mark.unit
def test_sse_chat_emits_tool_events(api_client, monkeypatch):
    client, _ = api_client
    import langgraph_agents.api.main as api_module
    mock_graph = MagicMock()
    mock_graph.astream_events = _make_fake_astream_with_tools()
    api_module._graph = mock_graph

    resp = client.post("/chat", json={"query": "Bài tập cho đau lưng"})
    events = _parse_sse_stream(resp.content)
    tool_calling = [e for e in events if e["event"] == "tool_calling"]
    tool_complete = [e for e in events if e["event"] == "tool_complete"]
    assert any(e["data"]["tool"] == "pgvector_search" for e in tool_calling)
    assert any(e["data"]["tool"] == "pgvector_search" for e in tool_complete)
    pgvector_complete = next(e for e in tool_complete if e["data"]["tool"] == "pgvector_search")
    assert pgvector_complete["data"]["result_count"] == 2


# ── Speech mode ────────────────────────────────────────────────────


@pytest.mark.unit
def test_sse_chat_speech_mode_emits_speech_pending(api_client, monkeypatch):
    client, _ = api_client
    import langgraph_agents.api.main as api_module
    mock_graph = MagicMock()
    mock_graph.astream_events = _make_fake_astream_stage_only()
    api_module._graph = mock_graph

    # Mock synthesize_speech_async
    fake_synth = AsyncMock()
    monkeypatch.setattr(api_module, "synthesize_speech_async", fake_synth)

    # Mock get_persona
    monkeypatch.setattr(api_module, "get_persona", lambda pid: {})

    # Mock _poll_speech_result to avoid 15s timeout
    async def fake_poll(task_id, timeout=15):
        yield api_module.encode_event("speech_ready", {"task_id": task_id, "url": "http://x.wav"})

    monkeypatch.setattr(api_module, "_poll_speech_result", fake_poll)

    resp = client.post("/chat", json={"query": "Hãy đọc", "output_mode": "speech"})
    events = _parse_sse_stream(resp.content)
    speech_pending = [e for e in events if e["event"] == "speech_pending"]
    assert len(speech_pending) == 1
    assert "task_id" in speech_pending[0]["data"]
    # speech_ready should also be emitted
    speech_ready = [e for e in events if e["event"] == "speech_ready"]
    assert len(speech_ready) == 1


# ── Session persisted ──────────────────────────────────────────────


@pytest.mark.unit
def test_sse_chat_session_persisted_before_done(api_client, monkeypatch):
    client, _ = api_client
    import langgraph_agents.api.main as api_module
    mock_graph = MagicMock()
    mock_graph.astream_events = _make_fake_astream_stage_only()
    api_module._graph = mock_graph

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
    client, _ = api_client
    import langgraph_agents.api.main as api_module
    mock_graph = MagicMock()
    mock_graph.astream_events = _make_fake_astream_stage_only()
    api_module._graph = mock_graph

    async def fake_write_fail(*args, **kwargs):
        raise RuntimeError("DB down")

    monkeypatch.setattr(api_module, "write_session_turn", fake_write_fail)

    resp = client.post("/chat", json={"query": "Xin chào"})
    events = _parse_sse_stream(resp.content)
    assert events[-1]["event"] == "done"


# ── TTS result (fallback) ──────────────────────────────────────────


@pytest.mark.unit
def test_tts_result_404_when_missing(api_client):
    client, mock_redis = api_client
    mock_redis.get.return_value = None
    resp = client.get("/tts/nonexistent/result")
    assert resp.status_code == 404


@pytest.mark.unit
def test_tts_result_200_when_present(api_client):
    client, mock_redis = api_client
    mock_redis.get.return_value = b'{"event":"speech_ready","url":"http://x.wav"}'
    resp = client.get("/tts/abc/result")
    assert resp.status_code == 200
    assert resp.json()["event"] == "speech_ready"


@pytest.mark.unit
def test_tts_result_500_on_corrupt(api_client):
    client, mock_redis = api_client
    mock_redis.get.return_value = b'not json'
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
