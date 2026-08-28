"""Tests for SSE /chat endpoint + session endpoints (Phase 5 + P0.1/P0.2)."""

import json
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from langchain_core.messages import ToolMessage


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
    """Fixture: mock graph + redis + a fixed authenticated user.

    The identity override is what lets these tests POST /chat at all. Every
    route now takes its user from a verified Bearer token — there is no
    anonymous path in any environment — so without it the request is rejected
    before the graph is ever reached and every assertion below sees an empty
    stream.

    dependency_overrides is the right seam for that: it belongs to this app
    object, so it cannot leak out of the test process the way an environment
    variable could. See api/auth.py.
    """
    mock_redis = MagicMock()
    mock_redis.get = AsyncMock(return_value=None)

    mock_graph = MagicMock()
    mock_graph.astream = _make_fake_astream_stage_only()

    _set_graph(mock_graph)
    _set_redis(mock_redis)

    from langgraph_agents.api.main import create_app
    from langgraph_agents.api.auth import current_user_id, override_user
    from fastapi.testclient import TestClient
    app = create_app()
    # override_user rather than a bare lambda: the override replaces everything
    # current_user_id does, including binding the user for row-level security.
    app.dependency_overrides[current_user_id] = override_user(
        "00000000-0000-0000-0000-000000000001"
    )
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
    # VIENEU_TTS_URL is what switches TTS on since 21-08 — see main.tts_enabled.
    # The deployed agent runs without it (TTS is unhosted), so this test has to
    # say explicitly that it wants the enabled path.
    monkeypatch.setenv("VIENEU_TTS_URL", "http://localhost:5000")
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


@pytest.mark.unit
def test_sse_chat_speech_mode_without_tts_configured(api_client, monkeypatch):
    """No VIENEU_TTS_URL: say so once and finish, never promise audio.

    speech_pending is a promise that speech_ready or speech_failed follows. With
    no TTS service, nothing can follow — the UI would sit on a spinner forever.
    Worse on Lambda, where a streaming invocation is billed for its FULL duration
    even after the client disconnects: the old code would have spent 130 seconds
    of memory polling for a result nothing was going to write.

    The `done` assertion is the other half. The stream must still terminate
    normally, because a caller that asked for speech and cannot have it should
    still get its answer.
    """
    client, _, mock_graph = api_client
    mock_graph.astream = _make_fake_astream_stage_only()
    _set_graph(mock_graph)

    import langgraph_agents.api.main as api_module
    monkeypatch.delenv("VIENEU_TTS_URL", raising=False)
    # If the disabled branch leaks, this blows up instead of silently passing.
    monkeypatch.setattr(api_module, "synthesize_speech_async", None)

    resp = client.post("/chat", json={"query": "Hãy đọc", "output_mode": "speech"})
    events = _parse_sse_stream(resp.content)
    kinds = [e["event"] for e in events]

    assert "speech_disabled" in kinds
    assert "speech_pending" not in kinds
    assert kinds[-1] == "done"
    assert [e for e in events if e["event"] == "done"][0]["data"]["speech_task_id"] is None


@pytest.mark.unit
def test_tts_endpoint_503_when_not_configured(api_client, monkeypatch):
    """POST /tts must refuse rather than hand back an id nothing will fulfil."""
    client, _, _ = api_client
    monkeypatch.delenv("VIENEU_TTS_URL", raising=False)

    resp = client.post("/tts", json={"text": "xin chào", "persona_id": "eca_default"})
    assert resp.status_code == 503


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


# ── Kimodo job id capture (R26) ─────────────────────────────────────
#
# kimodo runs as its own graph step before synthesizer and is deliberately
# NOT in _STAGE_NODES, so it must never produce a "stage" SSE event — but its
# job_id (for queued/cache_hit) must still reach write_session_turn. These
# tests exercise main.py's dedicated capture branch that makes that true
# without touching the wire contract.


def _kimodo_tool_message(payload: dict) -> ToolMessage:
    return ToolMessage(
        content=json.dumps(payload),
        tool_call_id="kimodo_motion",
        name="generate_motion",
    )


def _make_fake_astream_with_kimodo(kimodo_payload: dict):
    async def fake_stream(state, config, stream_mode=None):
        yield ("updates", {"memory": {}})
        yield ("updates", {"planner": {}})
        yield ("updates", {"kimodo": {"messages": [_kimodo_tool_message(kimodo_payload)]}})
        yield ("updates", {"synthesizer": {
            "final_answer": "Here's a stretch for you.",
            "intent": "exercise_recommendation",
            "total_tokens": 10,
        }})

    return fake_stream


@pytest.mark.unit
@pytest.mark.parametrize("state", ["queued", "cache_hit"])
def test_kimodo_job_id_reaches_write_session_turn(api_client, monkeypatch, state):
    """queued/cache_hit job ids must be captured and passed through to persistence."""
    client, _, mock_graph = api_client
    mock_graph.astream = _make_fake_astream_with_kimodo({"state": state, "job_id": "job-abc-123"})
    _set_graph(mock_graph)

    import langgraph_agents.api.main as api_module
    captured = {}

    async def fake_write(*args, **kwargs):
        captured.update(kwargs)

    monkeypatch.setattr(api_module, "write_session_turn", fake_write)

    resp = client.post("/chat", json={"query": "show me a stretch"})
    assert resp.status_code == 200
    events = _parse_sse_stream(resp.content)
    assert events[-1]["event"] == "done"
    assert captured.get("motion_job_id") == "job-abc-123"


@pytest.mark.unit
@pytest.mark.parametrize("state", ["busy", "unavailable"])
def test_kimodo_no_job_id_for_busy_or_unavailable(api_client, monkeypatch, state):
    """busy/unavailable carry no job_id — motion_job_id must stay None, not crash."""
    client, _, mock_graph = api_client
    payload = {"state": state}
    if state == "busy":
        payload["retry_after_seconds"] = 30
    mock_graph.astream = _make_fake_astream_with_kimodo(payload)
    _set_graph(mock_graph)

    import langgraph_agents.api.main as api_module
    captured = {}

    async def fake_write(*args, **kwargs):
        captured.update(kwargs)

    monkeypatch.setattr(api_module, "write_session_turn", fake_write)

    resp = client.post("/chat", json={"query": "show me a stretch"})
    assert resp.status_code == 200
    events = _parse_sse_stream(resp.content)
    assert events[-1]["event"] == "done"
    assert captured.get("motion_job_id") is None


@pytest.mark.unit
def test_kimodo_never_emits_a_stage_event(api_client, monkeypatch):
    """kimodo is deliberately absent from _STAGE_NODES (R26) — persisting the
    job id must not put a new event on the wire that no client listens for."""
    client, _, mock_graph = api_client
    mock_graph.astream = _make_fake_astream_with_kimodo({"state": "queued", "job_id": "job-xyz"})
    _set_graph(mock_graph)

    resp = client.post("/chat", json={"query": "show me a stretch"})
    events = _parse_sse_stream(resp.content)
    stage_nodes = {e["data"]["node"] for e in events if e["event"] == "stage"}
    assert "kimodo" not in stage_nodes
    # sanity: the nodes that ARE stage nodes still show up as before
    assert "memory" in stage_nodes
    assert "planner" in stage_nodes
    assert "synthesizer" in stage_nodes


@pytest.mark.unit
def test_kimodo_malformed_content_does_not_break_the_stream(api_client, monkeypatch):
    """A ToolMessage with non-JSON content must be swallowed, not raised —
    a missing motion id is a far smaller failure than a broken chat turn."""
    client, _, mock_graph = api_client

    async def fake_stream(state, config, stream_mode=None):
        yield ("updates", {"memory": {}})
        yield ("updates", {"kimodo": {"messages": [
            ToolMessage(content="not json at all", tool_call_id="x", name="generate_motion"),
        ]}})
        yield ("updates", {"synthesizer": {
            "final_answer": "Still works.",
            "intent": "conversation",
            "total_tokens": 3,
        }})

    mock_graph.astream = fake_stream
    _set_graph(mock_graph)

    resp = client.post("/chat", json={"query": "Xin chào"})
    assert resp.status_code == 200
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
        assert req.output_mode == "text"
        # No user_id: identity comes from the token, never the body.
        assert not hasattr(req, "user_id")

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
