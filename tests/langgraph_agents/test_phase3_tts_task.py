"""Tests for async TTS task (v2.4.1: BackgroundTasks pattern)."""

import json
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from langgraph_agents.services.exceptions import ServiceUnavailableError
from langgraph_agents.services.vieneu_tts import tasks


@pytest.mark.unit
@pytest.mark.asyncio
async def test_synthesize_async_success_persists_speech_ready():
    fake_client = MagicMock()
    fake_client.synthesize = AsyncMock(return_value={"audio_url": "http://x/a.wav"})

    fake_redis = AsyncMock()
    fake_redis.setex = AsyncMock()
    fake_redis.aclose = AsyncMock()

    with patch.object(tasks, "get_vieneu_tts_client", return_value=fake_client):
        with patch.object(tasks.aioredis, "from_url", return_value=fake_redis):
            await tasks.synthesize_speech_async("hello", task_id="t1")

    fake_redis.setex.assert_awaited_once()
    args = fake_redis.setex.await_args[0]
    assert args[0] == "task_result:t1"
    assert args[1] == 3600
    payload = json.loads(args[2])
    assert payload["event"] == "speech_ready"
    assert payload["url"] == "http://x/a.wav"


@pytest.mark.unit
@pytest.mark.asyncio
async def test_synthesize_async_service_down_persists_speech_failed():
    fake_client = MagicMock()
    fake_client.synthesize = AsyncMock(
        side_effect=ServiceUnavailableError("vieneu_tts", "circuit open")
    )

    fake_redis = AsyncMock()
    fake_redis.setex = AsyncMock()
    fake_redis.aclose = AsyncMock()

    with patch.object(tasks, "get_vieneu_tts_client", return_value=fake_client):
        with patch.object(tasks.aioredis, "from_url", return_value=fake_redis):
            await tasks.synthesize_speech_async("hello", task_id="t2")

    payload = json.loads(fake_redis.setex.await_args[0][2])
    assert payload["event"] == "speech_failed"
    assert "circuit open" in payload["error"]


@pytest.mark.unit
@pytest.mark.asyncio
async def test_synthesize_async_unexpected_error_caught():
    fake_client = MagicMock()
    fake_client.synthesize = AsyncMock(side_effect=RuntimeError("boom"))

    fake_redis = AsyncMock()
    fake_redis.setex = AsyncMock()
    fake_redis.aclose = AsyncMock()

    with patch.object(tasks, "get_vieneu_tts_client", return_value=fake_client):
        with patch.object(tasks.aioredis, "from_url", return_value=fake_redis):
            await tasks.synthesize_speech_async("hello", task_id="t3")

    payload = json.loads(fake_redis.setex.await_args[0][2])
    assert payload["event"] == "speech_failed"
    assert "boom" in payload["error"]


@pytest.mark.unit
@pytest.mark.asyncio
async def test_synthesize_async_redis_down_does_not_raise():
    """Persist failure should not propagate to caller (BackgroundTasks has no retry)."""
    fake_client = MagicMock()
    fake_client.synthesize = AsyncMock(return_value={"audio_url": "http://x/a.wav"})

    fake_redis = AsyncMock()
    fake_redis.setex = AsyncMock(side_effect=ConnectionError("redis down"))
    fake_redis.aclose = AsyncMock()

    with patch.object(tasks, "get_vieneu_tts_client", return_value=fake_client):
        with patch.object(tasks.aioredis, "from_url", return_value=fake_redis):
            # Should not raise
            await tasks.synthesize_speech_async("hello", task_id="t4")
