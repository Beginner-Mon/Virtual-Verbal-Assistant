"""Contract tests for the short-term-memory store (shared/stm.py).

Three things are pinned here, each because getting it wrong fails silently:

1. **A dead backend reads as a miss, never as an exception.** STM is a cache
   over PostgreSQL; the fallback in nodes/memory.py only runs if this layer
   returns None instead of raising.
2. **An expired DynamoDB item is invisible.** DynamoDB's TTL sweeper runs days
   late, so without an application-side check a conversation that should have
   gone cold after two hours keeps its memory for up to 48 more.
3. **No hardcoded localhost survives.** Four call sites each carried their own
   copy of `redis://localhost:6379` before 21-08. On Lambda that is not a
   configuration mistake, it is a guaranteed miss on every read.
"""

from __future__ import annotations

import json
import time
from unittest.mock import MagicMock, patch

import pytest

from langgraph_agents.shared import stm as stm_module


@pytest.fixture(autouse=True)
def _reset_store():
    """The store is a process-wide singleton; tests change its configuration."""
    stm_module.reset_stm()
    yield
    stm_module.reset_stm()


# ── Backend selection ─────────────────────────────────────────────────────────


@pytest.mark.unit
def test_default_backend_is_redis(monkeypatch):
    monkeypatch.delenv("STM_BACKEND", raising=False)
    assert isinstance(stm_module.get_stm(), stm_module.RedisStore)


@pytest.mark.unit
@pytest.mark.parametrize("value,expected", [
    ("redis", stm_module.RedisStore),
    ("dynamodb", stm_module.DynamoStore),
    ("none", stm_module.NullStore),
    ("  DynamoDB  ", stm_module.DynamoStore),
])
def test_backend_selection(monkeypatch, value, expected):
    monkeypatch.setenv("STM_BACKEND", value)
    monkeypatch.setenv("STM_TABLE", "test-table")
    assert isinstance(stm_module.get_stm(), expected)


@pytest.mark.unit
def test_unknown_backend_raises(monkeypatch):
    """A typo must fail at startup, not fall back to a default.

    Falling back is worse than failing here: on Lambda the default is Redis on
    localhost, which does not exist, so the service would run with every read
    missing while reporting itself healthy.
    """
    monkeypatch.setenv("STM_BACKEND", "dynamo")      # plausible typo
    with pytest.raises(ValueError, match="dynamo"):
        stm_module.get_stm()


@pytest.mark.unit
def test_redis_url_comes_from_environment(monkeypatch):
    monkeypatch.setenv("STM_BACKEND", "redis")
    monkeypatch.setenv("REDIS_URL", "redis://cache.example.com:6379/2")
    store = stm_module.get_stm()
    assert store._url == "redis://cache.example.com:6379/2"


# ── Degradation ───────────────────────────────────────────────────────────────


class _ExplodingStore(stm_module._Store):
    """A backend where every operation fails, to exercise the base class."""

    def __init__(self):
        super().__init__()
        self.calls = 0

    async def get(self, session_id):
        try:
            self.calls += 1
            raise ConnectionError("backend down")
        except Exception as exc:                                    # noqa: BLE001
            self._degrade("get", exc)
            return None


@pytest.mark.unit
@pytest.mark.asyncio
async def test_get_returns_none_when_backend_raises():
    store = _ExplodingStore()
    assert await store.get("s1") is None


@pytest.mark.unit
@pytest.mark.asyncio
async def test_outage_is_logged_once_not_per_request():
    """CloudWatch ingestion is billed per GB.

    A cache that is down for an hour would otherwise write one warning per
    request, turning a degraded cache into a bill — while burying the one line
    that mattered.
    """
    store = _ExplodingStore()
    with patch.object(stm_module.logger, "warning") as mock_warn:
        for _ in range(50):
            await store.get("s1")
        assert store.calls == 50
        assert mock_warn.call_count == 1


@pytest.mark.unit
@pytest.mark.asyncio
async def test_null_store_is_a_working_cache_that_holds_nothing():
    """STM_BACKEND=none must be usable, not merely inert.

    This is what lets the agent deploy before the cache decision is made, so a
    write followed by a read has to return a miss rather than fail.
    """
    store = stm_module.NullStore()
    await store.set("s1", [{"q": "a", "a": "b", "ts": ""}])
    assert await store.get("s1") is None
    await store.delete("s1")


# ── The DynamoDB TTL trap ─────────────────────────────────────────────────────


def _dynamo_store_returning(item):
    store = stm_module.DynamoStore("test-table")
    table = MagicMock()
    table.get_item.return_value = {"Item": item} if item is not None else {}
    store._table = table
    return store, table


@pytest.mark.unit
@pytest.mark.asyncio
async def test_dynamo_reads_a_live_item():
    payload = [{"q": "xin chào", "a": "Chào bạn!", "ts": ""}]
    store, _ = _dynamo_store_returning({
        "session_id": "s1",
        "value": json.dumps(payload),
        "expires_at": int(time.time()) + 3600,
    })
    assert await store.get("s1") == payload


@pytest.mark.unit
@pytest.mark.asyncio
async def test_dynamo_hides_an_expired_item_the_sweeper_has_not_collected():
    """THE test in this file.

    AWS deletes expired items "within a few days", typically within 48 hours —
    not at the timestamp. Without the application-side check, this item would be
    served and the agent would answer using a conversation it was supposed to
    have forgotten two days earlier. No error, no log: just a wrong answer.
    """
    store, _ = _dynamo_store_returning({
        "session_id": "s1",
        "value": json.dumps([{"q": "stale", "a": "stale", "ts": ""}]),
        "expires_at": int(time.time()) - 1,          # one second past due
    })
    assert await store.get("s1") is None


@pytest.mark.unit
@pytest.mark.asyncio
async def test_dynamo_missing_item_is_a_miss():
    store, _ = _dynamo_store_returning(None)
    assert await store.get("s1") is None


@pytest.mark.unit
@pytest.mark.asyncio
async def test_dynamo_writes_expires_at_as_an_integer():
    """DynamoDB ignores a TTL attribute that is not a Number — silently.

    A float or a string leaves the item with no expiry at all, and nothing
    surfaces that except a table that grows forever.
    """
    store = stm_module.DynamoStore("test-table")
    table = MagicMock()
    store._table = table

    before = int(time.time())
    await store.set("s1", [{"q": "a", "a": "b", "ts": ""}], ttl=7200)

    item = table.put_item.call_args.kwargs["Item"]
    assert isinstance(item["expires_at"], int)
    assert before + 7200 <= item["expires_at"] <= before + 7201 + 5
    assert json.loads(item["value"]) == [{"q": "a", "a": "b", "ts": ""}]


# ── No hardcoded localhost anywhere ───────────────────────────────────────────


@pytest.mark.unit
def test_no_call_site_hardcodes_a_redis_url():
    """The gap this refactor closed must not reopen quietly.

    `redis://localhost` appeared in four modules before 21-08. Deployed, each
    one is a read that always misses — and the service still answers, just from
    PostgreSQL every time, so nothing fails loudly enough to notice.

    Three modules are exempt, and all three read REDIS_URL from the environment
    before falling back: shared/stm.py, api/main.py (TTS results), and
    services/vieneu_tts/tasks.py (the writer for those results). A default for
    local development is fine; a value that CANNOT be overridden is not.

    Comment lines are skipped. celery_app.py is dead code kept as comments — the
    Celery decision was reversed in v2.4.1 — and flagging it would push someone
    toward adding a blanket file exemption, which is how this kind of check stops
    catching the thing it was written for.
    """
    from pathlib import Path

    package = Path(stm_module.__file__).resolve().parents[1]
    exempt = {"stm.py", "main.py", "tasks.py"}

    offenders = []
    for path in package.rglob("*.py"):
        if path.name in exempt or "local_tests" in path.parts:
            continue
        for lineno, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
            if line.lstrip().startswith("#"):
                continue
            if "redis://" in line:
                offenders.append(f"{path.relative_to(package).as_posix()}:{lineno}")

    assert not offenders, (
        f"These lines hardcode a Redis URL: {offenders}. Use "
        f"shared.stm.get_stm(), which reads REDIS_URL and can be pointed at "
        f"DynamoDB when deployed."
    )
