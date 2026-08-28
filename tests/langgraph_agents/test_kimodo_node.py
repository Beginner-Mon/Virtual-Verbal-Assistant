"""kimodo_node giờ ghi một row DynamoDB, không gọi mạng tới Kimodo nữa.

Import ở module scope là load-bearing: pytest.ini đặt filterwarnings = error và chuỗi
import của langgraph phát LangChainPendingDeprecationWarning, không nằm trong danh sách
ignore. Import trong hàm test sẽ làm mọi assertion đỏ vì lý do không liên quan.
"""
from __future__ import annotations

import json
import time as _time

import pytest

from langgraph_agents.nodes.kimodo import DEFAULT_DURATION, DEFAULT_STEPS, MODEL, kimodo_node
from vva_motion.jobs import MAX_QUEUE_DEPTH, compute_job_id, enqueue, write_heartbeat

HASH_SECRET = "test-secret"

CONFIG = {"configurable": {"request_id": "r1", "query": "nâng hai tay qua đầu",
                           "session_id": "s1"}}


def _content(result):
    return json.loads(result["messages"][0].content)


@pytest.fixture(autouse=True)
def _hash_secret(monkeypatch):
    # Read lazily inside kimodo_node (not module scope), so this only needs to
    # be present by the time the node runs, not by import time.
    monkeypatch.setenv("MOTION_HASH_SECRET", HASH_SECRET)


@pytest.mark.unit
async def test_unavailable_when_no_heartbeat(table, monkeypatch):
    monkeypatch.setattr("langgraph_agents.nodes.kimodo._table", lambda: table)
    out = _content(await kimodo_node({"resolved_query": "nâng hai tay"}, CONFIG))
    assert out["state"] == "unavailable"
    assert table.scan()["Count"] == 0          # hàng đợi không bao giờ được nạp rác


@pytest.mark.unit
async def test_queued_with_position_and_eta(table, monkeypatch):
    monkeypatch.setattr("langgraph_agents.nodes.kimodo._table", lambda: table)
    write_heartbeat(table)
    out = _content(await kimodo_node({"resolved_query": "nâng hai tay"}, CONFIG))
    assert out["state"] == "queued"
    assert out["queue_position"] == 1 and out["eta_seconds"] == 5


@pytest.mark.unit
async def test_second_identical_request_while_queued_reuses_same_job(table, monkeypatch):
    """Second call sees the first job still `status="queued"` — this exercises
    the queued-dedup branch, NOT cache_hit. (`existing["status"] in ("queued",
    "processing")` in the node.) Renamed from a name that claimed cache_hit;
    see test_cache_hit_when_job_already_done below for the real thing."""
    monkeypatch.setattr("langgraph_agents.nodes.kimodo._table", lambda: table)
    write_heartbeat(table)
    first = _content(await kimodo_node({"resolved_query": "nâng hai tay"}, CONFIG))
    second = _content(await kimodo_node({"resolved_query": "NÂNG HAI TAY"}, CONFIG))
    assert first["state"] == "queued" and second["state"] == "queued"
    assert second["job_id"] == first["job_id"]
    assert table.scan()["Count"] == 2          # 1 job + 1 heartbeat, KHÔNG phải 2 job


@pytest.mark.unit
async def test_cache_hit_when_job_already_done(table, monkeypatch):
    """The whole point of cache_hit: a row already marked `done` (GPU already
    rendered this exact request, same prompt/duration/steps/model) skips a
    redundant render entirely — the node must not enqueue anything new."""
    monkeypatch.setattr("langgraph_agents.nodes.kimodo._table", lambda: table)
    write_heartbeat(table)
    job_id = compute_job_id(HASH_SECRET, "nâng hai tay", DEFAULT_DURATION, DEFAULT_STEPS, MODEL)
    table.put_item(Item={
        "job_id": job_id, "status": "done", "s3_key": "motions/abc.npz", "created_at": 0,
        # read_status enforces expires_at now, so a hand-written row needs one.
        # enqueue() always writes it; a fixture without it is not a real row.
        "expires_at": int(_time.time()) + 3600,
    })
    out = _content(await kimodo_node({"resolved_query": "nâng hai tay"}, CONFIG))
    assert out["state"] == "cache_hit"
    assert out["job_id"] == job_id
    assert table.scan()["Count"] == 2          # heartbeat + the pre-seeded row, KHÔNG job mới


@pytest.mark.unit
async def test_expired_done_row_re_renders_instead_of_claiming_a_cache_hit(
    table, monkeypatch,
):
    """The other half of the two-clocks problem.

    S3 deletes motions/* reliably at 24h; DynamoDB TTL is best-effort within
    48h. If an expired `done` row still counted as a cache hit, this node would
    enqueue nothing and answer with a job id whose file no longer exists — and
    it would do that for every future request with the same prompt, because the
    row that causes it is the row that never gets replaced. That prompt becomes
    permanently un-renderable, with no error anywhere.

    Note the row is REPLACED, not added to: same HMAC job_id, so the count stays
    at heartbeat + one job.
    """
    monkeypatch.setattr("langgraph_agents.nodes.kimodo._table", lambda: table)
    write_heartbeat(table)
    job_id = compute_job_id(HASH_SECRET, "nâng hai tay", DEFAULT_DURATION, DEFAULT_STEPS, MODEL)
    table.put_item(Item={
        "job_id": job_id, "status": "done", "s3_key": "motions/gone.npz",
        "created_at": 0, "expires_at": int(_time.time()) - 1,
    })
    out = _content(await kimodo_node({"resolved_query": "nâng hai tay"}, CONFIG))
    assert out["state"] != "cache_hit"
    assert out["job_id"] == job_id
    assert table.scan()["Count"] == 2


@pytest.mark.unit
async def test_busy_when_queue_full(table, monkeypatch):
    monkeypatch.setattr("langgraph_agents.nodes.kimodo._table", lambda: table)
    write_heartbeat(table)
    for i in range(MAX_QUEUE_DEPTH):
        enqueue(table, f"filler{i}", prompt=f"p{i}")
    out = _content(await kimodo_node({"resolved_query": "nâng hai tay"}, CONFIG))
    assert out["state"] == "busy"
