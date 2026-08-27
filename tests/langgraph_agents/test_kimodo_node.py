"""kimodo_node giờ ghi một row DynamoDB, không gọi mạng tới Kimodo nữa.

Import ở module scope là load-bearing: pytest.ini đặt filterwarnings = error và chuỗi
import của langgraph phát LangChainPendingDeprecationWarning, không nằm trong danh sách
ignore. Import trong hàm test sẽ làm mọi assertion đỏ vì lý do không liên quan.
"""
from __future__ import annotations

import json

import pytest

from langgraph_agents.nodes.kimodo import kimodo_node
from vva_motion.jobs import MAX_QUEUE_DEPTH, enqueue, write_heartbeat

CONFIG = {"configurable": {"request_id": "r1", "query": "nâng hai tay qua đầu",
                           "session_id": "s1"}}


def _content(result):
    return json.loads(result["messages"][0].content)


@pytest.fixture(autouse=True)
def _hash_secret(monkeypatch):
    # Read lazily inside kimodo_node (not module scope), so this only needs to
    # be present by the time the node runs, not by import time.
    monkeypatch.setenv("MOTION_HASH_SECRET", "test-secret")


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
async def test_second_identical_request_is_cache_hit_not_new_job(table, monkeypatch):
    monkeypatch.setattr("langgraph_agents.nodes.kimodo._table", lambda: table)
    write_heartbeat(table)
    first = _content(await kimodo_node({"resolved_query": "nâng hai tay"}, CONFIG))
    second = _content(await kimodo_node({"resolved_query": "NÂNG HAI TAY"}, CONFIG))
    assert second["job_id"] == first["job_id"]
    assert table.scan()["Count"] == 2          # 1 job + 1 heartbeat, KHÔNG phải 2 job


@pytest.mark.unit
async def test_busy_when_queue_full(table, monkeypatch):
    monkeypatch.setattr("langgraph_agents.nodes.kimodo._table", lambda: table)
    write_heartbeat(table)
    for i in range(MAX_QUEUE_DEPTH):
        enqueue(table, f"filler{i}", prompt=f"p{i}")
    out = _content(await kimodo_node({"resolved_query": "nâng hai tay"}, CONFIG))
    assert out["state"] == "busy"
