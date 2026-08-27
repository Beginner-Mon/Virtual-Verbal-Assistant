import pytest
import time as _time

from vva_motion.jobs import (
    canonical_request, compute_job_id, enqueue,
    MAX_QUEUE_DEPTH, queue_depth, read_status, worker_alive, write_heartbeat,
)

SECRET = "test-secret"


@pytest.mark.unit
def test_canonical_ignores_case_and_padding():
    assert canonical_request("  Nâng Hai Tay  ", 3.0, 100, "m") == \
           canonical_request("nâng hai tay", 3.0, 100, "m")


@pytest.mark.unit
def test_same_request_same_id():
    a = compute_job_id(SECRET, "nâng hai tay", 3.0, 100, "m")
    b = compute_job_id(SECRET, "NÂNG HAI TAY", 3.0, 100, "m")
    assert a == b and len(a) == 32


@pytest.mark.unit
def test_different_params_different_id():
    a = compute_job_id(SECRET, "nâng hai tay", 3.0, 100, "m")
    assert a != compute_job_id(SECRET, "nâng hai tay", 5.0, 100, "m")


@pytest.mark.unit
def test_secret_changes_id():
    """Không có secret thì key suy ra được từ prompt — đây là bài chứng minh HMAC có tác dụng."""
    a = compute_job_id(SECRET, "nâng hai tay", 3.0, 100, "m")
    assert a != compute_job_id("other-secret", "nâng hai tay", 3.0, 100, "m")


@pytest.mark.unit
def test_enqueue_dedupes(table):
    jid = compute_job_id(SECRET, "nâng hai tay", 3.0, 100, "m")
    assert enqueue(table, jid, prompt="nâng hai tay", session_id="s1") == "created"
    assert enqueue(table, jid, prompt="nâng hai tay", session_id="s2") == "exists"
    assert table.scan()["Count"] == 1


@pytest.mark.unit
def test_lease_expired_reads_as_failed(table):
    """Cùng một trạng thái row, chạy hai lần: heartbeat tươi (worker treo) và
    heartbeat cũ (worker chết). CẢ HAI phải ra failed — luật lease đứng độc lập
    với heartbeat."""
    table.put_item(Item={
        "job_id": "j1", "status": "processing", "created_at": 0,
        "lease_until": int(_time.time()) - 1, "retry_count": 0,
    })
    write_heartbeat(table)                       # heartbeat TƯƠI
    assert read_status(table, "j1")["status"] == "failed"

    table.delete_item(Key={"job_id": "worker#heartbeat"})   # heartbeat CŨ
    assert read_status(table, "j1")["status"] == "failed"


@pytest.mark.unit
def test_lease_still_valid_reads_as_processing(table):
    table.put_item(Item={
        "job_id": "j2", "status": "processing", "created_at": 0,
        "lease_until": int(_time.time()) + 60, "retry_count": 0,
    })
    assert read_status(table, "j2")["status"] == "processing"


@pytest.mark.unit
def test_worker_alive_reflects_heartbeat(table):
    assert worker_alive(table) is False
    write_heartbeat(table)
    assert worker_alive(table) is True


@pytest.mark.unit
def test_queue_depth_counts_only_queued(table):
    for i in range(3):
        enqueue(table, f"q{i}", prompt="p")
    table.put_item(Item={"job_id": "done1", "status": "done", "created_at": 0})
    assert queue_depth(table) == 3
    assert queue_depth(table) < MAX_QUEUE_DEPTH
