import pytest
import time as _time

from vva_motion.jobs import (
    canonical_request, compute_job_id, enqueue,
    MAX_QUEUE_DEPTH, queue_depth, read_status, worker_alive, write_heartbeat,
    claim_next_job, complete_job, fail_job, recover_abandoned_jobs,
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


@pytest.mark.unit
def test_claim_moves_queued_to_processing_with_lease(table):
    enqueue(table, "c1", prompt="p")
    job = claim_next_job(table)
    assert job["job_id"] == "c1"
    row = table.get_item(Key={"job_id": "c1"})["Item"]
    assert row["status"] == "processing"
    assert int(row["lease_until"]) > int(_time.time())


@pytest.mark.unit
def test_claim_never_touches_processing_or_done(table):
    """Vòng poll chỉ biết 'queued'. Job sập KHÔNG được nhặt trực tiếp —
    nó phải đi qua recover_abandoned_jobs() để về queued trước."""
    table.put_item(Item={"job_id": "p1", "status": "processing",
                         "created_at": 0, "lease_until": 0})
    table.put_item(Item={"job_id": "d1", "status": "done", "created_at": 0})
    assert claim_next_job(table) is None


@pytest.mark.unit
def test_recover_resets_expired_lease_and_gives_up_after_3(table):
    table.put_item(Item={"job_id": "r1", "status": "processing", "created_at": 0,
                         "lease_until": int(_time.time()) - 1, "retry_count": 0})
    assert recover_abandoned_jobs(table) == 1
    row = table.get_item(Key={"job_id": "r1"})["Item"]
    assert row["status"] == "queued" and int(row["retry_count"]) == 1

    for _ in range(3):
        table.update_item(Key={"job_id": "r1"},
                          UpdateExpression="SET #s=:p, lease_until=:l",
                          ExpressionAttributeNames={"#s": "status"},
                          ExpressionAttributeValues={":p": "processing",
                                                     ":l": int(_time.time()) - 1})
        recover_abandoned_jobs(table)
    assert table.get_item(Key={"job_id": "r1"})["Item"]["status"] == "failed"


@pytest.mark.unit
def test_recover_leaves_valid_lease_alone(table):
    table.put_item(Item={"job_id": "r2", "status": "processing", "created_at": 0,
                         "lease_until": int(_time.time()) + 60, "retry_count": 0})
    assert recover_abandoned_jobs(table) == 0
    assert table.get_item(Key={"job_id": "r2"})["Item"]["status"] == "processing"
