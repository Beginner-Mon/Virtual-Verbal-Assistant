from decimal import Decimal

import pytest
import time as _time

from vva_motion.jobs import (
    canonical_request, compute_job_id, enqueue,
    HEARTBEAT_KEY, MAX_QUEUE_DEPTH, queue_depth, read_status, worker_alive,
    write_heartbeat, claim_next_job, complete_job, fail_job, recover_abandoned_jobs,
)

SECRET = "test-secret"

# Hand-written rows must carry expires_at, because read_status now enforces it:
# a row past its time reads as absent no matter what its status says. enqueue()
# always writes one, so a fixture without it is not a row this system produces.
FRESH = int(_time.time()) + 3600


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
def test_enqueue_coerces_float_fields_to_decimal(table):
    """RULING R16 — boto3's DynamoDB serializer rejects Python `float` with
    'Float types are not supported. Use Decimal types instead.'

    kimodo_node passes DEFAULT_DURATION = 3.0 (a plain float) straight into
    enqueue(). Task 6's test worked around this AT THE CALL SITE with
    Decimal("3.0") — fine for that test, but the next caller (kimodo_node, in
    production) would hit the raw boto3 exception on the first real request.
    The fix belongs in enqueue() itself, not in every caller.
    """
    jid = compute_job_id(SECRET, "nâng hai tay", 3.0, 100, "m")
    enqueue(table, jid, prompt="nâng hai tay", duration=3.0, steps=100)
    row = table.get_item(Key={"job_id": jid})["Item"]
    assert row["duration"] == Decimal("3.0")
    assert isinstance(row["duration"], Decimal)


@pytest.mark.unit
def test_enqueue_replaces_an_expired_row(table):
    """RULING R27 — the write half of the two-clocks problem.

    read_status() refuses to return an expired row, so kimodo_node walks past
    its cache_hit branch and calls enqueue(). With a bare
    `attribute_not_exists(job_id)` the row is still physically there, the write
    is rejected, and the node answers `queued` carrying a job id whose row
    still says `done`. claim_next_job() queries status=queued, so it never sees
    it: that prompt is permanently un-renderable until DynamoDB's TTL sweeper
    gets to it, which AWS only promises "within a few days". Nothing errors.

    The rule that closes it: enqueue() must be able to replace exactly the rows
    read_status() refuses to return.
    """
    jid = compute_job_id(SECRET, "nâng hai tay", 3.0, 100, "m")
    table.put_item(Item={
        "job_id": jid, "status": "done", "created_at": 0,
        "expires_at": int(_time.time()) - 1, "s3_key": "motions/gone.bvh",
    })
    assert enqueue(table, jid, prompt="nâng hai tay") == "created"

    row = table.get_item(Key={"job_id": jid})["Item"]
    assert row["status"] == "queued"
    assert "s3_key" not in row              # PutItem thay cả item, không merge
    assert int(row["expires_at"]) > int(_time.time())
    assert table.scan()["Count"] == 1       # thay thế, không phải thêm row mới


@pytest.mark.unit
def test_enqueue_replaces_a_row_that_has_no_expires_at(table):
    """read_status() counts a missing expires_at as expired: everything
    enqueue() writes has one, so its absence means the row did not come from
    this module. The two rules must agree, or such a row is unreadable and
    unwritable at the same time — the exact state that strands a prompt."""
    table.put_item(Item={"job_id": "stray", "status": "done", "created_at": 0})
    assert enqueue(table, "stray", prompt="p") == "created"
    assert table.get_item(Key={"job_id": "stray"})["Item"]["status"] == "queued"


@pytest.mark.unit
def test_enqueue_still_refuses_a_live_row(table):
    """The dedupe the whole design rests on. Widening the condition for expired
    rows must not widen it for rows that are still good — a `done` row inside
    its TTL is a cache hit worth real GPU seconds."""
    jid = compute_job_id(SECRET, "nâng hai tay", 3.0, 100, "m")
    table.put_item(Item={
        "job_id": jid, "status": "done", "created_at": 0,
        "expires_at": FRESH, "s3_key": "motions/live.bvh",
    })
    assert enqueue(table, jid, prompt="nâng hai tay") == "exists"
    row = table.get_item(Key={"job_id": jid})["Item"]
    assert row["status"] == "done" and row["s3_key"] == "motions/live.bvh"


@pytest.mark.unit
def test_enqueue_never_overwrites_the_heartbeat(table):
    """read_status() has a THIRD way of returning None — a row with no
    `status` — and the condition deliberately does NOT cover it.

    The only row that hits it is the heartbeat, which is not expired and whose
    loss makes every kimodo_node call answer `unavailable` until the worker's
    next beat. Symmetry with read_status() is worth less than never writing
    over that row. A real job_id is 32 hex chars and can never collide with
    `worker#heartbeat`, so nothing legitimate is turned away by stopping here.
    """
    write_heartbeat(table)
    assert enqueue(table, HEARTBEAT_KEY, prompt="p") == "exists"
    assert worker_alive(table) is True


@pytest.mark.unit
def test_queue_depth_ignores_expired_rows(table):
    """A worker that was off for a day leaves expired `queued` rows behind.
    Counting them turns MAX_QUEUE_DEPTH into a lock: the node answers `busy` to
    everyone while nothing is actually waiting."""
    for i in range(3):
        enqueue(table, f"q{i}", prompt="p")
    table.put_item(Item={"job_id": "stale", "status": "queued", "created_at": 0,
                         "expires_at": int(_time.time()) - 1})
    assert queue_depth(table) == 3


@pytest.mark.unit
def test_claim_skips_expired_queued_rows_and_takes_the_live_one(table):
    """Rendering a day-old request costs GPU seconds for an answer nobody is
    still waiting for. The stale row sorts FIRST on the GSI (created_at=0), so
    a claim loop that does not check expiry picks it every time."""
    table.put_item(Item={"job_id": "stale", "status": "queued", "created_at": 0,
                         "expires_at": int(_time.time()) - 1})
    enqueue(table, "live", prompt="p")
    job = claim_next_job(table)
    assert job is not None and job["job_id"] == "live"


@pytest.mark.unit
def test_claim_returns_none_when_only_expired_rows_wait(table):
    table.put_item(Item={"job_id": "stale", "status": "queued", "created_at": 0,
                         "expires_at": int(_time.time()) - 1})
    assert claim_next_job(table) is None


@pytest.mark.unit
def test_lease_expired_reads_as_failed(table):
    """Cùng một trạng thái row, chạy hai lần: heartbeat tươi (worker treo) và
    heartbeat cũ (worker chết). CẢ HAI phải ra failed — luật lease đứng độc lập
    với heartbeat."""
    table.put_item(Item={
        "job_id": "j1", "status": "processing", "created_at": 0, "expires_at": FRESH,
        "lease_until": int(_time.time()) - 1, "retry_count": 0,
    })
    write_heartbeat(table)                       # heartbeat TƯƠI
    assert read_status(table, "j1")["status"] == "failed"

    table.delete_item(Key={"job_id": "worker#heartbeat"})   # heartbeat CŨ
    assert read_status(table, "j1")["status"] == "failed"


@pytest.mark.unit
def test_lease_still_valid_reads_as_processing(table):
    table.put_item(Item={
        "job_id": "j2", "status": "processing", "created_at": 0, "expires_at": FRESH,
        "lease_until": int(_time.time()) + 60, "retry_count": 0,
    })
    assert read_status(table, "j2")["status"] == "processing"


@pytest.mark.unit
def test_expired_row_reads_as_absent(table):
    """THE test for the two clocks disagreeing.

    asset_stack.py's S3 lifecycle rule deletes motions/* reliably at 24h.
    DynamoDB TTL is a background sweeper AWS documents as running "within a few
    days", typically 48h. In that window the row still says `done` while the
    .bvh is already gone — and nothing errors. kimodo_node would report a
    cache_hit and enqueue nothing, so that exact prompt becomes permanently
    un-renderable, while motion_status hands out a signed URL to a 404.

    Same rule shared/stm.py already enforces on its own reads
    (test_stm.py::test_dynamo_hides_an_expired_item_the_sweeper_has_not_collected).
    """
    table.put_item(Item={
        "job_id": "old", "status": "done", "created_at": 0,
        "expires_at": int(_time.time()) - 1, "s3_key": "motions/old.bvh",
    })
    assert read_status(table, "old") is None


@pytest.mark.unit
def test_row_without_expires_at_reads_as_absent(table):
    """Every row enqueue() writes has one. Its absence means the row did not
    come from this module, and guessing that it is still valid is the failure
    mode this check exists to prevent."""
    table.put_item(Item={"job_id": "stray", "status": "done", "created_at": 0})
    assert read_status(table, "stray") is None


@pytest.mark.unit
def test_heartbeat_row_is_not_a_job(table):
    """The heartbeat shares this table under a reserved key and carries only
    last_seen. read_status used to do item["status"] on it and raise KeyError —
    reachable from an authed public route as GET /motion/worker%23heartbeat,
    i.e. a 500 any signed-in user could trigger."""
    write_heartbeat(table)
    assert read_status(table, "worker#heartbeat") is None
    assert worker_alive(table) is True        # ...and worker_alive still reads it


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
