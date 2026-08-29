"""GET /motion/{job_id} handler — reads a job row, decides what the caller is
allowed to know, and signs a CloudFront URL only once the file actually exists.

`_table` and `sign_url` are monkeypatched at the module level rather than
constructed for real: `_table()` would need AWS credentials + MOTION_TABLE, and
`sign_url()` would need a real RSA private key. Both are read from the
environment lazily, at the point of use, so patching the module-level names is
enough to isolate this test from AWS entirely — see the module docstring in
motion_status.py for why the env reads are lazy.
"""
import time as _time

import pytest

from langgraph_agents.api.motion_status import motion_status
from vva_motion.jobs import enqueue, write_heartbeat

# Hand-written rows must carry expires_at, because read_status now enforces it:
# a row past its time reads as absent no matter what its status says. enqueue()
# always writes one, so a fixture without it is not a row this system produces.
FRESH = int(_time.time()) + 3600


@pytest.mark.unit
def test_done_returns_signed_url(table, monkeypatch):
    monkeypatch.setattr("langgraph_agents.api.motion_status._table", lambda: table)
    monkeypatch.setattr("langgraph_agents.api.motion_status.sign_url",
                        lambda key: f"https://cdn/{key}?Signature=x")
    table.put_item(Item={"job_id": "d1", "status": "done", "created_at": 0,
                         "expires_at": FRESH, "s3_key": "motions/d1.bvh"})
    out = motion_status("d1")
    assert out["status"] == "done" and "Signature=" in out["url"]


@pytest.mark.unit
def test_done_without_an_s3_key_does_not_500(table, monkeypatch):
    """complete_job writes status and s3_key in one UpdateExpression, so this
    should be impossible — but this is a public authed route, and the
    alternative to a guard is an unhandled KeyError reaching the client as a
    500 for a condition the caller cannot act on."""
    monkeypatch.setattr("langgraph_agents.api.motion_status._table", lambda: table)

    def _explode(key):
        raise AssertionError("sign_url must not be called without a key")

    monkeypatch.setattr("langgraph_agents.api.motion_status.sign_url", _explode)
    table.put_item(Item={"job_id": "d2", "status": "done", "created_at": 0,
                         "expires_at": FRESH})
    out = motion_status("d2")
    assert out["status"] == "failed" and "url" not in out


@pytest.mark.unit
def test_the_heartbeat_row_is_not_a_job(table, monkeypatch):
    """The heartbeat shares this table under the reserved key
    `worker#heartbeat` and carries only last_seen. It is reachable from the
    authed public route as GET /motion/worker%23heartbeat, and reading it used
    to raise KeyError('status') — a 500 any signed-in user could trigger."""
    monkeypatch.setattr("langgraph_agents.api.motion_status._table", lambda: table)
    write_heartbeat(table)
    assert motion_status("worker#heartbeat")["status"] == "not_found"


@pytest.mark.unit
def test_expired_row_reads_as_not_found_even_when_done(table, monkeypatch):
    """S3 deletes motions/* reliably at 24h; DynamoDB TTL is best-effort within
    48h. In that window the row says done and the file is gone, so a signed URL
    would point at a 404 — and kimodo_node would call it a cache hit and
    enqueue nothing, making the prompt permanently un-renderable."""
    monkeypatch.setattr("langgraph_agents.api.motion_status._table", lambda: table)

    def _explode(key):
        raise AssertionError("must not sign a URL for a file S3 has deleted")

    monkeypatch.setattr("langgraph_agents.api.motion_status.sign_url", _explode)
    table.put_item(Item={"job_id": "old", "status": "done", "created_at": 0,
                         "expires_at": int(_time.time()) - 1,
                         "s3_key": "motions/old.bvh"})
    assert motion_status("old")["status"] == "not_found"


@pytest.mark.unit
def test_expired_lease_reports_failed_not_processing(table, monkeypatch):
    monkeypatch.setattr("langgraph_agents.api.motion_status._table", lambda: table)
    table.put_item(Item={"job_id": "s1", "status": "processing", "created_at": 0,
                         "expires_at": FRESH, "lease_until": int(_time.time()) - 1})
    assert motion_status("s1")["status"] == "failed"


@pytest.mark.unit
def test_unknown_job_returns_not_found(table, monkeypatch):
    monkeypatch.setattr("langgraph_agents.api.motion_status._table", lambda: table)
    assert motion_status("nope")["status"] == "not_found"


@pytest.mark.unit
def test_queued_never_leaks_a_url(table, monkeypatch):
    """URL chỉ được phát khi done — tải sớm sẽ 404 và CloudFront cache cả 404."""
    monkeypatch.setattr("langgraph_agents.api.motion_status._table", lambda: table)
    enqueue(table, "q9", prompt="p")
    assert "url" not in motion_status("q9")
