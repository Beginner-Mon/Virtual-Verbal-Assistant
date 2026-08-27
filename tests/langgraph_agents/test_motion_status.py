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
from vva_motion.jobs import enqueue


@pytest.mark.unit
def test_done_returns_signed_url(table, monkeypatch):
    monkeypatch.setattr("langgraph_agents.api.motion_status._table", lambda: table)
    monkeypatch.setattr("langgraph_agents.api.motion_status.sign_url",
                        lambda key: f"https://cdn/{key}?Signature=x")
    table.put_item(Item={"job_id": "d1", "status": "done",
                         "created_at": 0, "s3_key": "motions/d1.bvh"})
    out = motion_status("d1")
    assert out["status"] == "done" and "Signature=" in out["url"]


@pytest.mark.unit
def test_expired_lease_reports_failed_not_processing(table, monkeypatch):
    monkeypatch.setattr("langgraph_agents.api.motion_status._table", lambda: table)
    table.put_item(Item={"job_id": "s1", "status": "processing", "created_at": 0,
                         "lease_until": int(_time.time()) - 1})
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
