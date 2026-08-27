import pytest

from vva_motion.jobs import (
    canonical_request, compute_job_id, enqueue,
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
