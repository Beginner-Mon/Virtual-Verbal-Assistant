from decimal import Decimal

import boto3
import pytest
from moto import mock_aws

import worker
from vva_motion.jobs import enqueue, read_status
from worker import poll_forever, run_once


class FakeEngine:
    """No GPU needed: replaces exactly the surface run_once uses."""
    def __init__(self, raises=False):
        self.raises = raises
        self.calls = 0

    def generate(self, prompt, duration, steps):
        self.calls += 1
        if self.raises:
            raise RuntimeError("cuda oom")
        return {"posed_joints": None}

    def save_outputs(self, output, out_dir, base_name):
        npz = f"{out_dir}/{base_name}.npz"
        bvh = f"{out_dir}/{base_name}.bvh"
        for p in (npz, bvh):
            with open(p, "w", encoding="utf-8") as fh:
                fh.write("x")
        return npz, bvh


@pytest.mark.unit
def test_run_once_uploads_and_marks_done(table, tmp_path):
    with mock_aws():
        s3 = boto3.client("s3", region_name="us-east-1")
        s3.create_bucket(Bucket="assets")
        enqueue(table, "w1", prompt="nâng hai tay", duration=Decimal("3.0"), steps=100)

        assert run_once(table, FakeEngine(), "assets", s3, out_dir=str(tmp_path)) is True

        assert read_status(table, "w1")["status"] == "done"
        keys = {o["Key"] for o in s3.list_objects_v2(Bucket="assets")["Contents"]}
        assert keys == {"motions/w1.npz", "motions/w1.bvh"}


@pytest.mark.unit
def test_run_once_marks_failed_on_exception(table, tmp_path):
    with mock_aws():
        s3 = boto3.client("s3", region_name="us-east-1")
        s3.create_bucket(Bucket="assets")
        enqueue(table, "w2", prompt="p", duration=Decimal("3.0"), steps=100)

        assert run_once(table, FakeEngine(raises=True), "assets", s3,
                        out_dir=str(tmp_path)) is True

        row = read_status(table, "w2")
        assert row["status"] == "failed" and "cuda oom" in row["reason"]


@pytest.mark.unit
def test_run_once_returns_false_when_queue_empty(table, tmp_path):
    with mock_aws():
        s3 = boto3.client("s3", region_name="us-east-1")
        s3.create_bucket(Bucket="assets")
        assert run_once(table, FakeEngine(), "assets", s3, out_dir=str(tmp_path)) is False


class _ThrowingTable:
    """Every UpdateItem fails. Wraps a real moto table so claim_next_job's
    query still works and there is a job to fail on."""
    def __init__(self, real):
        self._real = real

    def __getattr__(self, name):
        return getattr(self._real, name)

    def update_item(self, **kwargs):
        if "SET #s = :f" in kwargs.get("UpdateExpression", ""):
            raise RuntimeError("ProvisionedThroughputExceededException")
        return self._real.update_item(**kwargs)


@pytest.mark.unit
def test_fail_job_throwing_does_not_escape_run_once(table, tmp_path):
    """fail_job is itself an UpdateItem and can throw — a throttle, a transient
    5xx, expired credentials. Unguarded it propagates out of the handler that
    exists to stop exceptions propagating, killing the process from inside the
    recovery path. Nothing restarts this task (no ECS Service, by design), so
    that is a g5.xlarge billing at full rate with nobody polling."""
    with mock_aws():
        s3 = boto3.client("s3", region_name="us-east-1")
        s3.create_bucket(Bucket="assets")
        enqueue(table, "w3", prompt="p", duration=Decimal("3.0"), steps=100)

        assert run_once(_ThrowingTable(table), FakeEngine(raises=True), "assets", s3,
                        out_dir=str(tmp_path)) is True

        # The row is not lost: still `processing`, lease expires in
        # LEASE_SECONDS, recover_abandoned_jobs requeues it on the next start.
        assert table.get_item(Key={"job_id": "w3"})["Item"]["status"] == "processing"


class _Stop(BaseException):
    """BaseException, so `except Exception` in the loop cannot swallow the
    thing being used to escape an infinite loop."""


@pytest.mark.unit
def test_poll_loop_survives_an_error_outside_run_once_s_try(table, monkeypatch):
    """claim_next_job() sits BEFORE run_once's try, so a DynamoDB throttle or an
    expired credential comes straight out of the loop. Before the guard, that
    ended the process — silently, because there is no ECS Service to report a
    crash loop.

    The loop is escaped by counting sleeps, not by letting the error through:
    the point is that the error does NOT end it.
    """
    calls = {"run": 0, "sleep": 0}

    def _run_once(*a, **kw):
        calls["run"] += 1
        if calls["run"] == 1:
            raise RuntimeError("ThrottlingException")
        return False                                   # queue empty

    def _sleep(_seconds):
        calls["sleep"] += 1
        if calls["sleep"] == 2:
            raise _Stop

    monkeypatch.setattr(worker, "run_once", _run_once)
    monkeypatch.setattr(worker.time, "sleep", _sleep)

    with pytest.raises(_Stop):
        poll_forever(table, FakeEngine(), "assets", None)

    assert calls["run"] == 2, "the loop stopped at the first error"
