from decimal import Decimal

import boto3
import pytest
from moto import mock_aws

from vva_motion.jobs import enqueue, read_status
from worker import run_once


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
