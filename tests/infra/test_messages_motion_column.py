from pathlib import Path

import pytest

SCHEMA = Path("infra/sql/init_schema.sql").read_text(encoding="utf-8")


@pytest.mark.unit
def test_messages_carries_motion_job_id():
    """Refresh must not lose an in-flight motion: ChatContext.tsx:215 already
    restores history from Postgres, so the job id has to travel with the
    message row it belongs to."""
    block = SCHEMA.split("CREATE TABLE IF NOT EXISTS messages")[1].split(");")[0]
    assert "motion_job_id" in block
