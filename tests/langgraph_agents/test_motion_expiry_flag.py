"""A restored `motion_job_id` must say whether it still points at anything.

The three clocks do not agree, and only one of them is authoritative:

    messages.motion_job_id   forever      — lives as long as the message
    DynamoDB job row         24h TTL      — best-effort, AWS promises ~48h
    S3 motions/*.bvh         1 day        — lifecycle rule, reliable

So a day after the turn, every stored job id points at nothing. Asking
`GET /motion/{job_id}` cannot tell you that: a swept row, an expired row and a
job that never existed all answer 404 identically — the evidence is gone.

Postgres has what DynamoDB does not: `created_at`. The age of the message
decides it, locally, with no round trip, and it is the only way to separate
"this expired" from "this failed" — which matters because the user-facing
sentence is different and the wrong one reads as a broken system.

The S3 rule is the binding clock, not the DynamoDB TTL: the file is what the
browser fetches, and lifecycle deletes it on time while TTL only promises to
get around to it.
"""

from datetime import datetime, timedelta, timezone

import pytest

from langgraph_agents.db.session_store import MOTION_TTL_SECONDS, motion_expired
from vva_motion.jobs import TTL_SECONDS as QUEUE_TTL_SECONDS


@pytest.mark.unit
def test_the_two_ttl_constants_have_not_drifted():
    """session_store carries its own copy of the queue's TTL, because it is
    served by two deployments and only one of them ships vva_motion: the CRUD
    Lambda is a zip of agenticRAG/langgraph_agents alone, where importing it is
    a cold-start crash that takes /sessions and /me/memory down.

    Duplication is the cost of that boundary; silent drift is not. If the queue
    changes its TTL and this copy does not, every restored motion is labelled
    with the wrong lifetime and nothing else complains.
    """
    assert MOTION_TTL_SECONDS == QUEUE_TTL_SECONDS


TTL_SECONDS = MOTION_TTL_SECONDS


def _ago(**kw) -> datetime:
    return datetime.now(timezone.utc) - timedelta(**kw)


@pytest.mark.unit
def test_fresh_turn_is_not_expired():
    assert motion_expired(_ago(minutes=5)) is False


@pytest.mark.unit
def test_just_inside_the_window_is_not_expired():
    assert motion_expired(_ago(seconds=TTL_SECONDS - 60)) is False


@pytest.mark.unit
def test_past_the_window_is_expired():
    assert motion_expired(_ago(seconds=TTL_SECONDS + 60)) is True


@pytest.mark.unit
def test_old_conversation_is_expired():
    assert motion_expired(_ago(days=3)) is True


@pytest.mark.unit
def test_naive_timestamp_is_read_as_utc_not_local():
    """asyncpg returns tz-aware values, but a hand-built row or a different
    driver may not. Treating a naive UTC timestamp as local time shifts it by
    the machine's offset — seven hours here — which silently reclassifies
    everything near the boundary."""
    naive = datetime.utcnow() - timedelta(seconds=TTL_SECONDS + 60)
    assert motion_expired(naive) is True


@pytest.mark.unit
def test_no_timestamp_is_treated_as_expired():
    """Unknown age cannot be assumed fresh: promising a motion that is not
    there costs a poll and a wrong message, while calling a live one expired
    costs a replay the user can trigger again."""
    assert motion_expired(None) is True
