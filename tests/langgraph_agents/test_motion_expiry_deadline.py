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

from langgraph_agents.db.session_store import (
    MOTION_TTL_SECONDS, _shape_message, motion_expires_at,
)
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


class _Row(dict):
    """asyncpg Record stand-in: `_shape_message` probes it with `.keys()`."""
    def __init__(self, **kw):
        super().__init__(**kw)


def _deadline(created_at):
    """Parse what the API would send, back into a datetime."""
    return datetime.fromisoformat(motion_expires_at(created_at))


@pytest.mark.unit
def test_the_deadline_is_the_turn_plus_the_ttl():
    made = _ago(minutes=5)
    assert _deadline(made) == made + timedelta(seconds=MOTION_TTL_SECONDS)


@pytest.mark.unit
def test_a_fresh_turn_has_not_reached_its_deadline():
    assert _deadline(_ago(minutes=5)) > datetime.now(timezone.utc)


@pytest.mark.unit
def test_an_old_turn_is_already_past_its_deadline():
    assert _deadline(_ago(days=3)) < datetime.now(timezone.utc)


@pytest.mark.unit
def test_the_deadline_does_not_go_stale_the_way_a_boolean_would():
    """The reason this is an instant and not `motion_expired: true|false`.

    A boolean is computed once, at request time, and answers a question whose
    answer changes: a payload built at 10:00 says `false` and a tab left open
    until the next morning is still holding that `false`. The instant is the
    same value whenever it is read, so the client compares it against its own
    clock at the moment it actually needs to decide.
    """
    made = _ago(hours=23)
    first = _deadline(made)
    later = _deadline(made)
    assert first == later
    # Still live now, definitively gone two hours from now — one value, both
    # answers, no refetch.
    now = datetime.now(timezone.utc)
    assert first > now
    assert first < now + timedelta(hours=2)


@pytest.mark.unit
def test_naive_timestamp_is_read_as_utc_not_local():
    """asyncpg returns tz-aware values, but a hand-built row or a different
    driver may not. Treating a naive UTC timestamp as local time shifts the
    deadline by the machine's offset — seven hours here."""
    naive = datetime.utcnow() - timedelta(hours=1)
    assert _deadline(naive) == naive.replace(tzinfo=timezone.utc) + timedelta(
        seconds=MOTION_TTL_SECONDS
    )


@pytest.mark.unit
def test_the_flag_is_absent_when_there_is_no_motion():
    """Most messages have no motion — it is an occasional extra, never part of
    a chat turn. Stamping every one of them with `motion_expired` says
    something false about a thing that does not exist, and puts a key on every
    row of every history payload to describe nothing.

    The flag belongs to the job id: present together, absent together.
    """
    rows = [
        _Row(role="user", content="cho tôi xem squat", token_count=None, extras=None),
        _Row(role="assistant", content="đây", token_count=12,
             extras='{"motion": {"job_id": "a72fb4b3"}}'),
        # Extras present but for another subsystem — the whole point of the
        # JSONB column. Still no motion, so still no motion keys.
        _Row(role="assistant", content="chỉ là chữ", token_count=8,
             extras='{"tts": {"lang": "vi"}}'),
    ]
    shaped = [_shape_message(r, _ago(minutes=5)) for r in rows]

    assert "motion_job_id" not in shaped[0] and "motion_expires_at" not in shaped[0]
    assert shaped[1]["motion_job_id"] == "a72fb4b3"
    assert shaped[1]["motion_expires_at"] is not None
    assert "motion_job_id" not in shaped[2] and "motion_expires_at" not in shaped[2]


@pytest.mark.unit
def test_no_timestamp_yields_no_deadline():
    """Unknown age cannot be assumed fresh. None means "assume gone": promising
    a motion that is not there costs a poll and a wrong message, while treating
    a live one as gone costs a replay the user can trigger again."""
    assert motion_expires_at(None) is None
