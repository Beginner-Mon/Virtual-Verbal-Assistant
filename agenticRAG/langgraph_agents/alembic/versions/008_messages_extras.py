"""Add messages.extras — one JSONB column for every optional per-feature field.

Revision ID: 008_messages_extras
Revises: 007_rls
Create Date: 2026-08-28

WHY JSONB AND NOT A COLUMN PER FEATURE
--------------------------------------
This started as `motion_job_id TEXT`. Motion generation is asynchronous — the
agent writes a DynamoDB job row and answers immediately, a GPU worker renders
and uploads to S3, and the browser polls GET /motion/{job_id} — so a refresh
mid-render needs the id carried on the message it belongs to.

One column for that one feature is the wrong shape, and N said so before it
shipped. Motion is not part of a chat turn; it is an occasional extra, and it
is not the last one. TTS already wants to record which language and voice
answered. The next feature will want its own field. Each as its own column
gives `messages` a growing tail of nullable columns belonging to subsystems
that have nothing to do with each other, and a migration every time.

`extras` holds all of them:

    {"motion": {"job_id": "a72fb4b3f33c1e71114d552bb6f48d3e"},
     "tts":    {"lang": "vi", "voice": "..."}}

NULL when a message has none, which is the large majority. Adding a field later
is a code change, not a migration.

The trade this accepts: no column-level constraint, and filtering by a nested
value needs `extras->'motion'->>'job_id'` plus a GIN index to be fast. Neither
costs anything here, because nothing filters on these — they are read back with
the message that owns them and never searched. That is precisely the case JSONB
is for; a column would win if these were query predicates, and they are not.

`token_count` stays a column. It is on every message, for every turn, and is
aggregated — universal and queried, the opposite of these.

WHY THIS IS A MIGRATION AND NOT A LINE IN A .sql FILE
------------------------------------------------------
The original column was added to three files, none of which reach a database:

  * infra/sql/init_schema.sql              — no runner anywhere in the repo
  * langgraph_agents/db/init_schema.sql    — what `python -m
    langgraph_agents.db.init_schema` actually runs, and it has no `messages`
    table at all (its `conversations.messages` is a JSONB column from the
    pre-M4 shape)
  * infra/sql/migrations/002_messages_motion_job_id.sql — no runner, no 001,
    and the number collided with alembic's own 002_m4_fresh_schema

Meanwhile db/session_store.py had already started SELECTing and INSERTing it
unconditionally. Deploying that combination does not fail to add a feature; it
BREAKS a working one: every GET /sessions/{id} raises UndefinedColumn, and every
/chat turn's write_session_turn fails into `session_persist_failed` — where it
is swallowed, so chat keeps answering while silently persisting nothing.
Alembic is the only schema system here that runs, so this is where it has to be.

Additive and nullable, so it is safe in either deploy order: the old Lambda
ignores a column it does not select, and the new one needs it present before its
first turn. Ship the migration first.

No RLS work needed. 007_rls put a policy on `messages` as a whole
(messages_owner, FOR ALL, reached through conversations.user_id) — policies are
per-table, not per-column, so a new column is covered by the existing one and
eca_user's table-level GRANT already includes it.

Named `extras`, not `metadata`: SQLAlchemy's declarative base reserves
`metadata` on model classes, so a future ORM model for this table could not
name the attribute after the column.
"""

from typing import Sequence, Union

from alembic import op

revision: str = "008_messages_extras"
down_revision: Union[str, None] = "007_rls"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    # One statement per op.execute() — asyncpg rejects a batch with "cannot
    # insert multiple commands into a prepared statement". See 005/006, and the
    # note in 004, which never ran at all for exactly this reason.
    op.execute(
        "ALTER TABLE messages ADD COLUMN IF NOT EXISTS extras JSONB"
    )
    op.execute(
        "COMMENT ON COLUMN messages.extras IS "
        "'Optional per-feature fields for this message, or NULL. Namespaced by "
        "subsystem: {\"motion\": {\"job_id\": ...}}, {\"tts\": {\"lang\": ...}}. "
        "Not searched — read back with the message that owns it, which is why "
        "it is JSONB rather than a column per feature. Motion job ids are not a "
        "foreign key: the job lives in DynamoDB and expires at 24h, well before "
        "the message does, so a stored id may point at nothing.'"
    )


def downgrade() -> None:
    op.execute("ALTER TABLE messages DROP COLUMN IF EXISTS extras")
