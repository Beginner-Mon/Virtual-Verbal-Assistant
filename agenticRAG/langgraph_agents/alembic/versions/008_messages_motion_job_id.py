"""Add messages.motion_job_id — the Kimodo job id a refresh needs to resume polling.

Revision ID: 008_messages_motion_job_id
Revises: 007_rls
Create Date: 2026-08-28

Motion generation is asynchronous: the agent writes a DynamoDB job row and
answers immediately, a GPU worker renders and uploads to S3, and the browser
polls GET /motion/{job_id}. Refresh the page mid-render and the job id was gone
— the clip finished and nothing was left that knew to ask for it. Carrying the
id on the message row it belongs to reuses the "Postgres stays the source of
truth" history restore (ChatContext.tsx:215) instead of inventing a client-side
job registry that a refresh would clear anyway.

WHY THIS IS A MIGRATION AND NOT A LINE IN A .sql FILE
------------------------------------------------------
The column was first added to three files, none of which reach a database:

  * infra/sql/init_schema.sql              — no runner anywhere in the repo
  * langgraph_agents/db/init_schema.sql    — what `python -m
    langgraph_agents.db.init_schema` actually runs, and it has no `messages`
    table at all (its `conversations.messages` is a JSONB column from the
    pre-M4 shape)
  * infra/sql/migrations/002_messages_motion_job_id.sql — no runner, no 001,
    and the number collided with alembic's own 002_m4_fresh_schema

Meanwhile db/session_store.py had already started SELECTing and INSERTing the
column unconditionally. Deploying that combination does not fail to add a
feature; it BREAKS a working one: every GET /sessions/{id} raises UndefinedColumn,
and every /chat turn's write_session_turn fails into `session_persist_failed` —
where it is swallowed, so chat keeps answering while silently persisting nothing.
Alembic is the only schema system here that runs, so this is where the column
has to be.

Additive and nullable, so it is safe in either deploy order: the old Lambda
ignores a column it does not select, and the new one needs it present before its
first turn. Ship the migration first.

No RLS work needed. 007_rls put a policy on `messages` as a whole
(messages_owner, FOR ALL, reached through conversations.user_id) — policies are
per-table, not per-column, so a new column is covered by the existing one and
eca_user's table-level GRANT already includes it.
"""

from typing import Sequence, Union

from alembic import op

revision: str = "008_messages_motion_job_id"
down_revision: Union[str, None] = "007_rls"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    # One statement per op.execute() — asyncpg rejects a batch with "cannot
    # insert multiple commands into a prepared statement". See 005/006, and the
    # note in 004, which never ran at all for exactly this reason.
    #
    # IF NOT EXISTS: this branch's dead infra/sql/migrations/002_*.sql may have
    # been applied by hand against a developer database before it was deleted.
    op.execute(
        "ALTER TABLE messages ADD COLUMN IF NOT EXISTS motion_job_id TEXT"
    )
    op.execute(
        "COMMENT ON COLUMN messages.motion_job_id IS "
        "'Kimodo motion job id for this turn, or NULL. Assistant rows only — "
        "the user row of a turn is always NULL. Lets a page refresh resume "
        "polling GET /motion/{job_id} for a clip still on the GPU. Not a "
        "foreign key: the job lives in DynamoDB and its row expires at 24h, "
        "well before the message does.'"
    )


def downgrade() -> None:
    op.execute("ALTER TABLE messages DROP COLUMN IF EXISTS motion_job_id")
