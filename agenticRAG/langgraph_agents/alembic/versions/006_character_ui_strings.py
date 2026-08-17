"""Add characters.ui_strings — chat-surface copy per character.

Revision ID: 006_character_ui_strings
Revises: 005_characters
Create Date: 2026-08-17

Picking a character changed how the assistant *answered* but not a single word
around the answer: the opening greeting introduced "ECA" in English, the stage
labels and the error line were hard-coded, and none of them moved when the
avatar did.

That copy could not ride along in `characters.persona`. `persona` is the system
prompt and is deliberately absent from the catalog Lambda's _PUBLIC_COLUMNS, so
anything the browser needs has to be a column of its own. Hence a second JSONB:
`persona` stays server-side, `ui_strings` is public.

Authored in personas/<slug>.md under `## UI Strings` and pushed here by
scripts/sync_personas_to_db.py, so the character is still defined in exactly one
file.

Additive with a default, so the catalog keeps answering through the deploy gap
between this migration and the Lambda that selects the new column.
"""

from typing import Sequence, Union

from alembic import op

revision: str = "006_character_ui_strings"
down_revision: Union[str, None] = "005_characters"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    # One statement per op.execute() — asyncpg rejects a batch with "cannot
    # insert multiple commands into a prepared statement". See 005 and the note
    # in 004, which never ran at all for exactly this reason.
    op.execute(
        "ALTER TABLE characters "
        "ADD COLUMN IF NOT EXISTS ui_strings JSONB NOT NULL DEFAULT '{}'"
    )
    op.execute(
        "COMMENT ON COLUMN characters.ui_strings IS "
        "'Chat-surface copy shown to the user: greeting, placeholder, stage "
        "labels, error lines. Parsed from the ## UI Strings section of "
        "personas/<slug>.md. PUBLIC — served by the catalog Lambda, unlike "
        "persona. Never put prompt text here.'"
    )


def downgrade() -> None:
    op.execute("ALTER TABLE characters DROP COLUMN IF EXISTS ui_strings")
