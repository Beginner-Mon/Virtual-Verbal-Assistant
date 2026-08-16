"""Add characters table — single source of truth for a VRM character.

Revision ID: 005_characters
Revises: 004_demo_billing
Create Date: 2026-08-16

Until now the four aspects of a "character" lived in four places with nothing
tying them together: the .vrm binary was bundled into the Vite build, the
avatar profile was a TypeScript const, the persona was a markdown file in the
backend, and the blendShape manifest was generated TypeScript. Adding a
character meant editing four files across two directories, and nothing
guaranteed that bronya.vrm was ever paired with a bronya persona.

One row per character, so `slug` is the only identifier anything needs to
carry. `persona` mirrors the parsed shape of personas/*.md — the markdown
files stay in the repo as the authoring format and the offline fallback, and
the seed script converts them via nodes._persona_loader._load_persona.

No FK to users: characters are a global catalog, not user data, so GDPR
erasure (db/gdpr.py delete_user) is unaffected in either direction.
"""

from typing import Sequence, Union

from alembic import op

revision: str = "005_characters"
down_revision: Union[str, None] = "004_demo_billing"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    # One statement per op.execute() — asyncpg rejects a batch with "cannot
    # insert multiple commands into a prepared statement". Migration 004 was
    # written as a single multi-statement string and therefore never ran at all;
    # see the note there.
    op.execute(
        """
        CREATE TABLE IF NOT EXISTS characters (
            id              UUID PRIMARY KEY DEFAULT gen_random_uuid(),
            slug            TEXT UNIQUE NOT NULL,
            display_name    TEXT NOT NULL,
            description     TEXT,

            vrm_url         TEXT NOT NULL,
            vrm_metadata    JSONB NOT NULL DEFAULT '{}',
            avatar_profile  JSONB NOT NULL DEFAULT '{}',
            persona         JSONB NOT NULL DEFAULT '{}',

            voice_provider  TEXT DEFAULT 'vieneu',
            voice_id        TEXT,
            voice_language  TEXT DEFAULT 'vi',

            thumbnail_url   TEXT,
            is_active       BOOLEAN NOT NULL DEFAULT true,
            sort_order      INT NOT NULL DEFAULT 0,

            created_at      TIMESTAMPTZ NOT NULL DEFAULT now(),
            updated_at      TIMESTAMPTZ NOT NULL DEFAULT now()
        )
        """
    )
    op.execute(
        "COMMENT ON COLUMN characters.slug IS "
        "'Stable id used as ChatRequest.persona_id and as the S3 key prefix.'"
    )
    op.execute(
        "COMMENT ON COLUMN characters.persona IS "
        "'Parsed personas/<slug>.md — identity, personality, behavioral_rules, "
        "response_formatting, safety_templates, voice_identity. Never served to clients.'"
    )
    op.execute(
        "COMMENT ON COLUMN characters.vrm_metadata IS "
        "'Auto-extracted from the .vrm GLB by scripts/upload_characters_to_s3.py. "
        "incompatible_reasons drives the greyed-out state in the avatar picker.'"
    )
    op.execute(
        "COMMENT ON COLUMN characters.vrm_url IS "
        "'CloudFront URL. The key embeds a content hash, so replacing a model "
        "changes this URL and no cache invalidation is ever required.'"
    )
    op.execute(
        """
        CREATE INDEX IF NOT EXISTS idx_characters_active
            ON characters (is_active, sort_order)
        """
    )


def downgrade() -> None:
    op.execute("DROP TABLE IF EXISTS characters;")
