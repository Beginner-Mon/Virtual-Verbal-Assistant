"""User preferences — cross-device synced UI prefs (avatar_bg, default character).

Revision ID: 009_user_preferences
Revises: 008_messages_extras
Create Date: 2026-09-01

WHY THIS TABLE EXISTS
---------------------
Frontend had three contexts that were localStorage-only (ThemeContext,
GraphicsContext, AvatarBgContext) and MotionContext.selectedVrmId that was
ephemeral. Owner requires:
  - avatar_bg + default character synced across all devices (pick on phone →
    appears on desktop instead of Anne default). Theme/graphics stay
    device-specific (per-browser localStorage) — see docs/plans/user-preferences-plan.md.
  - preferences are UI-only, no PHI. Clinical facts stay in user_memory.

WHY NEON, WHY HYBRID
--------------------
Scale is 100 users × 3 devices = 300 rows, QPS ~0 (Q7 rare, Q9 small) —
Neon already hosts users/conversations with RLS. DynamoDB would be a second
store for 60KB of data. Hybrid: stable fields as typed columns (FK + CHECK) +
extensible UI flags in prefs JSONB (no migration per flag).

SCALE NOTE: 1 row/user, ~200 bytes. 10k users = 2 MB. No extra compute.

IDOR: /me/preferences takes no user_id — identity from Bearer token only
(api/auth.py:207 current_user_id). RLS is second fence.

See docs/plans/user-preferences-plan.md v2.1, ADR-007.
"""

from typing import Sequence, Union

from alembic import op

revision: str = "009_user_preferences"
down_revision: Union[str, None] = "008_messages_extras"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None

APP_ROLE = "eca_user"


def upgrade() -> None:
    # One statement per op.execute() — asyncpg rejects batched prepared statements.
    op.execute(
        """
        CREATE TABLE IF NOT EXISTS user_preferences (
            user_id UUID PRIMARY KEY REFERENCES users(id) ON DELETE CASCADE,
            avatar_bg TEXT NOT NULL DEFAULT 'slate'
                CHECK (avatar_bg IN ('slate','violet','blue','emerald','amber','rose','cyan','indigo')),
            selected_character_slug TEXT REFERENCES characters(slug) ON DELETE SET NULL,
            display_name TEXT,
            prefs JSONB NOT NULL DEFAULT '{}'::jsonb,
            version INT NOT NULL DEFAULT 1,
            updated_at TIMESTAMPTZ NOT NULL DEFAULT now(),
            created_at TIMESTAMPTZ NOT NULL DEFAULT now()
        )
        """
    )
    op.execute(
        "COMMENT ON TABLE user_preferences IS "
        "'Cross-device synced UI prefs (UI-only, no PHI). 1 row/user. "
        "avatar_bg + selected_character_slug are synced; theme/graphics stay "
        "in localStorage per device. prefs holds extensible flags like "
        "{notifications:{email:true}, locale:vi} — not health data (see user_memory).'"
    )
    op.execute(
        "COMMENT ON COLUMN user_preferences.prefs IS "
        "'Extensible UI flags only — notifications/locale etc. 8KB max, depth<=2. "
        "Must not contain PHI (injury_history etc) — those belong in user_memory.'"
    )
    op.execute("CREATE INDEX IF NOT EXISTS idx_user_prefs_updated ON user_preferences(updated_at)")

    # RLS — same pattern as 007_rls OWNED_DIRECTLY
    op.execute('ALTER TABLE "user_preferences" ENABLE ROW LEVEL SECURITY')
    op.execute(
        'CREATE POLICY user_preferences_owner ON "user_preferences" FOR ALL '
        "USING (user_id = current_setting('app.user_id')::uuid) "
        "WITH CHECK (user_id = current_setting('app.user_id')::uuid)"
    )
    op.execute(f'GRANT SELECT, INSERT, UPDATE, DELETE ON "user_preferences" TO "{APP_ROLE}"')


def downgrade() -> None:
    op.execute('DROP POLICY IF EXISTS user_preferences_owner ON "user_preferences"')
    op.execute('ALTER TABLE IF EXISTS "user_preferences" DISABLE ROW LEVEL SECURITY')
    op.execute(f'REVOKE ALL ON "user_preferences" FROM "{APP_ROLE}"')
    op.execute("DROP INDEX IF EXISTS idx_user_prefs_updated")
    op.execute("DROP TABLE IF EXISTS user_preferences")
