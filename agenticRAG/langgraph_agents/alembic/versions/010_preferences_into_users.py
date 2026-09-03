"""Fold user_preferences into a single JSONB column on users.

Revision ID: 010_preferences_into_users
Revises: 009_user_preferences
Create Date: 2026-09-02

WHY THIS REVERSES 009
---------------------
009 gave preferences their own table for a relationship that is strictly 1:1
with users. Everything that table needed, `users` already had:

  - row-level security and the four grants (007_rls.py OWNED_DIRECTLY)
  - a row created for it in five separate code paths (routes_crud,
    session_store x2, billing, and 009's own lazy seed)
  - deletion, via `DELETE FROM users` in db/gdpr.py

So the second table bought a second RLS policy, a JOIN, a seed step that made
`GET /me/preferences` perform writes, and a third copy of `display_name`. It is
cheaper to carry two columns' worth of UI state on the row that already exists.

WHY ONE JSONB COLUMN AND NOT TYPED COLUMNS
------------------------------------------
Preferences are an open set: nobody can list today which settings will want to
follow a user across devices tomorrow. A typed column per setting means a
migration per setting. JSONB means a field added to api/schemas.py SyncedPrefs
and nothing else — and that model, with extra="forbid", is what keeps the column
from becoming a dumping ground.

WHY NO FOREIGN KEY ON selected_character_slug
---------------------------------------------
009 had `REFERENCES characters(slug) ON DELETE SET NULL`. It never earned its
place: the catalog soft-deletes (characters.is_active), so the SET NULL branch
cannot fire, and the FK cannot see is_active — which is the case that actually
occurs. The guard that does work is the explicit
`SELECT 1 FROM characters WHERE slug = $1 AND is_active` in routes_preferences,
and it catches both a missing slug and a disabled one. A stale slug surviving a
hard delete would 404 on GET /characters/{slug} and the UI already falls back to
its default character, which is the same thing a disabled character does today.

display_name is dropped rather than carried across: Cognito is its source of
truth (ProfileContent.tsx reads custom:displayName from the token), so both
users.display_name and user_preferences.display_name were copies nobody read.

See docs/plans/preferences-v3-plan.md, ADR-008.
"""

from typing import Sequence, Union

from alembic import op

revision: str = "010_preferences_into_users"
down_revision: Union[str, None] = "009_user_preferences"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None

APP_ROLE = "eca_user"


def upgrade() -> None:
    # One statement per op.execute() — asyncpg rejects batched prepared
    # statements. 004 never ran at all for exactly this reason.
    op.execute(
        "ALTER TABLE users ADD COLUMN IF NOT EXISTS preferences JSONB NOT NULL "
        "DEFAULT '{}'::jsonb"
    )
    op.execute(
        "COMMENT ON COLUMN users.preferences IS "
        "'Cross-device synced UI preferences (UI-only, no PHI). The set of keys "
        "is defined by SyncedPrefs in api/schemas.py, which forbids anything it "
        "does not declare. Clinical facts live in user_memory.'"
    )

    # Carry 009's rows over. jsonb_strip_nulls so a user who never picked a
    # character gets {} rather than {"selected_character_slug": null}, and the
    # `|| p.prefs` keeps whatever 009's open column held. display_name is
    # deliberately not carried — Cognito owns it.
    op.execute(
        """
        UPDATE users u SET preferences =
            jsonb_strip_nulls(jsonb_build_object(
                'avatar_bg',               p.avatar_bg,
                'selected_character_slug', p.selected_character_slug
            )) || COALESCE(p.prefs, '{}'::jsonb)
        FROM user_preferences p
        WHERE p.user_id = u.id
        """
    )

    op.execute('DROP POLICY IF EXISTS user_preferences_owner ON "user_preferences"')
    op.execute("DROP INDEX IF EXISTS idx_user_prefs_updated")
    op.execute('DROP TABLE IF EXISTS "user_preferences"')


def downgrade() -> None:
    # Rebuild 009's table and unpack the column back into it. avatar_bg is
    # coalesced to 'slate' because 009's CHECK rejects NULL, and any colour the
    # column picked up after 009 was dropped would fail that CHECK — the
    # downgrade takes the default rather than the migration failing outright.
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
        """
        INSERT INTO user_preferences (user_id, avatar_bg, selected_character_slug, prefs)
        SELECT u.id,
               CASE WHEN u.preferences->>'avatar_bg' IN
                        ('slate','violet','blue','emerald','amber','rose','cyan','indigo')
                    THEN u.preferences->>'avatar_bg' ELSE 'slate' END,
               CASE WHEN EXISTS (
                        SELECT 1 FROM characters c
                        WHERE c.slug = u.preferences->>'selected_character_slug')
                    THEN u.preferences->>'selected_character_slug' END,
               (u.preferences - 'avatar_bg' - 'selected_character_slug')
        FROM users u
        ON CONFLICT (user_id) DO NOTHING
        """
    )
    op.execute("CREATE INDEX IF NOT EXISTS idx_user_prefs_updated ON user_preferences(updated_at)")
    op.execute('ALTER TABLE "user_preferences" ENABLE ROW LEVEL SECURITY')
    op.execute(
        'CREATE POLICY user_preferences_owner ON "user_preferences" FOR ALL '
        "USING (user_id = current_setting('app.user_id')::uuid) "
        "WITH CHECK (user_id = current_setting('app.user_id')::uuid)"
    )
    op.execute(f'GRANT SELECT, INSERT, UPDATE, DELETE ON "user_preferences" TO "{APP_ROLE}"')
    op.execute("ALTER TABLE users DROP COLUMN IF EXISTS preferences")
