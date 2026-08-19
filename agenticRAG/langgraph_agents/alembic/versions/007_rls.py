"""Row-level security, and the least-privilege role the application connects as.

Revision ID: 007_rls
Revises: 006_character_ui_strings
Create Date: 2026-08-18

Every CRUD query already carries `WHERE user_id = $1::uuid`. This does not close
a hole that is open today — it moves the guarantee from "whoever writes the next
handler remembers" to "the database will not return the rows". Two routes make
that worth doing now: GET and DELETE /sessions/{session_id} take the id from the
URL, and dropping the user_id predicate from either is a one-line change that
reads as correct.

TWO THINGS THIS MIGRATION DEPENDS ON, both easy to get wrong:

1. **The application must not connect as the table owner.** PostgreSQL exempts a
   table's owner from its policies, so running this while the app still uses
   neondb_owner leaves RLS enabled, listed by \\d+, and completely inert. The app
   connects as `eca_user`, created out of band (no password in a migration) with
   NOBYPASSRLS and no group membership.

   Note the role could not simply be made in the Neon console: console-created
   roles join `neon_superuser`, which carries pg_read_all_data/pg_write_all_data
   and BYPASSRLS, and neondb_owner has no ADMIN OPTION to narrow them afterwards.
   Measured 18-08 on this database.

2. **`current_setting('app.user_id')` takes no second argument.** With
   `, true` a missing setting returns NULL, `user_id = NULL` is NULL, and the
   query returns zero rows — so a handler that forgot PostgresClient.user_scope()
   silently shows the user an empty account. Without it, the same mistake raises
   `unrecognized configuration parameter` on the spot. Fail loud is the whole
   point; test_rls_policies.py fails if the second argument comes back.

Not covered by RLS: characters, kb_embeddings, documents. They belong to nobody,
so there are no rows to filter by owner — they are restricted by GRANT instead
(eca_user reads, only the owner writes).

ONE PLACE STILL TO UPDATE, deliberately left alone: api/billing.py. Billing is
another author's unfinished work and is off by default
(BILLING_SANDBOX_ENABLED=false), so nothing breaks today. When it is picked up,
`POST /billing/webhook` will need the user bound before it writes
billing_accounts:

    from langgraph_agents.db.postgres import bind_request_user
    bind_request_user(uid)          # uid from client_reference_id or
                                    # metadata.vva_user_id in the Stripe event

Stripe's signature proves who SENT the event, not whose account it concerns —
the account id is inside the event, put there by this service when it created
the object. Without the bind, the writes fail loudly rather than touching the
wrong row.
"""

from typing import Sequence, Union

from alembic import op

revision: str = "007_rls"
down_revision: Union[str, None] = "006_character_ui_strings"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None

APP_ROLE = "eca_user"

# Tables whose rows belong to one user, keyed directly.
OWNED_DIRECTLY = {
    "conversations": "user_id",
    "user_memory": "user_id",
    "billing_accounts": "user_id",
    "users": "id",
}

# Tables that reach their owner through conversations. Neither has a user_id
# column — see 002_m4_fresh_schema.
OWNED_VIA_SESSION = ("messages", "summaries")

# Shared reference data. No owner, so no policy — a GRANT is the right tool.
#
# `documents` is here because of a JOIN, not a FROM: tools/pgvector_tool.py
# reads kb_embeddings and joins documents for the source title. An earlier
# version of this list was built by grepping for `FROM <table>` and missed it,
# which broke knowledge-base search with a permission error inside the retry
# loop — visible only as kb_search_error in the logs.
SYSTEM_READONLY = ("characters", "kb_embeddings", "documents")

# A ledger, not user data: Stripe's delivery ids, written to make webhook
# processing idempotent. No user column to filter by, so no policy — but the
# application does have to append to it, which read-only would prevent. No
# UPDATE or DELETE: an event that has been seen stays seen.
SYSTEM_APPEND_ONLY = ("billing_webhook_events",)

ALL_USER_TABLES = tuple(OWNED_DIRECTLY) + OWNED_VIA_SESSION


def upgrade() -> None:
    # One statement per op.execute(): asyncpg rejects a batch with "cannot
    # insert multiple commands into a prepared statement". 004 never ran at all
    # for exactly this reason.

    # ── Grants ──────────────────────────────────────────────────────────
    op.execute(f'GRANT USAGE ON SCHEMA public TO "{APP_ROLE}"')

    for table in ALL_USER_TABLES:
        op.execute(
            f'GRANT SELECT, INSERT, UPDATE, DELETE ON "{table}" TO "{APP_ROLE}"'
        )

    for table in SYSTEM_READONLY:
        op.execute(f'GRANT SELECT ON "{table}" TO "{APP_ROLE}"')

    for table in SYSTEM_APPEND_ONLY:
        op.execute(f'GRANT SELECT, INSERT ON "{table}" TO "{APP_ROLE}"')

    # messages.seq_id is BIGSERIAL; without this every insert fails on the
    # sequence rather than on the table, which reads as an unrelated error.
    op.execute(f'GRANT USAGE ON ALL SEQUENCES IN SCHEMA public TO "{APP_ROLE}"')

    # Deliberately NO `ALTER DEFAULT PRIVILEGES`. A table added later must be
    # granted on purpose; the alternative is new tables becoming readable by the
    # application the moment somebody creates them.

    # ── Policies ────────────────────────────────────────────────────────
    for table, column in OWNED_DIRECTLY.items():
        op.execute(f'ALTER TABLE "{table}" ENABLE ROW LEVEL SECURITY')
        # FOR ALL with both USING and WITH CHECK: USING filters what is read and
        # what may be updated or deleted, WITH CHECK constrains what may be
        # written. Omitting WITH CHECK would let a user insert rows owned by
        # somebody else and then be unable to see them.
        op.execute(
            f'CREATE POLICY {table}_owner ON "{table}" FOR ALL '
            f"USING ({column} = current_setting('app.user_id')::uuid) "
            f"WITH CHECK ({column} = current_setting('app.user_id')::uuid)"
        )

    for table in OWNED_VIA_SESSION:
        op.execute(f'ALTER TABLE "{table}" ENABLE ROW LEVEL SECURITY')
        predicate = (
            f"EXISTS (SELECT 1 FROM conversations c "
            f'WHERE c.session_id = "{table}".session_id '
            f"AND c.user_id = current_setting('app.user_id')::uuid)"
        )
        op.execute(
            f'CREATE POLICY {table}_owner ON "{table}" FOR ALL '
            f"USING ({predicate}) WITH CHECK ({predicate})"
        )


def downgrade() -> None:
    for table in ALL_USER_TABLES:
        op.execute(f'DROP POLICY IF EXISTS {table}_owner ON "{table}"')
        op.execute(f'ALTER TABLE "{table}" DISABLE ROW LEVEL SECURITY')

    for table in ALL_USER_TABLES:
        op.execute(f'REVOKE ALL ON "{table}" FROM "{APP_ROLE}"')

    for table in SYSTEM_READONLY + SYSTEM_APPEND_ONLY:
        op.execute(f'REVOKE ALL ON "{table}" FROM "{APP_ROLE}"')

    op.execute(f'REVOKE ALL ON ALL SEQUENCES IN SCHEMA public FROM "{APP_ROLE}"')
    op.execute(f'REVOKE USAGE ON SCHEMA public FROM "{APP_ROLE}"')
