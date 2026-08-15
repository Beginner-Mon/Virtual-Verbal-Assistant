"""Add zero-cost demo billing state.

Revision ID: 004_demo_billing
Revises: 003_kb_hnsw
Create Date: 2026-08-15
"""

from typing import Sequence, Union

from alembic import op

revision: str = "004_demo_billing"
down_revision: Union[str, None] = "003_kb_hnsw"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    op.execute(
        """
        CREATE TABLE IF NOT EXISTS billing_accounts (
            user_id                     UUID PRIMARY KEY REFERENCES users(id) ON DELETE CASCADE,
            access_plan                 TEXT NOT NULL DEFAULT 'DEMO'
                                        CHECK (access_plan IN ('FREE', 'DEMO')),
            stripe_customer_id          TEXT UNIQUE,
            stripe_subscription_id      TEXT UNIQUE,
            stripe_subscription_status  TEXT,
            created_at                  TIMESTAMPTZ NOT NULL DEFAULT now(),
            updated_at                  TIMESTAMPTZ NOT NULL DEFAULT now()
        );
        COMMENT ON COLUMN billing_accounts.access_plan IS
            'Internal entitlement only; deliberately restricted to FREE/DEMO.';

        CREATE TABLE IF NOT EXISTS billing_webhook_events (
            event_id       TEXT PRIMARY KEY,
            event_type     TEXT NOT NULL,
            livemode       BOOLEAN NOT NULL DEFAULT false CHECK (livemode = false),
            processed_at   TIMESTAMPTZ NOT NULL DEFAULT now()
        );
        COMMENT ON TABLE billing_webhook_events IS
            'Stripe sandbox webhook receipts only; live events are rejected by code and schema.';
        """
    )


def downgrade() -> None:
    op.execute("DROP TABLE IF EXISTS billing_webhook_events;")
    op.execute("DROP TABLE IF EXISTS billing_accounts;")
