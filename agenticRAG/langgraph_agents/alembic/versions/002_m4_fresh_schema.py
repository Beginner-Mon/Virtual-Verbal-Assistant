"""M.4 Fresh Schema — REUPDATE_PLAN.md §M.4

Revision ID: 002_m4_fresh
Revises: (none — fresh rebuild)
Create Date: 2026-06-05

Full schema replacement per 33 decisions D1-D33:
- users (auth_provider, auth_subject, display_name)
- conversations (session_id PK, user_id FK, title)
- messages (seq_id BIGSERIAL, NO user_id — D19)
- summaries (embedding vector(384), status for GDPR, NO user_id — D19)
- user_memory (fact_text, category, valid flag — D14)
- documents (external_id TEXT fixes bug A2, NO user_id)
- kb_embeddings (FK documents fixes bug A4, chunk_index)

Data = test → fresh rebuild, no backfill.
"""
from typing import Sequence, Union

from alembic import op

# revision identifiers, used by Alembic.
revision: str = "002_m4_fresh"
down_revision: Union[str, None] = None
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


# ── Raw SQL for full upgrade ─────────────────────────────────────────────
_UPGRADE_SQL = """
-- ═══════════════════════════════════════════════════════════════════════
-- M.4 FRESH SCHEMA (REUPDATE_PLAN.md §M.4)
-- 33 decisions D1-D33 encoded in schema.
-- ═══════════════════════════════════════════════════════════════════════

CREATE EXTENSION IF NOT EXISTS vector;

-- ── Drop old tables (fresh rebuild, data = test) ─────────────────────
DROP TABLE IF EXISTS kb_embeddings CASCADE;
DROP TABLE IF EXISTS user_memory CASCADE;
DROP TABLE IF EXISTS summaries CASCADE;
DROP TABLE IF EXISTS messages CASCADE;
DROP TABLE IF EXISTS embeddings CASCADE;
DROP TABLE IF EXISTS documents CASCADE;
DROP TABLE IF EXISTS conversations CASCADE;
DROP TABLE IF EXISTS users CASCADE;

-- ── users ────────────────────────────────────────────────────────────
CREATE TABLE users (
    id            UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    auth_provider TEXT,
    auth_subject  TEXT,
    display_name  TEXT,
    created_at    TIMESTAMPTZ DEFAULT now(),
    updated_at    TIMESTAMPTZ DEFAULT now(),
    UNIQUE (auth_provider, auth_subject)
);

-- ── conversations (session_id IS the PK now) ─────────────────────────
CREATE TABLE conversations (
    session_id  UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    user_id     UUID NOT NULL REFERENCES users(id) ON DELETE CASCADE,
    title       TEXT,
    created_at  TIMESTAMPTZ DEFAULT now(),
    updated_at  TIMESTAMPTZ DEFAULT now()
);

-- ── messages (NO user_id — D19; seq_id BIGSERIAL global ordering) ────
CREATE TABLE messages (
    id           UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    seq_id       BIGSERIAL,
    session_id   UUID NOT NULL REFERENCES conversations(session_id) ON DELETE CASCADE,
    role         TEXT NOT NULL CHECK (role IN ('user', 'assistant')),
    content      TEXT NOT NULL,
    token_count  INTEGER,
    created_at   TIMESTAMPTZ DEFAULT now()
);

-- ── summaries (chunk-based, frozen on write, NO user_id — D19) ───────
-- status: 'active' | 'dirty' (GDPR mark-dirty — M.8)
CREATE TABLE summaries (
    id               UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    session_id       UUID NOT NULL REFERENCES conversations(session_id) ON DELETE CASCADE,
    summary_text     TEXT NOT NULL,
    covers_from_seq  BIGINT NOT NULL,
    covers_up_to_seq BIGINT NOT NULL,
    embedding        vector(384),
    status           TEXT NOT NULL DEFAULT 'active',
    created_at       TIMESTAMPTZ DEFAULT now(),
    CONSTRAINT uq_chunk UNIQUE (session_id, covers_from_seq)
);

-- ── user_memory (fact-based profile, Tier 1 always-on — M.6) ─────────
CREATE TABLE user_memory (
    id          UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    user_id     UUID NOT NULL REFERENCES users(id) ON DELETE CASCADE,
    fact_text   TEXT NOT NULL,
    category    TEXT,
    valid       BOOLEAN DEFAULT true,
    created_at  TIMESTAMPTZ DEFAULT now()
);

-- ── documents (KB source, external_id TEXT fixes bug A2) ─────────────
CREATE TABLE documents (
    id          UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    source_type TEXT NOT NULL,
    external_id TEXT,
    title       TEXT,
    metadata    JSONB DEFAULT '{}',
    created_at  TIMESTAMPTZ DEFAULT now()
);

-- ── kb_embeddings (FK documents fixes bug A4, chunk_index — M.4) ─────
CREATE TABLE kb_embeddings (
    id          UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    document_id UUID NOT NULL REFERENCES documents(id) ON DELETE CASCADE,
    chunk_index INT NOT NULL DEFAULT 0,
    content     TEXT NOT NULL,
    embedding   vector(384) NOT NULL,
    created_at  TIMESTAMPTZ DEFAULT now()
);

-- ── Indexes ──────────────────────────────────────────────────────────
CREATE INDEX idx_messages_session_seq
    ON messages (session_id, seq_id);

CREATE INDEX idx_conversations_user_updated
    ON conversations (user_id, updated_at DESC);

CREATE INDEX idx_summaries_session
    ON summaries (session_id, covers_up_to_seq);

-- HNSW (no training data needed, works on empty table)
CREATE INDEX idx_summaries_embedding
    ON summaries USING hnsw (embedding vector_cosine_ops);

CREATE INDEX idx_user_memory_user
    ON user_memory (user_id) WHERE valid = true;

CREATE INDEX idx_kb_emb_document
    ON kb_embeddings (document_id);
"""


_DOWNGRADE_SQL = """
DROP TABLE IF EXISTS kb_embeddings CASCADE;
DROP TABLE IF EXISTS documents CASCADE;
DROP TABLE IF EXISTS user_memory CASCADE;
DROP TABLE IF EXISTS summaries CASCADE;
DROP TABLE IF EXISTS messages CASCADE;
DROP TABLE IF EXISTS conversations CASCADE;
DROP TABLE IF EXISTS users CASCADE;
"""


def upgrade() -> None:
    for stmt in _split_sql(_UPGRADE_SQL):
        op.execute(stmt)


def downgrade() -> None:
    for stmt in _split_sql(_DOWNGRADE_SQL):
        op.execute(stmt)


def _split_sql(sql: str) -> list:
    """Split multi-statement SQL for asyncpg. Skips comments and empty lines."""
    import re
    # Remove block comments and line comments
    cleaned = re.sub(r'/\*.*?\*/', '', sql, flags=re.DOTALL)
    cleaned = re.sub(r'--.*?$', '', cleaned, flags=re.MULTILINE)
    statements = []
    for s in cleaned.split(';'):
        s = s.strip()
        if s:
            statements.append(s)
    return statements
