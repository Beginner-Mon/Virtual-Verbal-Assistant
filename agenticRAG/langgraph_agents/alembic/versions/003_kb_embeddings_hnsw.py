"""Add the missing HNSW index on kb_embeddings.embedding

Revision ID: 003_kb_hnsw
Revises: 002_m4_fresh
Create Date: 2026-07-30

Migration 002 created an HNSW index for `summaries.embedding` but only a btree
on `kb_embeddings (document_id)` — the vector index for the knowledge base was
omitted. `kb_search` orders by `embedding <=> $1` (cosine), so without it every
KB lookup is a sequential scan over the whole table.

That went unnoticed because `kb_embeddings` was empty until the pgvector ingest
existed (scripts/ingest_kb_pgvector.py). At the current ~2.9k rows a seq scan is
only milliseconds, but it degrades linearly as the corpus grows — and the stack
already documents HNSW as the retrieval index.

`vector_cosine_ops` matches the `<=>` operator used by the query; an index built
for a different op class would simply be ignored by the planner.
"""
from typing import Sequence, Union

from alembic import op

# revision identifiers, used by Alembic.
revision: str = "003_kb_hnsw"
down_revision: Union[str, None] = "002_m4_fresh"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    # HNSW needs no training data — safe on an empty or populated table.
    op.execute(
        """
        CREATE INDEX IF NOT EXISTS idx_kb_emb_embedding
            ON kb_embeddings USING hnsw (embedding vector_cosine_ops);
        """
    )


def downgrade() -> None:
    op.execute("DROP INDEX IF EXISTS idx_kb_emb_embedding;")
