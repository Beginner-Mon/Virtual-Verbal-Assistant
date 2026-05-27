"""pgvector operations — vector similarity search via PostgreSQL.

Replaces ChromaDB backend. Uses the 'embeddings' table with
IVFFlat index for cosine similarity search.
"""

from __future__ import annotations

import json
import uuid
from typing import Optional

from langgraph_agents.db.postgres import PostgresClient


def _coerce_metadata(value) -> dict:
    """asyncpg may return JSONB as dict (if codec registered) or as str (default)."""
    if value is None:
        return {}
    if isinstance(value, dict):
        return value
    if isinstance(value, (bytes, bytearray)):
        try:
            return json.loads(value.decode("utf-8"))
        except Exception:
            return {}
    if isinstance(value, str):
        try:
            return json.loads(value)
        except Exception:
            return {}
    return {}


class VectorBackend:
    """pgvector-based vector similarity search."""

    def __init__(self, pg: PostgresClient):
        self.pg = pg

    async def _ensure_pool(self):
        await self.pg.connect()

    async def search(
        self,
        query_embedding: list[float],
        top_k: int = 5,
        source_type: Optional[str] = None,
        source_id: Optional[str] = None,
        user_id: Optional[str] = None,
    ) -> list[dict]:
        """Cosine similarity search against embeddings table."""
        await self._ensure_pool()

        rows = await self.pg.fetch(
            """SELECT id, source_type, content, metadata,
                      1 - (embedding <=> $1::vector) AS similarity
               FROM embeddings
               WHERE ($2::text IS NULL OR source_type = $2)
                 AND ($4::uuid IS NULL OR source_id::uuid = $4::uuid)
               ORDER BY embedding <=> $1::vector
               LIMIT $3""",
            query_embedding,
            source_type,
            top_k,
            source_id,
        )

        return [
            {
                "id": str(row["id"]),
                "content": row["content"],
                "metadata": {**_coerce_metadata(row["metadata"]), "source_type": row["source_type"]},
                "similarity": float(row["similarity"]),
            }
            for row in rows
        ]

    async def insert(
        self,
        content: str,
        embedding: list[float],
        source_type: str,
        source_id: str,
        metadata: Optional[dict] = None,
    ) -> str:
        """Insert a new embedding record. Returns the new record's UUID."""
        await self._ensure_pool()

        new_id = str(uuid.uuid4())
        await self.pg.execute(
            """INSERT INTO embeddings (id, source_type, source_id, content, embedding, metadata)
               VALUES ($1, $2, $3, $4, $5, $6::jsonb)""",
            new_id,
            source_type,
            source_id,
            content,
            embedding,
            json.dumps(metadata or {}),
        )
        return new_id

    async def delete_by_source(self, source_type: str, source_id: str) -> int:
        """Delete all embeddings for a given source. Returns count deleted."""
        result = await self.pg.execute(
            "DELETE FROM embeddings WHERE source_type = $1 AND source_id = $2",
            source_type,
            source_id,
        )
        # asyncpg returns a string like "DELETE N" — parse count
        try:
            return int(result.split()[-1])
        except (ValueError, IndexError):
            return 0
