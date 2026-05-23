"""Shared singletons for LangGraph nodes — avoids re-creating heavy objects.

Import from here in any node that needs EmbeddingService or PostgresClient.
Both are lazy-initialized on first use.
"""

_embedding_service = None
_pg_client = None


def get_embedding_service():
    global _embedding_service
    if _embedding_service is None:
        from memory.embedding_service import EmbeddingService
        _embedding_service = EmbeddingService()
    return _embedding_service


def get_pg_client():
    global _pg_client
    if _pg_client is None:
        from langgraph_agents.db.postgres import PostgresClient
        _pg_client = PostgresClient()
    return _pg_client
