"""Shared singletons for LangGraph nodes — avoids re-creating heavy objects.

Import from here in any node that needs E5EmbeddingService or PostgresClient.
Both are lazy-initialized on first use.
"""

_embedding_service = None
_pg_client = None


def get_embedding_service():
    """Get e5-small embedding service (D10). Returns E5EmbeddingService.

    Replaces old memory.embedding_service.EmbeddingService.
    Auto-applies query:/passage: prefix per e5 model requirements.
    """
    global _embedding_service
    if _embedding_service is None:
        from langgraph_agents.shared.embedding import get_embedding
        _embedding_service = get_embedding()
    return _embedding_service


def get_pg_client():
    """Get async PostgreSQL client singleton."""
    global _pg_client
    if _pg_client is None:
        from langgraph_agents.db.postgres import PostgresClient
        _pg_client = PostgresClient()
    return _pg_client
