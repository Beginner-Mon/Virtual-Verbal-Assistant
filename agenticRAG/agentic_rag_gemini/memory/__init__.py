"""Memory utilities — Phase 2.5: only EmbeddingService + DocumentStore survive.

VectorStore / MemoryManager / SessionStore (ChromaDB + Firebase backends) were
removed; langgraph_agents owns vector + memory + session now.
"""

from memory.embedding_service import EmbeddingService

__all__ = ["EmbeddingService"]
