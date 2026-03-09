"""DocumentRetrievalTool — wraps DocumentStore with a clean single-method interface."""

from typing import List, Dict, Any, Optional

from memory.document_store import DocumentStore
from utils.logger import get_logger

logger = get_logger(__name__)


class DocumentRetrievalTool:
    """Tool that searches uploaded documents via DocumentStore.

    The OrchestratorAgent calls this tool when the decision includes
    document retrieval (actions: call_llm, hybrid).
    """

    def __init__(self, document_store: DocumentStore) -> None:
        """Inject the shared DocumentStore instance.

        Args:
            document_store: Shared DocumentStore instance.
        """
        self._document_store = document_store

    def search_documents(
        self,
        query: str,
        user_id: Optional[str] = None,
        top_k: int = 5,
        max_chunks_per_document: int = 3,
    ) -> List[Dict[str, Any]]:
        """Search documents for chunks relevant to the given query.

        Args:
            query:                  Search query (the user's question).
            user_id:                Optional user ID to restrict search to
                                    that user's documents only.
            top_k:                  Maximum number of chunks to return.
            max_chunks_per_document: Maximum chunks kept per unique document
                                    (deduplication handled by DocumentStore).

        Returns:
            List of chunk dicts, each containing at minimum:
            {"document": str, "source_type": str, "similarity": float, "metadata": dict}
        """
        logger.debug(f"[DocumentRetrievalTool] search_documents query={query[:60]}...")
        try:
            results = self._document_store.search_documents(
                query=query,
                user_id=user_id,
                top_k=top_k,
                max_chunks_per_document=max_chunks_per_document,
            )
            logger.debug(f"[DocumentRetrievalTool] retrieved {len(results)} chunks")
            return results
        except Exception as exc:
            logger.error(f"[DocumentRetrievalTool] search_documents failed: {exc}")
            return []
