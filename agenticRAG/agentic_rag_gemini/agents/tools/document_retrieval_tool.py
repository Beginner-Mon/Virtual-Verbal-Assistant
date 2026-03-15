"""DocumentRetrievalTool — wraps DocumentStore with a clean single-method interface.

Supports multiple search methods:
1. Vector search (semantic similarity via embeddings) - DEFAULT
2. Fuzzy matching (keyword matching with typo tolerance)
3. Hybrid (combines both methods)
"""

from typing import List, Dict, Any, Optional

from memory.document_store import DocumentStore
from agents.tools.fuzzy_document_retriever import get_fuzzy_retriever
from utils.logger import get_logger

logger = get_logger(__name__)


class DocumentRetrievalTool:
    """Tool that searches documents via multiple methods.

    The OrchestratorAgent calls this tool when the decision includes
    document retrieval (actions: call_llm, hybrid, knowledge_query).
    
    Supports:
    - Vector search (semantic, requires embeddings)
    - Fuzzy matching (keyword, typo-tolerant)
    - Hybrid (combines both)
    """

    def __init__(self, document_store: DocumentStore) -> None:
        """Inject the shared DocumentStore instance.

        Args:
            document_store: Shared DocumentStore instance.
        """
        self._document_store = document_store
        self._fuzzy_retriever = get_fuzzy_retriever()

    def search_documents(
        self,
        query: str,
        user_id: Optional[str] = None,
        top_k: int = 5,
        max_chunks_per_document: int = 3,
        search_method: str = "vector",  # or "fuzzy" or "hybrid"
    ) -> List[Dict[str, Any]]:
        """Search documents using specified method.

        Args:
            query:                  Search query (the user's question).
            user_id:                Optional user ID to restrict search to
                                    that user's documents only.
            top_k:                  Maximum number of chunks/documents to return.
            max_chunks_per_document: Maximum chunks kept per unique document
                                    (only for vector search).
            search_method:          "vector" (default), "fuzzy", or "hybrid"

        Returns:
            List of result dicts, each containing at minimum:
            {"document": str, "source_type": str, "similarity": float, "metadata": dict}
            
            Additional fields for fuzzy search:
            {"score": float, "title": str, "target_body_part": str, ...}
        """
        logger.debug(
            f"[DocumentRetrievalTool] search_documents "
            f"query='{query[:60]}...', method={search_method}"
        )
        
        # Choose search method
        if search_method == "fuzzy":
            return self._search_fuzzy(query, top_k)
        elif search_method == "hybrid":
            return self._search_hybrid(query, user_id, top_k, max_chunks_per_document)
        else:  # "vector" or default
            return self._search_vector(query, user_id, top_k, max_chunks_per_document)
    
    def _search_vector(
        self,
        query: str,
        user_id: Optional[str] = None,
        top_k: int = 5,
        max_chunks_per_document: int = 3,
    ) -> List[Dict[str, Any]]:
        """Vector search using embeddings (semantic similarity).
        
        Args:
            query: Search query
            user_id: Optional user restriction
            top_k: Number of results to return
            max_chunks_per_document: Max chunks per document
            
        Returns:
            Vector search results
        """
        try:
            results = self._document_store.search_documents(
                query=query,
                user_id=user_id,
                top_k=top_k,
                max_chunks_per_document=max_chunks_per_document,
            )
            logger.debug(f"[DocumentRetrievalTool] Vector search retrieved {len(results)} chunks")
            return results
        except Exception as exc:
            logger.error(f"[DocumentRetrievalTool] Vector search failed: {exc}")
            return []
    
    def _search_fuzzy(
        self,
        query: str,
        top_k: int = 5,
    ) -> List[Dict[str, Any]]:
        """Fuzzy matching search (keyword with typo tolerance).
        
        Uses rapidfuzz for fuzzy matching against documents.txt.
        More suitable for exact exercise names with typo tolerance.
        
        Args:
            query: Search query
            top_k: Number of results to return
            
        Returns:
            Fuzzy match results with scores (0-100)
        """
        try:
            results = self._fuzzy_retriever.search_fuzzy(query, top_k=top_k)
            logger.debug(f"[DocumentRetrievalTool] Fuzzy search retrieved {len(results)} results")
            
            # Add source_type for consistency with vector search
            for result in results:
                result["source_type"] = "fuzzy_match"
                if "similarity" not in result:
                    result["similarity"] = result.get("score", 0) / 100.0
            
            return results
        except Exception as exc:
            logger.error(f"[DocumentRetrievalTool] Fuzzy search failed: {exc}")
            return []
    
    def _search_hybrid(
        self,
        query: str,
        user_id: Optional[str] = None,
        top_k: int = 5,
        max_chunks_per_document: int = 3,
    ) -> List[Dict[str, Any]]:
        """Hybrid search combining fuzzy + vector results.
        
        Uses both fuzzy matching and vector search, then combines and 
        re-ranks results for best coverage.
        
        Args:
            query: Search query
            user_id: Optional user restriction
            top_k: Number of results to return
            max_chunks_per_document: Max chunks per document (vector search)
            
        Returns:
            Combined and re-ranked results
        """
        try:
            # Get vector results
            vector_results = self._search_vector(
                query, user_id, top_k, max_chunks_per_document
            )
            
            # Get fuzzy results  
            fuzzy_results = self._search_fuzzy(query, top_k)
            
            # Combine them
            results = self._fuzzy_retriever.search_hybrid(
                query,
                vector_results=vector_results,
                top_k=top_k,
                vector_weight=0.6,  # 60% weight to vector, 40% to fuzzy
            )
            
            logger.debug(
                f"[DocumentRetrievalTool] Hybrid search combined "
                f"{len(vector_results)} vector + {len(fuzzy_results)} fuzzy "
                f"→ {len(results)} final results"
            )
            
            # Add source_type for tracking
            for result in results:
                result["source_type"] = "hybrid"
            
            return results
        except Exception as exc:
            logger.error(f"[DocumentRetrievalTool] Hybrid search failed: {exc}")
            return []
