"""Enhanced Document Retrieval Tool with Fuzzy Matching Support.

This module extends document retrieval with:
1. Vector-based search (semantic similarity via embeddings)
2. Fuzzy matching (keyword matching with typo tolerance)
3. Hybrid search (combining both methods)

Similar to ExerciseDetector, uses rapidfuzz for fuzzy matching.
"""

import re
import os
from typing import List, Dict, Any, Optional, Tuple
from pathlib import Path

from rapidfuzz import fuzz, process

from memory.document_store import DocumentStore
from utils.logger import get_logger

logger = get_logger(__name__)


class FuzzyDocumentRetriever:
    """Retrieve documents using fuzzy matching with typo tolerance.
    
    Loads exercise data from documents.txt and provides fuzzy search
    similar to ExerciseDetector but for general document retrieval.
    """
    
    def __init__(self, similarity_threshold: float = 70.0):
        """Initialize fuzzy retriever.
        
        Args:
            similarity_threshold: Minimum fuzzy match score (0-100) to accept
        """
        self.similarity_threshold = similarity_threshold
        self.documents: List[Dict[str, Any]] = []
        self.document_texts: List[str] = []
        
        self._load_documents()
        logger.info(
            f"FuzzyDocumentRetriever initialized with {len(self.documents)} documents, "
            f"threshold={self.similarity_threshold}%"
        )
    
    def _load_documents(self) -> None:
        """Load documents from documents.txt file."""
        try:
            # Path to documents.txt
            doc_path = os.path.join(
                os.path.dirname(os.path.dirname(__file__)),
                "data",
                "documents.txt"
            )
            
            if not os.path.exists(doc_path):
                logger.warning(f"documents.txt not found at {doc_path}")
                return
            
            with open(doc_path, 'r', encoding='utf-8') as f:
                content = f.read()
            
            # Parse documents (separated by "---")
            raw_docs = content.split("---")
            
            for i, doc_text in enumerate(raw_docs):
                doc_text = doc_text.strip()
                if not doc_text:
                    continue
                
                # Extract exercise name (first line often contains it)
                lines = doc_text.split('\n')
                title = lines[0].strip() if lines else f"Document {i}"
                
                # Extract key info (Type, Target Body Part, etc.)
                doc_dict = {
                    "id": i,
                    "title": title,
                    "content": doc_text,
                    "source": "documents.txt"
                }
                
                # Extract metadata fields
                for line in lines:
                    if line.startswith("Type:"):
                        doc_dict["type"] = line.replace("Type:", "").strip()
                    elif line.startswith("Target Body Part:"):
                        doc_dict["target_body_part"] = line.replace("Target Body Part:", "").strip()
                    elif line.startswith("Equipment:"):
                        doc_dict["equipment"] = line.replace("Equipment:", "").strip()
                    elif line.startswith("Difficulty Level:"):
                        doc_dict["difficulty"] = line.replace("Difficulty Level:", "").strip()
                
                self.documents.append(doc_dict)
                self.document_texts.append(doc_text)
            
            logger.info(f"Loaded {len(self.documents)} documents from documents.txt")
            
        except Exception as e:
            logger.error(f"Failed to load documents: {e}")
    
    def search_fuzzy(
        self,
        query: str,
        top_k: int = 5,
        match_scorer: str = "token_set_ratio",
    ) -> List[Dict[str, Any]]:
        """Search documents using fuzzy matching with typo tolerance.
        
        Args:
            query: Search query (e.g., "chin tuck", "shoulder excercise")
            top_k: Number of results to return
            match_scorer: Scoring algorithm:
                - "token_set_ratio": Best for phrase matching (default)
                - "ratio": Substring matching
                - "partial_ratio": Partial string matching
                - "token_sort_ratio": Ignores word order
        
        Returns:
            List of matching documents sorted by relevance:
            [{title, content, score, source, ...}, ...]
        """
        if not query or not self.documents:
            return []
        
        logger.debug(f"Fuzzy search: query='{query}', scorer={match_scorer}, threshold={self.similarity_threshold}%")
        
        try:
            # Choose scorer function
            scorer_map = {
                "token_set_ratio": fuzz.token_set_ratio,
                "ratio": fuzz.ratio,
                "partial_ratio": fuzz.partial_ratio,
                "token_sort_ratio": fuzz.token_sort_ratio,
            }
            scorer = scorer_map.get(match_scorer, fuzz.token_set_ratio)
            
            # Extract potential phrases from query for better matching
            search_phrases = self._extract_search_phrases(query)
            logger.debug(f"Search phrases extracted: {search_phrases}")
            
            # Score each document
            scored_results = []
            
            for doc in self.documents:
                doc_title = doc.get("title", "")
                doc_content = doc.get("content", "")
                
                # Score against title (highest priority)
                title_scores = [
                    scorer(phrase, doc_title)
                    for phrase in search_phrases
                ]
                title_score = max(title_scores) if title_scores else 0
                
                # Score against body part if available (high priority)
                target_body_part = doc.get("target_body_part", "")
                content_score = 0
                if target_body_part:
                    content_scores = [
                        scorer(phrase, target_body_part)
                        for phrase in search_phrases
                    ]
                    content_score = max(content_scores) if content_scores else 0
                
                # Use weighted score: title is primary, body part is secondary
                final_score = max(title_score, content_score * 0.8)
                
                if final_score >= self.similarity_threshold:
                    result = {
                        "title": doc_title,
                        "content": doc_content,
                        "score": final_score,
                        "source": doc.get("source", "documents.txt"),
                        "metadata": {k: v for k, v in doc.items() 
                                   if k not in ["title", "content", "source"]},
                        "document": doc_content,  # For compatibility with RAGPipeline
                        "similarity": final_score / 100.0,  # Also provide as 0-1 range
                    }
                    scored_results.append(result)
            
            # Sort by score descending
            scored_results.sort(key=lambda x: x["score"], reverse=True)
            results = scored_results[:top_k]
            
            logger.info(
                f"Fuzzy search returned {len(results)}/{len(self.documents)} documents "
                f"(top score: {results[0]['score']:.1f}% if results else 'N/A')"
            )
            
            return results
            
        except Exception as e:
            logger.error(f"Fuzzy search failed: {e}")
            return []
    
    def search_hybrid(
        self,
        query: str,
        vector_results: Optional[List[Dict[str, Any]]] = None,
        top_k: int = 5,
        vector_weight: float = 0.6,
    ) -> List[Dict[str, Any]]:
        """Hybrid search combining fuzzy matching + vector search results.
        
        Useful for combining best of both worlds:
        - Vector search: semantic understanding but requires embeddings
        - Fuzzy matching: exact keywords but keyword-based
        
        Args:
            query: Search query
            vector_results: Results from vector search (from DocumentStore)
            top_k: Number of results to return
            vector_weight: Weight for vector results (0-1)
            
        Returns:
            Combined and re-ranked results
        """
        # Get fuzzy results
        fuzzy_results = self.search_fuzzy(query, top_k=top_k)
        
        if not vector_results:
            return fuzzy_results
        
        # Create a map for deduplication
        result_map = {}
        
        # Add fuzzy results with fuzzy score
        fuzzy_weight = 1.0 - vector_weight
        for result in fuzzy_results:
            key = result["title"]
            result_map[key] = {
                **result,
                "fuzzy_score": result.get("score", 0),
                "vector_score": 0,
                "combined_score": result.get("score", 0) * fuzzy_weight,
            }
        
        # Add/merge vector results with vector score
        for result in vector_results[:top_k]:
            key = result.get("document", "").split('\n')[0]  # Extract first line as title
            vec_score = (result.get("similarity", 0) * 100)  # Convert 0-1 to 0-100
            
            if key in result_map:
                # Merge: both fuzzy and vector found this result
                result_map[key]["vector_score"] = vec_score
                result_map[key]["combined_score"] = (
                    result_map[key]["fuzzy_score"] * fuzzy_weight +
                    vec_score * vector_weight
                )
            else:
                # Only vector found this result
                result_map[key] = {
                    **result,
                    "fuzzy_score": 0,
                    "vector_score": vec_score,
                    "combined_score": vec_score * vector_weight,
                    "title": key,
                }
        
        # Sort by combined score
        combined_results = sorted(
            result_map.values(),
            key=lambda x: x["combined_score"],
            reverse=True
        )
        
        logger.info(
            f"Hybrid search combined {len(fuzzy_results)} fuzzy + "
            f"{len(vector_results)} vector results → {len(combined_results)} final results"
        )
        
        return combined_results[:top_k]
    
    def _extract_search_phrases(self, query: str) -> List[str]:
        """Extract search phrases from query for better fuzzy matching.
        
        Args:
            query: User query string
            
        Returns:
            List of phrases to fuzzy match
        """
        phrases = []
        
        # Full query
        phrases.append(query.lower().strip())
        
        # Extract phrases from common patterns
        patterns = [
            r'(?:exercise|workout|movement):\s*([a-z\s-]+)',
            r'(?:what|how|show|teach|demonstrate).*?(?:is|are|do)\s+(?:a|an|the|)?([a-z\s-]+)',
            r'([a-z\s-]+)\s+(?:exercise|workout|movement)',
        ]
        
        for pattern in patterns:
            matches = re.findall(pattern, query.lower(), re.IGNORECASE)
            phrases.extend(matches)
        
        # Also add individual words (for simple searches like "curl")
        words = query.lower().split()
        phrases.extend([w for w in words if len(w) >= 3])
        
        # Remove duplicates and empty strings
        phrases = list(set(p.strip() for p in phrases if p.strip()))
        
        return phrases
    
    def get_exercise_count(self) -> int:
        """Get total number of loaded documents.
        
        Returns:
            Number of exercises/documents
        """
        return len(self.documents)
    
    def list_documents(self, limit: int = 10) -> List[str]:
        """Get list of document titles.
        
        Args:
            limit: Maximum number to return
            
        Returns:
            List of document titles
        """
        return [doc.get("title", f"Doc {i}") 
                for i, doc in enumerate(self.documents[:limit])]


# Global instance for single initialization
_fuzzy_retriever: Optional[FuzzyDocumentRetriever] = None


def get_fuzzy_retriever() -> FuzzyDocumentRetriever:
    """Get or create global fuzzy retriever instance.
    
    Returns:
        FuzzyDocumentRetriever instance
    """
    global _fuzzy_retriever
    if _fuzzy_retriever is None:
        _fuzzy_retriever = FuzzyDocumentRetriever(similarity_threshold=70.0)
    return _fuzzy_retriever
