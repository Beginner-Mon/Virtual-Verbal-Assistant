"""Web Search Service - Search the web using DuckDuckGo API.

This module provides web search functionality as a fallback when local documents
don't provide enough context for answering user queries.

DuckDuckGo is used because:
- Free with no API key required
- Privacy-focused
- Good quality results
"""

import time
from typing import List, Dict, Optional
from utils.logger import get_logger

logger = get_logger(__name__)


def _load_ddgs_class():
    """Lazy-load the DDGS class at call time (not module import time).
    
    This avoids issues with Streamlit's ScriptRunner caching module-level
    imports: if the package is installed after the process starts, the
    module-level variable would remain False forever.
    """
    try:
        from ddgs import DDGS
        logger.debug("Loaded DDGS from 'ddgs' package")
        return DDGS
    except Exception as e1:
        logger.debug(f"ddgs not available: {type(e1).__name__}: {e1}")
    
    try:
        from duckduckgo_search import DDGS
        logger.debug("Loaded DDGS from deprecated 'duckduckgo_search' package")
        return DDGS
    except Exception as e2:
        logger.debug(f"duckduckgo_search not available: {type(e2).__name__}: {e2}")
    
    return None


class WebSearchService:
    """Search the web using DuckDuckGo (free, no API key required).
    
    This service provides web search functionality for the RAG pipeline
    when local documents don't have enough context.
    
    Example:
        service = WebSearchService()
        results = service.search("exercises for back pain relief")
        context = service.search_and_summarize("yoga stretches for stress")
    """
    
    def __init__(self, max_results: int = 5, timeout: int = 10):
        """Initialize web search service.
        
        Args:
            max_results: Maximum number of search results to return
            timeout: Request timeout in seconds
        """
        self.max_results = max_results
        self.timeout = timeout
        # Lazy-load: resolve the DDGS class at init time
        self._ddgs_class = _load_ddgs_class()
        self.enabled = self._ddgs_class is not None
        
        if self.enabled:
            logger.info(f"WebSearchService initialized (max_results={max_results})")
        else:
            logger.warning("WebSearchService disabled — install with: pip install ddgs")
    
    def is_available(self) -> bool:
        """Check if web search is available.
        
        Returns:
            True if ddgs is installed and service is enabled
        """
        # Re-check on every call in case the package was installed after init
        if not self.enabled:
            self._ddgs_class = _load_ddgs_class()
            self.enabled = self._ddgs_class is not None
        return self.enabled
    
    def search(
        self, 
        query: str, 
        max_results: Optional[int] = None,
        region: str = "wt-wt"  # Worldwide
    ) -> List[Dict[str, str]]:
        """Search the web and return results.
        
        Args:
            query: Search query
            max_results: Override default max results
            region: DuckDuckGo region code (default worldwide)
            
        Returns:
            List of dicts with keys: title, url, snippet
        """
        if not self.is_available():
            logger.warning("Web search attempted but service is disabled")
            return []
        
        max_results = max_results or self.max_results
        
        try:
            logger.info(f"Web search: '{query[:50]}...' (max={max_results})")
            
            ddgs = self._ddgs_class()
            results = list(ddgs.text(
                query,
                region=region,
                max_results=max_results
            ))
            
            # Format results
            formatted = []
            for r in results:
                formatted.append({
                    "title": r.get("title", ""),
                    "url": r.get("href", r.get("link", "")),
                    "snippet": r.get("body", r.get("snippet", ""))
                })
            
            logger.info(f"Web search returned {len(formatted)} results")
            return formatted
            
        except Exception as e:
            logger.error(f"Web search failed: {type(e).__name__}: {str(e)}")
            return []
    
    def search_and_summarize(
        self, 
        query: str,
        max_results: Optional[int] = None
    ) -> str:
        """Search and return formatted context string for RAG.
        
        This method formats search results into a context string suitable
        for injection into the RAG prompt.
        
        Args:
            query: Search query
            max_results: Override default max results
            
        Returns:
            Formatted context string with search results
        """
        results = self.search(query, max_results)
        
        if not results:
            return ""
        
        # Format results as context
        context_parts = ["### Web Search Results\n"]
        
        for i, r in enumerate(results, 1):
            title = r.get("title", "Untitled")
            url = r.get("url", "")
            snippet = r.get("snippet", "")
            
            context_parts.append(f"**{i}. {title}**")
            if snippet:
                context_parts.append(f"   {snippet}")
            if url:
                context_parts.append(f"   Source: {url}")
            context_parts.append("")
        
        return "\n".join(context_parts)
    
    def search_health_topics(
        self, 
        query: str,
        max_results: Optional[int] = None
    ) -> str:
        """Search specifically for health/exercise related topics.
        
        Adds health-focused keywords to improve result relevance.
        
        Args:
            query: Search query
            max_results: Override default max results
            
        Returns:
            Formatted context string with search results
        """
        # Enhance query for health topics
        enhanced_query = f"{query} exercises tutorial guide"
        return self.search_and_summarize(enhanced_query, max_results)


# Singleton instance
_web_search_service: Optional[WebSearchService] = None


def get_web_search_service(max_results: int = 5, timeout: int = 10) -> WebSearchService:
    """Get the singleton WebSearchService instance.
    
    Args:
        max_results: Maximum number of search results
        timeout: Request timeout in seconds
        
    Returns:
        WebSearchService instance
    """
    global _web_search_service
    if _web_search_service is None:
        _web_search_service = WebSearchService(max_results=max_results, timeout=timeout)
    return _web_search_service


if __name__ == "__main__":
    # Test the service
    service = WebSearchService()
    
    if service.is_available():
        print("Testing web search...")
        results = service.search("exercises for lower back pain relief")
        
        print(f"\nFound {len(results)} results:\n")
        for r in results:
            print(f"- {r['title']}")
            print(f"  {r['snippet'][:100]}...")
            print(f"  {r['url']}\n")
        
        print("\n--- Formatted Context ---")
        context = service.search_and_summarize("yoga stretches for stress relief")
        print(context)
    else:
        print("Web search not available. Install: pip install ddgs")
