"""WebSearchTool — wraps WebSearchService with a clean single-method interface."""

from typing import Optional

from utils.web_search import WebSearchService
from utils.logger import get_logger

logger = get_logger(__name__)


class WebSearchTool:
    """Tool that performs web searches via WebSearchService (DuckDuckGo).

    The OrchestratorAgent calls this tool only when the decision is
    'hybrid' — meaning local context is insufficient and an external
    web search is needed to supplement the answer.
    """

    def __init__(self, web_service: WebSearchService) -> None:
        """Inject the shared WebSearchService instance.

        Args:
            web_service: Shared WebSearchService instance.
        """
        self._web_service = web_service

    def search_web(self, query: str, max_results: Optional[int] = None) -> str:
        """Search the web and return a formatted context string.

        Args:
            query:       The query to search for (typically the user's question).
            max_results: Override the service default max result count.

        Returns:
            Formatted multi-line context string ready for injection into the RAG
            prompt, or an empty string if the service is unavailable or the
            search fails.
        """
        logger.debug(f"[WebSearchTool] search_web query={query[:60]}...")
        try:
            if not self._web_service.is_available():
                logger.warning("[WebSearchTool] WebSearchService is not available")
                return ""
            result = self._web_service.search_and_summarize(
                query=query,
                max_results=max_results,
            )
            logger.debug(
                f"[WebSearchTool] search_web returned {len(result)} chars"
            )
            return result
        except Exception as exc:
            logger.error(f"[WebSearchTool] search_web failed: {exc}")
            return ""
