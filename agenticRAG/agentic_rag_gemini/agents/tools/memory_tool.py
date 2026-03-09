"""MemoryTool — wraps MemoryManager with a clean single-method interface."""

from typing import List, Dict, Any, Optional

from memory.memory_manager import MemoryManager
from utils.logger import get_logger

logger = get_logger(__name__)


class MemoryTool:
    """Tool that retrieves relevant user memory from the MemoryManager.

    The OrchestratorAgent calls this tool when the decision includes
    memory retrieval (actions: retrieve_memory, call_llm, hybrid).
    """

    def __init__(self, memory_manager: MemoryManager) -> None:
        """Inject the shared MemoryManager instance.

        Args:
            memory_manager: Shared MemoryManager instance.
        """
        self._memory_manager = memory_manager

    def retrieve_memory(
        self,
        user_id: str,
        query: str,
        top_k: int = 5,
        memory_types: Optional[List[str]] = None,
    ) -> List[Dict[str, Any]]:
        """Retrieve semantically relevant memory items for the given user and query.

        Args:
            user_id:      User identifier.
            query:        The current user query (used for semantic search).
            top_k:        Maximum number of memory items to return.
            memory_types: Optional filter by memory type
                          (e.g. ["interaction", "summary"]).

        Returns:
            List of memory item dicts, each containing at minimum:
            {"document": str, "source_type": str, "similarity": float, "metadata": dict}
        """
        logger.debug(f"[MemoryTool] retrieve_memory user={user_id} query={query[:60]}...")
        try:
            results = self._memory_manager.retrieve_relevant_memory(
                user_id=user_id,
                query=query,
                top_k=top_k,
                memory_types=memory_types,
            )
            logger.debug(f"[MemoryTool] retrieved {len(results)} items")
            return results
        except Exception as exc:
            logger.error(f"[MemoryTool] retrieve_memory failed: {exc}")
            return []
