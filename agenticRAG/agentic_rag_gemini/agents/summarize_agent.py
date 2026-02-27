"""Summarize Agent — Condenses chat sessions into concise summaries.

Called when the user switches away from a session or starts a new conversation.
The summary is embedded into the chat_summaries ChromaDB collection so the LLM
can recall context from past sessions via fuzzy matching.
"""

from typing import List, Dict, Any, Optional

from utils.logger import get_logger
from utils.gemini_client import GeminiClientWrapper
from utils.prompt_templates import SESSION_SUMMARY_PROMPTS
from memory.embedding_service import EmbeddingService
from memory.vector_store import VectorStore

logger = get_logger(__name__)


class SummarizeAgent:
    """Generates and stores session summaries."""

    def __init__(
        self,
        client: Optional[GeminiClientWrapper] = None,
        vector_store: Optional[VectorStore] = None,
        embedding_service: Optional[EmbeddingService] = None,
    ):
        self.client = client or GeminiClientWrapper()
        self.vector_store = vector_store
        self.embedding_service = embedding_service

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def summarize_session(self, messages: List[Dict[str, Any]]) -> str:
        """Summarize a list of chat messages into a concise paragraph.

        Args:
            messages: List of message dicts with 'role' and 'content'.

        Returns:
            Summary string (3-5 sentences).
        """
        if not messages:
            return ""

        transcript = self._format_transcript(messages)
        prompt = SESSION_SUMMARY_PROMPTS["summarize"].format(transcript=transcript)

        try:
            summary = self.client.generate(
                prompt=prompt,
                system_instruction=SESSION_SUMMARY_PROMPTS["system"],
            )
            summary = summary.strip()
            logger.info(f"Generated session summary ({len(summary)} chars)")
            return summary
        except Exception as exc:
            logger.error(f"Summarization failed: {exc}")
            # Fallback: return a naive summary from the first user message
            first_user_msg = next(
                (m["content"] for m in messages if m.get("role") == "user"), ""
            )
            return f"Conversation about: {first_user_msg[:200]}"

    def store_summary(
        self,
        user_id: str,
        session_id: str,
        summary: str,
        session_title: str = "",
    ) -> Optional[str]:
        """Embed the summary in the chat_summaries vector collection.

        Args:
            user_id: User identifier.
            session_id: Session identifier.
            summary: Summary text to embed.
            session_title: Optional title for metadata.

        Returns:
            Document ID or None on failure.
        """
        if not self.vector_store or not self.embedding_service:
            logger.warning("Vector store or embedding service not set — skipping storage")
            return None

        try:
            embedding = self.embedding_service.embed_texts(summary)

            metadata = {
                "user_id": user_id,
                "session_id": session_id,
                "title": session_title,
                "type": "chat_summary",
                "timestamp": __import__("datetime").datetime.now().isoformat(),
            }

            ids = self.vector_store.add_chat_summary(
                document=summary,
                embedding=embedding,
                metadata=metadata,
                id=f"summary_{session_id}",
            )
            logger.info(f"Stored summary for session {session_id}")
            return ids
        except Exception as exc:
            logger.error(f"Failed to store session summary: {exc}")
            return None

    def summarize_and_store(
        self,
        user_id: str,
        session_id: str,
        messages: List[Dict[str, Any]],
        session_title: str = "",
    ) -> Optional[str]:
        """Convenience: summarize + embed in one call.

        Returns:
            The generated summary string, or None on failure.
        """
        summary = self.summarize_session(messages)
        if not summary:
            return None

        self.store_summary(user_id, session_id, summary, session_title)
        return summary

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _format_transcript(messages: List[Dict[str, Any]]) -> str:
        """Format messages into a readable transcript for the LLM."""
        lines = []
        for msg in messages:
            role = msg.get("role", "unknown").capitalize()
            content = msg.get("content", "")
            lines.append(f"{role}: {content}")
        return "\n".join(lines)
