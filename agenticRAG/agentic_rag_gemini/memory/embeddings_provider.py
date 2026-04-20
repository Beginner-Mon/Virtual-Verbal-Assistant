"""LangChain embeddings provider.

Replaces the custom EmbeddingService with LangChain's HuggingFaceEmbeddings.
Provides a singleton factory and backwards-compatible EmbeddingService class.

Usage:
    from memory.embeddings_provider import get_embeddings, EmbeddingService

    embeddings = get_embeddings()                     # LangChain Embeddings
    vec = embeddings.embed_query("hello")             # single query
    vecs = embeddings.embed_documents(["a", "b"])     # batch

    # Backwards-compatible:
    svc = EmbeddingService()
    vec = svc.embed_texts("hello")
    vecs = svc.embed_texts(["a", "b"])
"""

import os
from typing import List, Optional, Union

import numpy as np
from langchain_huggingface import HuggingFaceEmbeddings

from config import get_config
from utils.logger import get_logger

logger = get_logger(__name__)

# ---------------------------------------------------------------------------
# Singleton
# ---------------------------------------------------------------------------

_embeddings: Optional[HuggingFaceEmbeddings] = None


def get_embeddings(
    model_name: Optional[str] = None,
) -> HuggingFaceEmbeddings:
    """Get or create the global HuggingFaceEmbeddings singleton.

    Args:
        model_name: HuggingFace model name. Defaults to config value.

    Returns:
        HuggingFaceEmbeddings instance.
    """
    global _embeddings

    if model_name is None:
        model_name = get_config().embedding.model

    if _embeddings is not None:
        return _embeddings

    _embeddings = HuggingFaceEmbeddings(
        model_name=model_name,
        model_kwargs={"device": "cpu"},
        encode_kwargs={"normalize_embeddings": True},
    )

    logger.info("Embeddings provider: initialized HuggingFaceEmbeddings (model=%s)", model_name)
    return _embeddings


# ---------------------------------------------------------------------------
# Backwards-compatible EmbeddingService
# ---------------------------------------------------------------------------


class EmbeddingService:
    """Drop-in replacement for the old EmbeddingService.

    Wraps LangChain's HuggingFaceEmbeddings with the same public API:
      - embed_texts(str) -> List[float]
      - embed_texts(List[str]) -> List[List[float]]
      - compute_similarity(vec1, vec2) -> float
      - get_embedding_dimension() -> int
      - count_tokens(text) -> int
      - truncate_text(text, max_tokens) -> str
    """

    def __init__(self, config=None):
        """Initialize embedding service.

        Args:
            config: Configuration object. If None, loads from default config.
        """
        self.config = config or get_config().embedding
        self.model_name = self.config.model

        self._lc_embeddings = get_embeddings(self.model_name)

        # Determine dimension by embedding a test string
        test_vec = self._lc_embeddings.embed_query("dimension probe")
        self.embedding_dim = len(test_vec)
        self.model_type = "sentence_transformer"

        logger.info(
            "EmbeddingService (LangChain): model=%s dim=%d",
            self.model_name, self.embedding_dim,
        )

    def embed_texts(
        self,
        texts: Union[str, List[str]],
        batch_size: int = None,
    ) -> Union[List[float], List[List[float]]]:
        """Generate embeddings for one or more texts.

        Args:
            texts: Single text string or list of texts.
            batch_size: Ignored (LangChain handles batching internally).

        Returns:
            Single embedding vector (if input is string) or list of embeddings.
        """
        single_input = isinstance(texts, str)
        if single_input:
            vec = self._lc_embeddings.embed_query(texts)
            return vec

        vecs = self._lc_embeddings.embed_documents(texts)
        return vecs

    def embed_query(self, text: str) -> List[float]:
        """Generate embedding for a single query text.

        Args:
            text: Query text.

        Returns:
            Embedding vector.
        """
        return self._lc_embeddings.embed_query(text)

    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        """Generate embeddings for a list of documents.

        Args:
            texts: List of document texts.

        Returns:
            List of embedding vectors.
        """
        return self._lc_embeddings.embed_documents(texts)

    def compute_similarity(
        self,
        embedding1: List[float],
        embedding2: List[float],
    ) -> float:
        """Compute cosine similarity between two embeddings.

        Args:
            embedding1: First embedding vector.
            embedding2: Second embedding vector.

        Returns:
            Cosine similarity score (0 to 1).
        """
        vec1 = np.array(embedding1)
        vec2 = np.array(embedding2)
        similarity = np.dot(vec1, vec2) / (np.linalg.norm(vec1) * np.linalg.norm(vec2))
        return float(similarity)

    def get_embedding_dimension(self) -> int:
        """Get the dimension of embeddings produced by this service.

        Returns:
            Embedding dimension.
        """
        return self.embedding_dim

    def count_tokens(self, text: str) -> int:
        """Count tokens in text (approximate via word count).

        Args:
            text: Input text.

        Returns:
            Number of tokens.
        """
        try:
            import tiktoken
            encoding = tiktoken.get_encoding("cl100k_base")
            return len(encoding.encode(text))
        except Exception:
            return len(text.split())

    def truncate_text(self, text: str, max_tokens: int) -> str:
        """Truncate text to fit within token limit.

        Args:
            text: Input text.
            max_tokens: Maximum number of tokens.

        Returns:
            Truncated text.
        """
        try:
            import tiktoken
            encoding = tiktoken.get_encoding("cl100k_base")
            tokens = encoding.encode(text)
            if len(tokens) <= max_tokens:
                return text
            return encoding.decode(tokens[:max_tokens])
        except Exception:
            words = text.split()
            if len(words) <= max_tokens:
                return text
            return " ".join(words[:max_tokens])
