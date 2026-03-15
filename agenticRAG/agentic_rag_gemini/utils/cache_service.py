"""Redis caching layer for embeddings, knowledge retrieval, and user memory.

Provides three cache types:
  A) Embedding cache    — key: emb:sha256(normalized_query | model)
  B) Knowledge cache    — key: ret:sha256(normalized_query | top_k | collection)
  C) Memory cache       — key: mem:sha256(user_id | normalized_query)

Safe fallback: if Redis is unreachable, all methods silently no-op.
"""

import hashlib
import pickle
import re
import unicodedata
from typing import Any, Dict, List, Optional

from utils.logger import get_logger

logger = get_logger(__name__)

# ---------------------------------------------------------------------------
# Redis import with graceful fallback
# ---------------------------------------------------------------------------
try:
    import redis
except ImportError:
    redis = None  # type: ignore[assignment]
    logger.warning("redis package not installed — caching disabled")


# ---------------------------------------------------------------------------
# Query normalisation
# ---------------------------------------------------------------------------

def normalize_query(text: str) -> str:
    """Normalize a query string for consistent cache keys.

    Steps: lowercase → strip → collapse whitespace → remove punctuation → NFC.

    Examples:
        "What exercises  help Knee Pain?" → "what exercises help knee pain"
        "  NECK   pain!!! " → "neck pain"
    """
    text = text.lower().strip()
    text = re.sub(r"\s+", " ", text)
    text = re.sub(r"[^\w\s]", "", text)
    return unicodedata.normalize("NFC", text)


# ---------------------------------------------------------------------------
# CacheService
# ---------------------------------------------------------------------------

class CacheService:
    """Redis-backed cache with safe fallback when Redis is unavailable."""

    def __init__(
        self,
        host: str = "localhost",
        port: int = 6379,
        db: int = 0,
        embedding_ttl: int = 21_600,   # 6 hours
        retrieval_ttl: int = 3_600,    # 1 hour
        memory_ttl: int = 1_800,       # 30 minutes
    ):
        self._embedding_ttl = embedding_ttl
        self._retrieval_ttl = retrieval_ttl
        self._memory_ttl = memory_ttl
        self._available = False

        if redis is None:
            logger.warning("CacheService disabled — redis package not installed")
            return

        try:
            self._redis: Any = redis.Redis(
                host=host,
                port=port,
                db=db,
                decode_responses=False,
                socket_connect_timeout=2,
            )
            self._redis.ping()
            self._available = True
            logger.info(f"Redis cache connected ({host}:{port}/{db})")
        except Exception as exc:
            self._redis = None
            logger.warning(f"Redis unavailable — caching disabled: {exc}")

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _hash(text: str) -> str:
        """SHA-256 hash for cache key."""
        return hashlib.sha256(text.encode()).hexdigest()

    # ------------------------------------------------------------------
    # A) Embedding Cache
    #    Key includes model name → safe if embedding model changes.
    # ------------------------------------------------------------------

    def get_embedding(self, query: str, model: str) -> Optional[List[float]]:
        """Look up a cached embedding vector.

        Returns None on cache miss or if Redis is unavailable.
        """
        if not self._available:
            return None
        try:
            nq = normalize_query(query)
            data = self._redis.get(f"emb:{self._hash(f'{nq}|{model}')}")
            return pickle.loads(data) if data else None
        except Exception as exc:
            logger.debug(f"Embedding cache read error: {exc}")
            return None

    def set_embedding(
        self, query: str, model: str, embedding: List[float]
    ) -> None:
        """Store an embedding vector in the cache."""
        if not self._available:
            return
        try:
            nq = normalize_query(query)
            key = f"emb:{self._hash(f'{nq}|{model}')}"
            self._redis.setex(key, self._embedding_ttl, pickle.dumps(embedding))
        except Exception as exc:
            logger.debug(f"Embedding cache write error: {exc}")

    # ------------------------------------------------------------------
    # B) Knowledge Retrieval Cache
    #    NOT user-specific: knowledge base results are identical for all
    #    users, so excluding user_id enables cross-user cache reuse.
    # ------------------------------------------------------------------

    def get_retrieval(
        self,
        query: str,
        top_k: int,
        collection: str = "default",
    ) -> Optional[List[Dict]]:
        """Look up cached knowledge retrieval results.

        Returns None on cache miss or if Redis is unavailable.
        """
        if not self._available:
            return None
        try:
            nq = normalize_query(query)
            key = f"ret:{self._hash(f'{nq}|{top_k}|{collection}')}"
            data = self._redis.get(key)
            return pickle.loads(data) if data else None
        except Exception as exc:
            logger.debug(f"Retrieval cache read error: {exc}")
            return None

    def set_retrieval(
        self,
        query: str,
        top_k: int,
        results: List[Dict],
        collection: str = "default",
    ) -> None:
        """Store knowledge retrieval results in the cache."""
        if not self._available:
            return
        try:
            nq = normalize_query(query)
            key = f"ret:{self._hash(f'{nq}|{top_k}|{collection}')}"
            self._redis.setex(key, self._retrieval_ttl, pickle.dumps(results))
        except Exception as exc:
            logger.debug(f"Retrieval cache write error: {exc}")

    # ------------------------------------------------------------------
    # C) Memory Retrieval Cache (user-specific)
    #    Memory depends on user identity → cache key MUST include user_id.
    # ------------------------------------------------------------------

    def get_memory(
        self, user_id: str, query: str
    ) -> Optional[List[Dict]]:
        """Look up cached memory retrieval results for a specific user.

        Returns None on cache miss or if Redis is unavailable.
        """
        if not self._available:
            return None
        try:
            nq = normalize_query(query)
            key = f"mem:{self._hash(f'{user_id}|{nq}')}"
            data = self._redis.get(key)
            return pickle.loads(data) if data else None
        except Exception as exc:
            logger.debug(f"Memory cache read error: {exc}")
            return None

    def set_memory(
        self, user_id: str, query: str, results: List[Dict]
    ) -> None:
        """Store memory retrieval results for a specific user."""
        if not self._available:
            return
        try:
            nq = normalize_query(query)
            key = f"mem:{self._hash(f'{user_id}|{nq}')}"
            self._redis.setex(key, self._memory_ttl, pickle.dumps(results))
        except Exception as exc:
            logger.debug(f"Memory cache write error: {exc}")
