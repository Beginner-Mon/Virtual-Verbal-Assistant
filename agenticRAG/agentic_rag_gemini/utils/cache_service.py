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
        orchestrator_ttl: int = 3_600, # 1 hour
        motion_prompt_ttl: int = 604_800,  # 7 days — HumanML3D vocabulary is static
    ):
        self._embedding_ttl = embedding_ttl
        self._retrieval_ttl = retrieval_ttl
        self._memory_ttl = memory_ttl
        self._orchestrator_ttl = orchestrator_ttl
        self._motion_prompt_ttl = motion_prompt_ttl
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

    # ------------------------------------------------------------------
    # D) Orchestrator Decision Cache (per-user)
    #    Cache key = f"{user_id}:{query.lower().strip()}" (as requested)
    # ------------------------------------------------------------------

    def get_orchestrator(self, user_id: str, query: str) -> Optional[Dict[str, Any]]:
        """Look up cached orchestrator decision for a specific user."""
        if not self._available:
            return None
        try:
            # Using specific key format requested by user
            clean_query = query.lower().strip()
            key = f"orc:{user_id}:{clean_query}"
            data = self._redis.get(key)
            return pickle.loads(data) if data else None
        except Exception as exc:
            logger.debug(f"Orchestrator cache read error: {exc}")
            return None

    def set_orchestrator(self, user_id: str, query: str, decision: Dict[str, Any]) -> None:
        """Store orchestrator decision in the cache."""
        if not self._available:
            return
        try:
            clean_query = query.lower().strip()
            key = f"orc:{user_id}:{clean_query}"
            self._redis.setex(key, self._orchestrator_ttl, pickle.dumps(decision))
        except Exception as exc:
            logger.debug(f"Orchestrator cache write error: {exc}")

    # ------------------------------------------------------------------
    # E) Semantic Cache (Query Transformation & HyDE)
    #    Caches the expanded query and generated HyDE document to skip LLM calls.
    # ------------------------------------------------------------------

    def get_semantic_transformation(self, query: str) -> Optional[Dict[str, str]]:
        """Look up cached query expansion and HyDE documents.

        Returns None on cache miss or if Redis is unavailable.
        """
        if not self._available:
            return None
        try:
            nq = normalize_query(query)
            key = f"sem:{self._hash(nq)}"
            data = self._redis.get(key)
            return pickle.loads(data) if data else None
        except Exception as exc:
            logger.debug(f"Semantic cache read error: {exc}")
            return None

    def set_semantic_transformation(self, query: str, transformation: Dict[str, str]) -> None:
        """Store query expansion and HyDE documents in the cache."""
        if not self._available:
            return
        try:
            nq = normalize_query(query)
            key = f"sem:{self._hash(nq)}"
            self._redis.setex(key, self._motion_prompt_ttl, pickle.dumps(transformation))
        except Exception as exc:
            logger.debug(f"Semantic cache write error: {exc}")

    # ------------------------------------------------------------------
    # F) Motion Prompt Cache (Semantic Bridge)
    #    NOT user-specific: the HumanML3D vocabulary mapping is identical
    #    for all users, so we cache by query text only.
    #    TTL: 7 days (the knowledge base is static).
    # ------------------------------------------------------------------

    def get_motion_prompt(self, query: str) -> Optional[str]:
        """Look up a cached Semantic Bridge rewritten prompt.

        Returns None on cache miss or if Redis is unavailable.
        """
        if not self._available:
            return None
        try:
            nq = normalize_query(query)
            key = f"mpt:{self._hash(nq)}"
            data = self._redis.get(key)
            if data:
                logger.debug(f"Motion prompt cache HIT for: '{query[:40]}'")
                return data.decode("utf-8")
            return None
        except Exception as exc:
            logger.debug(f"Motion prompt cache read error: {exc}")
            return None

    def set_motion_prompt(self, query: str, rewritten: str) -> None:
        """Store a Semantic Bridge rewritten prompt in the cache."""
        if not self._available:
            return
        try:
            nq = normalize_query(query)
            key = f"mpt:{self._hash(nq)}"
            self._redis.setex(key, self._motion_prompt_ttl, rewritten.encode("utf-8"))
            logger.debug(f"Motion prompt cached: '{query[:40]}' → '{rewritten[:40]}'")
        except Exception as exc:
            logger.debug(f"Motion prompt cache write error: {exc}")

    # ------------------------------------------------------------------
    # F) Motion Generation Result Cache
    #    Cache key = f"{prompt}|{duration}|{num_steps}"
    #    Stores the final relative video url (e.g. "/static/videos/...")
    # ------------------------------------------------------------------

    def get_motion_result(self, prompt: str, duration: float, num_steps: int) -> Optional[str]:
        """Look up a ready-to-use motion video URL by its generation parameters."""
        if not self._available:
            return None
        try:
            key_str = f"{prompt}|{duration}|{num_steps}"
            key = f"mres:{self._hash(key_str)}"
            data = self._redis.get(key)
            if data:
                logger.debug(f"Motion result Cache HIT for: '{prompt[:40]}'")
                return data.decode("utf-8")
            return None
        except Exception as exc:
            logger.debug(f"Motion result cache read error: {exc}")
            return None

    def set_motion_result(self, prompt: str, duration: float, num_steps: int, video_url: str) -> None:
        """Store a generated motion video URL."""
        if not self._available:
            return
        try:
            key_str = f"{prompt}|{duration}|{num_steps}"
            key = f"mres:{self._hash(key_str)}"
            self._redis.setex(key, self._motion_prompt_ttl, video_url.encode("utf-8"))
            logger.debug(f"Motion result cached: '{prompt[:40]}' → {video_url}")
        except Exception as exc:
            logger.debug(f"Motion result cache write error: {exc}")
