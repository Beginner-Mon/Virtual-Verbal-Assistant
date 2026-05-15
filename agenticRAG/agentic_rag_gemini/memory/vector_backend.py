"""Vector store backend abstraction.

Until now the codebase had three call-sites that knew about ChromaDB and two that
knew about Pinecone, with ad-hoc env-var gating in each.  This module unifies
that via a small ``VectorBackend`` ABC plus per-collection routing.

Routing policy (default, configurable via env):
    - ``humanml3d_library`` and ``medquad_library`` (large, read-mostly) →
      Pinecone if credentials are present, else Chroma, else JSONL fallback.
    - ``user_*`` collections (per-user docs / memory) → Chroma always.

The adapters are thin: they expose only the operations actually used by the
project (similarity ``query``, optional ``upsert``, ``count``, ``health``).
Heavyweight legacy stores (``vector_store.py``, ``document_store.py``) keep
their existing internals; this module is opt-in for new call sites and for
``MotionCandidateRetriever``-style use cases.
"""

from __future__ import annotations

import os
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Sequence

from utils.logger import get_logger

logger = get_logger(__name__)


# ── Public types ──────────────────────────────────────────────────────────────


@dataclass
class VectorMatch:
    """Single similarity-search hit, normalised across backends."""

    id: str
    score: float
    text: str = ""
    metadata: Dict[str, Any] = None  # type: ignore[assignment]

    def __post_init__(self) -> None:
        if self.metadata is None:
            self.metadata = {}


class VectorBackend(ABC):
    """Read-mostly vector store adapter."""

    name: str = "abstract"

    @abstractmethod
    def query(self, collection: str, embedding: Sequence[float], top_k: int = 5) -> List[VectorMatch]:
        ...

    @abstractmethod
    def count(self, collection: str) -> int:
        ...

    @abstractmethod
    def health(self) -> Dict[str, Any]:
        ...


# ── ChromaDB adapter ──────────────────────────────────────────────────────────


class ChromaBackend(VectorBackend):
    name = "chromadb"

    def __init__(self, host: Optional[str] = None, port: Optional[int] = None) -> None:
        self._host = host or os.getenv("CHROMA_HOST", "localhost")
        self._port = int(port or os.getenv("CHROMA_PORT", "8100"))
        self._client = None
        self._collections: Dict[str, Any] = {}

    def _client_or_init(self):
        if self._client is not None:
            return self._client
        try:
            import chromadb
            from chromadb.config import Settings

            self._client = chromadb.HttpClient(
                host=self._host,
                port=self._port,
                settings=Settings(anonymized_telemetry=False),
            )
            self._client.heartbeat()
            return self._client
        except Exception as exc:  # noqa: BLE001
            logger.warning("ChromaBackend: HttpClient init failed (%s)", exc)
            self._client = None
            return None

    def _get_collection(self, name: str):
        if name in self._collections:
            return self._collections[name]
        client = self._client_or_init()
        if client is None:
            return None
        try:
            col = client.get_collection(name=name, embedding_function=None)
            self._collections[name] = col
            return col
        except Exception as exc:  # noqa: BLE001
            logger.warning("ChromaBackend: collection '%s' unavailable (%s)", name, exc)
            return None

    def query(self, collection: str, embedding: Sequence[float], top_k: int = 5) -> List[VectorMatch]:
        col = self._get_collection(collection)
        if col is None:
            return []
        try:
            results = col.query(query_embeddings=[list(embedding)], n_results=int(top_k))
        except Exception as exc:  # noqa: BLE001
            logger.warning("ChromaBackend.query failed for '%s': %s", collection, exc)
            return []

        out: List[VectorMatch] = []
        ids = (results.get("ids") or [[]])[0]
        docs = (results.get("documents") or [[]])[0]
        dists = (results.get("distances") or [[]])[0] if "distances" in results else [0.0] * len(ids)
        metas = (results.get("metadatas") or [[]])[0] if "metadatas" in results else [{}] * len(ids)
        for i, _id in enumerate(ids):
            dist = float(dists[i]) if i < len(dists) else 0.0
            similarity = max(0.0, 1.0 - dist / 2.0)
            out.append(
                VectorMatch(
                    id=str(_id),
                    score=similarity,
                    text=str(docs[i]) if i < len(docs) else "",
                    metadata=dict(metas[i]) if i < len(metas) and metas[i] else {},
                )
            )
        return out

    def count(self, collection: str) -> int:
        col = self._get_collection(collection)
        if col is None:
            return 0
        try:
            return int(col.count())
        except Exception:  # noqa: BLE001
            return 0

    def health(self) -> Dict[str, Any]:
        client = self._client_or_init()
        return {
            "backend": self.name,
            "host": self._host,
            "port": self._port,
            "available": client is not None,
        }


# ── Pinecone adapter ──────────────────────────────────────────────────────────


class PineconeBackend(VectorBackend):
    name = "pinecone"

    def __init__(
        self,
        api_key: Optional[str] = None,
        index_name: Optional[str] = None,
        index_host: Optional[str] = None,
    ) -> None:
        self._api_key = api_key or os.getenv("PINECONE_API_KEY")
        self._index_name = index_name or os.getenv("PINECONE_INDEX_NAME", "kinetichat")
        self._index_host = index_host or os.getenv("PINECONE_INDEX_HOST")
        self._index = None

    def _index_or_init(self):
        if self._index is not None:
            return self._index
        if not self._api_key:
            logger.info("PineconeBackend: PINECONE_API_KEY not set — backend disabled")
            return None
        try:
            from pinecone import Pinecone

            pc = Pinecone(api_key=self._api_key)
            if self._index_host:
                self._index = pc.Index(name=self._index_name, host=self._index_host)
            else:
                self._index = pc.Index(name=self._index_name)
            return self._index
        except Exception as exc:  # noqa: BLE001
            logger.warning("PineconeBackend: init failed (%s)", exc)
            self._index = None
            return None

    def query(self, collection: str, embedding: Sequence[float], top_k: int = 5) -> List[VectorMatch]:
        index = self._index_or_init()
        if index is None:
            return []
        try:
            res = index.query(
                vector=list(embedding),
                top_k=int(top_k),
                namespace=collection,
                include_metadata=True,
            )
        except Exception as exc:  # noqa: BLE001
            logger.warning("PineconeBackend.query failed for '%s': %s", collection, exc)
            return []

        out: List[VectorMatch] = []
        for match in res.get("matches", []) or []:
            meta = match.get("metadata") or {}
            out.append(
                VectorMatch(
                    id=str(match.get("id", "")),
                    score=float(match.get("score", 0.0)),
                    text=str(meta.get("text", "")),
                    metadata=dict(meta),
                )
            )
        return out

    def count(self, collection: str) -> int:
        index = self._index_or_init()
        if index is None:
            return 0
        try:
            stats = index.describe_index_stats()
            ns = stats.get("namespaces", {}) or {}
            return int((ns.get(collection) or {}).get("vector_count", 0))
        except Exception:  # noqa: BLE001
            return 0

    def health(self) -> Dict[str, Any]:
        return {
            "backend": self.name,
            "index_name": self._index_name,
            "configured": bool(self._api_key),
            "available": self._index_or_init() is not None,
        }


# ── Hybrid router ─────────────────────────────────────────────────────────────


# Collection name prefixes that should always go to Chroma (per-user data).
_USER_COLLECTION_PREFIXES = ("user_",)

# Public read-mostly collections that should prefer Pinecone.
_PUBLIC_COLLECTIONS = {"humanml3d_library", "medquad_library"}


class HybridVectorBackend(VectorBackend):
    """Routes per-collection between Chroma and Pinecone with graceful fallback."""

    name = "hybrid"

    def __init__(
        self,
        chroma: Optional[ChromaBackend] = None,
        pinecone: Optional[PineconeBackend] = None,
        prefer_pinecone_for: Optional[set] = None,
    ) -> None:
        self._chroma = chroma or ChromaBackend()
        self._pinecone = pinecone or PineconeBackend()
        self._prefer_pinecone_for = (
            prefer_pinecone_for if prefer_pinecone_for is not None else _PUBLIC_COLLECTIONS
        )

    def _backend_for(self, collection: str) -> VectorBackend:
        # Per-user collections always go to Chroma (privacy + locality).
        if any(collection.startswith(p) for p in _USER_COLLECTION_PREFIXES):
            return self._chroma
        # Public collections prefer Pinecone if the index is reachable.
        if collection in self._prefer_pinecone_for and self._pinecone._index_or_init() is not None:
            return self._pinecone
        # Otherwise fall back to Chroma.
        return self._chroma

    def query(self, collection: str, embedding: Sequence[float], top_k: int = 5) -> List[VectorMatch]:
        primary = self._backend_for(collection)
        results = primary.query(collection, embedding, top_k)
        if results or primary is self._chroma:
            return results
        # If Pinecone returned empty, give Chroma a shot for resilience.
        logger.info(
            "HybridVectorBackend: '%s' empty on %s, retrying ChromaBackend",
            collection,
            primary.name,
        )
        return self._chroma.query(collection, embedding, top_k)

    def count(self, collection: str) -> int:
        return self._backend_for(collection).count(collection)

    def health(self) -> Dict[str, Any]:
        return {
            "backend": self.name,
            "chroma": self._chroma.health(),
            "pinecone": self._pinecone.health(),
            "prefer_pinecone_for": sorted(self._prefer_pinecone_for),
        }


# ── Factory ──────────────────────────────────────────────────────────────────


_default_backend: Optional[VectorBackend] = None


def get_vector_backend() -> VectorBackend:
    """Return the process-wide vector backend, selected by ``VECTOR_DB_TYPE``.

    Values:
        ``hybrid``   — HybridVectorBackend (recommended; respects per-collection routing).
        ``pinecone`` — Pinecone only.
        ``chromadb`` — Chroma only (default for backward compatibility).
    """
    global _default_backend
    if _default_backend is not None:
        return _default_backend

    kind = os.getenv("VECTOR_DB_TYPE", "chromadb").strip().lower()
    if kind == "pinecone":
        _default_backend = PineconeBackend()
    elif kind == "hybrid":
        _default_backend = HybridVectorBackend()
    else:
        _default_backend = ChromaBackend()

    logger.info("VectorBackend: selected '%s'", _default_backend.name)
    return _default_backend


def reset_vector_backend() -> None:
    """Drop the cached backend — used in tests."""
    global _default_backend
    _default_backend = None


__all__ = [
    "ChromaBackend",
    "HybridVectorBackend",
    "PineconeBackend",
    "VectorBackend",
    "VectorMatch",
    "get_vector_backend",
    "reset_vector_backend",
]
