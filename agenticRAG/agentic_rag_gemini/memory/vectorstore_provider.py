"""LangChain Pinecone vector store provider.

Replaces the custom dual-backend VectorStore (881 lines) with a Pinecone-only
provider using langchain-pinecone.

Provides lazy-initialized PineconeVectorStore instances for each namespace:
  - conversations
  - documents
  - chat_summaries
  - humanml3d_library  (public / read-only)

User isolation is achieved via Pinecone metadata filters.

Usage:
    from memory.vectorstore_provider import (
        get_conversations_store,
        get_documents_store,
        get_chat_summaries_store,
        get_public_store,
        VectorStore,               # backwards-compat class
    )
"""

import os
import uuid
from datetime import datetime
from typing import Any, Dict, List, Optional

from langchain_pinecone import PineconeVectorStore as LCPineconeVectorStore
from pinecone import Pinecone

from memory.embeddings_provider import get_embeddings
from config import get_config
from utils.logger import get_logger

logger = get_logger(__name__)

# ---------------------------------------------------------------------------
# Namespace constants (matching current production data)
# ---------------------------------------------------------------------------
NS_CONVERSATIONS = "conversations"
NS_DOCUMENTS = "documents"
NS_CHAT_SUMMARIES = "chat_summaries"
NS_PUBLIC = "humanml3d_library"

DEFAULT_GUEST_USER_ID = "guest"

# ---------------------------------------------------------------------------
# Singleton state
# ---------------------------------------------------------------------------

_pinecone_client: Optional[Pinecone] = None
_pinecone_index = None
_index_name: Optional[str] = None

_stores: Dict[str, LCPineconeVectorStore] = {}


def _init_pinecone():
    """Initialize the Pinecone client and index singleton."""
    global _pinecone_client, _pinecone_index, _index_name

    if _pinecone_index is not None:
        return

    config = get_config().vector_database
    pc_config = config.pinecone

    api_key = os.getenv("PINECONE_API_KEY") or pc_config.get("api_key", "")
    _index_name = os.getenv("PINECONE_INDEX_NAME") or pc_config.get("index_name", "kinetichat")
    index_host = os.getenv("PINECONE_INDEX_HOST") or pc_config.get("index_host", "")

    if not api_key:
        raise ValueError(
            "Pinecone API key is required. "
            "Set PINECONE_API_KEY env var or pinecone.api_key in config.yaml"
        )

    _pinecone_client = Pinecone(api_key=api_key)

    if index_host:
        _pinecone_index = _pinecone_client.Index(name=_index_name, host=index_host)
    else:
        _pinecone_index = _pinecone_client.Index(name=_index_name)

    stats = _pinecone_index.describe_index_stats()
    logger.info(
        "Pinecone connected: index=%s, total_vectors=%d, namespaces=%s",
        _index_name,
        stats.get("total_vector_count", 0),
        list(stats.get("namespaces", {}).keys()),
    )


def _get_store(namespace: str) -> LCPineconeVectorStore:
    """Get or create a PineconeVectorStore for the given namespace.

    Args:
        namespace: Pinecone namespace name.

    Returns:
        LCPineconeVectorStore instance.
    """
    if namespace in _stores:
        return _stores[namespace]

    _init_pinecone()

    store = LCPineconeVectorStore(
        index=_pinecone_index,
        embedding=get_embeddings(),
        namespace=namespace,
        text_key="text",
    )

    _stores[namespace] = store
    logger.info("VectorStore provider: created store for namespace='%s'", namespace)
    return store


# ---------------------------------------------------------------------------
# Public namespace accessors
# ---------------------------------------------------------------------------


def get_conversations_store() -> LCPineconeVectorStore:
    """Get the vector store for conversation memory."""
    return _get_store(NS_CONVERSATIONS)


def get_documents_store() -> LCPineconeVectorStore:
    """Get the vector store for uploaded documents."""
    return _get_store(NS_DOCUMENTS)


def get_chat_summaries_store() -> LCPineconeVectorStore:
    """Get the vector store for chat session summaries."""
    return _get_store(NS_CHAT_SUMMARIES)


def get_public_store() -> LCPineconeVectorStore:
    """Get the vector store for the shared HumanML3D library."""
    return _get_store(NS_PUBLIC)


def get_pinecone_index():
    """Get the raw Pinecone index (for advanced operations)."""
    _init_pinecone()
    return _pinecone_index


# ---------------------------------------------------------------------------
# Helper: resolve user_id
# ---------------------------------------------------------------------------


def _resolve_user_id(user_id: Optional[str]) -> str:
    """Normalize user_id, defaulting to 'guest'."""
    value = str(user_id or DEFAULT_GUEST_USER_ID).strip()
    return value or DEFAULT_GUEST_USER_ID


# ---------------------------------------------------------------------------
# Backwards-compatible VectorStore class
# ---------------------------------------------------------------------------


class VectorStore:
    """Backwards-compatible wrapper matching the old VectorStore API.

    Delegates to LangChain PineconeVectorStore instances per namespace.
    All ChromaDB code paths have been removed; Pinecone is the sole backend.
    """

    def __init__(self, config=None):
        """Initialize vector store.

        Args:
            config: Configuration object (ignored — uses global config).
        """
        _init_pinecone()
        self.db_type = "pinecone"
        logger.info("VectorStore (LangChain): initialized with Pinecone backend")

    # ── add_documents ────────────────────────────────────────────────

    def add_documents(
        self,
        documents: List[str],
        embeddings: List[List[float]],
        metadata: Optional[List[Dict[str, Any]]] = None,
        ids: Optional[List[str]] = None,
        collection_type: str = "conversations",
        user_id: Optional[str] = None,
    ) -> List[str]:
        """Add documents to vector store.

        Args:
            documents:       List of text documents.
            embeddings:      List of embedding vectors.
            metadata:        Optional metadata dicts per document.
            ids:             Optional document IDs.
            collection_type: Namespace type (conversations, documents, etc.)
            user_id:         User identifier for scoping.

        Returns:
            List of document IDs.
        """
        if ids is None:
            ids = [str(uuid.uuid4()) for _ in documents]

        if metadata is None:
            metadata = [{} for _ in documents]
        else:
            metadata = [m or {} for m in metadata]

        for meta in metadata:
            if "timestamp" not in meta:
                meta["timestamp"] = datetime.now().isoformat()

        resolved_uid = _resolve_user_id(user_id or metadata[0].get("user_id"))

        ns = self._map_namespace(collection_type)
        index = get_pinecone_index()

        vectors = []
        for id_, doc, embedding, meta in zip(ids, documents, embeddings, metadata):
            pinecone_meta = {
                "text": doc,
                "user_id": resolved_uid,
                **meta,
            }
            vectors.append((id_, embedding, pinecone_meta))

        index.upsert(vectors=vectors, namespace=ns)

        logger.info(
            "Added %d documents (namespace=%s, user_id=%s)",
            len(documents), ns, resolved_uid,
        )
        return ids

    # ── search (conversations) ───────────────────────────────────────

    def search(
        self,
        query_embedding: List[float],
        top_k: int = 5,
        filter_metadata: Optional[Dict[str, Any]] = None,
        user_id: Optional[str] = None,
    ) -> List[Dict[str, Any]]:
        """Search for similar documents in conversations namespace."""
        return self._pinecone_search(
            collection_type="conversations",
            query_embedding=query_embedding,
            top_k=top_k,
            filter_metadata=filter_metadata,
            user_id=user_id,
        )

    # ── add_document (single, documents namespace) ───────────────────

    def add_document(
        self,
        document: str,
        embedding: List[float],
        metadata: Optional[Dict[str, Any]] = None,
        id: Optional[str] = None,
    ) -> str:
        """Add a single document to the documents collection."""
        ids = self.add_documents(
            documents=[document],
            embeddings=[embedding],
            metadata=[metadata],
            ids=[id] if id else None,
            collection_type="documents",
        )
        return ids[0]

    # ── add_public_documents ─────────────────────────────────────────

    def add_public_documents(
        self,
        documents: List[str],
        embeddings: List[List[float]],
        metadata: Optional[List[Dict[str, Any]]] = None,
        ids: Optional[List[str]] = None,
    ) -> List[str]:
        """Add shared read-only library entries to the public namespace."""
        if ids is None:
            ids = [str(uuid.uuid4()) for _ in documents]
        if metadata is None:
            metadata = [{} for _ in documents]

        index = get_pinecone_index()
        vectors = []
        for id_, doc, embedding, meta in zip(ids, documents, embeddings, metadata):
            vectors.append((id_, embedding, {"text": doc, **(meta or {})}))

        index.upsert(vectors=vectors, namespace=NS_PUBLIC)
        logger.info("Added %d entries to public namespace", len(documents))
        return ids

    # ── search_documents ─────────────────────────────────────────────

    def search_documents(
        self,
        query_embedding: List[float],
        top_k: int = 5,
        filter_metadata: Optional[Dict[str, Any]] = None,
        user_id: Optional[str] = None,
        include_public: bool = False,
    ) -> List[Dict[str, Any]]:
        """Search documents namespace, optionally including public library."""
        results = self._pinecone_search(
            collection_type="documents",
            query_embedding=query_embedding,
            top_k=top_k,
            filter_metadata=filter_metadata,
            user_id=user_id,
        )

        if include_public:
            public_results = self._pinecone_search(
                collection_type="public",
                query_embedding=query_embedding,
                top_k=top_k,
            )
            for r in public_results:
                r["metadata"]["library_scope"] = "public"
            results.extend(public_results)
            results.sort(key=lambda x: x.get("similarity") or 0.0, reverse=True)
            return results[:top_k]

        return results

    # ── add_chat_summary ─────────────────────────────────────────────

    def add_chat_summary(
        self,
        summary: str,
        embedding: List[float],
        metadata: Optional[Dict[str, Any]] = None,
        id: Optional[str] = None,
        user_id: Optional[str] = None,
    ) -> str:
        """Add a single chat summary."""
        if id is None:
            id = str(uuid.uuid4())
        if metadata is None:
            metadata = {}
        if "timestamp" not in metadata:
            metadata["timestamp"] = datetime.now().isoformat()

        resolved_uid = _resolve_user_id(user_id or metadata.get("user_id"))
        index = get_pinecone_index()

        pinecone_meta = {
            "text": summary,
            "user_id": resolved_uid,
            **metadata,
        }
        index.upsert(
            vectors=[(id, embedding, pinecone_meta)],
            namespace=NS_CHAT_SUMMARIES,
        )
        logger.info("Added chat summary ID=%s", id)
        return id

    # ── search_chat_summaries ────────────────────────────────────────

    def search_chat_summaries(
        self,
        query_embedding: List[float],
        top_k: int = 5,
        filter_metadata: Optional[Dict[str, Any]] = None,
        user_id: Optional[str] = None,
    ) -> List[Dict[str, Any]]:
        """Search chat summaries namespace."""
        return self._pinecone_search(
            collection_type="chat_summaries",
            query_embedding=query_embedding,
            top_k=top_k,
            filter_metadata=filter_metadata,
            user_id=user_id,
        )

    # ── delete_documents ─────────────────────────────────────────────

    def delete_documents(
        self,
        ids: List[str],
        user_id: Optional[str] = None,
        collection_type: str = "conversations",
    ) -> bool:
        """Delete documents from vector store."""
        ns = self._map_namespace(collection_type)
        index = get_pinecone_index()
        index.delete(ids=ids, namespace=ns)
        logger.info("Deleted %d documents from namespace=%s", len(ids), ns)
        return True

    # ── clear_all_data ───────────────────────────────────────────────

    def clear_all_data(self) -> bool:
        """Clear all data from all namespaces."""
        try:
            index = get_pinecone_index()
            for ns in (NS_CONVERSATIONS, NS_DOCUMENTS, NS_CHAT_SUMMARIES):
                try:
                    index.delete(delete_all=True, namespace=ns)
                    logger.info("Cleared Pinecone namespace: %s", ns)
                except Exception as e:
                    logger.warning("Error clearing namespace %s: %s", ns, e)
            logger.info("Successfully cleared all Pinecone data")
            return True
        except Exception as e:
            logger.error("Failed to clear vector store data: %s", e)
            return False

    # ── reset_collections ────────────────────────────────────────────

    def reset_collections(self) -> bool:
        """Reset namespaces by deleting all data."""
        return self.clear_all_data()

    # ── update_metadata ──────────────────────────────────────────────

    def update_metadata(
        self,
        id: str,
        metadata: Dict[str, Any],
        user_id: Optional[str] = None,
    ) -> bool:
        """Update metadata for a document."""
        index = get_pinecone_index()
        result = index.fetch(ids=[id], namespace=NS_CONVERSATIONS)
        vectors = result.get("vectors", {})
        if id in vectors:
            existing = vectors[id]
            merged_meta = {**(existing.get("metadata", {})), **metadata}
            index.upsert(
                vectors=[(id, existing.get("values", []), merged_meta)],
                namespace=NS_CONVERSATIONS,
            )
        logger.info("Updated metadata for document %s", id)
        return True

    # ── get_documents_collection / get_conversations_collection ──────

    def get_documents_collection(self, user_id: Optional[str] = None):
        """Get the documents store (for backwards compat)."""
        return get_documents_store()

    def get_conversations_collection(self, user_id: Optional[str] = None):
        """Get the conversations store (for backwards compat)."""
        return get_conversations_store()

    # ── Internal helpers ─────────────────────────────────────────────

    def _map_namespace(self, collection_type: str) -> str:
        """Map logical collection type to Pinecone namespace."""
        mapping = {
            "conversations": NS_CONVERSATIONS,
            "documents": NS_DOCUMENTS,
            "chat_summaries": NS_CHAT_SUMMARIES,
            "public": NS_PUBLIC,
        }
        return mapping.get(collection_type, NS_CONVERSATIONS)

    def _pinecone_search(
        self,
        collection_type: str,
        query_embedding: List[float],
        top_k: int = 5,
        filter_metadata: Optional[Dict[str, Any]] = None,
        user_id: Optional[str] = None,
    ) -> List[Dict[str, Any]]:
        """Generic Pinecone similarity search with metadata filters."""
        ns = self._map_namespace(collection_type)
        index = get_pinecone_index()

        pinecone_filter: Dict[str, Any] = {}

        # User-scoped filter (skip for public namespace)
        if collection_type != "public":
            resolved_uid = _resolve_user_id(
                user_id or (filter_metadata or {}).get("user_id")
            )
            pinecone_filter["user_id"] = {"$eq": resolved_uid}

        # Extra metadata filters
        extra = dict(filter_metadata or {})
        extra.pop("user_id", None)
        for k, v in extra.items():
            pinecone_filter[k] = {"$eq": v}

        results = index.query(
            vector=query_embedding,
            top_k=top_k,
            namespace=ns,
            filter=pinecone_filter if pinecone_filter else None,
            include_metadata=True,
        )

        formatted: List[Dict[str, Any]] = []
        for match in results.get("matches", []):
            meta = match.get("metadata", {})
            formatted.append({
                "id": match["id"],
                "document": meta.get("text", ""),
                "metadata": {k: v for k, v in meta.items() if k != "text"},
                "distance": 1 - match["score"],
                "similarity": match["score"],
            })
        return formatted
