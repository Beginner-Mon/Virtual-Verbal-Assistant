"""Vector Store implementation for semantic memory storage.

This module provides vector database operations for storing and retrieving
embeddings of user conversations and context.

Supports two backends:
  - **ChromaDB** (local Docker or PersistentClient)
  - **Qdrant**   (local or Qdrant Cloud)

When using Qdrant, four named collections are created:
  conversations, documents, chat_summaries, humanml3d_library
User isolation is achieved via a ``user_id`` metadata filter (payload field)
rather than per-user collections.
"""

from typing import List, Dict, Any, Optional
from datetime import datetime
import uuid
import os

from config import get_config
from utils.logger import get_logger


logger = get_logger(__name__)


PUBLIC_COLLECTION_NAME = "humanml3d_library"
DEFAULT_GUEST_USER_ID = "guest"

# Qdrant collection names
_QDRANT_CONVERSATIONS = "conversations"
_QDRANT_DOCUMENTS = "documents"
_QDRANT_CHAT_SUMMARIES = "chat_summaries"
_QDRANT_PUBLIC = PUBLIC_COLLECTION_NAME


class VectorStore:
    """Vector database wrapper for memory storage and retrieval.
    
    Supports ChromaDB and Qdrant backends for storing document embeddings
    and performing similarity search.
    """
    
    def __init__(self, config=None):
        """Initialize vector store.
        
        Args:
            config: Configuration object. If None, loads from default config.
        """
        self.config = config or get_config().vector_database
        self.db_type = self.config.type
        
        if self.db_type == "chromadb":
            self._init_chromadb()
        elif self.db_type == "qdrant":
            self._init_qdrant()
        else:
            raise ValueError(f"Unsupported vector database type: {self.db_type}")
        
        logger.info(f"Initialized VectorStore with backend: {self.db_type}")

    # ==================================================================
    # ChromaDB initialisation (unchanged from original)
    # ==================================================================

    def _init_chromadb(self):
        """Initialize ChromaDB client and collections."""
        import chromadb
        from chromadb.config import Settings

        chroma_config = self.config.chromadb
        mode = str(chroma_config.get("mode", "http")).strip().lower()

        settings = Settings(
            anonymized_telemetry=False,
            allow_reset=False
        )

        if mode == "http":
            host = chroma_config.get("host", "localhost")
            port = int(chroma_config.get("port", 8100))
            self.client = chromadb.HttpClient(
                host=host,
                port=port,
                ssl=bool(chroma_config.get("ssl", False)),
                headers=chroma_config.get("headers") or {},
                settings=settings,
            )
            self._chroma_endpoint = f"{host}:{port}"
            logger.info(
                "Using ChromaDB HttpClient mode host=%s port=%s",
                host,
                port,
            )
        else:
            persist_directory = chroma_config.get("persist_directory", "./data/vector_store")
            self.client = chromadb.PersistentClient(
                path=persist_directory,
                settings=settings,
            )
            self._chroma_endpoint = persist_directory
            logger.info(
                "Using ChromaDB PersistentClient mode path=%s",
                persist_directory,
            )
        
        try:
            self.client.heartbeat()
        except Exception as exc:
            raise RuntimeError(
                "Cannot connect to ChromaDB server at "
                f"{self._chroma_endpoint}. Ensure ChromaDB is running "
                "(docker compose up -d chromadb)."
            ) from exc

        bootstrap_user = self._resolve_user_id(None)
        self.collection = self._get_chroma_collection("conversations", bootstrap_user, create_if_missing=True)
        self.documents_collection = self._get_chroma_collection("documents", bootstrap_user, create_if_missing=True)
        self.chat_summaries_collection = self._get_chroma_collection("chat_summaries", bootstrap_user, create_if_missing=True)
        self.public_collection = self._get_public_collection(create_if_missing=True)

        logger.info(
            "ChromaDB tenant bootstrap completed for user=%s with public collection=%s",
            bootstrap_user,
            PUBLIC_COLLECTION_NAME,
        )

    # ==================================================================
    # Qdrant initialisation (rewritten for cloud support)
    # ==================================================================

    def _init_qdrant(self):
        """Initialize Qdrant client and collections.

        Reads ``qdrant.url`` and ``qdrant.api_key`` from config / env.
        Creates four collections if they don't exist:
            conversations, documents, chat_summaries, humanml3d_library
        """
        from qdrant_client import QdrantClient
        from qdrant_client.models import Distance, VectorParams

        qdrant_config = self.config.qdrant

        url = os.getenv("QDRANT_URL") or qdrant_config.get("url", "http://localhost:6333")
        api_key = os.getenv("QDRANT_API_KEY") or qdrant_config.get("api_key") or None
        vector_size = int(qdrant_config.get("vector_size", 384))

        self.client = QdrantClient(url=url, api_key=api_key, timeout=30)
        self._qdrant_vector_size = vector_size
        logger.info("Qdrant client connected to %s", url)

        # Ensure all required collections exist
        existing = {c.name for c in self.client.get_collections().collections}

        for coll_name in (_QDRANT_CONVERSATIONS, _QDRANT_DOCUMENTS,
                          _QDRANT_CHAT_SUMMARIES, _QDRANT_PUBLIC):
            if coll_name not in existing:
                self.client.create_collection(
                    collection_name=coll_name,
                    vectors_config=VectorParams(
                        size=vector_size,
                        distance=Distance.COSINE,
                    ),
                )
                logger.info("Created Qdrant collection: %s", coll_name)
            else:
                logger.info("Qdrant collection already exists: %s", coll_name)

    # ==================================================================
    # Helpers
    # ==================================================================

    def _resolve_user_id(self, user_id: Optional[str]) -> str:
        value = str(user_id or DEFAULT_GUEST_USER_ID).strip()
        return value or DEFAULT_GUEST_USER_ID

    def _sanitize_user_id(self, user_id: str) -> str:
        return "".join(ch if ch.isalnum() or ch in {"_", "-"} else "_" for ch in user_id)

    def _base_collection_name(self, user_id: str) -> str:
        return f"user_{self._sanitize_user_id(user_id)}_collection"

    def _collection_name(self, collection_type: str, user_id: str) -> str:
        base = self._base_collection_name(user_id)
        if collection_type == "documents":
            return f"{base}_documents"
        if collection_type == "chat_summaries":
            return f"{base}_chat_summaries"
        return base

    def _get_chroma_collection(self, collection_type: str, user_id: str, create_if_missing: bool = True):
        name = self._collection_name(collection_type, user_id)
        metadata_map = {
            "conversations": {"description": f"Conversation memory for user {user_id}"},
            "documents": {"description": f"Uploaded document knowledge for user {user_id}"},
            "chat_summaries": {"description": f"Chat summaries for user {user_id}"},
        }
        if create_if_missing:
            return self.client.get_or_create_collection(
                name=name,
                metadata=metadata_map.get(collection_type, {"description": f"Collection for user {user_id}"}),
                embedding_function=None,
            )
        return self.client.get_collection(name=name, embedding_function=None)

    def _get_public_collection(self, create_if_missing: bool = True):
        if create_if_missing:
            return self.client.get_or_create_collection(
                name=PUBLIC_COLLECTION_NAME,
                metadata={"description": "Shared read-only HumanML3D knowledge library"},
                embedding_function=None,
            )
        return self.client.get_collection(name=PUBLIC_COLLECTION_NAME, embedding_function=None)

    # ------------------------------------------------------------------
    # Qdrant helper: map collection_type → Qdrant collection name
    # ------------------------------------------------------------------

    def _qdrant_collection(self, collection_type: str) -> str:
        """Return the Qdrant collection name for a logical collection type."""
        mapping = {
            "conversations": _QDRANT_CONVERSATIONS,
            "documents": _QDRANT_DOCUMENTS,
            "chat_summaries": _QDRANT_CHAT_SUMMARIES,
            "public": _QDRANT_PUBLIC,
        }
        return mapping.get(collection_type, _QDRANT_CONVERSATIONS)

    @staticmethod
    def _qdrant_uuid(raw_id: str) -> str:
        """Ensure *raw_id* is a valid UUID string (Qdrant requires UUIDs for named vectors).

        If the incoming ID is already a valid UUID it is returned as-is;
        otherwise we derive a deterministic UUID-5 from it.
        """
        try:
            uuid.UUID(raw_id)
            return raw_id
        except (ValueError, AttributeError):
            return str(uuid.uuid5(uuid.NAMESPACE_URL, str(raw_id)))

    # ==================================================================
    # add_documents
    # ==================================================================

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
            documents: List of document texts
            embeddings: List of embedding vectors
            metadata: List of metadata dictionaries for each document
            ids: Optional list of document IDs. Generated if not provided.
            collection_type: "conversations" or "documents"
            
        Returns:
            List of document IDs
        """
        if ids is None:
            ids = [str(uuid.uuid4()) for _ in documents]
        
        if metadata is None:
            metadata = [{} for _ in documents]
        else:
            metadata = [m or {} for m in metadata]
        
        # Add timestamp to metadata
        for meta in metadata:
            if "timestamp" not in meta:
                meta["timestamp"] = datetime.now().isoformat()
        
        resolved_user_id = self._resolve_user_id(user_id or metadata[0].get("user_id"))

        # Select collection based on type
        if self.db_type == "chromadb":
            collection = self._get_chroma_collection(collection_type, resolved_user_id, create_if_missing=True)
            collection.add(
                documents=documents,
                embeddings=embeddings,
                metadatas=metadata,
                ids=ids
            )
        elif self.db_type == "qdrant":
            from qdrant_client.models import PointStruct

            coll = self._qdrant_collection(collection_type)
            points = []
            for i, (id_, doc, embedding, meta) in enumerate(zip(ids, documents, embeddings, metadata)):
                payload = {
                    "text": doc,
                    "user_id": resolved_user_id,
                    **meta,
                }
                points.append(
                    PointStruct(
                        id=self._qdrant_uuid(id_),
                        vector=embedding,
                        payload=payload,
                    )
                )
            self.client.upsert(collection_name=coll, points=points)
        
        logger.info(
            "Added %d documents to vector store (collection_type=%s user_id=%s)",
            len(documents),
            collection_type,
            resolved_user_id,
        )
        return ids

    # ==================================================================
    # search (conversations)
    # ==================================================================

    def search(
        self,
        query_embedding: List[float],
        top_k: int = 5,
        filter_metadata: Optional[Dict[str, Any]] = None,
        user_id: Optional[str] = None,
    ) -> List[Dict[str, Any]]:
        """Search for similar documents in vector store."""
        if self.db_type == "chromadb":
            user_from_filter = (filter_metadata or {}).get("user_id")
            resolved_user_id = self._resolve_user_id(user_id or user_from_filter)
            collection = self._get_chroma_collection("conversations", resolved_user_id, create_if_missing=True)

            # Handle multiple filter conditions with $and operator
            where_clause = dict(filter_metadata or {})
            where_clause.pop("user_id", None)
            if not where_clause:
                where_clause = None
            if filter_metadata and len(filter_metadata) > 1:
                where_clause = {"$and": [{k: v} for k, v in where_clause.items()]}
            
            results = collection.query(
                query_embeddings=[query_embedding],
                n_results=top_k,
                where=where_clause
            )
            
            # Format results
            formatted_results = []
            for i in range(len(results["ids"][0])):
                formatted_results.append({
                    "id": results["ids"][0][i],
                    "document": results["documents"][0][i],
                    "metadata": results["metadatas"][0][i],
                    "distance": results["distances"][0][i] if "distances" in results else None,
                    "similarity": max(0.0, 1.0 - (results["distances"][0][i] / 2.0)) if "distances" in results else None
                })
            
            return formatted_results
        
        elif self.db_type == "qdrant":
            return self._qdrant_search(
                collection_type="conversations",
                query_embedding=query_embedding,
                top_k=top_k,
                filter_metadata=filter_metadata,
                user_id=user_id,
            )
        
        # Fallback for unsupported db types
        logger.warning(f"Unsupported database type for search: {self.db_type}")
        return []

    # ==================================================================
    # add_document (single, to documents collection)
    # ==================================================================

    def add_document(
        self,
        document: str,
        embedding: List[float],
        metadata: Optional[Dict[str, Any]] = None,
        id: Optional[str] = None
    ) -> str:
        """Add a single document to the documents collection."""
        ids = self.add_documents(
            documents=[document],
            embeddings=[embedding],
            metadata=[metadata],
            ids=[id] if id else None,
            collection_type="documents"
        )
        return ids[0]

    # ==================================================================
    # add_public_documents
    # ==================================================================

    def add_public_documents(
        self,
        documents: List[str],
        embeddings: List[List[float]],
        metadata: Optional[List[Dict[str, Any]]] = None,
        ids: Optional[List[str]] = None,
    ) -> List[str]:
        """Add shared read-only library entries to the public collection."""
        if ids is None:
            ids = [str(uuid.uuid4()) for _ in documents]
        if metadata is None:
            metadata = [{} for _ in documents]

        if self.db_type == "chromadb":
            collection = self._get_public_collection(create_if_missing=True)
            collection.add(
                documents=documents,
                embeddings=embeddings,
                metadatas=metadata,
                ids=ids,
            )
        elif self.db_type == "qdrant":
            from qdrant_client.models import PointStruct

            points = []
            for id_, doc, embedding, meta in zip(ids, documents, embeddings, metadata):
                points.append(
                    PointStruct(
                        id=self._qdrant_uuid(id_),
                        vector=embedding,
                        payload={"text": doc, **(meta or {})},
                    )
                )
            self.client.upsert(collection_name=_QDRANT_PUBLIC, points=points)

        logger.info("Added %d entries to public collection %s", len(documents), PUBLIC_COLLECTION_NAME)
        return ids

    # ==================================================================
    # search_documents
    # ==================================================================

    def search_documents(
        self,
        query_embedding: List[float],
        top_k: int = 5,
        filter_metadata: Optional[Dict[str, Any]] = None,
        user_id: Optional[str] = None,
        include_public: bool = False,
    ) -> List[Dict[str, Any]]:
        """Search documents collection specifically."""
        if self.db_type == "chromadb":
            user_from_filter = (filter_metadata or {}).get("user_id")
            resolved_user_id = self._resolve_user_id(user_id or user_from_filter)
            documents_collection = self._get_chroma_collection("documents", resolved_user_id, create_if_missing=True)

            # Handle multiple filter conditions with $and operator
            where_clause = dict(filter_metadata or {})
            where_clause.pop("user_id", None)
            if not where_clause:
                where_clause = None
            if filter_metadata and len(filter_metadata) > 1:
                where_clause = {"$and": [{k: v} for k, v in where_clause.items()]}
            
            results = documents_collection.query(
                query_embeddings=[query_embedding],
                n_results=top_k,
                where=where_clause
            )
            
            # Format results
            formatted_results = []
            for i in range(len(results["ids"][0])):
                formatted_results.append({
                    "id": results["ids"][0][i],
                    "document": results["documents"][0][i],
                    "metadata": results["metadatas"][0][i],
                    "distance": results["distances"][0][i] if "distances" in results else None,
                    "similarity": max(0.0, 1.0 - (results["distances"][0][i] / 2.0)) if "distances" in results else None
                })
            
            if include_public:
                public_collection = self._get_public_collection(create_if_missing=True)
                public_results = public_collection.query(
                    query_embeddings=[query_embedding],
                    n_results=top_k,
                )
                for i in range(len(public_results["ids"][0])):
                    formatted_results.append({
                        "id": public_results["ids"][0][i],
                        "document": public_results["documents"][0][i],
                        "metadata": {
                            **(public_results["metadatas"][0][i] or {}),
                            "library_scope": "public",
                        },
                        "distance": public_results["distances"][0][i] if "distances" in public_results else None,
                        "similarity": max(0.0, 1.0 - (public_results["distances"][0][i] / 2.0)) if "distances" in public_results else None,
                    })
                formatted_results.sort(key=lambda item: item.get("similarity") or 0.0, reverse=True)
                return formatted_results[:top_k]

            return formatted_results

        elif self.db_type == "qdrant":
            results = self._qdrant_search(
                collection_type="documents",
                query_embedding=query_embedding,
                top_k=top_k,
                filter_metadata=filter_metadata,
                user_id=user_id,
            )

            if include_public:
                public_results = self._qdrant_search(
                    collection_type="public",
                    query_embedding=query_embedding,
                    top_k=top_k,
                )
                for r in public_results:
                    r["metadata"]["library_scope"] = "public"
                results.extend(public_results)
                results.sort(key=lambda item: item.get("similarity") or 0.0, reverse=True)
                return results[:top_k]

            return results

        return []

    # ==================================================================
    # add_chat_summary
    # ==================================================================

    def add_chat_summary(
        self,
        summary: str,
        embedding: List[float],
        metadata: Optional[Dict[str, Any]] = None,
        id: Optional[str] = None,
        user_id: Optional[str] = None,
    ) -> str:
        """Add a single chat summary to the chat summaries collection."""
        if id is None:
            id = str(uuid.uuid4())
        
        if metadata is None:
            metadata = {}
        
        if "timestamp" not in metadata:
            metadata["timestamp"] = datetime.now().isoformat()

        resolved_user_id = self._resolve_user_id(user_id or metadata.get("user_id"))
        
        if self.db_type == "chromadb":
            chat_collection = self._get_chroma_collection("chat_summaries", resolved_user_id, create_if_missing=True)
            chat_collection.add(
                documents=[summary],
                embeddings=[embedding],
                metadatas=[metadata],
                ids=[id]
            )
        elif self.db_type == "qdrant":
            from qdrant_client.models import PointStruct

            payload = {
                "text": summary,
                "user_id": resolved_user_id,
                **metadata,
            }
            self.client.upsert(
                collection_name=_QDRANT_CHAT_SUMMARIES,
                points=[
                    PointStruct(
                        id=self._qdrant_uuid(id),
                        vector=embedding,
                        payload=payload,
                    )
                ],
            )
        
        logger.info(f"Added chat summary with ID: {id}")
        return id

    # ==================================================================
    # search_chat_summaries
    # ==================================================================

    def search_chat_summaries(
        self,
        query_embedding: List[float],
        top_k: int = 5,
        filter_metadata: Optional[Dict[str, Any]] = None,
        user_id: Optional[str] = None,
    ) -> List[Dict[str, Any]]:
        """Search chat summaries collection specifically."""
        if self.db_type == "chromadb":
            user_from_filter = (filter_metadata or {}).get("user_id")
            resolved_user_id = self._resolve_user_id(user_id or user_from_filter)
            summaries_collection = self._get_chroma_collection("chat_summaries", resolved_user_id, create_if_missing=True)

            where_clause = dict(filter_metadata or {})
            where_clause.pop("user_id", None)
            if not where_clause:
                where_clause = None
            if filter_metadata and len(filter_metadata) > 1:
                where_clause = {"$and": [{k: v} for k, v in where_clause.items()]}
            
            results = summaries_collection.query(
                query_embeddings=[query_embedding],
                n_results=top_k,
                where=where_clause
            )
            
            formatted_results = []
            for i in range(len(results["ids"][0])):
                formatted_results.append({
                    "id": results["ids"][0][i],
                    "document": results["documents"][0][i],
                    "metadata": results["metadatas"][0][i],
                    "distance": results["distances"][0][i] if "distances" in results else None,
                    "similarity": max(0.0, 1.0 - (results["distances"][0][i] / 2.0)) if "distances" in results else None
                })
            
            return formatted_results
        
        elif self.db_type == "qdrant":
            return self._qdrant_search(
                collection_type="chat_summaries",
                query_embedding=query_embedding,
                top_k=top_k,
                filter_metadata=filter_metadata,
                user_id=user_id,
            )
        
        return []

    # ==================================================================
    # Qdrant unified search helper
    # ==================================================================

    def _qdrant_search(
        self,
        collection_type: str,
        query_embedding: List[float],
        top_k: int = 5,
        filter_metadata: Optional[Dict[str, Any]] = None,
        user_id: Optional[str] = None,
    ) -> List[Dict[str, Any]]:
        """Generic Qdrant similarity search with optional user_id + metadata filter."""
        from qdrant_client.models import Filter, FieldCondition, MatchValue

        coll = self._qdrant_collection(collection_type)

        conditions: list = []
        # User-scoped filter (skip for public collection)
        if collection_type != "public":
            resolved_uid = self._resolve_user_id(
                user_id or (filter_metadata or {}).get("user_id")
            )
            conditions.append(
                FieldCondition(key="user_id", match=MatchValue(value=resolved_uid))
            )

        # Extra metadata filters
        extra = dict(filter_metadata or {})
        extra.pop("user_id", None)
        for k, v in extra.items():
            conditions.append(FieldCondition(key=k, match=MatchValue(value=v)))

        query_filter = Filter(must=conditions) if conditions else None

        results = self.client.search(
            collection_name=coll,
            query_vector=query_embedding,
            limit=top_k,
            query_filter=query_filter,
        )

        formatted: List[Dict[str, Any]] = []
        for point in results:
            payload = point.payload or {}
            formatted.append({
                "id": str(point.id),
                "document": payload.get("text", ""),
                "metadata": {k: v for k, v in payload.items() if k != "text"},
                "distance": 1 - point.score,
                "similarity": point.score,
            })
        return formatted

    # ==================================================================
    # get_documents_collection / get_conversations_collection
    # ==================================================================

    def get_documents_collection(self, user_id: Optional[str] = None):
        """Get the documents collection object."""
        if self.db_type == "chromadb":
            return self._get_chroma_collection("documents", self._resolve_user_id(user_id), create_if_missing=True)
        return None
    
    def get_conversations_collection(self, user_id: Optional[str] = None):
        """Get the conversations collection object."""
        if self.db_type == "chromadb":
            return self._get_chroma_collection("conversations", self._resolve_user_id(user_id), create_if_missing=True)
        return None

    # ==================================================================
    # delete_documents
    # ==================================================================

    def delete_documents(self, ids: List[str], user_id: Optional[str] = None, collection_type: str = "conversations") -> bool:
        """Delete documents from vector store."""
        if self.db_type == "chromadb":
            collection = self._get_chroma_collection(collection_type, self._resolve_user_id(user_id), create_if_missing=True)
            collection.delete(ids=ids)
        elif self.db_type == "qdrant":
            from qdrant_client.models import PointIdsList
            coll = self._qdrant_collection(collection_type)
            qdrant_ids = [self._qdrant_uuid(i) for i in ids]
            self.client.delete(
                collection_name=coll,
                points_selector=PointIdsList(points=qdrant_ids),
            )
        
        logger.info(f"Deleted {len(ids)} documents from vector store")
        return True

    # ==================================================================
    # clear_all_data
    # ==================================================================

    def clear_all_data(self) -> bool:
        """Clear all data from all collections."""
        try:
            if self.db_type == "chromadb":
                logger.info("Clearing all ChromaDB data...")
                
                for label, coll in [
                    ("conversations", self.collection),
                    ("documents", self.documents_collection),
                    ("chat_summaries", self.chat_summaries_collection),
                ]:
                    try:
                        data = coll.get()
                        coll_ids = data.get("ids", [])
                        if coll_ids:
                            logger.info(f"Deleting {len(coll_ids)} {label} records")
                            coll.delete(ids=coll_ids)
                        else:
                            logger.info(f"No {label} records to delete")
                    except Exception as e:
                        logger.warning(f"Error clearing {label} collection: {e}")
                
                conv_remaining = self.collection.count()
                doc_remaining = self.documents_collection.count()
                sum_remaining = self.chat_summaries_collection.count()
                
                if conv_remaining == 0 and doc_remaining == 0 and sum_remaining == 0:
                    logger.info("✅ Successfully cleared all ChromaDB data")
                    return True
                else:
                    logger.warning(f"⚠️ Data may remain - Conversations: {conv_remaining}, Documents: {doc_remaining}, Summaries: {sum_remaining}")
                    return False
                    
            elif self.db_type == "qdrant":
                from qdrant_client.models import Filter
                logger.info("Clearing all Qdrant data...")

                for coll_name in (_QDRANT_CONVERSATIONS, _QDRANT_DOCUMENTS, _QDRANT_CHAT_SUMMARIES):
                    try:
                        # Delete all points by scrolling
                        self.client.delete(
                            collection_name=coll_name,
                            points_selector=Filter(must=[]),
                        )
                        logger.info("Cleared Qdrant collection: %s", coll_name)
                    except Exception as e:
                        logger.warning("Error clearing Qdrant collection %s: %s", coll_name, e)

                logger.info("✅ Successfully cleared all Qdrant data")
                return True
            
            logger.error(f"Unsupported database type: {self.db_type}")
            return False
            
        except Exception as e:
            logger.error(f"Failed to clear vector store data: {e}")
            return False

    # ==================================================================
    # reset_collections
    # ==================================================================

    def reset_collections(self) -> bool:
        """Reset collections completely (delete and recreate)."""
        try:
            if self.db_type == "chromadb":
                logger.info("Resetting ChromaDB collections...")
                
                conversations_collection_name = self.collection.name
                documents_collection_name = self.documents_collection.name
                summaries_collection_name = self.chat_summaries_collection.name
                
                for name in (conversations_collection_name, documents_collection_name, summaries_collection_name):
                    try:
                        self.client.delete_collection(name=name)
                        logger.info(f"Deleted collection: {name}")
                    except Exception as e:
                        logger.warning(f"Failed to delete collection {name}: {e}")
                
                self.collection = self.client.get_or_create_collection(
                    name=conversations_collection_name,
                    metadata={"description": "KineticChat conversation history and summaries"},
                    embedding_function=None
                )
                self.documents_collection = self.client.get_or_create_collection(
                    name=documents_collection_name,
                    metadata={"description": "KineticChat uploaded documents and knowledge base"},
                    embedding_function=None
                )
                self.chat_summaries_collection = self.client.get_or_create_collection(
                    name=summaries_collection_name,
                    metadata={"description": "KineticChat chat session summaries for memory recall"},
                    embedding_function=None
                )
                
                logger.info("✅ Successfully reset ChromaDB collections")
                return True

            elif self.db_type == "qdrant":
                from qdrant_client.models import Distance, VectorParams
                logger.info("Resetting Qdrant collections...")

                for coll_name in (_QDRANT_CONVERSATIONS, _QDRANT_DOCUMENTS, _QDRANT_CHAT_SUMMARIES):
                    try:
                        self.client.delete_collection(collection_name=coll_name)
                        logger.info("Deleted Qdrant collection: %s", coll_name)
                    except Exception:
                        pass
                    self.client.create_collection(
                        collection_name=coll_name,
                        vectors_config=VectorParams(
                            size=self._qdrant_vector_size,
                            distance=Distance.COSINE,
                        ),
                    )
                    logger.info("Recreated Qdrant collection: %s", coll_name)

                logger.info("✅ Successfully reset Qdrant collections")
                return True
                
            else:
                logger.warning(f"Collection reset not implemented for {self.db_type}")
                return False
                
        except Exception as e:
            logger.error(f"Failed to reset collections: {e}")
            return False

    # ==================================================================
    # update_metadata
    # ==================================================================

    def update_metadata(self, id: str, metadata: Dict[str, Any], user_id: Optional[str] = None) -> bool:
        """Update metadata for a document."""
        if self.db_type == "chromadb":
            collection = self._get_chroma_collection("conversations", self._resolve_user_id(user_id), create_if_missing=True)
            collection.update(
                ids=[id],
                metadatas=[metadata]
            )
        elif self.db_type == "qdrant":
            self.client.set_payload(
                collection_name=_QDRANT_CONVERSATIONS,
                payload=metadata,
                points=[self._qdrant_uuid(id)],
            )
        
        logger.info(f"Updated metadata for document: {id}")
        return True

    # ==================================================================
    # get_document
    # ==================================================================

    def get_document(self, id: str) -> Optional[Dict[str, Any]]:
        """Retrieve a document by ID."""
        if self.db_type == "chromadb":
            result = self.collection.get(ids=[id])
            if result["ids"]:
                return {
                    "id": result["ids"][0],
                    "document": result["documents"][0],
                    "metadata": result["metadatas"][0]
                }
        elif self.db_type == "qdrant":
            result = self.client.retrieve(
                collection_name=_QDRANT_CONVERSATIONS,
                ids=[self._qdrant_uuid(id)],
            )
            if result:
                return {
                    "id": str(result[0].id),
                    "document": result[0].payload.get("text", ""),
                    "metadata": {k: v for k, v in result[0].payload.items() if k != "text"}
                }
        
        return None

    # ==================================================================
    # count_documents
    # ==================================================================

    def count_documents(self, filter_metadata: Optional[Dict[str, Any]] = None) -> int:
        """Count documents in vector store."""
        if self.db_type == "chromadb":
            return self.collection.count()
        elif self.db_type == "qdrant":
            info = self.client.get_collection(_QDRANT_CONVERSATIONS)
            return info.points_count
        
        return 0


if __name__ == "__main__":
    # Example usage
    from memory.embedding_service import EmbeddingService
    
    # Initialize
    vector_store = VectorStore()
    embedding_service = EmbeddingService()
    
    # Add sample documents
    documents = [
        "User mentioned having neck pain from desk work",
        "User prefers exercises without equipment",
        "User has limited space for exercises"
    ]
    
    embeddings = embedding_service.embed_texts(documents)
    
    metadata = [
        {"user_id": "user_123", "type": "physical_context"},
        {"user_id": "user_123", "type": "preference"},
        {"user_id": "user_123", "type": "constraint"}
    ]
    
    ids = vector_store.add_documents(documents, embeddings, metadata)
    print(f"Added documents with IDs: {ids}")
    
    # Search
    query = "What are user's physical issues?"
    query_embedding = embedding_service.embed_texts([query])[0]
    results = vector_store.search(query_embedding, top_k=3)
    
    print("\nSearch results:")
    for result in results:
        print(f"- {result['document']} (similarity: {result['similarity']:.3f})")
