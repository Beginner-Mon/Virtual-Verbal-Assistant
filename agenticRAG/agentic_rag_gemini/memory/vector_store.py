"""Vector Store implementation for semantic memory storage.

This module provides vector database operations for storing and retrieving
embeddings of user conversations and context.
"""

from typing import List, Dict, Any, Optional
from datetime import datetime
import uuid

import chromadb
from chromadb.config import Settings

from config import get_config
from utils.logger import get_logger


logger = get_logger(__name__)


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
    
    def _init_chromadb(self):
        """Initialize ChromaDB client and collections."""
        chroma_config = self.config.chromadb
        
        self.client = chromadb.PersistentClient(
            path=chroma_config["persist_directory"],
            settings=Settings(
                anonymized_telemetry=False,
                allow_reset=False
            )
        )
        
        # Get or create conversations collection with cosine distance
        conversations_collection_name = chroma_config["collection_name"]
        self.collection = self.client.get_or_create_collection(
            name=conversations_collection_name,
            metadata={"description": "KineticChat conversation history and summaries"},
            embedding_function=None  # Use custom embeddings
        )
        
        # Get or create documents collection with cosine distance (separate storage for uploaded documents)
        documents_collection_name = f"{conversations_collection_name}_documents"
        self.documents_collection = self.client.get_or_create_collection(
            name=documents_collection_name,
            metadata={"description": "KineticChat uploaded documents and knowledge base"},
            embedding_function=None  # Use custom embeddings
        )
        
        logger.info(f"ChromaDB collections: {conversations_collection_name}, {documents_collection_name}")
    
    def _init_qdrant(self):
        """Initialize Qdrant client and collection."""
        from qdrant_client import QdrantClient
        from qdrant_client.models import Distance, VectorParams
        
        qdrant_config = self.config.qdrant
        
        self.client = QdrantClient(url=qdrant_config["url"])
        
        # Create collection if not exists
        collections = self.client.get_collections().collections
        collection_names = [c.name for c in collections]
        
        if qdrant_config["collection_name"] not in collection_names:
            self.client.create_collection(
                collection_name=qdrant_config["collection_name"],
                vectors_config=VectorParams(
                    size=qdrant_config["vector_size"],
                    distance=Distance.COSINE
                )
            )
        
        self.collection_name = qdrant_config["collection_name"]
        logger.info(f"Qdrant collection: {self.collection_name}")
    
    def add_documents(
        self,
        documents: List[str],
        embeddings: List[List[float]],
        metadata: Optional[List[Dict[str, Any]]] = None,
        ids: Optional[List[str]] = None,
        collection_type: str = "conversations"
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
        
        # Add timestamp to metadata
        for meta in metadata:
            if "timestamp" not in meta:
                meta["timestamp"] = datetime.now().isoformat()
        
        # Select collection based on type
        if self.db_type == "chromadb":
            collection = self.documents_collection if collection_type == "documents" else self.collection
            collection.add(
                documents=documents,
                embeddings=embeddings,
                metadatas=metadata,
                ids=ids
            )
        elif self.db_type == "qdrant":
            from qdrant_client.models import PointStruct
            
            points = [
                PointStruct(
                    id=id_,
                    vector=embedding,
                    payload={"text": doc, **meta}
                )
                for id_, doc, embedding, meta in zip(ids, documents, embeddings, metadata)
            ]
            
            self.client.upsert(
                collection_name=self.collection_name,
                points=points
            )
        
        logger.info(f"Added {len(documents)} documents to vector store")
        return ids
    
    def search(
        self,
        query_embedding: List[float],
        top_k: int = 5,
        filter_metadata: Optional[Dict[str, Any]] = None
    ) -> List[Dict[str, Any]]:
        """Search for similar documents in vector store.
        
        Args:
            query_embedding: Query embedding vector
            top_k: Number of results to return
            filter_metadata: Optional metadata filters
            
        Returns:
            List of documents with similarity scores and metadata
        """
        if self.db_type == "chromadb":
            # Handle multiple filter conditions with $and operator
            where_clause = filter_metadata
            if filter_metadata and len(filter_metadata) > 1:
                where_clause = {"$and": [{k: v} for k, v in filter_metadata.items()]}
            
            results = self.collection.query(
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
            from qdrant_client.models import Filter, FieldCondition, MatchValue
            
            # Build filter if provided
            query_filter = None
            if filter_metadata:
                conditions = [
                    FieldCondition(key=k, match=MatchValue(value=v))
                    for k, v in filter_metadata.items()
                ]
                query_filter = Filter(must=conditions)
            
            results = self.client.search(
                collection_name=self.collection_name,
                query_vector=query_embedding,
                limit=top_k,
                query_filter=query_filter
            )
            
            # Format results
            formatted_results = []
            for result in results:
                formatted_results.append({
                    "id": str(result.id),
                    "document": result.payload.get("text", ""),
                    "metadata": {k: v for k, v in result.payload.items() if k != "text"},
                    "distance": 1 - result.score,  # Convert score to distance
                    "similarity": result.score
                })
            
            return formatted_results
        
        # Fallback for unsupported db types
        logger.warning(f"Unsupported database type for search: {self.db_type}")
        return []
    
    def add_document(
        self,
        document: str,
        embedding: List[float],
        metadata: Optional[Dict[str, Any]] = None,
        id: Optional[str] = None
    ) -> str:
        """Add a single document to the documents collection.
        
        Args:
            document: Document text
            embedding: Embedding vector
            metadata: Document metadata
            id: Optional document ID
            
        Returns:
            Document ID
        """
        ids = self.add_documents(
            documents=[document],
            embeddings=[embedding],
            metadata=[metadata],
            ids=[id] if id else None,
            collection_type="documents"
        )
        return ids[0]
    
    def search_documents(
        self,
        query_embedding: List[float],
        top_k: int = 5,
        filter_metadata: Optional[Dict[str, Any]] = None
    ) -> List[Dict[str, Any]]:
        """Search documents collection specifically.
        
        Args:
            query_embedding: Query embedding vector
            top_k: Number of results to return
            filter_metadata: Optional metadata filters
            
        Returns:
            List of documents with scores
        """
        if self.db_type == "chromadb":
            # Handle multiple filter conditions with $and operator
            where_clause = filter_metadata
            if filter_metadata and len(filter_metadata) > 1:
                where_clause = {"$and": [{k: v} for k, v in filter_metadata.items()]}
            
            results = self.documents_collection.query(
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
        
        # Similar Qdrant logic would go here...
        return []
    
    def get_documents_collection(self):
        """Get the documents collection object.
        
        Returns:
            Documents collection
        """
        if self.db_type == "chromadb":
            return self.documents_collection
        return None
    
    def get_conversations_collection(self):
        """Get the conversations collection object.
        
        Returns:
            Conversations collection
        """
        if self.db_type == "chromadb":
            return self.collection
        return None

    def delete_documents(self, ids: List[str]) -> bool:
        """Delete documents from vector store.
        
        Args:
            ids: List of document IDs to delete
            
        Returns:
            True if successful
        """
        if self.db_type == "chromadb":
            self.collection.delete(ids=ids)
        elif self.db_type == "qdrant":
            self.client.delete(
                collection_name=self.collection_name,
                points_selector=ids
            )
        
        logger.info(f"Deleted {len(ids)} documents from vector store")
        return True
    
    def clear_all_data(self) -> bool:
        """Clear all data from both collections (conversations and documents).
        
        Returns:
            True if successful
        """
        try:
            if self.db_type == "chromadb":
                logger.info("Clearing all ChromaDB data...")
                
                # Clear conversation collection
                try:
                    conv_data = self.collection.get()
                    conv_ids = conv_data.get("ids", [])
                    if conv_ids:
                        logger.info(f"Deleting {len(conv_ids)} conversation records")
                        self.collection.delete(ids=conv_ids)
                    else:
                        logger.info("No conversation records to delete")
                except Exception as e:
                    logger.warning(f"Error clearing conversation collection: {e}")
                
                # Clear documents collection
                try:
                    doc_data = self.documents_collection.get()
                    doc_ids = doc_data.get("ids", [])
                    if doc_ids:
                        logger.info(f"Deleting {len(doc_ids)} document records")
                        self.documents_collection.delete(ids=doc_ids)
                    else:
                        logger.info("No document records to delete")
                except Exception as e:
                    logger.warning(f"Error clearing documents collection: {e}")
                
                # Verify collections are empty
                conv_remaining = self.collection.count()
                doc_remaining = self.documents_collection.count()
                
                if conv_remaining == 0 and doc_remaining == 0:
                    logger.info("✅ Successfully cleared all ChromaDB data")
                    return True
                else:
                    logger.warning(f"⚠️ Data may remain - Conversations: {conv_remaining}, Documents: {doc_remaining}")
                    return False
                    
            elif self.db_type == "qdrant":
                logger.info("Clearing all Qdrant data...")
                
                # Clear conversation memory
                try:
                    self.client.delete(
                        collection_name=self.collection_name,
                        points_selector=models.Filter(
                            must=[]
                        )
                    )
                    logger.info("Cleared conversation collection")
                except Exception as e:
                    logger.warning(f"Error clearing conversation collection: {e}")
                
                # Clear documents
                try:
                    self.client.delete(
                        collection_name=f"{self.collection_name}_documents",
                        points_selector=models.Filter(
                            must=[]
                        )
                    )
                    logger.info("Cleared documents collection")
                except Exception as e:
                    logger.warning(f"Error clearing documents collection: {e}")
                
                logger.info("✅ Successfully cleared all Qdrant data")
                return True
            
            logger.error(f"Unsupported database type: {self.db_type}")
            return False
            
        except Exception as e:
            logger.error(f"Failed to clear vector store data: {e}")
            return False
    
    def reset_collections(self) -> bool:
        """Reset collections completely (delete and recreate) to fix schema issues.
        
        Returns:
            True if successful
        """
        try:
            if self.db_type == "chromadb":
                logger.info("Resetting ChromaDB collections...")
                
                # Get collection names
                conversations_collection_name = self.collection.name
                documents_collection_name = self.documents_collection.name
                
                # Delete existing collections
                try:
                    self.client.delete_collection(name=conversations_collection_name)
                    logger.info(f"Deleted collection: {conversations_collection_name}")
                except Exception as e:
                    logger.warning(f"Failed to delete conversation collection: {e}")
                
                try:
                    self.client.delete_collection(name=documents_collection_name)
                    logger.info(f"Deleted collection: {documents_collection_name}")
                except Exception as e:
                    logger.warning(f"Failed to delete documents collection: {e}")
                
                # Recreate collections
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
                
                logger.info("✅ Successfully reset ChromaDB collections")
                return True
                
            else:
                logger.warning(f"Collection reset not implemented for {self.db_type}")
                return False
                
        except Exception as e:
            logger.error(f"Failed to reset collections: {e}")
            return False
    
    def update_metadata(self, id: str, metadata: Dict[str, Any]) -> bool:
        """Update metadata for a document.
        
        Args:
            id: Document ID
            metadata: New metadata dictionary
            
        Returns:
            True if successful
        """
        if self.db_type == "chromadb":
            self.collection.update(
                ids=[id],
                metadatas=[metadata]
            )
        elif self.db_type == "qdrant":
            self.client.set_payload(
                collection_name=self.collection_name,
                payload=metadata,
                points=[id]
            )
        
        logger.info(f"Updated metadata for document: {id}")
        return True
    
    def get_document(self, id: str) -> Optional[Dict[str, Any]]:
        """Retrieve a document by ID.
        
        Args:
            id: Document ID
            
        Returns:
            Document dictionary or None if not found
        """
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
                collection_name=self.collection_name,
                ids=[id]
            )
            if result:
                return {
                    "id": str(result[0].id),
                    "document": result[0].payload.get("text", ""),
                    "metadata": {k: v for k, v in result[0].payload.items() if k != "text"}
                }
        
        return None
    
    def count_documents(self, filter_metadata: Optional[Dict[str, Any]] = None) -> int:
        """Count documents in vector store.
        
        Args:
            filter_metadata: Optional metadata filters
            
        Returns:
            Number of documents
        """
        if self.db_type == "chromadb":
            return self.collection.count()
        elif self.db_type == "qdrant":
            info = self.client.get_collection(self.collection_name)
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
