"""Document Store for managing uploaded documents with chunking support."""

import os
from datetime import datetime
from typing import List, Dict, Any, Optional

from memory.vector_store import VectorStore
from memory.embedding_service import EmbeddingService
from utils.logger import get_logger
from utils.document_loader import DocumentLoader

logger = get_logger(__name__)


class DocumentStore:
    """Store and manage uploaded documents with chunking for better retrieval."""
    
    def __init__(self, vector_store: VectorStore, embedding_service: EmbeddingService):
        """Initialize document store.
        
        Args:
            vector_store: Vector store for document embeddings
            embedding_service: Service for generating embeddings
        """
        self.vector_store = vector_store
        self.embedding_service = embedding_service
        self.document_loader = DocumentLoader()
        
        # Load chunking configuration
        from config import get_config
        config = get_config()
        
        # Chunking parameters from config
        self.enable_chunking = config.chunking.enable_chunking
        self.chunk_size = config.chunking.chunk_size
        self.chunk_overlap = config.chunking.chunk_overlap
        self.min_chunk_size = config.chunking.min_chunk_size
        self.search_multiplier = config.chunking.chunk_search_multiplier
        
        logger.info(f"DocumentStore initialized with chunking: {self.enable_chunking}, "
                   f"chunk_size: {self.chunk_size}, overlap: {self.chunk_overlap}")
    
    def store_document(
        self,
        user_id: str,
        document_content: str,
        filename: str,
        context_type: str = "uploaded_document"
    ) -> List[str]:
        """Store a document in the documents collection with chunking.
        
        Args:
            user_id: User identifier
            document_content: Full document text
            filename: Original filename
            context_type: Type of document content
            
        Returns:
            List of document chunk IDs
        """
        try:
            # Determine if document needs chunking
            content_length = len(document_content)
            
            if not self.enable_chunking or content_length <= self.min_chunk_size:
                # Store as single document for small content or if chunking disabled
                return self._store_single_document(
                    user_id, document_content, filename, context_type
                )
            else:
                # Chunk the document for better retrieval
                return self._store_chunked_document(
                    user_id, document_content, filename, context_type
                )
                
        except Exception as e:
            logger.error(f"Failed to store document '{filename}': {e}")
            raise
    
    def _store_single_document(
        self,
        user_id: str,
        document_content: str,
        filename: str,
        context_type: str
    ) -> List[str]:
        """Store a document as a single chunk."""
        # Generate embedding
        embedding = self.embedding_service.embed_texts(document_content)
        
        # Prepare metadata
        document_metadata = {
            "user_id": user_id,
            "filename": filename,
            "type": context_type,
            "timestamp": datetime.now().isoformat(),
            "content_length": len(document_content),
            "document_type": "uploaded_knowledge",
            "chunk_number": 0,
            "total_chunks": 1,
            "chunk_type": "single"
        }
        
        # Store in documents collection
        doc_ids = self.vector_store.add_documents(
            documents=[document_content],
            embeddings=[embedding],
            metadata=[document_metadata],
            collection_type="documents"
        )
        
        logger.info(f"Stored single document '{filename}' for user {user_id}")
        return doc_ids
    
    def _store_chunked_document(
        self,
        user_id: str,
        document_content: str,
        filename: str,
        context_type: str
    ) -> List[str]:
        """Store a document as multiple chunks."""
        # Create chunks using the document loader's chunking method
        chunks = self.document_loader._chunk_text(
            document_content, 
            self.chunk_size, 
            self.chunk_overlap
        )
        
        # Filter out very small chunks
        valid_chunks = [
            chunk for chunk in chunks 
            if len(chunk.strip()) >= self.min_chunk_size
        ]
        
        if not valid_chunks:
            logger.warning(f"No valid chunks found for document '{filename}'")
            return []
        
        # Generate embeddings for all chunks
        chunk_embeddings = self.embedding_service.embed_texts(valid_chunks)
        
        # Prepare documents and metadata for each chunk
        documents = []
        metadatas = []
        
        for i, chunk in enumerate(valid_chunks):
            # Calculate chunk position in original document
            start_pos = document_content.find(chunk[:50])  # Use first 50 chars to find position
            end_pos = start_pos + len(chunk)
            
            chunk_metadata = {
                "user_id": user_id,
                "filename": filename,
                "type": context_type,
                "timestamp": datetime.now().isoformat(),
                "content_length": len(chunk),
                "document_type": "uploaded_knowledge",
                "chunk_number": i,
                "total_chunks": len(valid_chunks),
                "chunk_type": "chunked",
                "start_position": start_pos,
                "end_position": end_pos,
                "parent_document": filename
            }
            
            documents.append(chunk)
            metadatas.append(chunk_metadata)
        
        # Store all chunks in batch
        doc_ids = self.vector_store.add_documents(
            documents=documents,
            embeddings=chunk_embeddings,
            metadata=metadatas,
            collection_type="documents"
        )
        
        logger.info(f"Stored {len(valid_chunks)} chunks for document '{filename}' for user {user_id}")
        return doc_ids
    
    def search_documents(
        self,
        query: str,
        user_id: Optional[str] = None,
        top_k: int = 5,
        max_chunks_per_document: int = 3
    ) -> List[Dict[str, Any]]:
        """Search documents for a given query with chunk deduplication.
        
        Args:
            query: Search query text
            user_id: Optional user ID to filter by
            top_k: Number of results to return
            max_chunks_per_document: Maximum chunks to return per document
            
        Returns:
            List of matching documents/chunks with scores
        """
        try:
            # Generate query embedding
            query_embedding = self.embedding_service.embed_texts(query)
            
            # Build filter if user_id provided
            filter_metadata = None
            if user_id:
                filter_metadata = {"user_id": user_id}
            
            # Search documents collection - get more results to allow deduplication
            search_limit = top_k * self.search_multiplier  # Use config multiplier
            results = self.vector_store.search_documents(
                query_embedding=query_embedding,
                top_k=search_limit,
                filter_metadata=filter_metadata
            )
            
            # Group chunks by document and select best ones
            deduplicated_results = self._deduplicate_chunks(
                results, max_chunks_per_document
            )
            
            # Limit to final top_k results
            final_results = deduplicated_results[:top_k]
            
            logger.info(f"Found {len(results)} raw results, deduplicated to {len(final_results)} for query")
            return final_results
            
        except Exception as e:
            logger.error(f"Failed to search documents: {e}")
            return []
    
    def _deduplicate_chunks(
        self, 
        results: List[Dict[str, Any]], 
        max_chunks_per_document: int
    ) -> List[Dict[str, Any]]:
        """Deduplicate chunks from the same document, keeping the best ones.
        
        Args:
            results: Raw search results
            max_chunks_per_document: Maximum chunks to keep per document
            
        Returns:
            Deduplicated results
        """
        # Group chunks by document
        document_groups = {}
        
        for result in results:
            filename = result.get("metadata", {}).get("filename", "unknown")
            chunk_type = result.get("metadata", {}).get("chunk_type", "single")
            
            # Create document key
            doc_key = f"{filename}_{chunk_type}"
            
            if doc_key not in document_groups:
                document_groups[doc_key] = []
            
            document_groups[doc_key].append(result)
        
        # Select best chunks from each document
        deduplicated_results = []
        
        for doc_key, chunks in document_groups.items():
            # Sort chunks by similarity score (descending)
            chunks.sort(key=lambda x: x.get("similarity", 0), reverse=True)
            
            # Take top chunks for this document
            best_chunks = chunks[:max_chunks_per_document]
            deduplicated_results.extend(best_chunks)
        
        # Sort all results by similarity score
        deduplicated_results.sort(key=lambda x: x.get("similarity", 0), reverse=True)
        
        return deduplicated_results
    
    def get_user_documents(
        self,
        user_id: str
    ) -> List[Dict[str, Any]]:
        """Get all documents for a user.
        
        Args:
            user_id: User identifier
            
        Returns:
            List of user's documents with metadata
        """
        try:
            # Query documents collection for user's documents
            documents_collection = self.vector_store.get_documents_collection()
            if documents_collection is None:
                logger.warning("Documents collection not available")
                return []
            
            results = documents_collection.get(
                where={"user_id": user_id}
            )
            
            # Format results
            documents = []
            if results and results["ids"]:
                for i in range(len(results["ids"])):
                    documents.append({
                        "id": results["ids"][i],
                        "filename": results["metadatas"][i].get("filename", "unknown"),
                        "content_preview": results["documents"][i][:200] + "...",
                        "metadata": results["metadatas"][i]
                    })
            
            logger.info(f"Retrieved {len(documents)} documents for user {user_id}")
            return documents
            
        except Exception as e:
            logger.error(f"Failed to get user documents: {e}")
            return []
    
    def delete_document(self, document_id: str) -> bool:
        """Delete a document.
        
        Args:
            document_id: Document ID to delete
            
        Returns:
            True if successful
        """
        try:
            documents_collection = self.vector_store.get_documents_collection()
            if documents_collection is None:
                logger.warning("Documents collection not available")
                return False
            
            documents_collection.delete(ids=[document_id])
            logger.info(f"Deleted document: {document_id}")
            return True
        except Exception as e:
            logger.error(f"Failed to delete document: {e}")
            return False
    
    def delete_user_documents(self, user_id: str) -> int:
        """Delete all documents for a user.
        
        Args:
            user_id: User identifier
            
        Returns:
            Number of documents deleted
        """
        try:
            # Get all user's documents
            documents_collection = self.vector_store.get_documents_collection()
            if documents_collection is None:
                logger.warning("Documents collection not available")
                return 0
            
            results = documents_collection.get(
                where={"user_id": user_id}
            )
            
            if not results or not results["ids"]:
                return 0
            
            # Delete all found documents
            doc_ids = results["ids"]
            documents_collection.delete(ids=doc_ids)
            
            logger.info(f"Deleted {len(doc_ids)} documents for user {user_id}")
            return len(doc_ids)
            
        except Exception as e:
            logger.error(f"Failed to delete user documents: {e}")
            return 0
    
    def count_documents(self, user_id: Optional[str] = None) -> int:
        """Count documents in store.
        
        Args:
            user_id: Optional user ID to count for specific user
            
        Returns:
            Number of documents
        """
        try:
            documents_collection = self.vector_store.get_documents_collection()
            if documents_collection is None:
                logger.warning("Documents collection not available")
                return 0
            
            if user_id:
                results = documents_collection.get(
                    where={"user_id": user_id}
                )
                return len(results["ids"]) if results and results["ids"] else 0
            else:
                # Count all documents
                return documents_collection.count()
        except Exception as e:
            logger.error(f"Failed to count documents: {e}")
            return 0
