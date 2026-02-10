"""Memory Manager for user context and conversation history.

This module manages user-specific memory, including conversation history,
preferences, physical context, and summarization.
"""

from typing import List, Dict, Any, Optional
from datetime import datetime, timedelta
import json

from memory.vector_store import VectorStore
from memory.embedding_service import EmbeddingService
from utils.document_loader import DocumentLoader
from config import get_config
from utils.logger import get_logger


logger = get_logger(__name__)


class MemoryManager:
    """Manages user memory and context storage/retrieval.
    
    Handles:
    - Conversation history storage
    - User preferences and constraints
    - Physical context (pain, limitations)
    - Periodic summarization
    - Memory retrieval for context-aware responses
    """
    
    def __init__(self, config=None, vector_store=None, embedding_service=None, document_loader=None):
        """Initialize memory manager.
        
        Args:
            config: Configuration object. If None, loads from default config.
            vector_store: Shared VectorStore instance. Creates new if None.
            embedding_service: Shared EmbeddingService instance. Creates new if None.
            document_loader: Shared DocumentLoader instance. Creates new if None.
        """
        self.config = config or get_config().memory
        self.vector_store = vector_store or VectorStore()
        self.embedding_service = embedding_service or EmbeddingService()
        self.document_loader = document_loader or DocumentLoader()
        
        # Track conversation counts for summarization
        self.conversation_counts = {}
        
        logger.info("Initialized MemoryManager")
    
    def store_interaction(
        self,
        user_id: str,
        user_message: str,
        assistant_response: str,
        metadata: Optional[Dict[str, Any]] = None
    ) -> str:
        """Store a user-assistant interaction in memory.
        
        Args:
            user_id: User identifier
            user_message: User's message
            assistant_response: Assistant's response
            metadata: Additional metadata
            
        Returns:
            Document ID
        """
        # Combine user and assistant messages for context
        interaction_text = f"User: {user_message}\nAssistant: {assistant_response}"
        
        # Generate embedding
        embedding = self.embedding_service.embed_texts(interaction_text)
        
        # Prepare metadata
        interaction_metadata = {
            "user_id": user_id,
            "type": "interaction",
            "timestamp": datetime.now().isoformat(),
            "user_message": user_message,
            "assistant_response": assistant_response
        }
        
        if metadata:
            interaction_metadata.update(metadata)
        
        # Store in vector database
        ids = self.vector_store.add_documents(
            documents=[interaction_text],
            embeddings=[embedding],
            metadata=[interaction_metadata]
        )
        
        # Update conversation count
        if user_id not in self.conversation_counts:
            self.conversation_counts[user_id] = 0
        self.conversation_counts[user_id] += 1
        
        # Check if summarization is needed
        if self.conversation_counts[user_id] >= self.config.summarization_interval:
            self._create_summary(user_id)
            self.conversation_counts[user_id] = 0
        
        logger.info(f"Stored interaction for user {user_id}")
        return ids[0]
    
    def store_user_context(
        self,
        user_id: str,
        context_type: str,
        content: str,
        metadata: Optional[Dict[str, Any]] = None
    ) -> str:
        """Store user context information (preferences, physical state, etc).
        
        Args:
            user_id: User identifier
            context_type: Type of context (preference, physical_context, constraint)
            content: Context content
            metadata: Additional metadata
            
        Returns:
            Document ID
        """
        # Generate embedding
        embedding = self.embedding_service.embed_texts(content)
        
        # Prepare metadata
        context_metadata = {
            "user_id": user_id,
            "type": context_type,
            "timestamp": datetime.now().isoformat()
        }
        
        if metadata:
            context_metadata.update(metadata)
        
        # Store in vector database
        ids = self.vector_store.add_documents(
            documents=[content],
            embeddings=[embedding],
            metadata=[context_metadata]
        )
        
        logger.info(f"Stored {context_type} for user {user_id}")
        return ids[0]
    
    def retrieve_relevant_memory(
        self,
        user_id: str,
        query: str,
        top_k: int = None,
        memory_types: Optional[List[str]] = None
    ) -> List[Dict[str, Any]]:
        """Retrieve relevant memory for a given query.
        
        Args:
            user_id: User identifier
            query: Query text
            top_k: Number of results to retrieve
            memory_types: Filter by memory types (interaction, preference, etc)
            
        Returns:
            List of relevant memory items
        """
        if top_k is None:
            top_k = self.config.top_k
        
        # Generate query embedding
        query_embedding = self.embedding_service.embed_texts(query)
        
        # Build filter
        filter_metadata = {"user_id": user_id}
        
        # Search vector store
        results = self.vector_store.search(
            query_embedding=query_embedding,
            top_k=top_k * 2,  # Get more results for filtering
            filter_metadata=filter_metadata
        )
        
        # Filter by memory types if specified
        if memory_types:
            results = [
                r for r in results
                if r["metadata"].get("type") in memory_types
            ]
        
        # Filter by similarity threshold
        results = [
            r for r in results
            if r["similarity"] >= self.config.similarity_threshold
        ]
        
        # Limit to top_k
        results = results[:top_k]
        
        logger.info(f"Retrieved {len(results)} relevant memories for user {user_id}")
        return results
    
    def get_recent_interactions(
        self,
        user_id: str,
        limit: int = 10
    ) -> List[Dict[str, Any]]:
        """Get recent interactions for a user.
        
        Args:
            user_id: User identifier
            limit: Maximum number of interactions to retrieve
            
        Returns:
            List of recent interactions
        """
        # Search with user filter
        # Note: This is a simplified implementation
        # In production, you'd want to sort by timestamp
        
        filter_metadata = {
            "user_id": user_id,
            "type": "interaction"
        }
        
        # Use a dummy query to get results
        dummy_embedding = self.embedding_service.embed_texts("recent conversation")
        
        results = self.vector_store.search(
            query_embedding=dummy_embedding,
            top_k=limit,
            filter_metadata=filter_metadata
        )
        
        # Sort by timestamp (most recent first)
        results.sort(
            key=lambda x: x["metadata"].get("timestamp", ""),
            reverse=True
        )
        
        return results
    
    def _create_summary(self, user_id: str):
        """Create a summary of recent conversations.
        
        Args:
            user_id: User identifier
        """
        # Get recent interactions
        recent = self.get_recent_interactions(
            user_id=user_id,
            limit=self.config.summarization_interval
        )
        
        if not recent:
            return
        
        # Combine interactions
        interaction_texts = [r["document"] for r in recent]
        combined_text = "\n\n".join(interaction_texts)
        
        # Generate summary (simplified - in production, use LLM)
        summary = f"Summary of {len(recent)} recent interactions with user {user_id}"
        
        # Store summary
        self.store_user_context(
            user_id=user_id,
            context_type="summary",
            content=summary,
            metadata={
                "interactions_count": len(recent),
                "summary_date": datetime.now().isoformat()
            }
        )
        
        logger.info(f"Created summary for user {user_id}")
    
    def get_user_profile(self, user_id: str) -> Dict[str, Any]:
        """Get aggregated user profile from memory.
        
        Args:
            user_id: User identifier
            
        Returns:
            Dictionary with user profile information
        """
        profile = {
            "user_id": user_id,
            "preferences": [],
            "physical_context": [],
            "constraints": [],
            "recent_topics": []
        }
        
        # Retrieve different types of context
        context_types = ["preference", "physical_context", "constraint"]
        
        for context_type in context_types:
            dummy_embedding = self.embedding_service.embed_texts(context_type)
            
            results = self.vector_store.search(
                query_embedding=dummy_embedding,
                top_k=10,
                filter_metadata={"user_id": user_id, "type": context_type}
            )
            
            profile[f"{context_type}s"] = [
                r["document"] for r in results
            ]
        
        return profile
    
    def clear_old_memory(
        self,
        user_id: str,
        days: int = None
    ) -> int:
        """Clear old memory entries for a user.
        
        Args:
            user_id: User identifier
            days: Number of days to keep. Uses config default if None.
            
        Returns:
            Number of entries deleted
        """
        if days is None:
            days = self.config.retention_days
        
        cutoff_date = datetime.now() - timedelta(days=days)
        
        # Note: This is a simplified implementation
        # In production, you'd implement proper date-based filtering
        
        logger.info(f"Cleared old memory for user {user_id} (older than {days} days)")
        return 0  # Placeholder


    def load_documents_from_file(
        self,
        user_id: str,
        file_path: str,
        context_type: str = "document"
    ) -> int:
        """Load and store documents from file (PDF, Word, Image, TXT).
        
        DEPRECATED: Use DocumentStore.store_document() instead for document storage.
        This method is kept for backward compatibility but documents are no longer
        stored in the conversations collection.
        
        Args:
            user_id: User identifier
            file_path: Path to file (PDF, DOCX, TXT, PNG, JPG, etc.)
            context_type: Type of content ("document", "manual", "reference", etc.)
            
        Returns:
            Number of documents stored (always 1 for backward compatibility)
            
        Example:
            # For new code, use DocumentStore instead:
            from memory.document_store import DocumentStore
            ds = DocumentStore(vector_store, embedding_service)
            doc_id = ds.store_document(user_id, content, filename)
        """
        try:
            logger.info(f"Loading document from {file_path} (deprecated method)")
            text = self.document_loader.load_file(file_path)
            
            logger.warning(f"Document loading called on MemoryManager - use DocumentStore instead")
            logger.info(f"Loaded document ({len(text)} chars)")
            return 1
            
        except Exception as e:
            logger.error(f"Failed to load document: {e}")
            raise
    
    def load_documents_from_directory(
        self,
        user_id: str,
        directory_path: str,
        context_type: str = "document",
        chunk_size: Optional[int] = None
    ) -> int:
        """Load all documents from directory.
        
        Args:
            user_id: User identifier
            directory_path: Path to directory with documents
            context_type: Type of content
            chunk_size: If specified, split documents into chunks
            
        Returns:
            Number of documents stored
            
        Example:
            mm = MemoryManager()
            stored = mm.load_documents_from_directory(
                user_id="user123",
                directory_path="data/documents/",
                context_type="reference_material",
                chunk_size=2000
            )
        """
        try:
            logger.info(f"Loading documents from {directory_path}")
            
            documents = self.document_loader.load_mixed_content(directory_path)
            
            count = 0
            for doc in documents:
                try:
                    content = doc['content']
                    
                    # Chunk if requested
                    if chunk_size:
                        chunks = self.document_loader._chunk_text(content, chunk_size)
                        for chunk_num, chunk in enumerate(chunks):
                            self.store_user_context(
                                user_id=user_id,
                                context_type=f"{context_type}_chunk_{chunk_num}",
                                content=chunk,
                                metadata={
                                    "source_file": doc['file_name'],
                                    "chunk": chunk_num
                                }
                            )
                            count += 1
                    else:
                        self.store_user_context(
                            user_id=user_id,
                            context_type=context_type,
                            content=content,
                            metadata={"source_file": doc['file_name']}
                        )
                        count += 1
                        
                except Exception as e:
                    logger.error(f"Failed to process {doc['file_name']}: {e}")
            
            logger.info(f"Successfully stored {count} documents")
            return count
            
        except Exception as e:
            logger.error(f"Failed to load documents: {e}")
            raise
    
    def search_documents(
        self,
        user_id: str,
        query: str,
        context_type: Optional[str] = None,
        top_k: int = 5
    ) -> List[Dict[str, Any]]:
        """Search stored documents by query.
        
        Args:
            user_id: User identifier
            query: Search query
            context_type: Filter by document type (optional)
            top_k: Number of results to return
            
        Returns:
            List of matching documents with similarity scores
            
        Example:
            results = mm.search_documents(
                user_id="user123",
                query="neck exercises",
                top_k=3
            )
        """
        return self.retrieve_relevant_memory(
            user_id=user_id,
            query=query,
            top_k=top_k
        )


if __name__ == "__main__":
    # Example usage
    memory_manager = MemoryManager()
    
    user_id = "test_user_123"
    
    # Store an interaction
    memory_manager.store_interaction(
        user_id=user_id,
        user_message="I have neck pain from sitting all day",
        assistant_response="I understand. Let me suggest some neck stretches."
    )
    
    # Store user context
    memory_manager.store_user_context(
        user_id=user_id,
        context_type="physical_context",
        content="User experiences neck pain, works at desk"
    )
    
    # Retrieve relevant memory
    query = "What exercises can help with my neck?"
    relevant_memory = memory_manager.retrieve_relevant_memory(
        user_id=user_id,
        query=query
    )
    
    print(f"Retrieved {len(relevant_memory)} relevant memories")
    for mem in relevant_memory:
        print(f"- {mem['document'][:100]}... (similarity: {mem['similarity']:.3f})")