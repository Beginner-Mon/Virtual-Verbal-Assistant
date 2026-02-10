"""Embedding service for converting text to vector representations.

This module provides text embedding functionality using various models
including Sentence Transformers and Gemini embeddings.
"""

from typing import List, Union
import numpy as np

from sentence_transformers import SentenceTransformer
import tiktoken

from config import get_config
from utils.logger import get_logger


logger = get_logger(__name__)


class EmbeddingService:
    """Service for generating text embeddings.
    
    Supports multiple embedding models:
    - Sentence Transformers (local, offline) - RECOMMENDED
    - Gemini embeddings (API-based)
    """
    
    def __init__(self, config=None):
        """Initialize embedding service.
        
        Args:
            config: Configuration object. If None, loads from default config.
        """
        self.config = config or get_config().embedding
        self.model_name = self.config.model
        
        # Initialize model based on type
        if "text-embedding" in self.model_name or "embedding" in self.model_name:
            self._init_gemini_embeddings()
        else:
            self._init_sentence_transformer()
        
        logger.info(f"Initialized EmbeddingService with model: {self.model_name}")
    
    def _init_sentence_transformer(self):
        """Initialize Sentence Transformer model."""
        self.model = SentenceTransformer(self.model_name)
        self.embedding_dim = self.model.get_sentence_embedding_dimension()
        self.model_type = "sentence_transformer"
        
        logger.info(f"Loaded Sentence Transformer: {self.model_name} (dim={self.embedding_dim})")
    
    def _init_gemini_embeddings(self):
        """Initialize Gemini embeddings."""
        from utils.gemini_client import GeminiClientWrapper
        
        self.client = GeminiClientWrapper()
        self.model_type = "gemini"
        
        # Gemini text-embedding-004 has 768 dimensions
        self.embedding_dim = 768
        
        logger.info(f"Using Gemini embeddings: {self.model_name} (dim={self.embedding_dim})")
    
    def embed_texts(
        self,
        texts: Union[str, List[str]],
        batch_size: int = None
    ) -> Union[List[float], List[List[float]]]:
        """Generate embeddings for one or more texts.
        
        Args:
            texts: Single text string or list of texts
            batch_size: Batch size for processing. If None, uses config default.
            
        Returns:
            Single embedding vector (if input is string) or list of embeddings
        """
        # Handle single text input
        single_input = isinstance(texts, str)
        if single_input:
            texts = [texts]
        
        # Use config batch size if not specified
        if batch_size is None:
            batch_size = self.config.batch_size
        
        # Generate embeddings based on model type
        if self.model_type == "sentence_transformer":
            embeddings = self._embed_sentence_transformer(texts, batch_size)
        elif self.model_type == "gemini":
            embeddings = self._embed_gemini(texts)
        else:
            raise ValueError(f"Unknown model type: {self.model_type}")
        
        # Return single embedding if single input
        if single_input:
            return embeddings[0]
        
        return embeddings
    
    def _embed_sentence_transformer(
        self,
        texts: List[str],
        batch_size: int
    ) -> List[List[float]]:
        """Generate embeddings using Sentence Transformer.
        
        Args:
            texts: List of texts
            batch_size: Batch size for processing
            
        Returns:
            List of embedding vectors
        """
        embeddings = self.model.encode(
            texts,
            batch_size=batch_size,
            show_progress_bar=False,
            convert_to_numpy=True
        )
        
        # Convert to list of lists
        embeddings = embeddings.tolist()
        
        logger.debug(f"Generated {len(embeddings)} embeddings")
        return embeddings
    
    def _embed_gemini(self, texts: List[str]) -> List[List[float]]:
        """Generate embeddings using Gemini API.
        
        Args:
            texts: List of texts
            
        Returns:
            List of embedding vectors
        """
        try:
            response = self.client.embeddings.create(
                model=self.model_name,
                input=texts
            )
            
            embeddings = [item.embedding for item in response.data]
            
            logger.debug(f"Generated {len(embeddings)} Gemini embeddings")
            return embeddings
            
        except Exception as e:
            logger.error(f"Error generating Gemini embeddings: {str(e)}")
            raise
    
    def compute_similarity(
        self,
        embedding1: List[float],
        embedding2: List[float]
    ) -> float:
        """Compute cosine similarity between two embeddings.
        
        Args:
            embedding1: First embedding vector
            embedding2: Second embedding vector
            
        Returns:
            Cosine similarity score (0 to 1)
        """
        vec1 = np.array(embedding1)
        vec2 = np.array(embedding2)
        
        # Cosine similarity
        similarity = np.dot(vec1, vec2) / (np.linalg.norm(vec1) * np.linalg.norm(vec2))
        
        return float(similarity)
    
    def get_embedding_dimension(self) -> int:
        """Get the dimension of embeddings produced by this service.
        
        Returns:
            Embedding dimension
        """
        return self.embedding_dim
    
    def count_tokens(self, text: str) -> int:
        """Count tokens in text (useful for managing context length).
        
        Args:
            text: Input text
            
        Returns:
            Number of tokens
        """
        try:
            # Use tiktoken for Gemini models (similar tokenization)
            if self.model_type == "gemini":
                encoding = tiktoken.get_encoding("cl100k_base")
                return len(encoding.encode(text))
            else:
                # Approximate for other models (rough estimate)
                return len(text.split())
        except Exception as e:
            logger.warning(f"Error counting tokens: {str(e)}")
            return len(text.split())  # Fallback to word count
    
    def truncate_text(self, text: str, max_tokens: int) -> str:
        """Truncate text to fit within token limit.
        
        Args:
            text: Input text
            max_tokens: Maximum number of tokens
            
        Returns:
            Truncated text
        """
        if self.model_type == "gemini":
            encoding = tiktoken.get_encoding("cl100k_base")
            tokens = encoding.encode(text)
            
            if len(tokens) <= max_tokens:
                return text
            
            truncated_tokens = tokens[:max_tokens]
            return encoding.decode(truncated_tokens)
        else:
            # Simple word-based truncation for other models
            words = text.split()
            if len(words) <= max_tokens:
                return text
            return " ".join(words[:max_tokens])


if __name__ == "__main__":
    # Example usage
    embedding_service = EmbeddingService()
    
    # Single text
    text = "I have neck pain from working at my computer"
    embedding = embedding_service.embed_texts(text)
    print(f"Embedding dimension: {len(embedding)}")
    
    # Multiple texts
    texts = [
        "Neck pain from desk work",
        "Lower back discomfort",
        "Shoulder tension"
    ]
    
    embeddings = embedding_service.embed_texts(texts)
    print(f"\nGenerated {len(embeddings)} embeddings")
    
    # Compute similarity
    similarity = embedding_service.compute_similarity(embeddings[0], embeddings[1])
    print(f"\nSimilarity between text 1 and 2: {similarity:.3f}")
    
    # Token counting
    long_text = "This is a longer text that might exceed token limits. " * 20
    token_count = embedding_service.count_tokens(long_text)
    print(f"\nToken count: {token_count}")
    
    truncated = embedding_service.truncate_text(long_text, max_tokens=50)
    print(f"Truncated token count: {embedding_service.count_tokens(truncated)}")
