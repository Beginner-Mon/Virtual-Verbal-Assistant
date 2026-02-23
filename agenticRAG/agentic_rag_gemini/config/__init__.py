"""Configuration management for Agentic RAG system."""

import os
from pathlib import Path
from typing import Any, Dict

import yaml
from dotenv import load_dotenv
from pydantic import BaseModel, Field


# Load environment variables
load_dotenv()


class OrchestratorConfig(BaseModel):
    """Orchestrator agent configuration."""
    model: str = Field(default="gpt-3.5-turbo")
    temperature: float = Field(default=0.1)
    max_tokens: int = Field(default=500)
    memory_retrieval_threshold: float = Field(default=0.6)
    llm_call_threshold: float = Field(default=0.7)
    motion_generation_threshold: float = Field(default=0.8)
    system_prompt: str = Field(default="")


class LLMConfig(BaseModel):
    """LLM configuration for response generation."""
    model: str = Field(default="gpt-4-turbo-preview")
    temperature: float = Field(default=0.7)
    max_tokens: int = Field(default=1000)
    enable_validation: bool = Field(default=True)
    max_retries: int = Field(default=3)
    retry_delay: float = Field(default=1.0)
    system_prompt: str = Field(default="")


class EmbeddingConfig(BaseModel):
    """Embedding model configuration."""
    model: str = Field(default="sentence-transformers/all-MiniLM-L6-v2")
    dimension: int = Field(default=384)
    batch_size: int = Field(default=32)


class VectorDatabaseConfig(BaseModel):
    """Vector database configuration."""
    type: str = Field(default="chromadb")
    chromadb: Dict[str, Any] = Field(default_factory=dict)
    qdrant: Dict[str, Any] = Field(default_factory=dict)


class MemoryConfig(BaseModel):
    """Memory management configuration."""
    max_items_per_user: int = Field(default=100)
    retention_days: int = Field(default=90)
    top_k: int = Field(default=5)
    similarity_threshold: float = Field(default=0.75)
    enable_reranking: bool = Field(default=False)
    summarization_interval: int = Field(default=5)
    summary_max_length: int = Field(default=500)
    max_chat_sessions: int = Field(default=5)  # Number of chat sessions to keep (1-10)


class RAGConfig(BaseModel):
    """RAG pipeline configuration."""
    top_k_documents: int = Field(default=5)
    similarity_threshold: float = Field(default=0.7)
    enable_query_expansion: bool = Field(default=True)
    query_expansion_method: str = Field(default="llm")
    max_context_length: int = Field(default=2000)
    include_metadata: bool = Field(default=True)
    max_chunks_per_document: int = Field(default=3)
    enable_query_reformulation: bool = Field(default=True)
    max_reformulation_attempts: int = Field(default=2)
    reformulation_quality_threshold: float = Field(default=0.3)
    enable_iterative_reflection: bool = Field(default=True)
    max_reflection_iterations: int = Field(default=1)


class ChunkingConfig(BaseModel):
    """Document chunking configuration."""
    enable_chunking: bool = Field(default=True)
    chunk_size: int = Field(default=1500)
    chunk_overlap: int = Field(default=150)
    min_chunk_size: int = Field(default=300)
    chunk_search_multiplier: int = Field(default=3)


class ValidationConfig(BaseModel):
    """Response validation configuration."""
    enable_safety_check: bool = Field(default=True)
    enable_factuality_check: bool = Field(default=False)
    enable_relevance_check: bool = Field(default=True)
    unsafe_keywords: list = Field(default_factory=list)
    min_response_length: int = Field(default=50)
    max_response_length: int = Field(default=1500)


class RetryFallbackConfig(BaseModel):
    """Retry and fallback configuration."""
    max_retries: int = Field(default=3)
    retry_delay: float = Field(default=1.0)
    exponential_backoff: bool = Field(default=True)
    enable_fallback: bool = Field(default=True)
    fallback_messages: Dict[str, str] = Field(default_factory=dict)


class LoggingConfig(BaseModel):
    """Logging configuration."""
    level: str = Field(default="INFO")
    format: str = Field(default="json")
    output_file: str = Field(default="logs/agentic_rag.log")
    log_orchestrator_decisions: bool = Field(default=True)
    log_memory_retrievals: bool = Field(default=True)
    log_llm_calls: bool = Field(default=True)


class Config(BaseModel):
    """Main configuration class."""
    orchestrator: OrchestratorConfig
    llm: LLMConfig
    embedding: EmbeddingConfig
    vector_database: VectorDatabaseConfig
    memory: MemoryConfig
    rag: RAGConfig
    chunking: ChunkingConfig
    validation: ValidationConfig
    retry_fallback: RetryFallbackConfig
    logging: LoggingConfig


def load_config(config_path: str = None) -> Config:
    """Load configuration from YAML file and environment variables.
    
    Args:
        config_path: Path to config YAML file. If None, uses default path.
        
    Returns:
        Config object with loaded settings.
    """
    if config_path is None:
        config_path = Path(__file__).parent / "config.yaml"
    
    with open(config_path, 'r') as f:
        config_dict = yaml.safe_load(f)
    
    # Override with environment variables where applicable
    if os.getenv("ORCHESTRATOR_MODEL"):
        config_dict["orchestrator"]["model"] = os.getenv("ORCHESTRATOR_MODEL")
    
    if os.getenv("LLM_MODEL"):
        config_dict["llm"]["model"] = os.getenv("LLM_MODEL")
    
    if os.getenv("EMBEDDING_MODEL"):
        config_dict["embedding"]["model"] = os.getenv("EMBEDDING_MODEL")
    
    if os.getenv("VECTOR_DB_TYPE"):
        config_dict["vector_database"]["type"] = os.getenv("VECTOR_DB_TYPE")
    
    return Config(**config_dict)


# Global config instance
_config: Config = None


def get_config() -> Config:
    """Get the global configuration instance.
    
    Returns:
        Config object.
    """
    global _config
    if _config is None:
        _config = load_config()
    return _config


def reload_config(config_path: str = None):
    """Reload configuration from file.
    
    Args:
        config_path: Path to config YAML file.
    """
    global _config
    _config = load_config(config_path)
