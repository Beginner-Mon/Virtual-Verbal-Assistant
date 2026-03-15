"""Local Orchestrator Configuration.

This module defines configuration settings for the local Qwen2.5-3B orchestrator.

MODEL OPTIONS:
  - qwen2.5:3b        [Current] Size: 3B, Speed: ~15-30s first inference, good quality
  - phi:latest        [RECOMMENDED] Size: 2.7B, Speed: ~5-10s first inference, good quality  
  - tinyllama:latest  [FAST] Size: 1.1B, Speed: ~2-3s first inference, lower quality
  - mistral:latest    [POWERFUL] Size: 7B, Speed: ~30-60s, best quality (slower)

RECOMMENDATION:
  If experiencing >40s latency, switch to `phi:latest` for 3-5x speedup.
  If need sub-5s routing, use `tinyllama:latest` (with quality trade-off).
"""

from pydantic import BaseModel, Field


class LocalOrchestratorConfig(BaseModel):
    """Local orchestrator configuration."""
    model: str = Field(
        default="qwen2.5:3b",
        description="Model name for local orchestrator (phi:latest recommended for speed)"
    )
    temperature: float = Field(
        default=0.1,
        description="Sampling temperature for deterministic responses (keep low for routing)"
    )
    max_tokens: int = Field(
        default=256,
        description="Maximum tokens for routing decision (256 is sufficient for JSON)"
    )
    timeout: int = Field(
        default=30,
        description="Timeout in seconds for model requests (reduce to 15 for fast fail)"
    )
    fallback_to_api: bool = Field(
        default=True,
        description="Fallback to API orchestrator if local fails or times out"
    )
    confidence_threshold: float = Field(
        default=0.7,
        description="Minimum confidence for routing decisions"
    )


class OllamaConfig(BaseModel):
    """Ollama service configuration."""
    base_url: str = Field(
        default="http://localhost:11434",
        description="Ollama service base URL"
    )
    timeout: int = Field(
        default=30,
        description="Timeout in seconds for Ollama requests (reduce to 15 for fast fail)"
    )
