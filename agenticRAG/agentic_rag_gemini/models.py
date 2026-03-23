"""Shared Pydantic models for AgenticRAG API.

These models are defined here (not in api_server.py or response_templates.py)
to avoid duplicate class definitions that would cause Pydantic v2 type-mismatch errors.
"""

from typing import Optional
from pydantic import BaseModel, Field


class VoicePrompt(BaseModel):
    """Voice synthesis prompt."""

    text: str = Field(..., description="Text to synthesize")
    emotion: Optional[str] = Field(None, description="Detected or requested emotion")
    duration_estimate_seconds: float = Field(5.0, description="Estimated audio duration")


class MotionPrompt(BaseModel):
    """Motion generation prompt."""

    description: str = Field(..., description="Natural language motion description")
    primitive_sequence: Optional[str] = Field(
        None,
        description='Legacy primitive sequence (e.g., "walk*20,turn_left*10"). Optional.',
    )
    duration_seconds: Optional[float] = Field(
        None,
        description="Preferred target duration in seconds for motion generation.",
    )
    num_frames: int = Field(160, description="Number of frames to generate")
    fps: int = Field(30, description="Frames per second")
