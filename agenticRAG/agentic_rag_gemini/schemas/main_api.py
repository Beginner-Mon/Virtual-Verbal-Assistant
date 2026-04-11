from typing import Any, Dict, List, Literal, Optional

from pydantic import BaseModel, Field


class ConversationTurn(BaseModel):
    """Single conversation turn."""

    role: str = Field(..., description="'user' or 'assistant'")
    content: str = Field(..., description="Message content")


class QueryRequestCompat(BaseModel):
    """Compatibility request so 8080 can accept 8000-style payloads."""

    query: str = Field(..., description="User query")
    user_id: str = Field(default="default", description="User identifier")
    session_id: Optional[str] = Field(
        None,
        description="Session ID for server-managed history. When provided, conversation_history is ignored.",
    )
    conversation_history: Optional[List[ConversationTurn]] = Field(
        None, description="Previous conversation turns (legacy fallback, ignored when session_id is set)"
    )
    motion_duration_seconds: Optional[float] = Field(
        None,
        description="Compatibility field. Main API infers duration from downstream metadata.",
    )
    stream: bool = Field(False, description="Compatibility field. Streaming is not supported on 8080.")
    motion_job_enabled: bool = Field(True, description="Enable async motion job tracking.")



class AnswerRequest(BaseModel):
    """Request for a complete answer."""

    query: str = Field(..., description="User query")
    user_id: str = Field(default="default", description="User identifier")
    session_id: Optional[str] = Field(
        None,
        description="Session ID for server-managed history. When provided, conversation_history is ignored.",
    )
    motion_format: Literal["glb", "npz"] = Field(
        default="glb",
        description="Requested motion output format: 'glb' or 'npz'",
    )
    conversation_history: Optional[List[ConversationTurn]] = Field(
        None, description="Previous conversation turns (legacy fallback, ignored when session_id is set)"
    )


class MotionMetadata(BaseModel):
    """Motion output metadata returned from DART."""

    motion_file_url: str = Field(..., description="URL to download the generated motion file (.glb or .npz)")
    num_frames: int = Field(..., description="Total number of motion frames")
    fps: int = Field(..., description="Frames per second (always 30 for DART)")
    duration_seconds: float = Field(..., description="Total clip duration in seconds")
    text_prompt: str = Field(..., description="The prompt that was sent to DART")


class MotionJobStatus(BaseModel):
    """Async motion job status payload."""

    job_id: str = Field(..., description="Celery job identifier")
    status: str = Field(..., description="queued | processing | completed | failed")
    motion_file_url: Optional[str] = Field(None, description="Absolute GLB URL when completed")
    video_url: Optional[str] = Field(None, description="Rendered video or artifact URL when completed")
    error: Optional[str] = Field(None, description="Error details for failed jobs")
    stage: Optional[str] = Field(None, description="Current worker stage")
    timings_ms: Optional[Dict[str, float]] = Field(None, description="Timing details")



class TTSMetadata(BaseModel):
    """TTS output metadata returned from SpeechLLM."""

    audio_file: str = Field(..., description="Filename of the generated audio file")
    audio_url: str = Field(..., description="URL to download the audio file")
    text: str = Field(..., description="Text that was synthesized")
    emotion: Optional[str] = Field(None, description="Emotion used for synthesis")


class AnswerResponse(BaseModel):
    """Combined response from AgenticRAG + DART + TTS."""

    request_id: str = Field(..., description="Request identifier for status polling")
    status: str = Field(..., description="processing or completed")
    pending_services: List[str] = Field(default_factory=list, description="Background services still running")
    language: str = Field("other", description="Detected language code: en | vi | jp | other")
    selected_strategy: str = Field("unknown", description="The routing strategy, e.g., visualize_motion")
    progress_stage: str = Field("completed", description="Current execution stage for polling visibility")

    text_answer: str = Field(..., description="Text response from AgenticRAG")
    exercises: List[Dict[str, str]] = Field(
        default_factory=list, description="List of recommended exercises from AgenticRAG"
    )
    motion: Optional[MotionMetadata] = Field(None, description="Motion output from DART")
    motion_job: Optional[MotionJobStatus] = Field(None, description="Async motion job handle")
    tts: Optional[TTSMetadata] = Field(None, description="Speech output from TTS/SpeechLLM")
    generation_time_ms: float = Field(..., description="Total wall-clock time in ms")

    errors: Optional[Dict[str, str]] = Field(None, description="Per-service errors if any")
    debug: Optional[Dict[str, Any]] = Field(None, description="Detailed diagnostics for bottleneck analysis")


class UnifiedTaskResponseCompat(BaseModel):
    """Task envelope compatible with AgenticRAG /process_query polling contract."""

    task_id: str = Field(..., description="Unique task identifier")
    status: str = Field(..., description="processing | completed | failed")
    progress_stage: str = Field(..., description="queued | text_ready | motion_generation | completed | failed")
    result: Optional[Dict[str, Any]] = Field(None, description="Normalized result payload")
    error: Optional[str] = Field(None, description="Error text when task fails")
