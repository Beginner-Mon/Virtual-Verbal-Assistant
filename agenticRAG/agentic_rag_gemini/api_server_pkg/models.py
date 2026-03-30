"""Request/response models for the dedicated api_server package."""

from typing import Any, Dict, List, Optional

from pydantic import BaseModel, Field

class ConversationTurn(BaseModel):
    """Single conversation turn."""

    role: str = Field(..., description="Role: 'user' or 'assistant'")
    content: str = Field(..., description="Message content")


class QueryRequest(BaseModel):
    """Request to process a query."""

    query: str = Field(..., description="User's query")
    user_id: str = Field("guest", description="User identifier (optional, defaults to guest)")
    session_id: Optional[str] = Field(
        None,
        description=(
            "Session ID for server-managed conversation history. "
            "When provided, history is loaded from SessionStore and "
            "conversation_history field is ignored."
        ),
    )
    conversation_history: Optional[List[ConversationTurn]] = Field(
        None,
        description=(
            "Previous conversation turns (legacy/stateless fallback). "
            "Ignored when session_id is provided."
        ),
    )
    motion_duration_seconds: Optional[float] = Field(
        None,
        description="Requested motion clip duration in seconds for strict duration mode",
    )
    stream: bool = Field(False, description="Enable streaming response")


class OrchestratorDecision(BaseModel):
    """Orchestrator decision details."""

    action: str = Field(..., description="Action type: retrieve_memory, call_llm, generate_motion, hybrid, clarify")
    intent: Optional[str] = Field(None, description="Intent detected: conversation, knowledge_query, exercise_recommendation, visualize_motion, etc")
    confidence: float = Field(..., description="Confidence score 0.0-1.0")
    language: str = Field("other", description="Detected language code: en | vi | jp | other")
    reasoning: str = Field(..., description="Reasoning for decision")
    parameters: Dict[str, Any] = Field(default_factory=dict, description="Action parameters")
    tools_selected: List[str] = Field(
        default_factory=list,
        description="Tools selected by orchestrator: memory, documents, web_search, motion"
    )
    tools_executed: List[str] = Field(
        default_factory=list,
        description="Tools actually executed (may differ from selected if skipped)"
    )
    tools_failed: List[str] = Field(
        default_factory=list,
        description="Tools that failed during execution"
    )
    execution_time_ms: float = Field(
        0.0,
        description="Total orchestrator execution time in milliseconds"
    )
    debug_info: Dict[str, Any] = Field(
        default_factory=dict,
        description="Debug info: memory_results count, documents_results count, web_results count, llm_calls, errors"
    )


class MotionMetadata(BaseModel):
    """Motion generation result from DART/MotionGenerationTool."""

    motion_file: str = Field(..., description="Generated .npz motion file name")
    frames: int = Field(..., description="Total frame count")
    fps: int = Field(..., description="Frames per second")


class MotionJobStatus(BaseModel):
    """Async motion job status payload."""

    job_id: str = Field(..., description="Celery job identifier")
    status: str = Field(..., description="queued | processing | completed | failed")
    motion_file_url: Optional[str] = Field(None, description="Absolute GLB URL when completed")
    video_url: Optional[str] = Field(None, description="Rendered video or artifact URL when completed")
    error: Optional[str] = Field(None, description="Error details for failed jobs")
    stage: Optional[str] = Field(None, description="Current worker stage for queue processing")
    timings_ms: Optional[Dict[str, float]] = Field(None, description="Worker and queue timing buckets in ms")
    timeline_id: Optional[str] = Field(None, description="Timeline correlation identifier")


class QueryResponse(BaseModel):
    """Response from query processing."""

    query: str = Field(..., description="Original query")
    user_id: str = Field(..., description="User ID")
    language: str = Field("other", description="Detected language code: en | vi | jp | other")
    text_answer: str = Field(..., description="Generated text answer")
    exercises: List[Dict[str, str]] = Field(
        default_factory=list,
        description="Recommended exercises (name only). Empty list when none recommended."
    )
    exercise_motion_prompt: Optional[str] = Field(
        None,
        description=(
            "Exercise name to generate motion for, selected from the exercises list. "
            "Set only when the query implies visualization (show/demonstrate/animate). "
            "Intended for MotionGenerationTool once DART is integrated."
        ),
    )
    motion: Optional[MotionMetadata] = Field(
        None, description="Motion generation result from DART (set when exercise_motion_prompt is not null)"
    )
    motion_job: Optional[MotionJobStatus] = Field(
        None,
        description="Async motion job handle when motion queue mode is enabled",
    )
    orchestrator_decision: OrchestratorDecision = Field(
        ..., description="Orchestrator decision and reasoning"
    )
    motion_prompt: Optional[Any] = Field(
        None, description="Motion generation prompt if applicable"
    )
    voice_prompt: Optional[Any] = Field(
        None, description="Voice synthesis prompt if applicable"
    )
    metadata: Dict[str, Any] = Field(default_factory=dict, description="Additional metadata")

    # NEW DEBUG FIELDS
    pipeline_trace: Dict[str, Any] = Field(
        default_factory=dict,
        description=(
            "Execution trace: shows which path was taken and what was executed. "
            "Keys: orchestrator_type, intent, tools_invoked, memory_results_count, "
            "documents_results_count, web_results_count, llm_calls_count, rag_used, errors"
        )
    )
    performance: Dict[str, float] = Field(
        default_factory=dict,
        description="Timing info: orchestrator_ms, tools_ms, rag_ms, motion_ms, total_ms"
    )


class UnifiedTaskResponse(BaseModel):
    """Strict response contract for ECA UI 2.0 polling flow."""

    task_id: str = Field(..., description="Unique task identifier")
    status: str = Field(..., description="processing | completed | failed")
    progress_stage: str = Field(
        ..., description="queued | text_ready | motion_generation | completed | failed"
    )
    result: Optional[Dict[str, Any]] = Field(
        None,
        description="Task result payload including absolute motion_file_url when available",
    )
    error: Optional[str] = Field(None, description="Error details when task fails")


class ChatHistoryResponse(BaseModel):
    """File-backed chat history contract used by Official UI."""

    user_id: str
    messages: List[Dict[str, Any]] = Field(default_factory=list)


class SessionCreateRequest(BaseModel):
    """Request to create a new chat session."""

    user_id: str = Field("guest", description="User identifier")
    first_message: Optional[str] = Field(
        None, description="Optional first message to set the session title"
    )


class SessionMetaResponse(BaseModel):
    """Lightweight session metadata for listing."""

    session_id: str = Field(..., description="Session UUID")
    title: str = Field(..., description="Auto-generated session title")
    created_at: str = Field(..., description="ISO timestamp of session creation")
    updated_at: str = Field(..., description="ISO timestamp of last update")
    message_count: int = Field(..., description="Total messages in session")
    is_summarized: bool = Field(False, description="Whether session has been summarized to ChromaDB")
