from pydantic import BaseModel, Field
from typing import Literal, Optional


class ChatRequest(BaseModel):
    query: str
    user_id: str = "anonymous"
    session_id: str = "default"
    persona_id: str = "eca_default"
    output_mode: Literal["text", "speech", "both"] = "text"
    token_limit: Optional[int] = None


class ChatResponse(BaseModel):
    request_id: str
    final_answer: str
    intent: str
    confidence: float
    needs_clarification: bool = False
    speech_task_id: Optional[str] = None
    total_tokens: int = 0
    grader_result: Optional[str] = None
    grader_warning: Optional[str] = None
    errors: list[dict] = Field(default_factory=list)
