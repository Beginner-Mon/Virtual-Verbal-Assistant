from pydantic import BaseModel, Field
from typing import Literal, Optional


class ChatRequest(BaseModel):
    query: str
    user_id: str = "anonymous"
    session_id: str = "default"
    persona_id: str = Field(default="eca_default", pattern=r"^[A-Za-z0-9_-]{1,64}$")
    output_mode: Literal["text", "speech", "both"] = "text"
    token_limit: Optional[int] = None
    web_search: bool = False


class ChatResponse(BaseModel):
    request_id: str
    final_answer: str
    required_outputs: list[str] = Field(default_factory=list)
    needs_retrieval: bool = False
    needs_motion: bool = False
    needs_clarification: bool = False
    speech_task_id: Optional[str] = None
    total_tokens: int = 0
    grader_result: Optional[str] = None
    errors: list[dict] = Field(default_factory=list)


class TTSRequest(BaseModel):
    """Synthesize arbitrary text on demand — the per-message speaker button.

    Separate from `output_mode` on /chat, which decides whether a reply is voiced
    automatically as it is produced. This one voices text the user has already
    read and chose to hear, so it must never be implicit.
    """
    text: str = Field(min_length=1, max_length=5000)
    persona_id: str = Field(default="eca_default", pattern=r"^[A-Za-z0-9_-]{1,64}$")


class TTSTaskResponse(BaseModel):
    task_id: str


class SessionListItem(BaseModel):
    session_id: str
    created_at: str
    updated_at: str
    first_user_message_preview: str
    message_count: int


class SessionListResponse(BaseModel):
    sessions: list[SessionListItem]
    total: int


class SessionResumeResponse(BaseModel):
    session_id: str
    messages: list[dict]
    stm_populated: bool
    last_updated: str


class UserMemoryCreate(BaseModel):
    fact_text: str = Field(min_length=1, max_length=500)
    category: Optional[str] = Field(default=None, max_length=100)


class UserMemoryItem(BaseModel):
    id: str
    fact_text: str
    category: Optional[str] = None
    valid: bool = True
    created_at: str


class UserMemoryListResponse(BaseModel):
    facts: list[UserMemoryItem]
