from pydantic import BaseModel, Field
from typing import Literal, Optional


class ChatRequest(BaseModel):
    """A chat turn. Note there is no user_id — identity comes from the Bearer
    token via Depends(current_user_id), never from the request body."""
    query: str
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


# ── User preferences — cross-device synced UI prefs (no PHI) ───────────────


# Must stay in sync with the CHECK in 009_user_preferences.py
AvatarBg = Literal["slate", "violet", "blue", "emerald", "amber", "rose", "cyan", "indigo"]


class UserPreferencesOut(BaseModel):
    """What GET /me/preferences returns. prefs is UI-only — notifications/locale.

    Never contains PHI (injury_history etc) — those live in user_memory.
    """

    avatar_bg: AvatarBg = "slate"
    selected_character_slug: Optional[str] = None
    display_name: Optional[str] = None
    prefs: dict = Field(default_factory=dict)
    version: int = 1
    updated_at: str


class UserPreferencesPatch(BaseModel):
    """PATCH /me/preferences — all fields optional except version (optimistic lock)."""

    avatar_bg: Optional[AvatarBg] = None
    selected_character_slug: Optional[str] = Field(
        default=None, max_length=100, pattern=r"^[A-Za-z0-9_-]{1,64}$"
    )
    display_name: Optional[str] = Field(default=None, max_length=100)
    prefs: Optional[dict] = None
    version: int = Field(description="Expected current version; 409 if stale")
