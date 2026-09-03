from pydantic import BaseModel, ConfigDict, Field
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


class SyncedPrefs(BaseModel):
    """The whole of `users.preferences`, and the only thing allowed into it.

    `extra="forbid"` IS the PHI guard. What it replaced was a blocklist of key
    names — injury_history, fitness_level, age — which missed in both directions:
    it only looked at the top level, so {"a": {"injury_history": "x"}} went
    through (there was a test asserting it did), and it only looked at key names,
    so {"note": "thoát vị L4"} went through as well. A whitelist cannot miss
    either. A key that is not declared below is a 422 whatever it is called and
    however deep it sits, which is the property a healthcare project needs from
    a free-form column. Clinical facts belong in user_memory, which has the
    `valid` flag and the advisory semantics for them.

    Adding a synced preference means adding a field here and nothing else. The
    column is JSONB precisely so that set can grow without a migration.
    """

    model_config = ConfigDict(extra="forbid")

    # Deliberately not an enum. The palette lives in avatarPalette.ts alongside
    # the Tailwind classes and hex values the backend has no use for; declaring
    # the ids here too would mean a backend deploy per colour. The frontend ships
    # through Amplify and the backend through CDK, independently, so a colour
    # released to the UI first would 422 for whoever picked it. Unknown values
    # are inert: the UI looks the id up and falls back to 'slate'.
    avatar_bg: Optional[str] = Field(
        default=None, max_length=32, pattern=r"^[a-z][a-z0-9_-]{0,31}$"
    )
    selected_character_slug: Optional[str] = Field(
        default=None, max_length=64, pattern=r"^[A-Za-z0-9_-]{1,64}$"
    )


class UserPreferencesOut(BaseModel):
    """What GET/PATCH /me/preferences return.

    No `version`: writes are last-write-wins. Two devices changing two different
    preferences do not actually conflict — the merge is per key — so a version
    check would have manufactured a 409 where there was no disagreement, and the
    resolution for that 409 was going to be "take the other write" anyway.
    """

    preferences: "SyncedPrefs" = Field(default_factory=lambda: SyncedPrefs())
    updated_at: Optional[str] = None


class UserPreferencesPatch(BaseModel):
    """PATCH /me/preferences — a shallow merge over `users.preferences`.

    Only the keys actually present in the request are written; the rest of the
    stored object is left alone. Sending `selected_character_slug: null`
    explicitly is how a default character is cleared.
    """

    preferences: SyncedPrefs
