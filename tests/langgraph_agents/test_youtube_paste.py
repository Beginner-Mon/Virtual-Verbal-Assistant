"""Tests for youtube_transcript tool (FIX-YOUTUBE-PASTE.md Task 4).

Covers 8 spec cases:
  1. watch?v= and youtu.be/ URLs → same video_id
  2. Invalid URL → {found: false, reason: "invalid_url"}, no raise
  3. Short transcript → {found: true, truncated: false}
  4. Long transcript (> cap) → truncated: true, len <= cap
  5. NoTranscriptFound → {found: false, reason: "no_transcript"}, no raise
  6. Network error (generic Exception) → tool RAISES
  7. youtube_transcript in RETRIEVER_BASE_TOOLS
  8. Tool schema exposes only `url` (no extra LLM-visible fields)

Mock strategy: patch YouTubeTranscriptApi.get_transcript via
  `langgraph_agents.tools.pgvector_tool` namespace — no real network calls.
"""

import pytest
from unittest.mock import AsyncMock, patch, MagicMock

from youtube_transcript_api import NoTranscriptFound, TranscriptsDisabled

from langgraph_agents.tools.pgvector_tool import youtube_transcript, _YT_CHAR_CAP
from langgraph_agents.nodes.retriever_agent import RETRIEVER_BASE_TOOLS


# ── Helper: build a fake transcript entry list ────────────────────────────

def _make_entries(total_chars: int) -> list[dict]:
    """Build transcript entries whose joined text totals total_chars."""
    text = "a" * total_chars
    # Single entry is fine for testing join logic
    return [{"text": text, "start": 0.0, "duration": 5.0}]


# ── Patch target ──────────────────────────────────────────────────────────
# YouTubeTranscriptApi is imported lazily inside the tool function body, so
# we must patch it at the source module where Python resolves the name.
_PATCH_TARGET = "youtube_transcript_api.YouTubeTranscriptApi"


# ═══════════════════════════════════════════════════════════════════════════
# Case 1 — URL extraction: watch?v= and youtu.be/ → same video_id
# ═══════════════════════════════════════════════════════════════════════════

@pytest.mark.unit
@pytest.mark.asyncio
async def test_watch_and_youtu_be_same_video_id():
    """Both URL formats return the same video_id in the result."""
    fake_entries = _make_entries(100)

    with patch(_PATCH_TARGET) as mock_api:
        mock_api.get_transcript.return_value = fake_entries

        result_watch = await youtube_transcript.ainvoke(
            {"url": "https://www.youtube.com/watch?v=dQw4w9WgXcQ"}
        )
        result_youtu = await youtube_transcript.ainvoke(
            {"url": "https://youtu.be/dQw4w9WgXcQ"}
        )

    assert result_watch["found"] is True
    assert result_youtu["found"] is True
    assert result_watch["video_id"] == "dQw4w9WgXcQ"
    assert result_youtu["video_id"] == "dQw4w9WgXcQ"
    assert result_watch["video_id"] == result_youtu["video_id"]


# ═══════════════════════════════════════════════════════════════════════════
# Case 2 — Invalid URL → {found: false, reason: "invalid_url"}, no raise
# ═══════════════════════════════════════════════════════════════════════════

@pytest.mark.unit
@pytest.mark.asyncio
async def test_invalid_url_returns_dict_not_raise():
    """Non-YouTube URL → {found: false, reason: 'invalid_url'}, no exception."""
    result = await youtube_transcript.ainvoke({"url": "https://example.com/page"})

    assert result["found"] is False
    assert result["reason"] == "invalid_url"


@pytest.mark.unit
@pytest.mark.asyncio
async def test_bare_string_invalid_url():
    """Bare string with no video ID → {found: false, reason: 'invalid_url'}."""
    result = await youtube_transcript.ainvoke({"url": "not-a-url-at-all"})

    assert result["found"] is False
    assert result["reason"] == "invalid_url"


# ═══════════════════════════════════════════════════════════════════════════
# Case 3 — Short transcript → found: true, truncated: false
# ═══════════════════════════════════════════════════════════════════════════

@pytest.mark.unit
@pytest.mark.asyncio
async def test_short_transcript_not_truncated():
    """Transcript under cap → truncated: false, full text preserved."""
    short_entries = _make_entries(500)

    with patch(_PATCH_TARGET) as mock_api:
        mock_api.get_transcript.return_value = short_entries

        result = await youtube_transcript.ainvoke(
            {"url": "https://www.youtube.com/watch?v=SHORT123"}
        )

    assert result["found"] is True
    assert result["truncated"] is False
    assert len(result["transcript"]) == 500
    assert result["video_id"] == "SHORT123"


# ═══════════════════════════════════════════════════════════════════════════
# Case 4 — Long transcript (> cap) → truncated: true, len <= cap
# ═══════════════════════════════════════════════════════════════════════════

@pytest.mark.unit
@pytest.mark.asyncio
async def test_long_transcript_truncated():
    """Transcript over cap → truncated: true, len(transcript) <= _YT_CHAR_CAP."""
    long_entries = _make_entries(_YT_CHAR_CAP + 5000)

    with patch(_PATCH_TARGET) as mock_api:
        mock_api.get_transcript.return_value = long_entries

        result = await youtube_transcript.ainvoke(
            {"url": "https://www.youtube.com/watch?v=LONG1234"}
        )

    assert result["found"] is True
    assert result["truncated"] is True
    assert len(result["transcript"]) == _YT_CHAR_CAP


# ═══════════════════════════════════════════════════════════════════════════
# Case 5 — No captions → {found: false, reason: "no_transcript"}, no raise
# ═══════════════════════════════════════════════════════════════════════════

@pytest.mark.unit
@pytest.mark.asyncio
async def test_no_transcript_found_returns_dict_not_raise():
    """NoTranscriptFound → {found: false, reason: 'no_transcript'}, no exception."""
    with patch(_PATCH_TARGET) as mock_api:
        # NoTranscriptFound requires video_id positional arg
        mock_api.get_transcript.side_effect = NoTranscriptFound(
            "ABC123", ["vi", "en"], MagicMock()
        )

        result = await youtube_transcript.ainvoke(
            {"url": "https://www.youtube.com/watch?v=ABC123"}
        )

    assert result["found"] is False
    assert result["reason"] == "no_transcript"


@pytest.mark.unit
@pytest.mark.asyncio
async def test_transcripts_disabled_returns_dict_not_raise():
    """TranscriptsDisabled (subclass of CouldNotRetrieveTranscript) → no raise."""
    with patch(_PATCH_TARGET) as mock_api:
        mock_api.get_transcript.side_effect = TranscriptsDisabled("ABC456")

        result = await youtube_transcript.ainvoke(
            {"url": "https://www.youtube.com/watch?v=ABC456"}
        )

    assert result["found"] is False
    assert result["reason"] == "no_transcript"


# ═══════════════════════════════════════════════════════════════════════════
# Case 6 — Network / generic error → tool RAISES (D23 service error)
# ═══════════════════════════════════════════════════════════════════════════

@pytest.mark.unit
@pytest.mark.asyncio
async def test_network_error_raises():
    """Generic Exception (network failure) → tool re-raises for retriever to handle."""
    with patch(_PATCH_TARGET) as mock_api:
        mock_api.get_transcript.side_effect = ConnectionError("Network unreachable")

        with pytest.raises(ConnectionError):
            await youtube_transcript.ainvoke(
                {"url": "https://www.youtube.com/watch?v=NET1234"}
            )


@pytest.mark.unit
@pytest.mark.asyncio
async def test_generic_exception_raises():
    """Generic RuntimeError → tool re-raises (not silenced)."""
    with patch(_PATCH_TARGET) as mock_api:
        mock_api.get_transcript.side_effect = RuntimeError("Unexpected API error")

        with pytest.raises(RuntimeError):
            await youtube_transcript.ainvoke(
                {"url": "https://www.youtube.com/watch?v=ERR1234"}
            )


# ═══════════════════════════════════════════════════════════════════════════
# Case 7 — youtube_transcript registered in RETRIEVER_BASE_TOOLS
# ═══════════════════════════════════════════════════════════════════════════

@pytest.mark.unit
def test_youtube_transcript_in_retriever_base_tools():
    """youtube_transcript must be present in RETRIEVER_BASE_TOOLS."""
    tool_names = [t.name for t in RETRIEVER_BASE_TOOLS]
    assert "youtube_transcript" in tool_names


# ═══════════════════════════════════════════════════════════════════════════
# Case 8 — Tool schema exposes only `url` (no extra LLM-visible fields)
# ═══════════════════════════════════════════════════════════════════════════

@pytest.mark.unit
def test_tool_schema_exposes_only_url():
    """LangChain tool schema must have exactly one arg: url."""
    schema = youtube_transcript.args_schema.schema()
    props = schema.get("properties", {})
    assert set(props.keys()) == {"url"}, f"Unexpected schema properties: {set(props.keys())}"


@pytest.mark.unit
def test_tool_name_and_description():
    """Tool name must be 'youtube_transcript' and description must be non-empty."""
    assert youtube_transcript.name == "youtube_transcript"
    assert youtube_transcript.description
    assert "YouTube" in youtube_transcript.description
