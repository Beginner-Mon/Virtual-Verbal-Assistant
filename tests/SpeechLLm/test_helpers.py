"""
Unit tests for SpeechLLm helper functions and request schemas.

Tests cover:
- clean_text_for_tts  (text sanitization for TTS input)
- extract_voice_fields (script JSON parsing)
- TTSRequest / VoicePrompt Pydantic schemas
"""

import pytest


# ── clean_text_for_tts ────────────────────────────────────────────────────────

@pytest.mark.unit
class TestCleanTextForTTS:

    def _clean(self, text: str) -> str:
        from api_server import clean_text_for_tts
        return clean_text_for_tts(text)

    def test_removes_literal_backslash_n(self):
        """Literal \\n sequences (from JSON) should be replaced with spaces."""
        assert self._clean("hello\\nworld") == "hello world"

    def test_removes_newlines(self):
        """Actual newline characters should be replaced with spaces."""
        assert self._clean("hello\nworld") == "hello world"

    def test_removes_bullet_markers(self):
        """Markdown bullets (* and •) should be stripped."""
        assert self._clean("* item1 \u2022 item2") == "item1 item2"

    def test_removes_dash_separators(self):
        """Whitespace-dash-whitespace patterns should collapse."""
        assert self._clean("a - b") == "a b"

    def test_collapses_whitespace(self):
        """Multiple consecutive spaces should become a single space."""
        assert self._clean("a   b     c") == "a b c"

    def test_strips_edges(self):
        """Leading and trailing whitespace should be trimmed."""
        assert self._clean("  hello  ") == "hello"

    def test_combined_cleanup(self):
        """All formatting artifacts should be cleaned in a single pass."""
        text = "  * First\\n \u2022 Second\n- Third   "
        result = self._clean(text)
        assert result == "First Second Third"


# ── extract_voice_fields ──────────────────────────────────────────────────────

@pytest.mark.unit
class TestExtractVoiceFields:

    def _extract(self, script: dict):
        from main import extract_voice_fields
        return extract_voice_fields(script)

    def test_valid_script(self):
        """Full script dict should return correct (text, emotion, language)."""
        script = {
            "voice_prompt": {"text": "Hello there", "emotion": "happy"},
            "language": "en",
        }
        text, emotion, language = self._extract(script)

        assert text == "Hello there"
        assert emotion == "happy"
        assert language == "en"

    def test_missing_voice_prompt(self):
        """Missing voice_prompt key should return safe defaults."""
        text, emotion, language = self._extract({})

        assert text == ""
        assert emotion == "neutral"
        assert language == "en"

    def test_null_emotion_defaults_to_neutral(self):
        """None emotion should fall back to 'neutral'."""
        script = {
            "voice_prompt": {"text": "Hi", "emotion": None},
        }
        _, emotion, _ = self._extract(script)
        assert emotion == "neutral"

    def test_custom_language(self):
        """Non-English language should be extracted correctly."""
        script = {
            "voice_prompt": {"text": "Xin ch\u00e0o"},
            "language": "vi",
        }
        _, _, language = self._extract(script)
        assert language == "vi"


# ── TTSRequest / VoicePrompt schemas ──────────────────────────────────────────

@pytest.mark.unit
class TestTTSRequestSchema:

    def test_simple_mode_fields(self):
        """TTSRequest with text only should have correct defaults."""
        from api_server import TTSRequest

        req = TTSRequest(text="Hello")

        assert req.text == "Hello"
        assert req.language == "en"
        assert req.emotion is None
        assert req.voice_prompt is None

    def test_voice_prompt_mode(self):
        """TTSRequest with voice_prompt should populate nested fields."""
        from api_server import TTSRequest, VoicePrompt

        req = TTSRequest(
            voice_prompt=VoicePrompt(text="Do stretches", emotion="calm")
        )

        assert req.voice_prompt.text == "Do stretches"
        assert req.voice_prompt.emotion == "calm"

    def test_all_fields_optional(self):
        """TTSRequest() with no args should not raise."""
        from api_server import TTSRequest

        req = TTSRequest()

        assert req.text is None
        assert req.emotion is None
        assert req.voice_prompt is None
        assert req.language == "en"
        assert req.user_id is None
