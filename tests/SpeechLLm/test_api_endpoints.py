"""
Unit tests for the SpeechLLm API endpoints.

Tests cover:
- Health check
- Successful TTS synthesis (simple mode and LLM mode)
- Request validation (empty text, missing fields)
- Audio file serving (MP3, WAV, 404)
"""

import pytest
from unittest.mock import patch


# ── Health ────────────────────────────────────────────────────────────────────

@pytest.mark.unit
class TestHealthEndpoint:

    def test_returns_ok(self, tts_client):
        """GET /health should return 200 with status ok."""
        response = tts_client.get("/health")
        assert response.status_code == 200
        assert response.json() == {"status": "ok"}


# ── Synthesize ────────────────────────────────────────────────────────────────

@pytest.mark.unit
class TestSynthesizeEndpoint:

    def test_simple_mode_success(self, tts_client):
        """POST /synthesize with text+emotion should return synthesis metadata."""
        payload = {"text": "Hello world", "emotion": "happy", "language": "en"}
        response = tts_client.post("/synthesize", json=payload)
        
        if response.status_code != 200:
            print("ERROR RESPONSE:", response.json())

        assert response.status_code == 200
        data = response.json()
        assert data["message"] == "Synthesis complete"
        assert "audio_file" in data
        assert data["language"] == "en"
        assert data["emotion"] == "happy"
        assert "tts_time_sec" in data
        assert data["tts_provider"] == "coqui"

    def test_llm_mode_with_voice_prompt(self, tts_client):
        """POST /synthesize with voice_prompt object should extract text and emotion."""
        payload = {
            "voice_prompt": {"text": "Do some stretches", "emotion": "calm"},
            "language": "en", # Ensure language model is available
        }
        response = tts_client.post("/synthesize", json=payload)

        assert response.status_code == 200
        data = response.json()
        assert data["language"] == "en"
        assert data["emotion"] == "calm"
        assert data["tts_provider"] == "coqui"

    def test_empty_text_returns_400(self, tts_client):
        """POST /synthesize with empty text should return 400."""
        payload = {"text": ""}
        response = tts_client.post("/synthesize", json=payload)
        assert response.status_code == 400
        assert "empty" in response.json()["detail"].lower()

    def test_whitespace_only_text_returns_400(self, tts_client):
        """POST /synthesize with whitespace-only text should return 400."""
        payload = {"text": "   "}
        response = tts_client.post("/synthesize", json=payload)
        assert response.status_code == 400

    def test_no_text_at_all_returns_400(self, tts_client):
        """POST /synthesize with neither text nor voice_prompt should return 400."""
        payload = {"language": "en"}
        response = tts_client.post("/synthesize", json=payload)
        assert response.status_code == 400

    def test_default_language_is_en(self, tts_client):
        """Omitting language should default to 'en'."""
        payload = {"text": "Hello"}
        response = tts_client.post("/synthesize", json=payload)

        assert response.status_code == 200
        assert response.json()["language"] == "en"

    def test_default_emotion_is_neutral(self, tts_client):
        """Omitting emotion should default to 'neutral'."""
        payload = {"text": "Hello"}
        response = tts_client.post("/synthesize", json=payload)

        assert response.status_code == 200
        assert response.json()["emotion"] == "neutral"

    def test_tts_failure_returns_500(self, tts_client):
        """When TTS router raises, API should return 500."""
        # We can force Coqui to fail by passing an unsupported language
        payload = {"text": "Hello", "language": "unsupported_lang"}
        response = tts_client.post("/synthesize", json=payload)

        assert response.status_code == 500
        assert "TTS failed" in response.json()["detail"]

    def test_response_includes_request_id(self, tts_client):
        """user_id should be passed through as response request_id."""
        payload = {"text": "Hello", "user_id": "user_abc"}
        response = tts_client.post("/synthesize", json=payload)

        assert response.status_code == 200
        assert response.json()["request_id"] == "user_abc"


# ── Audio ─────────────────────────────────────────────────────────────────────

@pytest.mark.unit
class TestAudioEndpoint:

    def test_serve_mp3_file(self, tts_client):
        """GET /audio/test.mp3 should serve the file as audio/mpeg."""
        response = tts_client.get("/audio/test.mp3")
        assert response.status_code == 200
        assert "audio/mpeg" in response.headers["content-type"]

    def test_serve_wav_file(self, tts_client):
        """GET /audio/test.wav should serve the file as audio/wav."""
        response = tts_client.get("/audio/test.wav")
        assert response.status_code == 200
        assert "audio/wav" in response.headers["content-type"]

    def test_missing_file_returns_404(self, tts_client):
        """GET /audio/nonexistent.mp3 should return 404."""
        response = tts_client.get("/audio/nonexistent.mp3")
        assert response.status_code == 404
