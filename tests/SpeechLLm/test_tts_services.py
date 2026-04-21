"""
Unit tests for the SpeechLLm TTS service layer.

Tests cover:
- TTSRouter fallback logic (ElevenLabs → Coqui)
- ElevenLabsClient initialization and validation
- CoquiClient text chunking and validation
"""

import os
import pytest
from unittest.mock import MagicMock, patch


# ── TTSRouter ─────────────────────────────────────────────────────────────────

@pytest.mark.unit
class TestTTSRouter:

    def test_prefers_elevenlabs(self):
        """When both clients available, ElevenLabs should be called first."""
        from src.services.tts_router import TTSRouter

        mock_eleven = MagicMock()
        mock_eleven.synthesize.return_value = "/audio/eleven.mp3"
        mock_coqui = MagicMock()

        router = TTSRouter(eleven_client=mock_eleven, coqui_client=mock_coqui)
        result = router.synthesize("hello")

        assert result == "/audio/eleven.mp3"
        assert router.last_provider == "elevenlabs"
        mock_eleven.synthesize.assert_called_once_with("hello")
        mock_coqui.synthesize.assert_not_called()

    def test_fallback_to_coqui_on_quota_exceeded(self):
        """ElevenLabs 'quota_exceeded' error should trigger Coqui fallback."""
        from src.services.tts_router import TTSRouter

        mock_eleven = MagicMock()
        mock_eleven.synthesize.side_effect = Exception("quota_exceeded")
        mock_coqui = MagicMock()
        mock_coqui.synthesize.return_value = "/audio/coqui.wav"

        router = TTSRouter(eleven_client=mock_eleven, coqui_client=mock_coqui)
        result = router.synthesize("hello", language="en")

        assert result == "/audio/coqui.wav"
        assert router.last_provider == "coqui"

    def test_fallback_to_coqui_on_credits_error(self):
        """ElevenLabs 'credits' error should trigger Coqui fallback."""
        from src.services.tts_router import TTSRouter

        mock_eleven = MagicMock()
        mock_eleven.synthesize.side_effect = Exception("insufficient credits")
        mock_coqui = MagicMock()
        mock_coqui.synthesize.return_value = "/audio/coqui.wav"

        router = TTSRouter(eleven_client=mock_eleven, coqui_client=mock_coqui)
        result = router.synthesize("hello")

        assert result == "/audio/coqui.wav"
        assert router.last_provider == "coqui"

    def test_fallback_to_coqui_on_generic_error(self):
        """Any ElevenLabs failure should fall back to Coqui."""
        from src.services.tts_router import TTSRouter

        mock_eleven = MagicMock()
        mock_eleven.synthesize.side_effect = RuntimeError("connection timeout")
        mock_coqui = MagicMock()
        mock_coqui.synthesize.return_value = "/audio/coqui.wav"

        router = TTSRouter(eleven_client=mock_eleven, coqui_client=mock_coqui)
        result = router.synthesize("hello")

        assert result == "/audio/coqui.wav"
        assert router.last_provider == "coqui"

    def test_coqui_only_when_no_elevenlabs(self):
        """With eleven_client=None, Coqui should be used directly."""
        from src.services.tts_router import TTSRouter

        mock_coqui = MagicMock()
        mock_coqui.synthesize.return_value = "/audio/coqui.wav"

        router = TTSRouter(eleven_client=None, coqui_client=mock_coqui)
        result = router.synthesize("hello", language="vi")

        assert result == "/audio/coqui.wav"
        assert router.last_provider == "coqui"
        mock_coqui.synthesize.assert_called_once_with("hello", language="vi")

    def test_raises_when_both_unavailable(self):
        """With both clients None, should raise RuntimeError."""
        from src.services.tts_router import TTSRouter

        router = TTSRouter(eleven_client=None, coqui_client=None)
        with pytest.raises(RuntimeError, match="All TTS providers failed"):
            router.synthesize("hello")

    def test_language_passed_to_coqui_on_fallback(self):
        """Language parameter should be forwarded to Coqui on fallback."""
        from src.services.tts_router import TTSRouter

        mock_eleven = MagicMock()
        mock_eleven.synthesize.side_effect = Exception("error")
        mock_coqui = MagicMock()
        mock_coqui.synthesize.return_value = "/audio/coqui.wav"

        router = TTSRouter(eleven_client=mock_eleven, coqui_client=mock_coqui)
        router.synthesize("xin chào", language="vi")

        mock_coqui.synthesize.assert_called_once_with("xin chào", language="vi")


# ── ElevenLabsClient ──────────────────────────────────────────────────────────

@pytest.mark.unit
class TestElevenLabsClient:

    def test_missing_api_key_raises(self, monkeypatch):
        """Should raise ValueError when ELEVENLABS_API_KEY is not set."""
        from src.services.elevenlabs_client import ElevenLabsClient

        monkeypatch.delenv("ELEVENLABS_API_KEY", raising=False)
        with pytest.raises(ValueError, match="ELEVENLABS_API_KEY"):
            ElevenLabsClient()

    def test_empty_text_raises(self):
        """synthesize('') should raise ValueError."""
        from src.services.elevenlabs_client import ElevenLabsClient

        client = ElevenLabsClient()
        with pytest.raises(ValueError, match="empty"):
            client.synthesize("")

    def test_whitespace_text_raises(self):
        """synthesize('   ') should raise ValueError."""
        from src.services.elevenlabs_client import ElevenLabsClient

        client = ElevenLabsClient()
        with pytest.raises(ValueError, match="empty"):
            client.synthesize("   ")

    def test_estimate_duration_one_second(self):
        """16 000 bytes at 128 kbps should be ~1.0 second."""
        from src.services.elevenlabs_client import ElevenLabsClient

        client = ElevenLabsClient()
        assert client._estimate_duration(b"\x00" * 16000) == 1.0

    def test_estimate_duration_half_second(self):
        """8 000 bytes at 128 kbps should be ~0.5 seconds."""
        from src.services.elevenlabs_client import ElevenLabsClient

        client = ElevenLabsClient()
        assert client._estimate_duration(b"\x00" * 8000) == 0.5


# ── CoquiClient ──────────────────────────────────────────────────────────────

@pytest.mark.unit
class TestCoquiClient:

    def _make_client(self, **config_overrides):
        from src.services.coqui_client import CoquiClient

        config = {
            "use_gpu": False,
            "output_dir": "data/temp_audio",
            "speaker": None,
            "language_models": {
                "en": "tts_models/en/vctk/vits",
                "vi": "tts_models/vi/vivos/vits",
            },
        }
        config.update(config_overrides)
        return CoquiClient(config)

    def test_empty_text_raises(self):
        """synthesize('') should raise ValueError."""
        client = self._make_client()
        with pytest.raises(ValueError, match="empty"):
            client.synthesize("")

    def test_whitespace_text_raises(self):
        """synthesize('   ') should raise ValueError."""
        client = self._make_client()
        with pytest.raises(ValueError, match="empty"):
            client.synthesize("   ")

    def test_smart_chunks_splits_on_period(self):
        """_smart_chunks should split on sentence boundaries."""
        client = self._make_client()
        text = "A" * 150 + ". " + "B" * 50
        chunks = client._smart_chunks(text)

        assert len(chunks) == 2
        assert chunks[0].endswith(".")

    def test_smart_chunks_short_text_no_split(self):
        """Text under max_chars should remain a single chunk."""
        client = self._make_client()
        chunks = client._smart_chunks("Hello world.")

        assert len(chunks) == 1
        assert chunks[0] == "Hello world."

    def test_smart_chunks_no_period_splits_at_max(self):
        """Long text without periods should be split into multiple chunks."""
        client = self._make_client()
        text = "A" * 300
        chunks = client._smart_chunks(text, max_chars=200)

        assert len(chunks) >= 2
        assert "".join(chunks) == text  # no data lost

    def test_unsupported_language_raises(self):
        """Loading a model for an unsupported language should raise ValueError."""
        client = self._make_client()
        with pytest.raises(ValueError, match="Unsupported language"):
            client._load_model_for_language("zz")

    def test_gpu_fallback_to_cpu(self):
        """When GPU requested but CUDA unavailable, should fall back to CPU."""
        with patch("torch.cuda.is_available", return_value=False):
            with patch.dict(os.environ, {"COQUI_USE_GPU": "true"}):
                client = self._make_client()
                assert client.use_gpu is False
