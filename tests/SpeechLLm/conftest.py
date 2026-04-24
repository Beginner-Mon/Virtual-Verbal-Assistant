"""
Shared fixtures for SpeechLLm unit tests.

Handles:
- Stubbing ElevenLabs to force fallback to Coqui for local integration testing.
- Setting CWD to SpeechLLm root (required for configs/models.yaml at import time).
"""

import importlib.util
import os
import sys
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest
from fastapi.testclient import TestClient


# ---------------------------------------------------------------------------
# Stub ElevenLabs so we don't consume credits, forcing fallback to Coqui
# ---------------------------------------------------------------------------

_OPTIONAL_PACKAGES = {
    "elevenlabs": ["elevenlabs", "elevenlabs.client"],
}

for _top_pkg, _sub_mods in _OPTIONAL_PACKAGES.items():
    if importlib.util.find_spec(_top_pkg) is None:
        for _mod in _sub_mods:
            sys.modules.setdefault(_mod, MagicMock())


# ---------------------------------------------------------------------------
# Session-scoped CWD + environment fixture
# ---------------------------------------------------------------------------

@pytest.fixture(scope="session", autouse=True)
def _speechllm_env(request):
    """Set CWD to SpeechLLm root and provide a dummy API key for the session."""
    speechllm_root = Path(request.config.rootdir) / "SpeechLLm"
    original_cwd = os.getcwd()
    os.chdir(speechllm_root)

    original_key = os.environ.get("ELEVENLABS_API_KEY")
    os.environ["ELEVENLABS_API_KEY"] = "test_key_for_unit_tests"
    os.environ["COQUI_USE_GPU"] = "false"

    yield

    os.chdir(original_cwd)
    if original_key is None:
        os.environ.pop("ELEVENLABS_API_KEY", None)
    else:
        os.environ["ELEVENLABS_API_KEY"] = original_key


# ---------------------------------------------------------------------------
# Mock data factory
# ---------------------------------------------------------------------------

def make_synthesis_result(**overrides) -> dict:
    """
    Build a realistic /synthesize response dict.
    """
    defaults = {
        "message": "Synthesis complete",
        "audio_file": "test_audio.mp3",
        "language": "en",
        "emotion": "neutral",
        "tts_time_sec": 0.123,
        "tts_provider": "coqui",
        "request_id": None,
    }
    defaults.update(overrides)
    return defaults


# ---------------------------------------------------------------------------
# Core fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def synthesis_result():
    """Expose the default synthesis result dict for assertions."""
    return make_synthesis_result()


@pytest.fixture
def audio_dir(tmp_path):
    """Create a temp audio directory pre-populated with sample test files."""
    d = tmp_path / "audio"
    d.mkdir()
    # Minimal binary stubs — just enough for FileResponse to serve
    (d / "test.mp3").write_bytes(b"\xff\xfb\x90\x00" + b"\x00" * 100)
    (d / "test.wav").write_bytes(b"RIFF" + b"\x00" * 100)
    return d


@pytest.fixture
def force_coqui_fallback():
    """
    Patch ElevenLabsClient.synthesize to always fail, forcing the TTSRouter
    to fall back to Coqui TTS.
    """
    with patch("src.services.elevenlabs_client.ElevenLabsClient.synthesize", side_effect=Exception("Forced fallback for testing")):
        yield


@pytest.fixture
def tts_client(audio_dir, force_coqui_fallback):
    """
    Provide a FastAPI TestClient that uses the real TTSRouter,
    but with ElevenLabs forced to fail.
    """
    with patch("api_server.audio_dir", audio_dir):
        from api_server import app

        with TestClient(app) as client:
            yield client

