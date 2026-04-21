"""
Shared fixtures for SpeechLLm unit tests.

Handles:
- Stubbing heavy third-party dependencies (ElevenLabs SDK, Coqui TTS, PyTorch)
  only when they are not installed in the test environment.
- Setting CWD to SpeechLLm root (required for configs/models.yaml at import time)
- Providing mocked TTS services so tests run without API keys or GPU
"""

import importlib.util
import os
import sys
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest
from fastapi.testclient import TestClient


# ---------------------------------------------------------------------------
# Stub heavy third-party packages that may not be installed in the test
# environment.  Only packages that cannot be found are replaced with
# MagicMock stubs; if a package IS installed it is left untouched so
# that other test suites (e.g. DART integration tests) are unaffected.
# ---------------------------------------------------------------------------

_OPTIONAL_PACKAGES = {
    "elevenlabs": ["elevenlabs", "elevenlabs.client"],
    "TTS":        ["TTS", "TTS.api"],
    "torch":      ["torch"],
    "sounddevice": ["sounddevice"],
    "soundfile":  ["soundfile"],
}

for _top_pkg, _sub_mods in _OPTIONAL_PACKAGES.items():
    if importlib.util.find_spec(_top_pkg) is None:
        for _mod in _sub_mods:
            sys.modules.setdefault(_mod, MagicMock())


# ---------------------------------------------------------------------------
# Session-scoped CWD + environment fixture
#
# api_server.py calls load_yaml("configs/models.yaml") and
# ElevenLabsClient checks os.getenv("ELEVENLABS_API_KEY") at import time.
# Both must be available before the module is first imported.
# ---------------------------------------------------------------------------

@pytest.fixture(scope="session", autouse=True)
def _speechllm_env(request):
    """Set CWD to SpeechLLm root and provide a dummy API key for the session."""
    speechllm_root = Path(request.config.rootdir) / "SpeechLLm"
    original_cwd = os.getcwd()
    os.chdir(speechllm_root)

    original_key = os.environ.get("ELEVENLABS_API_KEY")
    os.environ["ELEVENLABS_API_KEY"] = "test_key_for_unit_tests"

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

    Call with no args for sensible defaults, or pass keyword overrides:
        make_synthesis_result(language="vi", tts_provider="coqui")
    """
    defaults = {
        "message": "Synthesis complete",
        "audio_file": "test_audio.mp3",
        "language": "en",
        "emotion": "neutral",
        "tts_time_sec": 0.123,
        "tts_provider": "elevenlabs",
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
def mock_tts_router():
    """
    Patch tts_router in api_server so /synthesize never calls
    real ElevenLabs or Coqui services.
    """
    with patch("api_server.tts_router") as mock_router:
        mock_router.synthesize.return_value = "data/temp_audio/test_audio.mp3"
        mock_router.last_provider = "elevenlabs"
        yield mock_router


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
def tts_client(mock_tts_router, audio_dir):
    """
    Provide a FastAPI TestClient with TTS services fully mocked.

    Patches:
      - api_server.tts_router  (via mock_tts_router dependency)
      - api_server.audio_dir   (via audio_dir tmp directory)
    """
    with patch("api_server.audio_dir", audio_dir):
        from api_server import app

        with TestClient(app) as client:
            yield client
