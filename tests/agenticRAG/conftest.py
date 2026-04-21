"""
Shared fixtures for agenticRAG unit tests.

Handles:
- Resetting the settings singleton between tests that manipulate env vars
- Providing mock data factories for realistic test payloads
- FastAPI TestClient with all downstream services mocked
"""

import os
import sys
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from fastapi.testclient import TestClient


# ---------------------------------------------------------------------------
# Session-scoped CWD + environment fixture
#
# main_api.py imports core.config.settings which imports utils.logger,
# which calls get_config() at import time to load config/config.yaml.
# CWD must point to the agenticRAG source root before any import.
# ---------------------------------------------------------------------------

@pytest.fixture(scope="session", autouse=True)
def _agenticrag_env(request):
    """Set CWD to agenticRAG source root and provide dummy env vars."""
    agenticrag_root = os.path.join(str(request.config.rootdir), "agenticRAG", "agentic_rag_gemini")
    original_cwd = os.getcwd()
    os.chdir(agenticrag_root)
    yield
    os.chdir(original_cwd)


# ---------------------------------------------------------------------------
# Reset the settings singleton between tests
# ---------------------------------------------------------------------------

@pytest.fixture(autouse=True)
def _reset_settings():
    """Reset cached MainAPISettings singleton so env var changes take effect."""
    import core.config.settings as settings_mod
    original = settings_mod._SETTINGS
    settings_mod._SETTINGS = None
    yield
    settings_mod._SETTINGS = original


# ---------------------------------------------------------------------------
# Mock data factories
# ---------------------------------------------------------------------------

def make_answer_response(**overrides) -> dict:
    """
    Build a realistic AnswerResponse dict.

    Call with no args for sensible defaults, or pass keyword overrides:
        make_answer_response(language="vi", status="processing")
    """
    from schemas.main_api import AnswerResponse

    defaults = {
        "request_id": "test_abc123",
        "status": "completed",
        "pending_services": [],
        "language": "en",
        "selected_strategy": "knowledge_query",
        "progress_stage": "completed",
        "text_answer": "Try gentle neck stretches.",
        "exercises": [{"name": "neck stretch", "reason": "neck pain relief"}],
        "motion": None,
        "motion_job": None,
        "tts": None,
        "generation_time_ms": 123.4,
        "errors": None,
        "debug": None,
    }
    defaults.update(overrides)
    return defaults


def make_rag_data(**overrides) -> dict:
    """
    Build a realistic AgenticRAG /query response dict.
    """
    defaults = {
        "text_answer": "Try gentle neck stretches.",
        "exercises": [{"name": "neck stretch", "reason": "neck pain relief"}],
        "language": "en",
        "motion": None,
        "motion_job": None,
        "exercise_motion_prompt": None,
        "orchestrator_decision": {
            "intent": "knowledge_query",
            "confidence": 0.95,
        },
    }
    defaults.update(overrides)
    return defaults


# ---------------------------------------------------------------------------
# Core fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def answer_response_data():
    """Expose the default answer response dict for assertions."""
    return make_answer_response()


@pytest.fixture
def mock_check_services_health():
    """Patch check_services_health so /health never makes real HTTP calls."""
    with patch("routers.health.check_services_health", new_callable=AsyncMock) as mock_fn:
        mock_fn.return_value = {"agenticrag": "ok", "dart": "ok"}
        yield mock_fn


@pytest.fixture
def main_api_client(mock_check_services_health):
    """
    Provide a FastAPI TestClient for main_api.py with health checks mocked.

    Note: The /answer endpoint requires additional mocking via inline patches.
    """
    from main_api import app

    with TestClient(app) as client:
        yield client
