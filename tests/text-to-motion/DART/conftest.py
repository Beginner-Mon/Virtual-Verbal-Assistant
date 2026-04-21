"""
Shared fixtures for DART (text-to-motion) unit tests.

Handles:
- Setting CWD to DART root (required by DART imports that use relative data/ paths)
- Providing a fully-mocked MotionGenerator so tests run without GPU or model weights
"""

import os
from pathlib import Path

import pytest
from unittest.mock import patch
from fastapi.testclient import TestClient


# ---------------------------------------------------------------------------
# Session-scoped CWD fixture
#
# DART source code loads SMPL-X model files via relative paths like
# `data/smplx_lockedhead_20230207/...` at import time. CWD must point
# to the DART source root before any DART module is imported.
# We derive the path from pytest's rootdir (set by pytest.ini location).
# ---------------------------------------------------------------------------

@pytest.fixture(scope="session", autouse=True)
def _dart_cwd(request):
    """Set CWD to DART source root for the session, then restore."""
    dart_root = Path(request.config.rootdir) / "text-to-motion" / "DART"
    original_cwd = os.getcwd()
    os.chdir(dart_root)
    yield
    os.chdir(original_cwd)


# ---------------------------------------------------------------------------
# Mock data factory
# ---------------------------------------------------------------------------

def make_generation_result(**overrides) -> dict:
    """
    Build a realistic MotionGenerator.generate() return value.

    Call with no args for sensible defaults, or pass keyword overrides:
        make_generation_result(num_frames=120, duration_seconds=4.0)
    """
    defaults = {
        "request_id": "test_abcd1234",
        "motion_file": "outputs/motion_test_abcd1234.npz",
        "num_frames": 60,
        "fps": 30,
        "duration_seconds": 2.0,
        "text_prompt": "walk forward",
    }
    defaults.update(overrides)
    return defaults


# ---------------------------------------------------------------------------
# Core fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def generation_result():
    """Expose the default generation result dict for assertions in tests."""
    return make_generation_result()


@pytest.fixture
def mock_motion_generator():
    """
    Patch MotionGenerator so that importing api_server never loads
    real PyTorch models, SMPL-X weights, or CLIP encoders.
    """
    with patch("api_server.MotionGenerator") as MockClass:
        mock_instance = MockClass.return_value
        mock_instance.load_models.return_value = None
        mock_instance.generate.return_value = make_generation_result()
        yield mock_instance


@pytest.fixture
def dart_client(mock_motion_generator):
    """Provide a FastAPI TestClient with the MotionGenerator fully mocked."""
    from api_server import app

    with TestClient(app) as client:
        yield client


@pytest.fixture(scope="module")
def real_dart_client():
    """
    Provide a FastAPI TestClient with the REAL MotionGenerator.

    This loads actual model weights (~2-5 min on first call).
    Only used by integration tests. Module-scoped so models load once.
    """
    from api_server import app

    with TestClient(app) as client:
        yield client
