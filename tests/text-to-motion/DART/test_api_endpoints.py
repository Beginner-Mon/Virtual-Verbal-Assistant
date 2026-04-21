"""
Unit tests for the DART text-to-motion API endpoints.

Tests cover:
- Health check
- Successful motion generation (GLB and NPZ formats)
- Request validation (invalid / missing fields)
- Download endpoint (404 for missing files)
- Service unavailability (503 when generator is None)
"""

import pytest
from unittest.mock import patch


# ── Health ────────────────────────────────────────────────────────────────────

@pytest.mark.unit
class TestHealthEndpoint:

    def test_returns_ok(self, dart_client):
        """GET /health should return 200 with status ok."""
        response = dart_client.get("/health")
        assert response.status_code == 200
        assert response.json() == {"status": "ok"}


# ── Generate ──────────────────────────────────────────────────────────────────

@pytest.mark.unit
class TestGenerateEndpoint:

    def test_success_glb_format(self, dart_client, mock_motion_generator, generation_result):
        """POST /generate with output_format=glb should invoke the model and GLB converter."""
        with patch("api_server.convert_npz_to_glb") as mock_convert:
            payload = {
                "text_prompt": "walk forward",
                "duration_seconds": 2.0,
                "guidance_scale": 5.0,
                "output_format": "glb",
                "gender": "female",
            }
            response = dart_client.post("/generate", json=payload)

            assert response.status_code == 200
            data = response.json()
            assert data["request_id"] == generation_result["request_id"]
            assert data["motion_file_url"].endswith(".glb")

            mock_motion_generator.generate.assert_called_once_with(
                text_prompt="walk forward",
                duration_seconds=2.0,
                guidance_scale=5.0,
                num_steps=50,
                seed=None,
                respacing="",
                gender="female",
            )
            mock_convert.assert_called_once()

    def test_success_npz_format(self, dart_client, mock_motion_generator, generation_result):
        """POST /generate with output_format=npz should skip GLB conversion."""
        with patch("api_server.convert_npz_to_glb") as mock_convert:
            payload = {
                "text_prompt": "run in circles",
                "output_format": "npz",
            }
            response = dart_client.post("/generate", json=payload)

            assert response.status_code == 200
            data = response.json()
            assert data["request_id"] == generation_result["request_id"]
            assert data["motion_file_url"].endswith(".npz")

            # GLB converter must NOT be called for npz output
            mock_convert.assert_not_called()

    def test_invalid_duration_too_high(self, dart_client):
        """duration_seconds above the schema max (120) should return 422."""
        payload = {
            "text_prompt": "walk forward",
            "duration_seconds": 200.0,
        }
        response = dart_client.post("/generate", json=payload)
        assert response.status_code == 422

    def test_missing_text_prompt(self, dart_client):
        """text_prompt is required — omitting it should return 422."""
        payload = {
            "duration_seconds": 3.0,
        }
        response = dart_client.post("/generate", json=payload)
        assert response.status_code == 422

    def test_service_unavailable(self, dart_client):
        """When the generator is None (not loaded yet), should return 503."""
        with patch("api_server.generator", None):
            payload = {
                "text_prompt": "wave hand",
            }
            response = dart_client.post("/generate", json=payload)
            assert response.status_code == 503


# ── Download ──────────────────────────────────────────────────────────────────

@pytest.mark.unit
class TestDownloadEndpoint:

    def test_missing_file_returns_404(self, dart_client):
        """GET /download with a nonexistent filename should return 404."""
        response = dart_client.get("/download/nonexistent_file.glb")
        assert response.status_code == 404
