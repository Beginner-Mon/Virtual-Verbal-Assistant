"""
Integration tests for the DART text-to-motion pipeline.

These tests use the REAL MotionGenerator — no mocking.
They verify that the full ML pipeline (diffusion model → SMPL-X → NPZ/GLB)
produces valid output.

Requirements:
- GPU or CPU with enough memory
- Model weights present at text-to-motion/DART/mld_denoiser/...
- Standing seed file at text-to-motion/DART/data/stand.pkl

Run with:
    pytest tests/text-to-motion/DART/test_integration.py -m integration
Skip with:
    pytest -m "not integration"
"""

import numpy as np
import pytest
from pathlib import Path


@pytest.mark.integration
@pytest.mark.slow
class TestMotionGeneration:
    """End-to-end tests that exercise the real diffusion model."""

    def test_health_with_real_model(self, real_dart_client):
        """Verify the server boots and models load successfully."""
        response = real_dart_client.get("/health")
        assert response.status_code == 200
        assert response.json() == {"status": "ok"}

    def test_generate_npz_produces_valid_motion(self, real_dart_client):
        """Generate a short motion clip and validate the NPZ file structure."""
        payload = {
            "text_prompt": "walk forward",
            "duration_seconds": 2.0,
            "output_format": "npz",
            "gender": "female",
        }
        response = real_dart_client.post("/generate", json=payload)

        assert response.status_code == 200
        data = response.json()

        # ── Response shape checks ────────────────────────────────────────
        assert data["num_frames"] > 0
        assert data["fps"] == 30
        assert data["duration_seconds"] > 0
        assert data["text_prompt"] == "walk forward"
        assert data["motion_file_url"].endswith(".npz")

        # ── Download and inspect the NPZ file ───────────────────────────
        download_url = data["motion_file_url"]
        dl_response = real_dart_client.get(download_url)
        assert dl_response.status_code == 200

        # Parse the NPZ binary content
        import io
        npz_data = np.load(io.BytesIO(dl_response.content))

        # Verify required keys exist
        required_keys = {"mocap_framerate", "gender", "betas", "poses", "trans"}
        assert required_keys.issubset(set(npz_data.files)), (
            f"Missing keys in NPZ: {required_keys - set(npz_data.files)}"
        )

        # ── Data shape validation ────────────────────────────────────────
        poses = npz_data["poses"]
        trans = npz_data["trans"]
        betas = npz_data["betas"]

        num_frames = data["num_frames"]

        # poses: (num_frames, 165) — 55 joints × 3 axis-angle
        assert poses.shape == (num_frames, 165), f"Unexpected poses shape: {poses.shape}"

        # trans: (num_frames, 3) — XYZ root translation per frame
        assert trans.shape == (num_frames, 3), f"Unexpected trans shape: {trans.shape}"

        # betas: (10,) — SMPL-X body shape parameters
        assert betas.shape == (10,), f"Unexpected betas shape: {betas.shape}"

        # framerate should be 30
        assert int(npz_data["mocap_framerate"]) == 30

    def test_generate_glb_produces_file(self, real_dart_client):
        """Generate a motion clip in GLB format and verify it's downloadable."""
        payload = {
            "text_prompt": "wave hand",
            "duration_seconds": 1.0,
            "output_format": "glb",
            "gender": "female",
        }
        response = real_dart_client.post("/generate", json=payload)

        assert response.status_code == 200
        data = response.json()
        assert data["motion_file_url"].endswith(".glb")

        # Verify the GLB file is downloadable and non-empty
        dl_response = real_dart_client.get(data["motion_file_url"])
        assert dl_response.status_code == 200
        assert len(dl_response.content) > 0, "GLB file is empty"

    def test_duration_maps_to_expected_frame_count(self, real_dart_client):
        """Verify that duration_seconds roughly maps to the correct number of frames."""
        payload = {
            "text_prompt": "stand still",
            "duration_seconds": 3.0,
            "output_format": "npz",
            "gender": "female",
        }
        response = real_dart_client.post("/generate", json=payload)

        assert response.status_code == 200
        data = response.json()

        # At 30 fps, 3 seconds ≈ 90 frames.
        # Allow generous tolerance since primitives are quantized.
        expected_frames = 3.0 * 30
        actual_frames = data["num_frames"]
        assert actual_frames > expected_frames * 0.5, (
            f"Too few frames: got {actual_frames}, expected ~{expected_frames}"
        )
        assert actual_frames < expected_frames * 2.0, (
            f"Too many frames: got {actual_frames}, expected ~{expected_frames}"
        )

    def test_seed_produces_deterministic_output(self, real_dart_client):
        """Two requests with the same seed should produce identical motion."""
        payload = {
            "text_prompt": "walk forward",
            "duration_seconds": 1.0,
            "output_format": "npz",
            "seed": 42,
            "gender": "female",
        }

        resp1 = real_dart_client.post("/generate", json=payload)
        resp2 = real_dart_client.post("/generate", json=payload)

        assert resp1.status_code == 200
        assert resp2.status_code == 200

        # Download both NPZ files
        import io
        npz1 = np.load(io.BytesIO(
            real_dart_client.get(resp1.json()["motion_file_url"]).content
        ))
        npz2 = np.load(io.BytesIO(
            real_dart_client.get(resp2.json()["motion_file_url"]).content
        ))

        # Poses and translations should be identical
        np.testing.assert_array_equal(npz1["poses"], npz2["poses"])
        np.testing.assert_array_equal(npz1["trans"], npz2["trans"])
