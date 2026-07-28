# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Kimodo MCP Server — exposes text-to-motion generation over the Model Context Protocol.

This server wraps the Kimodo motion generation model and exposes it as MCP tools
over Streamable HTTP. The model is pre-loaded into GPU memory at startup to
minimize latency for incoming requests.

Usage:
    python mcp_server.py

Environment Variables:
    MCP_PORT:               Port to serve on (default: 8000)
    MCP_OUTPUT_DIR:         Directory for generated files (default: /workspace/outputs)
    MCP_TTL_SECONDS:        Time-to-live for generated files in seconds (default: 3600)
    TEXT_ENCODER_MODE:      Text encoder mode: local, api, auto (default: local)
    TEXT_ENCODER_DEVICE:    Device for text encoder: cuda, cpu (default: auto)
    CHECKPOINT_DIR:         Local checkpoint directory (optional)
    HF_HOME:               HuggingFace cache directory (optional)
"""

import json
import logging
import os
import threading
import time
import uuid
from datetime import datetime, timezone
from pathlib import Path

import torch

from starlette.responses import Response
from starlette.requests import Request
from fastmcp import FastMCP

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

MCP_PORT = int(os.environ.get("MCP_PORT", "8000"))
OUTPUT_DIR = os.environ.get("MCP_OUTPUT_DIR", "/workspace/outputs")
TTL_SECONDS = int(os.environ.get("MCP_TTL_SECONDS", "3600"))

# Model to pre-load at startup
DEFAULT_MODEL_NAME = "Kimodo-SMPLX-RP-v1"

# Generation defaults
DEFAULT_DURATION = 3.0
DEFAULT_DIFFUSION_STEPS = 100
DEFAULT_NUM_SAMPLES = 1
MAX_DURATION = 5.0

# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger("kimodo-mcp")

# ---------------------------------------------------------------------------
# Global state
# ---------------------------------------------------------------------------

_model = None
_resolved_model_name = None
_model_fps = None
_skeleton = None
_device = None
_startup_time = None
_amass_converter = None


def _load_model():
    """Load the Kimodo model into GPU memory."""
    global _model, _resolved_model_name, _model_fps, _skeleton, _device, _startup_time, _amass_converter

    _device = "cuda:0" if torch.cuda.is_available() else "cpu"
    logger.info(f"Loading model {DEFAULT_MODEL_NAME} on device {_device}...")

    from kimodo import load_model
    from kimodo.model.registry import get_model_info

    _model, _resolved_model_name = load_model(
        DEFAULT_MODEL_NAME,
        device=_device,
        default_family="Kimodo",
        return_resolved_name=True,
    )
    _model_fps = _model.fps
    _skeleton = _model.skeleton

    info = get_model_info(_resolved_model_name)
    display = info.display_name if info else _resolved_model_name
    logger.info(f"Model loaded: {display} ({_resolved_model_name}) at {_model_fps} FPS")

    # Pre-build the AMASS converter for SMPL-X output
    from kimodo.exports.smplx import AMASSConverter

    _amass_converter = AMASSConverter(skeleton=_skeleton, fps=_model_fps)
    logger.info("AMASS converter initialized")

    _startup_time = time.time()
    logger.info("Model ready for inference")


# ---------------------------------------------------------------------------
# TTL cleanup
# ---------------------------------------------------------------------------

# Track generated files with their creation time
_generated_files: dict[str, float] = {}
_files_lock = threading.Lock()


def _register_file(path: str):
    """Register a generated file for TTL tracking."""
    with _files_lock:
        _generated_files[path] = time.time()


def _ttl_cleanup_loop():
    """Background thread that deletes expired files."""
    while True:
        time.sleep(30)  # Check every 30 seconds
        now = time.time()
        expired = []
        with _files_lock:
            for path, created_at in list(_generated_files.items()):
                if now - created_at > TTL_SECONDS:
                    expired.append(path)
                    del _generated_files[path]

        for path in expired:
            try:
                if os.path.exists(path):
                    os.remove(path)
                    logger.info(f"TTL cleanup: removed {path}")
            except OSError as e:
                logger.warning(f"TTL cleanup: failed to remove {path}: {e}")


# ---------------------------------------------------------------------------
# MCP Server
# ---------------------------------------------------------------------------

mcp = FastMCP(
    name="kimodo-motion-gen",
    instructions=(
        "Kimodo Motion Generation Server. "
        "Use the generate_motion tool to create 3D human motions from text prompts. "
        "The model generates SMPL-X compatible motion data in NPZ format."
    ),
)


# ── HTTP file download endpoint ──────────────────────────────────────
@mcp.custom_route("/files/{filename:path}", methods=["GET"])
async def download_file(request: Request) -> Response:
    """Serve generated motion NPZ files via HTTP."""
    filename = request.path_params["filename"]
    filepath = Path(OUTPUT_DIR) / filename
    if not filepath.resolve().is_relative_to(Path(OUTPUT_DIR).resolve()):
        return Response(status_code=403)
    if not filepath.exists():
        return Response(status_code=404)
    return Response(
        content=filepath.read_bytes(),
        media_type="application/octet-stream",
        headers={
            "Content-Disposition": f'attachment; filename="{filepath.name}"',
        },
    )


@mcp.tool()
def generate_motion(prompt: str) -> str:
    """Generate 3D human motion from a text prompt.

    Takes a natural language description of a motion (e.g., 'A person walks forward
    and waves') and generates a 3D motion sequence as SMPL-X compatible NPZ files.

    Args:
        prompt: Text describing the desired motion. Use periods to separate multiple
                sequential motions (e.g., 'A person walks forward. Then they sit down.').

    Returns:
        JSON string containing file paths, metadata, and expiration time for
        the generated motion files.
    """
    if _model is None:
        return json.dumps({"error": "Model not loaded yet. Server is still starting up."})

    duration = DEFAULT_DURATION
    diffusion_steps = DEFAULT_DIFFUSION_STEPS
    num_samples = DEFAULT_NUM_SAMPLES
    seed = None

    # Clamp duration
    if duration > MAX_DURATION:
        duration = MAX_DURATION

    # Parse multi-prompt (split on periods)
    texts = [text.strip() for text in prompt.split(".")]
    texts = [text + "." for text in texts if text]

    if not texts:
        return json.dumps({"error": "Empty prompt provided."})

    num_frames = [int(duration * _model_fps)] * len(texts)

    # Set seed if provided
    if seed is not None:
        from kimodo.tools import seed_everything

        seed_everything(seed)

    logger.info(f"Generating motion: prompts={texts}, duration={duration}s, "
                f"num_frames={num_frames}, diffusion_steps={diffusion_steps}")

    gen_start = time.time()

    try:
        output = _model(
            texts,
            num_frames,
            constraint_lst=[],
            num_denoising_steps=diffusion_steps,
            num_samples=num_samples,
            multi_prompt=True,
            num_transition_frames=5,
            post_processing=True,
            return_numpy=True,
        )
    except Exception as e:
        logger.error(f"Generation failed: {e}", exc_info=True)
        return json.dumps({"error": f"Generation failed: {str(e)}"})

    gen_elapsed = time.time() - gen_start
    logger.info(f"Generation completed in {gen_elapsed:.2f}s")

    # Save outputs
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    file_id = uuid.uuid4().hex[:12]
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    base_name = f"motion_{timestamp}_{file_id}"

    n_samples = int(output["posed_joints"].shape[0])

    from kimodo.exports.motion_io import save_kimodo_npz

    result_files = []

    if n_samples == 1:
        # Save Kimodo NPZ
        npz_path = os.path.join(OUTPUT_DIR, f"{base_name}.npz")
        single = {
            k: (v[0] if hasattr(v, "shape") and len(v.shape) > 0 and v.shape[0] == n_samples else v)
            for k, v in output.items()
        }
        save_kimodo_npz(npz_path, single)
        _register_file(npz_path)

        # Save AMASS NPZ (SMPL-X compatible)
        amass_path = os.path.join(OUTPUT_DIR, f"{base_name}_amass.npz")
        _amass_converter.convert_save_npz(output, amass_path)
        _register_file(amass_path)

        result_files.append({
            "kimodo_npz": npz_path,
            "amass_npz": amass_path,
        })
    else:
        sample_dir = os.path.join(OUTPUT_DIR, base_name)
        os.makedirs(sample_dir, exist_ok=True)
        for i in range(n_samples):
            single = {
                k: (v[i] if hasattr(v, "shape") and len(v.shape) > 0 and v.shape[0] == n_samples else v)
                for k, v in output.items()
            }
            npz_path = os.path.join(sample_dir, f"{base_name}_{i:02d}.npz")
            save_kimodo_npz(npz_path, single)
            _register_file(npz_path)

            amass_path = os.path.join(sample_dir, f"{base_name}_{i:02d}_amass.npz")
            _amass_converter.convert_save_npz(
                {k: (v[i:i+1] if hasattr(v, "shape") and len(v.shape) > 0 and v.shape[0] == n_samples else v)
                 for k, v in output.items()},
                amass_path,
            )
            _register_file(amass_path)

            result_files.append({
                "kimodo_npz": npz_path,
                "amass_npz": amass_path,
            })

    expires_at = datetime.fromtimestamp(
        time.time() + TTL_SECONDS, tz=timezone.utc
    ).isoformat()

    total_frames = sum(num_frames)

    return json.dumps({
        "status": "success",
        "files": result_files,
        "metadata": {
            "prompt": prompt,
            "texts": texts,
            "model": _resolved_model_name,
            "duration_seconds": duration,
            "total_frames": total_frames,
            "fps": _model_fps,
            "num_samples": n_samples,
            "diffusion_steps": diffusion_steps,
            "seed": seed,
            "generation_time_seconds": round(gen_elapsed, 2),
        },
        "expires_at": expires_at,
    })


@mcp.tool()
def list_models() -> str:
    """List all available Kimodo model variants and the currently loaded model.

    Returns:
        JSON string with available models and which one is currently loaded.
    """
    from kimodo import AVAILABLE_MODELS
    from kimodo.model.registry import get_model_info

    models = []
    for key in AVAILABLE_MODELS:
        info = get_model_info(key)
        if info:
            models.append({
                "short_key": info.short_key,
                "display_name": info.display_name,
                "repo_id": info.repo_id,
                "skeleton": info.skeleton,
                "dataset": info.dataset_ui_label,
                "version": info.version,
            })
        else:
            models.append({"short_key": key})

    return json.dumps({
        "available_models": models,
        "loaded_model": _resolved_model_name,
        "default_model": DEFAULT_MODEL_NAME,
    })


@mcp.tool()
def health_check() -> str:
    """Check the server's health status, loaded model info, and GPU memory usage.

    Returns:
        JSON string with server status, model info, GPU stats, and uptime.
    """
    status = {
        "status": "healthy" if _model is not None else "loading",
        "model_loaded": _model is not None,
        "loaded_model": _resolved_model_name,
        "device": str(_device) if _device else None,
        "uptime_seconds": round(time.time() - _startup_time, 1) if _startup_time else None,
    }

    # GPU info
    if torch.cuda.is_available():
        gpu_mem = torch.cuda.mem_get_info(0)
        status["gpu"] = {
            "name": torch.cuda.get_device_name(0),
            "memory_free_mb": round(gpu_mem[0] / 1024 / 1024, 1),
            "memory_total_mb": round(gpu_mem[1] / 1024 / 1024, 1),
            "memory_used_mb": round((gpu_mem[1] - gpu_mem[0]) / 1024 / 1024, 1),
        }
    else:
        status["gpu"] = None

    # File tracking
    with _files_lock:
        status["tracked_files"] = len(_generated_files)

    status["output_dir"] = OUTPUT_DIR
    status["ttl_seconds"] = TTL_SECONDS

    return json.dumps(status)


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    # Ensure output directory exists
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    # Load model before starting the server
    _load_model()

    # Start TTL cleanup thread
    cleanup_thread = threading.Thread(target=_ttl_cleanup_loop, daemon=True)
    cleanup_thread.start()
    logger.info(f"TTL cleanup thread started (TTL={TTL_SECONDS}s)")

    # Start MCP server
    logger.info(f"Starting MCP server on port {MCP_PORT}...")
    mcp.run(transport="streamable-http", host="0.0.0.0", port=MCP_PORT)
