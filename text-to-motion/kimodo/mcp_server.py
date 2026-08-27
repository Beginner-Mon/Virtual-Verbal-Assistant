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

from motion_engine import MotionEngine, build_base_name

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

# Model loading, generation and output saving live in MotionEngine (motion_engine.py) so
# worker.py (job-queue path) does not duplicate them. This is the only engine instance for
# the local MCP path's process lifetime.
_engine = None
_startup_time = None


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
    if _engine is None or _engine.model is None:
        return json.dumps({"error": "Model not loaded yet. Server is still starting up."})

    duration = DEFAULT_DURATION
    diffusion_steps = DEFAULT_DIFFUSION_STEPS
    seed = None

    # Clamp duration
    if duration > MAX_DURATION:
        duration = MAX_DURATION

    # Parse multi-prompt (split on periods) — mirrors MotionEngine.generate's own parsing,
    # done here too so we can validate and log before calling into the engine.
    texts = [text.strip() for text in prompt.split(".")]
    texts = [text + "." for text in texts if text]

    if not texts:
        return json.dumps({"error": "Empty prompt provided."})

    num_frames = [int(duration * _engine.model.fps)] * len(texts)

    # Set seed if provided
    if seed is not None:
        from kimodo.tools import seed_everything

        seed_everything(seed)

    logger.info(f"Generating motion: prompts={texts}, duration={duration}s, "
                f"num_frames={num_frames}, diffusion_steps={diffusion_steps}")

    gen_start = time.time()

    try:
        output = _engine.generate(prompt, duration, diffusion_steps)
    except Exception as e:
        logger.error(f"Generation failed: {e}", exc_info=True)
        return json.dumps({"error": f"Generation failed: {str(e)}"})

    gen_elapsed = time.time() - gen_start
    logger.info(f"Generation completed in {gen_elapsed:.2f}s")

    # Save outputs. The filename is the job id (not a timestamp+random suffix) so the
    # output URL is knowable before generation even starts and repeat requests can hit a
    # content cache — see motion_engine.build_base_name.
    job_id = uuid.uuid4().hex[:12]
    base_name = build_base_name(job_id)
    npz_path, bvh_path = _engine.save_outputs(output, OUTPUT_DIR, base_name)
    _register_file(npz_path)
    _register_file(bvh_path)

    n_samples = int(output["posed_joints"].shape[0])

    expires_at = datetime.fromtimestamp(
        time.time() + TTL_SECONDS, tz=timezone.utc
    ).isoformat()

    total_frames = sum(num_frames)

    return json.dumps({
        "status": "success",
        "files": {
            "npz": npz_path,
            "bvh": bvh_path,
        },
        "metadata": {
            "prompt": prompt,
            "texts": texts,
            "model": _engine.resolved_model_name,
            "duration_seconds": duration,
            "total_frames": total_frames,
            "fps": _engine.model.fps,
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
        "loaded_model": _engine.resolved_model_name if _engine else None,
        "default_model": DEFAULT_MODEL_NAME,
    })


@mcp.tool()
def health_check() -> str:
    """Check the server's health status, loaded model info, and GPU memory usage.

    Returns:
        JSON string with server status, model info, GPU stats, and uptime.
    """
    model_loaded = _engine is not None and _engine.model is not None
    status = {
        "status": "healthy" if model_loaded else "loading",
        "model_loaded": model_loaded,
        "loaded_model": _engine.resolved_model_name if _engine else None,
        "device": str(_engine.device) if _engine and _engine.device else None,
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
    logger.info(f"Loading model {DEFAULT_MODEL_NAME}...")
    _engine = MotionEngine(DEFAULT_MODEL_NAME)
    _engine.load()
    _startup_time = time.time()
    logger.info(
        f"Model loaded: {_engine.resolved_model_name} on device {_engine.device} "
        f"at {_engine.model.fps} FPS"
    )

    # Start TTL cleanup thread
    cleanup_thread = threading.Thread(target=_ttl_cleanup_loop, daemon=True)
    cleanup_thread.start()
    logger.info(f"TTL cleanup thread started (TTL={TTL_SECONDS}s)")

    # Start MCP server
    logger.info(f"Starting MCP server on port {MCP_PORT}...")
    mcp.run(transport="streamable-http", host="0.0.0.0", port=MCP_PORT)
