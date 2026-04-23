"""
MCP Server for DART Text-to-Motion Generation.

Lightweight bridge that delegates to the DART REST API (api_server.py)
via HTTP.  Runs in a separate Python 3.10+ venv so the DART conda
environment (Python 3.8) stays untouched.

Setup (one-time):
    cd /path/to/DART
    python3.10 -m venv .venv-mcp
    source .venv-mcp/bin/activate
    pip install "mcp[cli]" httpx

Usage:
    # Start DART API first (in another terminal):
    conda activate DART && python api_server.py

    # Then run MCP server:
    source .venv-mcp/bin/activate
    python mcp_server.py                   # stdio transport (production)
    mcp dev mcp_server.py                  # MCP Inspector (debugging)

Environment variables:
    DART_API_URL  - Base URL of the DART REST API (default: http://localhost:5001)
"""

import os
import sys
import json
import logging

import httpx
from mcp.server.fastmcp import FastMCP

# ── Logging (stderr only — stdout is reserved for MCP stdio protocol) ────────
logging.basicConfig(
    stream=sys.stderr,
    level=logging.INFO,
    format="%(asctime)s [%(name)s] %(levelname)s: %(message)s",
)
logger = logging.getLogger("dart-mcp")


# ── Configuration ────────────────────────────────────────────────────────────

DART_API_URL = os.getenv("DART_API_URL", "http://localhost:5001")


# ── MCP Server Instance ──────────────────────────────────────────────────────

mcp = FastMCP(
    "dart-motion",
    instructions=(
        "Generate 3D human body motion sequences from text descriptions "
        "using the DART diffusion model.  Returns downloadable GLB/NPZ files."
    ),
)


# ── Tools ─────────────────────────────────────────────────────────────────────


@mcp.tool()
async def generate_motion(
    text_prompt: str,
    duration_seconds: float | None = None,
    guidance_scale: float = 5.0,
    seed: int | None = None,
    respacing: str = "",
    gender: str = "female",
    output_format: str = "glb",
) -> str:
    """Generate a 3D human motion sequence from a text description.

    Calls the DART REST API to generate SMPL-X body motion and returns
    metadata with a download URL for the resulting file.

    Args:
        text_prompt: Motion description.  Supports several formats:
            - Simple action:   'walk forward'
            - Comma sequences: 'walk forward, turn left, sit down'
            - Repeat syntax:   'walk forward*12'  (12 motion primitives)
            - Use duration_seconds for time-based control instead.
        duration_seconds: Desired clip length in seconds (1.0 - 120.0).
            Ignored if the text_prompt uses explicit '*count' syntax.
        guidance_scale: Classifier-free guidance strength (1.0 - 12.0).
            Higher values produce motion more faithful to the text.
            Default: 5.0.
        seed: Random seed for reproducible generation.  Optional.
        respacing: Diffusion respacing schedule for faster inference.
            Examples: '' (full quality), 'ddim5' (~10x faster), '10'.
            Default: '' (best quality).
        gender: SMPL-X body model gender: 'female' or 'male'.
            Default: 'female'.
        output_format: Output file format:
            'glb' - ready for 3D viewers (Blender, three.js, web).
            'npz' - raw SMPL-X parameters.
            Default: 'glb'.

    Returns:
        JSON string containing:
        - request_id: Unique identifier for the generated motion.
        - download_url: Full HTTP URL to download the motion file.
        - num_frames: Total number of animation frames.
        - fps: Frames per second (always 30).
        - duration_seconds: Actual duration of the generated clip.
        - text_prompt: The original prompt used.
    """
    if not text_prompt or not text_prompt.strip():
        return json.dumps({"error": "text_prompt cannot be empty"})
    if gender not in ("female", "male"):
        return json.dumps({"error": f"gender must be 'female' or 'male', got '{gender}'"})
    if output_format not in ("glb", "npz"):
        return json.dumps({"error": f"output_format must be 'glb' or 'npz', got '{output_format}'"})

    # Build request payload matching MotionGenerationRequest schema
    payload: dict = {
        "text_prompt": text_prompt,
        "guidance_scale": guidance_scale,
        "respacing": respacing,
        "gender": gender,
        "output_format": output_format,
    }
    if duration_seconds is not None:
        payload["duration_seconds"] = duration_seconds
    if seed is not None:
        payload["seed"] = seed

    try:
        # Long timeout — model inference can take 10-60+ seconds
        async with httpx.AsyncClient(timeout=300.0) as client:
            resp = await client.post(f"{DART_API_URL}/generate", json=payload)
            resp.raise_for_status()
            result = resp.json()

        # Build full download URL from the relative path returned by API
        motion_file_url = result.get("motion_file_url", "")
        download_url = f"{DART_API_URL}{motion_file_url}"

        logger.info(
            "Generated motion: id=%s frames=%d duration=%.2fs url=%s",
            result.get("request_id"),
            result.get("num_frames", 0),
            result.get("duration_seconds", 0),
            download_url,
        )

        return json.dumps(
            {
                "request_id": result["request_id"],
                "download_url": download_url,
                "num_frames": result["num_frames"],
                "fps": result["fps"],
                "duration_seconds": result["duration_seconds"],
                "text_prompt": result["text_prompt"],
            },
            indent=2,
        )

    except httpx.ConnectError:
        return json.dumps({
            "error": (
                "Cannot connect to DART API server. "
                "Make sure it is running: conda activate DART && python api_server.py"
            ),
            "api_url": DART_API_URL,
        })
    except httpx.HTTPStatusError as exc:
        error_detail = exc.response.text
        try:
            error_detail = exc.response.json().get("detail", error_detail)
        except Exception:
            pass
        return json.dumps({
            "error": f"DART API error ({exc.response.status_code}): {error_detail}",
        })
    except Exception as exc:
        logger.error("generate_motion failed: %s", exc)
        return json.dumps({"error": str(exc)})


@mcp.tool()
async def list_generated_motions() -> str:
    """List generated motion files available for download.

    Checks if the DART API server is running and provides the download
    URL pattern for retrieving generated files.

    Returns:
        JSON with server status and download URL pattern.
    """
    try:
        async with httpx.AsyncClient(timeout=10.0) as client:
            resp = await client.get(f"{DART_API_URL}/health")
            resp.raise_for_status()

        return json.dumps({
            "server_status": "running",
            "api_url": DART_API_URL,
            "download_pattern": f"{DART_API_URL}/download/{{filename}}",
            "hint": (
                "Use generate_motion to create a motion, then use the "
                "download_url from the response to download it."
            ),
        }, indent=2)

    except httpx.ConnectError:
        return json.dumps({
            "error": "DART API server is not running.",
            "hint": "Start it: conda activate DART && python api_server.py",
        })
    except Exception as exc:
        return json.dumps({"error": str(exc)})


@mcp.tool()
async def health_check() -> str:
    """Check whether the DART API server is running and ready.

    Returns:
        JSON with connection status and server details.
    """
    try:
        async with httpx.AsyncClient(timeout=5.0) as client:
            resp = await client.get(f"{DART_API_URL}/health")
            resp.raise_for_status()
            data = resp.json()

        return json.dumps({
            "status": data.get("status", "unknown"),
            "connected": True,
            "api_url": DART_API_URL,
        }, indent=2)

    except httpx.ConnectError:
        return json.dumps({
            "status": "unreachable",
            "connected": False,
            "api_url": DART_API_URL,
            "hint": "Start the DART server: conda activate DART && python api_server.py",
        }, indent=2)
    except Exception as exc:
        return json.dumps({
            "status": "error",
            "connected": False,
            "error": str(exc),
            "api_url": DART_API_URL,
        }, indent=2)


# ── Entry Point ──────────────────────────────────────────────────────────────

if __name__ == "__main__":
    mcp.run(transport="stdio")
