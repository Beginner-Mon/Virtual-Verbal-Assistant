"""
REST API server for text-to-motion generation (MLD-style)

Generates offline SMPL-X .npz sequences from text prompts.
Uses the exact rollout logic from rollout_mld.py for 100% compatibility.
"""

import logging
import os
import uuid
import random
import copy
import socket
import json
import asyncio
import sys
import subprocess
import time
from urllib.request import urlopen
from urllib.error import URLError
from typing import Optional, Literal, Dict, Any
from pathlib import Path
from contextlib import asynccontextmanager

import torch
import numpy as np
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from pydantic import BaseModel, Field
import uvicorn
import tyro
from dataclasses import dataclass

# ── Imports from your project (exact same as rollout_mld.py) ───────────────
from mld.train_mld import create_gaussian_diffusion
from diffusion.respace import space_timesteps
from data_loaders.humanml.data.dataset import SinglePrimitiveDataset
from utils.misc_util import encode_text, compose_texts_with_and
from pytorch3d import transforms as pyt3d_transforms

# Import the proven loading + rollout helpers from rollout_mld.py
from mld.rollout_mld import load_mld

# ── Logging ──────────────────────────────────────────────────────────────────
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def _env_int(name: str, default: int) -> int:
    value = os.getenv(name)
    if value is None:
        return default
    try:
        return int(value)
    except ValueError:
        logger.warning("Invalid integer for %s=%r; using default %d", name, value, default)
        return default


def _env_bool(name: str, default: bool) -> bool:
    value = os.getenv(name)
    if value is None:
        return default
    return str(value).strip().lower() in {"1", "true", "yes", "on"}


def _is_port_in_use(host: str, port: int) -> bool:
    probe_host = "127.0.0.1" if host in ("0.0.0.0", "::") else host
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.settimeout(0.5)
        return sock.connect_ex((probe_host, port)) == 0


def _is_healthy_dart_running(host: str, port: int) -> bool:
    probe_host = "127.0.0.1" if host in ("0.0.0.0", "::") else host
    url = f"http://{probe_host}:{port}/health"
    try:
        with urlopen(url, timeout=1.0) as resp:
            if resp.status != 200:
                return False
            payload = resp.read().decode("utf-8", errors="ignore")
            data = json.loads(payload)
            return data.get("status") == "ok"
    except (URLError, TimeoutError, ValueError, OSError):
        return False

# ── Request / Response Models ────────────────────────────────────────────────

class MotionGenerationRequest(BaseModel):
    text_prompt: str = Field(
        ...,
        description=(
            "Text prompt. Supports MLD-style syntax: "
            "'walk forward' with duration_seconds, or legacy '*count' syntax like "
            "'walk forward*12'"
        )
    )
    duration_seconds: Optional[float] = Field(
        None,
        ge=1.0,
        le=120.0,
        description="Desired clip duration in seconds. Preferred over '*count' syntax.",
    )
   
    guidance_scale: float = Field(5.0, ge=1.0, le=12.0)
    num_steps: int = Field(50, ge=10, le=1000, description="Ignored – use respacing instead")
    seed: Optional[int] = Field(None)
    respacing: str = Field(
        "", 
        description="Optional diffusion respacing (examples: '', ddim5, 10).")
    gender: Literal["female", "male"] = Field(
        "female",
        description="Body gender for SMPL-X motion generation."
    )
    output_format: Literal["glb", "npz"] = Field(
        "glb",
        description="Output file format to return. Default is glb."
    )


class MotionGenerationResponse(BaseModel):
    request_id: str
    motion_file_url: str
    num_frames: int
    fps: int = 30
    duration_seconds: float
    text_prompt: str
    debug: Optional[Dict[str, Any]] = None


# ── Model & Diffusion Manager ────────────────────────────────────────────────

class MotionGenerator:
    def __init__(self, device: str = None):
        # Auto-detect device: use CUDA if available, otherwise fall back to CPU
        if device is None:
            device = "cuda" if torch.cuda.is_available() else "cpu"
        self.device = torch.device(device)
        
        # Ensure float32 for all operations on CPU (CPU doesn't support efficient float16)
        if self.device.type == 'cpu':
            torch.set_default_dtype(torch.float32)
            logger.info("Setting default dtype to float32 for CPU")
        
        self.denoiser_args = None
        self.vae_args = None
        self.denoiser = None
        self.vae = None
        self.dataset = None
        self.primitive_utility = None
        self.standing_seed_path = None

        logger.info(f"Initializing motion generator on {device}")
        if device == "cuda" and not torch.cuda.is_available():
            logger.warning("CUDA requested but not available. Falling back to CPU.")
            self.device = torch.device("cpu")
            torch.set_default_dtype(torch.float32)
        self.output_dir = Path("outputs")
        self.output_dir.mkdir(parents=True, exist_ok=True)

    def load_models(
        self,
        denoiser_ckpt: str = "mld_denoiser/mld_fps_clip_repeat_euler/checkpoint_300000.pt",
        standing_seed_path: str = "./data/stand.pkl",
    ):
        # Use the exact same loader as rollout_mld.py (correct VAE args, ClassifierFreeWrapper, etc.)
        self.denoiser_args, self.denoiser, self.vae_args, self.vae = load_mld(denoiser_ckpt, self.device)

        self.standing_seed_path = standing_seed_path
        # Dataset (exact same as rollout_mld.py)
        self.dataset = SinglePrimitiveDataset(
            cfg_path=self.vae_args.data_args.cfg_path,
            dataset_path=self.vae_args.data_args.data_dir,
            body_type=self.vae_args.data_args.body_type,
            sequence_path=standing_seed_path,
            batch_size=1,
            device=self.device,
            enforce_gender="female",
            enforce_zero_beta=1,
        )
        self.primitive_utility = self.dataset.primitive_utility

        logger.info("Motion models loaded successfully")

    @torch.no_grad()
    def generate(
        self,
        text_prompt: str,
        duration_seconds: Optional[float],
        guidance_scale: float,
        num_steps: int,          # unused (respacing controls steps)
        seed: Optional[int] = None,
        respacing: str = "",
        gender: Literal["female", "male"] = "female",
    ):
        if self.dataset is None:
            raise HTTPException(503, "Dataset not initialized")
        if self.standing_seed_path is None:
            raise HTTPException(500, "Standing seed path is not configured")

        # Dataset caches sequence metadata; refresh if requested gender changed.
        if self.dataset.enforce_gender != gender:
            self.dataset.enforce_gender = gender
            self.dataset.update_seq(self.standing_seed_path)

        if seed is not None:
            random.seed(seed)
            np.random.seed(seed)
            torch.manual_seed(seed)

        device = self.device
        batch_size = 1
        future_length = self.dataset.future_length
        history_length = self.dataset.history_length
        primitive_length = history_length + future_length

        # ── Parse text prompt into primitives ────────────────────────────────────
        texts = []
        total_primitives = 0
        parsed_actions = []
        explicit_count_used = False

        # Split by comma for sequence of actions
        segments = [s.strip() for s in text_prompt.split(',') if s.strip()]

        for segment in segments:
            if '*' in segment:
                action_part, count_str = segment.split('*', 1)
                try:
                    count = int(count_str.strip())
                except ValueError:
                    raise HTTPException(400, f"Invalid count in '{segment}': must be integer after *")
                explicit_count_used = True
            else:
                action_part = segment
                count = 1

            # Clean and compose the action text (handles "and" like in rollout)
            action = compose_texts_with_and(action_part.split(' and '))
            parsed_actions.append((action, count, '*' in segment))

        if duration_seconds is not None and not explicit_count_used:
            # Convert desired duration to primitive count using model future length.
            target_total = max(1, int(round((duration_seconds * 30.0) / float(future_length))))

            # Distribute target primitives across comma-separated actions.
            num_actions = len(parsed_actions)
            base = target_total // num_actions
            remainder = target_total % num_actions
            distributed = []
            for idx, (action, _, _) in enumerate(parsed_actions):
                c = base + (1 if idx < remainder else 0)
                distributed.append((action, max(1, c)))

            for action, count in distributed:
                texts.extend([action] * count)
                total_primitives += count
            logger.info(
                "duration_seconds=%.2f mapped to %d primitives (future_length=%d)",
                duration_seconds,
                total_primitives,
                future_length,
            )
        else:
            if duration_seconds is not None and explicit_count_used:
                logger.info("duration_seconds provided, but explicit '*count' syntax detected; using explicit counts")
            for action, count, _ in parsed_actions:
                texts.extend([action] * count)
                total_primitives += count

        if total_primitives == 0:
            raise HTTPException(400, "No valid motion primitives found in text_prompt")

        logger.info(f"Parsed prompt into {total_primitives} primitives")

        all_text_embedding = encode_text(
            self.dataset.clip_model,
            texts,
            force_empty_zero=True
        ).to(device)

        # === Initial standing seed ===
        batch = self.dataset.get_batch(batch_size=batch_size)
        input_motions, model_kwargs = batch[0]['motion_tensor_normalized'], {'y': batch[0]}
        del model_kwargs['y']['motion_tensor_normalized']
        gender = model_kwargs['y']['gender'][0]
        betas = model_kwargs['y']['betas'][:, :primitive_length, :].to(device)
        pelvis_delta = self.primitive_utility.calc_calibrate_offset({
            'betas': betas[:, 0, :],
            'gender': gender,
        })
        input_motions = input_motions.to(device)
        motion_tensor = input_motions.squeeze(2).permute(0, 2, 1)  # [B, T, D]
        history_motion = motion_tensor[:, :history_length, :]

        # === Diffusion with per-request respacing (exact as rollout_mld.py) ===
        raw_respacing = (respacing or "").strip()
        respacing = raw_respacing
        total_steps = int(self.denoiser_args.diffusion_args.diffusion_steps)

        # DDIM with full step count is equivalent to full schedule here.
        if respacing.lower().startswith("ddim"):
            try:
                desired = int(respacing[4:])
            except ValueError as exc:
                raise HTTPException(400, f"Invalid respacing format: '{raw_respacing}'") from exc
            if desired == total_steps:
                respacing = ""
            else:
                try:
                    space_timesteps(total_steps, respacing)
                except ValueError as exc:
                    raise HTTPException(
                        400,
                        f"Invalid respacing '{raw_respacing}' for diffusion_steps={total_steps}: {exc}",
                    ) from exc
        elif respacing:
            try:
                space_timesteps(total_steps, respacing)
            except (ValueError, TypeError) as exc:
                raise HTTPException(
                    400,
                    f"Invalid respacing '{raw_respacing}' for diffusion_steps={total_steps}: {exc}",
                ) from exc

        diffusion_args = copy.deepcopy(self.denoiser_args.diffusion_args)
        diffusion_args.respacing = respacing
        try:
            diffusion = create_gaussian_diffusion(diffusion_args)
        except ValueError as exc:
            raise HTTPException(
                400,
                f"Failed to create diffusion schedule (respacing='{raw_respacing}', "
                f"diffusion_steps={total_steps}): {exc}",
            ) from exc
        sample_fn = diffusion.p_sample_loop if respacing == '' else diffusion.ddim_sample_loop

        # === Rollout loop (100% copy of working code from rollout_mld.py) ===
        motion_sequences = None
        transf_rotmat = torch.eye(3, device=device).unsqueeze(0).repeat(batch_size, 1, 1)
        transf_transl = torch.zeros(3, device=device).reshape(1, 1, 3).repeat(batch_size, 1, 1)

        for segment_id in range(total_primitives):
            text_embedding = all_text_embedding[segment_id].expand(batch_size, -1)
            guidance_param = torch.ones(batch_size, *self.denoiser_args.model_args.noise_shape).to(device) * guidance_scale

            y = {
                'text_embedding': text_embedding,
                'history_motion_normalized': history_motion,
                'scale': guidance_param,
            }

            x_start_pred = sample_fn(
                self.denoiser,
                (batch_size, *self.denoiser_args.model_args.noise_shape),
                clip_denoised=False,
                model_kwargs={'y': y},
                skip_timesteps=0,
                init_image=None,
                progress=False,
                dump_steps=None,
                noise=None,
                const_noise=False,
            )

            latent_pred = x_start_pred.permute(1, 0, 2)
            future_motion_pred = self.vae.decode(
                latent_pred, history_motion, nfuture=future_length,
                scale_latent=self.denoiser_args.rescale_latent
            )

            future_frames = self.dataset.denormalize(future_motion_pred)
            all_frames = torch.cat([self.dataset.denormalize(history_motion), future_frames], dim=1)

            future_feature_dict = self.primitive_utility.tensor_to_dict(future_frames)
            future_feature_dict.update({
                'transf_rotmat': transf_rotmat,
                'transf_transl': transf_transl,
                'gender': gender,
                'betas': betas[:, :future_length, :] if segment_id > 0 else betas[:, :primitive_length, :],
                'pelvis_delta': pelvis_delta,
            })
            future_primitive_dict = self.primitive_utility.feature_dict_to_smpl_dict(future_feature_dict)
            future_primitive_dict = self.primitive_utility.transform_primitive_to_world(future_primitive_dict)

            if motion_sequences is None:
                motion_sequences = {k: v for k, v in future_primitive_dict.items()}
            else:
                for key in ['transl', 'global_orient', 'body_pose', 'betas', 'joints']:
                    motion_sequences[key] = torch.cat([motion_sequences[key], future_primitive_dict[key]], dim=1)

            # Update history for next step (exact blending from rollout_mld.py)
            new_history_frames = all_frames[:, -history_length:, :]
            history_feature_dict = self.primitive_utility.tensor_to_dict(new_history_frames)
            history_feature_dict.update({
                'transf_rotmat': transf_rotmat,
                'transf_transl': transf_transl,
                'gender': gender,
                'betas': betas[:, :history_length, :],
                'pelvis_delta': pelvis_delta,
            })
            canonicalized_history_primitive_dict, blended_feature_dict = self.primitive_utility.get_blended_feature(
                history_feature_dict, use_predicted_joints=0
            )
            transf_rotmat = canonicalized_history_primitive_dict['transf_rotmat']
            transf_transl = canonicalized_history_primitive_dict['transf_transl']
            history_motion = self.primitive_utility.dict_to_tensor(blended_feature_dict)
            history_motion = self.dataset.normalize(history_motion)

        # === Convert to SMPL-X .npz format (ready for Blender) – exact as rollout_mld.py ===
        # Remove batch dimension (we always use B=1)
        for key in list(motion_sequences.keys()):
            if torch.is_tensor(motion_sequences[key]):
                motion_sequences[key] = motion_sequences[key][0]

        sequence = motion_sequences
        poses = pyt3d_transforms.matrix_to_axis_angle(
            torch.cat([sequence['global_orient'].reshape(-1, 1, 3, 3), sequence['body_pose']], dim=1)
        ).reshape(-1, 22 * 3)
        poses = torch.cat([poses, torch.zeros(poses.shape[0], 99, dtype=poses.dtype, device=poses.device)], dim=1)

        data_dict = {
            'mocap_framerate': min(getattr(self.dataset, 'target_fps', 30), 30),
            'gender': sequence['gender'],
            'betas': sequence['betas'][0, :10].detach().cpu().numpy(),
            'poses': poses.detach().cpu().numpy(),
            'trans': sequence['transl'].detach().cpu().numpy(),
        }

        request_id = str(uuid.uuid4())[:12]
        npz_path = self.output_dir / f"motion_{request_id}.npz"
        np.savez(npz_path, **data_dict)

        return {
            "request_id": request_id,
            "motion_file": str(npz_path),
            "num_frames": poses.shape[0],
            "fps": 30,
            "duration_seconds": round(poses.shape[0] / 30.0, 2),
            "text_prompt": text_prompt,
        }


# ── FastAPI App ──────────────────────────────────────────────────────────────

app = FastAPI(title="Text-to-Motion API (MLD)", version="0.2")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],                   # ← easiest for local dev
    allow_credentials=True,
    allow_methods=["*", "OPTIONS"],        # important: OPTIONS must be allowed
    allow_headers=["*"],
    expose_headers=["Content-Disposition"],# useful if you add file downloads later
    max_age=600,
)
generator: Optional[MotionGenerator] = None
OUTPUT_DIR = Path("outputs")
OUTPUT_DIR.mkdir(exist_ok=True)
cleanup_task: Optional[asyncio.Task] = None
DART_INCLUDE_DEBUG = _env_bool("DART_INCLUDE_DEBUG", True)


def convert_npz_to_glb(npz_path: Path, glb_path: Path, gender: Literal["female", "male"]) -> None:
    script_path = Path(__file__).with_name("to_glb.py")
    model_path = Path(__file__).resolve().parent / "data" / "smplx_lockedhead_20230207" / "models_lockedhead"
    glb_export_mode = (os.getenv("DART_GLB_EXPORT_MODE", "single") or "single").strip().lower()
    if glb_export_mode not in {"single", "stack", "sequence"}:
        logger.warning("Invalid DART_GLB_EXPORT_MODE=%r, defaulting to 'single'", glb_export_mode)
        glb_export_mode = "single"

    if not script_path.exists():
        raise RuntimeError(f"GLB converter script not found: {script_path}")

    cmd = [
        sys.executable,
        str(script_path),
        "--npz_path",
        str(npz_path),
        "--out_path",
        str(glb_path),
        "--gender",
        gender,
        "--mode",
        glb_export_mode,
    ]

    if model_path.exists():
        cmd.extend(["--model_path", str(model_path)])

    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0:
        raise RuntimeError(
            f"NPZ→GLB conversion failed (exit={result.returncode}): {result.stderr.strip() or result.stdout.strip()}"
        )


def remove_expired_artifacts(ttl_seconds: int) -> None:
    now = time.time()
    for path in OUTPUT_DIR.glob("motion_*"):
        if path.suffix not in {".npz", ".glb"}:
            continue
        try:
            age_seconds = now - path.stat().st_mtime
            if age_seconds > ttl_seconds:
                path.unlink(missing_ok=True)
                logger.info(f"Deleted expired artifact: {path.name} (age={int(age_seconds)}s)")
        except Exception as exc:
            logger.warning(f"Failed to cleanup artifact {path}: {exc}")


async def cleanup_loop(ttl_seconds: int) -> None:
    while True:
        remove_expired_artifacts(ttl_seconds)
        await asyncio.sleep(30)


@app.on_event("startup")
async def startup_event():
    global generator, cleanup_task
    generator = MotionGenerator()
    generator.load_models(
        denoiser_ckpt=args.denoiser_checkpoint,
        standing_seed_path=args.standing_seed,
    )
    cleanup_task = asyncio.create_task(cleanup_loop(args.artifact_ttl_seconds))


@app.on_event("shutdown")
async def shutdown_event():
    global cleanup_task
    if cleanup_task:
        cleanup_task.cancel()
        try:
            await cleanup_task
        except asyncio.CancelledError:
            pass


@app.post("/generate", response_model=MotionGenerationResponse)
async def generate_motion(req: MotionGenerationRequest):
    if generator is None:
        raise HTTPException(503, "Service not ready")

    request_start_perf = time.perf_counter()
    debug_payload: Dict[str, Any] = {
        "request": {
            "output_format": req.output_format,
            "duration_seconds": req.duration_seconds,
            "respacing": req.respacing,
            "seed": req.seed,
        },
        "runtime": {
            "device": str(getattr(generator, "device", "unknown")),
            "cuda_available": bool(torch.cuda.is_available()),
        },
        "timings_ms": {},
    }

    model_generate_start_perf = time.perf_counter()
    result = generator.generate(
        text_prompt=req.text_prompt,
        duration_seconds=req.duration_seconds,
        guidance_scale=req.guidance_scale,
        num_steps=req.num_steps,
        seed=req.seed,
        respacing=req.respacing,
        gender=req.gender,
    )
    debug_payload["timings_ms"]["model_generate_ms"] = round(
        (time.perf_counter() - model_generate_start_perf) * 1000,
        3,
    )

    npz_file_name = Path(result["motion_file"]).name
    selected_file_name = npz_file_name

    if req.output_format == "glb":
        npz_path = OUTPUT_DIR / npz_file_name
        glb_file_name = f"{Path(npz_file_name).stem}.glb"
        glb_path = OUTPUT_DIR / glb_file_name
        try:
            glb_convert_start_perf = time.perf_counter()
            convert_npz_to_glb(npz_path=npz_path, glb_path=glb_path, gender=req.gender)
            debug_payload["timings_ms"]["glb_convert_ms"] = round(
                (time.perf_counter() - glb_convert_start_perf) * 1000,
                3,
            )
            selected_file_name = glb_file_name
            logger.info(f"Generated GLB artifact: {glb_file_name} from {npz_file_name}")
        except Exception as exc:
            logger.error(f"Failed to generate GLB for request {result['request_id']}: {exc}")
            raise HTTPException(500, f"GLB conversion failed: {exc}")
    else:
        debug_payload["timings_ms"]["glb_convert_ms"] = 0.0

    debug_payload["timings_ms"]["total_ms"] = round((time.perf_counter() - request_start_perf) * 1000, 3)

    logger.info(
        f"Motion request {result['request_id']} output_format={req.output_format} file={selected_file_name}"
    )

    response = MotionGenerationResponse(
        request_id=result["request_id"],
        motion_file_url=f"/download/{selected_file_name}",
        num_frames=result["num_frames"],
        fps=30,
        duration_seconds=result["duration_seconds"],
        text_prompt=result["text_prompt"],
        debug=debug_payload if DART_INCLUDE_DEBUG else None,
    )
    return response


@app.get("/download/{filename}")
async def download_file(filename: str):
    path = OUTPUT_DIR / filename
    if not path.exists():
        raise HTTPException(404, "File not found")
    return FileResponse(path, filename=filename, media_type="application/octet-stream")


@app.get("/health")
async def health():
    return {"status": "ok"}


# ── CLI Args ─────────────────────────────────────────────────────────────────

@dataclass
class ServerArgs:
    denoiser_checkpoint: str = "mld_denoiser/mld_fps_clip_repeat_euler/checkpoint_300000.pt"
    standing_seed: str = "./data/stand.pkl"
    host: str = os.getenv("DART_HOST", "0.0.0.0")
    port: int = _env_int("DART_PORT", 5001)
    artifact_ttl_seconds: int = _env_int("DART_ARTIFACT_TTL", 21600)  # 6 hours


# Module-level default (so startup can see it before __main__ block runs)
args: ServerArgs = ServerArgs()


if __name__ == "__main__":
    args = tyro.cli(ServerArgs)

    # Single-instance guard: avoid bind crash when DART is already running.
    if _is_port_in_use(args.host, args.port):
        if _is_healthy_dart_running(args.host, args.port):
            logger.info(
                "DART server already running at http://%s:%d (health=ok). Exiting.",
                "127.0.0.1" if args.host in ("0.0.0.0", "::") else args.host,
                args.port,
            )
            raise SystemExit(0)
        logger.error(
            "Port %d is already in use and not serving DART /health. "
            "Set DART_PORT to a free port or stop the conflicting process.",
            args.port,
        )
        raise SystemExit(1)

    uvicorn.run(app, host=args.host, port=args.port, log_level="info")
