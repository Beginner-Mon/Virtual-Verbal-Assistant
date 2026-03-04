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
from typing import Optional
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
from data_loaders.humanml.data.dataset import SinglePrimitiveDataset
from utils.misc_util import encode_text, compose_texts_with_and
from pytorch3d import transforms as pyt3d_transforms

# Import the proven loading + rollout helpers from rollout_mld.py
from mld.rollout_mld import load_mld

# ── Logging ──────────────────────────────────────────────────────────────────
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# ── Request / Response Models ────────────────────────────────────────────────

class MotionGenerationRequest(BaseModel):
    text_prompt: str = Field(
        ...,
        description=(
            "Text prompt. Supports MLD-style syntax: "
            "'walk forward*12' or 'wave hands*10,walk forward*5,cartwheel*8,...'"
        )
    )
   
    guidance_scale: float = Field(5.0, ge=1.0, le=12.0)
    num_steps: int = Field(50, ge=10, le=1000, description="Ignored – use respacing instead")
    seed: Optional[int] = Field(None)
    respacing: str = Field(
        "", 
        description="e.g. ddim50, ddim100, 250")


class MotionGenerationResponse(BaseModel):
    request_id: str
    motion_file_url: str
    num_frames: int
    fps: int = 30
    duration_seconds: float
    text_prompt: str


# ── Model & Diffusion Manager ────────────────────────────────────────────────

class MotionGenerator:
    def __init__(self, device: str = "cuda"):
        self.device = torch.device(device)
        self.denoiser_args = None
        self.vae_args = None
        self.denoiser = None
        self.vae = None
        self.dataset = None
        self.primitive_utility = None

        logger.info(f"Initializing motion generator on {device}")
        self.output_dir = Path("outputs")
        self.output_dir.mkdir(parents=True, exist_ok=True)

    def load_models(
        self,
        denoiser_ckpt: str = "mld_denoiser/mld_fps_clip_repeat_euler/checkpoint_300000.pt",
        standing_seed_path: str = "./data/stand.pkl",
    ):
        # Use the exact same loader as rollout_mld.py (correct VAE args, ClassifierFreeWrapper, etc.)
        self.denoiser_args, self.denoiser, self.vae_args, self.vae = load_mld(denoiser_ckpt, self.device)

        # Dataset (exact same as rollout_mld.py)
        self.dataset = SinglePrimitiveDataset(
            cfg_path=self.vae_args.data_args.cfg_path,
            dataset_path=self.vae_args.data_args.data_dir,
            body_type=self.vae_args.data_args.body_type,
            sequence_path=standing_seed_path,
            batch_size=1,
            device=self.device,
            enforce_gender="male",
            enforce_zero_beta=1,
        )
        self.primitive_utility = self.dataset.primitive_utility

        logger.info("Motion models loaded successfully")

    @torch.no_grad()
    def generate(
        self,
        text_prompt: str,
    
        guidance_scale: float,
        num_steps: int,          # unused (respacing controls steps)
        seed: Optional[int] = None,
        respacing: str = "",
    ):
        if seed is not None:
            random.seed(seed)
            np.random.seed(seed)
            torch.manual_seed(seed)

        device = self.device
        batch_size = 1
        future_length = self.dataset.future_length
        history_length = self.dataset.history_length
        primitive_length = history_length + future_length

        # ── Parse multi-action prompt like rollout_mld.py ────────────────────────
        texts = []
        total_primitives = 0

        # Split by comma for sequence of actions
        segments = [s.strip() for s in text_prompt.split(',') if s.strip()]

        for segment in segments:
            if '*' in segment:
                action_part, count_str = segment.split('*', 1)
                try:
                    count = int(count_str.strip())
                except ValueError:
                    raise HTTPException(400, f"Invalid count in '{segment}': must be integer after *")
            else:
                action_part = segment
                count = 1

            # Clean and compose the action text (handles "and" like in rollout)
            action = compose_texts_with_and(action_part.split(' and '))
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
        diffusion_args = copy.deepcopy(self.denoiser_args.diffusion_args)
        diffusion_args.respacing = respacing
        diffusion = create_gaussian_diffusion(diffusion_args)
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

generator: Optional[MotionGenerator] = None
OUTPUT_DIR = Path("outputs")
OUTPUT_DIR.mkdir(exist_ok=True)


@app.on_event("startup")
async def startup_event():
    global generator
    generator = MotionGenerator()
    generator.load_models(
        denoiser_ckpt=args.denoiser_checkpoint,
        standing_seed_path=args.standing_seed,
    )


@app.post("/generate", response_model=MotionGenerationResponse)
async def generate_motion(req: MotionGenerationRequest):
    if generator is None:
        raise HTTPException(503, "Service not ready")

    result = generator.generate(
        text_prompt=req.text_prompt,
        guidance_scale=req.guidance_scale,
        num_steps=req.num_steps,
        seed=req.seed,
        respacing=req.respacing,
    )

    response = MotionGenerationResponse(
        request_id=result["request_id"],
        motion_file_url=f"/download/{Path(result['motion_file']).name}",
        num_frames=result["num_frames"],
        fps=30,
        duration_seconds=result["duration_seconds"],
        text_prompt=result["text_prompt"],
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
    host: str = "0.0.0.0"
    port: int = 8000


# Module-level default (so startup can see it before __main__ block runs)
args: ServerArgs = ServerArgs()


if __name__ == "__main__":
    args = tyro.cli(ServerArgs)
    uvicorn.run(app, host=args.host, port=args.port, log_level="info")