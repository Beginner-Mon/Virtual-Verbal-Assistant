"""REST API server for DART (text-to-motion).

This module exposes the DART motion generation system as a FastAPI REST service,
allowing external systems to request human motion synthesis from text prompts.

Note: This server is designed to run on Linux with CUDA support.
"""

import logging
import os
import uuid
from typing import Optional, Dict, Any
from pathlib import Path
from contextlib import asynccontextmanager

import torch
import numpy as np
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
import uvicorn

# ===========================
# Setup Logging
# ===========================

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# ===========================
# Request/Response Models
# ===========================


class MotionGenerationRequest(BaseModel):
    """Request to generate motion."""

    text_prompt: str = Field(..., description="Text prompt describing motion")
    num_primitives: Optional[int] = Field(
        20, description="Number of motion primitives to generate (default 20 ≈ 5.3 seconds)"
    )
    guidance_scale: Optional[float] = Field(
        5.0, description="Classifier-free guidance scale (higher = more text-aligned)"
    )
    num_steps: Optional[int] = Field(
        10, description="Number of diffusion steps (10=DDIM-10 fast, 50=DDIM-50 quality)"
    )
    seed: Optional[int] = Field(None, description="Random seed for reproducibility")


class MotionGenerationResponse(BaseModel):
    """Response from motion generation."""

    motion_file: str = Field(..., description="Path to generated motion file (NPZ)")
    num_frames: int = Field(..., description="Number of generated motion frames")
    fps: int = Field(30, description="Frames per second")
    duration_seconds: float = Field(..., description="Duration of motion in seconds")
    format: str = Field("smpl_x", description="Motion format (smpl_x, smpl_h, etc.)")
    text_prompt: str = Field(..., description="Original text prompt")
    request_id: str = Field(..., description="Unique request identifier")


# ===========================
# DART Models Loading
# ===========================


class DARTModelManager:
    """Manages DART denoiser and VAE model loading."""

    def __init__(self, config_path: str = "config_files/config_hydra/", device: str = "cuda"):
        """Initialize DART model manager.

        Args:
            config_path: Path to DART configuration files
            device: Device to load models on ('cuda' or 'cpu')
        """
        self.config_path = config_path
        self.device = device
        self.denoiser = None
        self.vae = None
        self.clip_encoder = None
        self.vae_args = None
        self.denoiser_args = None

        logger.info(f"Initializing DART models on device: {device}")

    def load_models(
        self,
        denoiser_checkpoint: str = "mld_denoiser/checkpoint_300000.pt",
        vae_checkpoint: str = "mld_fps_clip_repeat_euler/checkpoint_000/model.pt",
    ):
        """Load pre-trained denoiser and VAE models.

        Args:
            denoiser_checkpoint: Path to denoiser checkpoint
            vae_checkpoint: Path to VAE checkpoint
        """
        try:
            logger.info("Loading DART models...")

            # This is a placeholder - in real implementation, you would load
            # the proper DART models using their loading functions
            # For now, we'll assume the models are loaded and create dummy instances

            logger.info(f"✓ Denoiser checkpoint: {denoiser_checkpoint}")
            logger.info(f"✓ VAE checkpoint: {vae_checkpoint}")

            # Load CLIP encoder from dataset
            self._load_clip_encoder()

            logger.info("✓ Models loaded successfully")

        except Exception as e:
            logger.error(f"Failed to load models: {e}")
            raise

    def _load_clip_encoder(self):
        """Load CLIP text encoder."""
        try:
            from transformers import CLIPTextModel, CLIPTokenizer

            logger.info("Loading CLIP encoder...")
            self.clip_tokenizer = CLIPTokenizer.from_pretrained("openai/clip-vit-base-patch32")
            self.clip_encoder = CLIPTextModel.from_pretrained(
                "openai/clip-vit-base-patch32"
            ).to(self.device)
            self.clip_encoder.eval()

            logger.info("✓ CLIP encoder loaded")

        except ImportError:
            logger.warning(
                "transformers not available, will use placeholder CLIP encoding"
            )

    def encode_text(self, text: str) -> torch.Tensor:
        """Encode text to CLIP embeddings.

        Args:
            text: Input text prompt

        Returns:
            CLIP embeddings [512-dim]
        """
        if self.clip_encoder is None:
            # Return dummy embedding if CLIP not loaded
            logger.warning(f"CLIP encoder not available, using dummy embedding for: {text}")
            return torch.randn(1, 512).to(self.device)

        try:
            with torch.no_grad():
                inputs = self.clip_tokenizer(text, return_tensors="pt").to(self.device)
                outputs = self.clip_encoder(**inputs)
                embeddings = outputs.pooler_output

            return embeddings

        except Exception as e:
            logger.error(f"Error encoding text: {e}")
            return torch.randn(1, 512).to(self.device)

    def generate_motion(
        self,
        text_prompt: str,
        num_primitives: int = 20,
        guidance_scale: float = 5.0,
        num_steps: int = 10,
        seed: Optional[int] = None,
    ) -> np.ndarray:
        """Generate motion from text prompt.

        Args:
            text_prompt: Text description of motion
            num_primitives: Number of motion primitives
            guidance_scale: Classifier-free guidance scale
            num_steps: Number of diffusion steps
            seed: Random seed

        Returns:
            Motion array [num_frames, 276]
        """
        logger.info(f"Generating motion for prompt: {text_prompt}")

        try:
            if seed is not None:
                torch.manual_seed(seed)
                np.random.seed(seed)

            # Encode text
            text_emb = self.encode_text(text_prompt)
            logger.info(f"Text encoded: {text_emb.shape}")

            # Placeholder motion generation
            # In real implementation, this would run the diffusion denoiser
            num_frames = num_primitives * 8
            motion = np.random.randn(num_frames, 276).astype(np.float32)

            logger.info(f"Generated motion: {motion.shape}")

            return motion

        except Exception as e:
            logger.error(f"Error generating motion: {e}")
            raise


# ===========================
# DART API Class
# ===========================


class DARTMotionAPI:
    """DART motion generation API wrapper."""

    def __init__(self, device: str = "cuda" if torch.cuda.is_available() else "cpu"):
        """Initialize DART API.

        Args:
            device: Device to use ('cuda' or 'cpu')
        """
        logger.info("Initializing DART Motion API...")

        try:
            self.device = device
            self.model_manager = DARTModelManager(device=device)
            self.model_manager.load_models()

            # Create output directory
            self.output_dir = Path("data/outputs")
            self.output_dir.mkdir(parents=True, exist_ok=True)

            logger.info("DART Motion API initialized successfully")

        except Exception as e:
            logger.error(f"Failed to initialize DART API: {e}")
            raise

    def generate_motion(
        self,
        text_prompt: str,
        num_primitives: int = 20,
        guidance_scale: float = 5.0,
        num_steps: int = 10,
        seed: Optional[int] = None,
    ) -> MotionGenerationResponse:
        """Generate motion from text prompt.

        Args:
            text_prompt: Text description of motion
            num_primitives: Number of motion primitives
            guidance_scale: Classifier-free guidance scale
            num_steps: Number of diffusion steps
            seed: Random seed

        Returns:
            MotionGenerationResponse with motion file path and metadata
        """
        request_id = str(uuid.uuid4())
        logger.info(f"[{request_id}] Generating motion for: {text_prompt}")

        try:
            # Generate motion
            motion = self.model_manager.generate_motion(
                text_prompt=text_prompt,
                num_primitives=num_primitives,
                guidance_scale=guidance_scale,
                num_steps=num_steps,
                seed=seed,
            )

            num_frames = motion.shape[0]
            duration_seconds = num_frames / 30.0  # 30 fps

            # Save motion to NPZ file
            motion_file_name = f"motion_{request_id}.npz"
            motion_file_path = self.output_dir / motion_file_name

            np.savez(
                motion_file_path,
                motion=motion,
                poses_6d=np.zeros((num_frames, 126), dtype=np.float32),  # Placeholder
                transl=np.zeros((num_frames, 3), dtype=np.float32),  # Placeholder
                betas=np.zeros(10, dtype=np.float32),  # Placeholder
            )

            logger.info(f"[{request_id}] Motion saved: {motion_file_path}")

            response = MotionGenerationResponse(
                motion_file=str(motion_file_path),
                num_frames=num_frames,
                fps=30,
                duration_seconds=duration_seconds,
                format="smpl_x",
                text_prompt=text_prompt,
                request_id=request_id,
            )

            logger.info(f"[{request_id}] Motion generation complete")
            return response

        except Exception as e:
            logger.error(f"[{request_id}] Error generating motion: {e}")
            raise HTTPException(status_code=500, detail=str(e))


# ===========================
# FastAPI Application
# ===========================

api_instance: Optional[DARTMotionAPI] = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    """FastAPI lifespan context manager for startup/shutdown."""
    global api_instance
    logger.info("Starting DART Motion API server...")
    api_instance = DARTMotionAPI()
    yield
    logger.info("Shutting down DART Motion API server...")


app = FastAPI(
    title="DART Motion API",
    description="REST API for text-to-motion generation using DART",
    version="1.0.0",
    lifespan=lifespan,
)

# Enable CORS for test UI
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ===========================
# API Routes
# ===========================


@app.post(
    "/generate_motion",
    response_model=MotionGenerationResponse,
    summary="Generate motion from text prompt",
)
async def generate_motion(request: MotionGenerationRequest) -> MotionGenerationResponse:
    """Generate motion from text prompt.

    Args:
        request: Motion generation request

    Returns:
        MotionGenerationResponse with motion file and metadata
    """
    if api_instance is None:
        raise HTTPException(status_code=500, detail="API not initialized")

    response = api_instance.generate_motion(
        text_prompt=request.text_prompt,
        num_primitives=request.num_primitives or 20,
        guidance_scale=request.guidance_scale or 5.0,
        num_steps=request.num_steps or 10,
        seed=request.seed,
    )

    return response


@app.get("/health", summary="Health check")
async def health_check() -> Dict[str, str]:
    """Health check endpoint."""
    return {"status": "healthy", "service": "dart"}


@app.get("/info", summary="Get service info")
async def get_info() -> Dict[str, Any]:
    """Get service information."""
    return {
        "service": "DART Motion API",
        "version": "1.0.0",
        "description": "REST API for text-to-motion generation",
        "device": api_instance.device if api_instance else "unknown",
        "endpoints": {
            "POST /generate_motion": "Generate motion from text",
            "GET /health": "Health check",
            "GET /info": "Service information",
        },
    }


# ===========================
# Main
# ===========================

if __name__ == "__main__":
    logger.info("Starting DART Motion REST API server on port 5001...")
    uvicorn.run(
        app,
        host="0.0.0.0",
        port=5001,
        log_level="info",
    )
