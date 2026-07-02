---
title: "DART — Text-to-Motion Architecture"
description: "Diffusion-based motion synthesis: MVAE, denoiser, CLIP encoding, autoregressive rollout."
tags:
  - dart
  - motion
  - diffusion
  - mvae
  - smpl-x
  - inference
---

# DART — Text-to-Motion Architecture

> Location: `text-to-motion/DART/`  
> Paper: [DART — ICLR 2025 Spotlight](https://arxiv.org/abs/2410.05260)

## Pipeline

```
Text Input ("walk forward")
    │
    ▼
CLIP Text Encoder (512-dim)
    │
    ▼
Motion Diffusion Denoiser
  - Input: noisy latent [1, 128] + text_emb [512] + history [2, 276]
  - DDIM denoising (10–50 steps)
  - Classifier-Free Guidance: scale 5.0
    │
    ▼
MVAE Decoder [1, 128] → [8, 276] motion primitive
    │
    ▼
Post-Processing → SMPL-X Body Model → NPZ / PKL / MP4
```

## Core Components

### MVAE (`model/mld_vae.py`)

- **Encoder**: `[T, 276]` → latent `(μ, σ)` → sample `[1, 128]`
- **Decoder**: `[1, 128]` → `[8, 276]` motion primitive
- **Loss**: `MSE_reconstruction + β·KL_divergence` (β = 0.0001)

### Diffusion Denoiser (`model/mld_denoiser.py`)

| Variant | Architecture | Use Case |
|---------|-------------|----------|
| **DenoiserMLP** | Dense(1704→512) + 2 MLP blocks | Faster inference |
| **DenoiserTransformer** | 8 layers, 4 heads | Higher quality, more VRAM |

### Motion Primitive System

- **Primitive** = 8 frames at 30fps (~0.27s)
- **Autoregressive**: each primitive uses previous as history
- Config: `history_length: 2`, `future_length: 8`, `body_dim: 276`

## Inference (`mld/rollout_mld.py`)

```python
def rollout(text_prompt, num_primitives=20):
    text_emb = clip_encoder(text_prompt)
    motion_history = standing_pose()
    for i in range(num_primitives):
        noisy_latent = torch.randn(1, 128)
        for t in reversed(range(10)):
            # Classifier-Free Guidance
            noise = noise_uncond + 5.0 * (noise_cond - noise_uncond)
            noisy_latent = ddim_step(noisy_latent, noise, t)
        new_primitive = mvae_decoder(noisy_latent)
        motion_history = new_primitive
        full_motion.append(new_primitive)
    return concatenate(full_motion)
```

## API (Port 5001)

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/health` | GET | Health check |
| `/generate` | POST | Generate motion from text |
| `/download/{filename}` | GET | Download `.npz` file |

### POST /generate Request

```json
{
  "text_prompt": "jump",
  "duration_seconds": 12,
  "guidance_scale": 5.0,
  "num_steps": 50,
  "respacing": "",
  "seed": null
}
```

## Output Formats

| Format | Content | Use Case |
|--------|---------|----------|
| `.npz` | SMPL-X parameters | Game engines, Blender |
| `.pkl` | Full Python motion dict | Downstream Python |
| `.mp4` | Rendered video | Visualization |

## Training Pipeline

1. **Train MVAE**: `python -m mld.train_mvae` (~12h on RTX 4090)
2. **Train Denoiser**: `python -m mld.train_mld` (~48h on RTX 4090)
3. **(Optional) RL Control**: `python -m control.train_reach_location_mld`

## Related Notes

- [[system-overview]] — Where DART fits in the service map
- [[api-contract]] — Gateway contract for motion requests
- [[troubleshooting]] — If DART returns errors or timeouts

---

#dart #motion #diffusion #mvae #smpl-x #inference
