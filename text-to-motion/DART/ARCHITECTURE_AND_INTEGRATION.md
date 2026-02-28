# DART: Diffusion-Based Autoregressive Motion Model
## Architecture, Integration Guide, and Technical Documentation

**Version:** 1.0  
**Last Updated:** February 28, 2026  
**Reference Paper:** [DART - ICLR 2025 Spotlight](https://arxiv.org/abs/2410.05260)  
**Project Website:** [https://zkf1997.github.io/DART/](https://zkf1997.github.io/DART/)

---

## Table of Contents

1. [Project Overview](#project-overview)
2. [Architecture Overview](#architecture-overview)
3. [Core Components](#core-components)
4. [Data Pipeline](#data-pipeline)
5. [Training Pipeline](#training-pipeline)
6. [Inference & Rollout Pipeline](#inference--rollout-pipeline)
7. [Key Applications & Use Cases](#key-applications--use-cases)
8. [Integration Guide](#integration-guide)
9. [Configuration System](#configuration-system)
10. [File Structure & Key Modules](#file-structure--key-modules)
11. [Performance Characteristics](#performance-characteristics)

---

## Project Overview

### What is DART?

DART (Diffusion-based Autoregressive Real-time Text-driven motion control) is a state-of-the-art deep learning framework for generating realistic human motion sequences from text descriptions. It combines:

- **Diffusion Models:** For high-quality motion generation with controllable denoising
- **Autoregressive Processing:** For real-time sequential motion composition
- **Latent Motion Spaces:** For efficient representation learning and generation
- **Classifier-Free Guidance:** For improved text-conditioned generation quality

### Core Capabilities

| Capability | Description |
|-----------|-------------|
| **Text-to-Motion Generation** | Generate human motion sequences from text prompts (e.g., "walk forward", "turn left") |
| **Motion Composition** | Combine multiple text-described actions into seamless motion sequences |
| **Motion In-betweening** | Generate natural transitions between keyframes conditioned on text |
| **Constrained Motion Synthesis** | Generate motions respecting spatial constraints, collision avoidance, contact points |
| **Real-time Control Policy** | Train policies for dynamic goal reaching with motion generation |
| **Trajectory Control** | Control sparse or dense joint trajectories while generating physically plausible motion |
| **Human-Scene Interaction** | Synthesize human motions that interact naturally with 3D environments |

### Supported Datasets

- **BABEL:** Action-annotated 3D motion dataset (primary training dataset)
- **HumanML3D:** Large-scale human motion dataset with text annotations
- **AMASS:** Archive of Motion Capture As Surface Shapes (multi-dataset compilation)
- **Custom Datasets:** Framework supports custom motion data with text annotations

---

## Architecture Overview

### High-Level System Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                        DART System Architecture                  │
├─────────────────────────────────────────────────────────────────┤
│                                                                   │
│  ┌──────────────────────────────────────────────────────────┐  │
│  │              Input Layer (Text or Constraints)           │  │
│  │  • Text Prompts  • Keyframes  • Joint Trajectories       │  │
│  └───────────────────────────┬──────────────────────────────┘  │
│                              │                                  │
│  ┌──────────────────────────────────────────────────────────┐  │
│  │          Text Encoding & Representation                  │  │
│  │  • CLIP Text Embeddings (512-dim)                        │  │
│  │  • Classifier-Free Guidance Conditioning                 │  │
│  └───────────────────────────┬──────────────────────────────┘  │
│                              │                                  │
│  ┌──────────────────────────────────────────────────────────┐  │
│  │      Motion Diffusion Denoiser (Core Model)             │  │
│  │  • Input: Noisy Motion Latent (128-dim)                 │  │
│  │  • Architecture: MLP or Transformer-based                │  │
│  │  • Process: Iterative Denoising (10-50 steps, DDIM)     │  │
│  │  • Output: Clean Motion Latent                           │  │
│  └───────────────────────────┬──────────────────────────────┘  │
│                              │                                  │
│  ┌──────────────────────────────────────────────────────────┐  │
│  │    Latent Motion Autoencoder (MVAE)                      │  │
│  │  • Encode: Motion Sequence → Latent Variables            │  │
│  │  • Decode: Latent Variables → Motion Sequence            │  │
│  │  • Frame Rate: 30 fps for BABEL, 20 fps for HML3D        │  │
│  └───────────────────────────┬──────────────────────────────┘  │
│                              │                                  │
│  ┌──────────────────────────────────────────────────────────┐  │
│  │    SMPL-X/SMPL-H Body Model & Visualization            │  │
│  │  • Parameters: global_orient, body_pose, transl, betas  │  │
│  │  • Outputs: 3D Joint Positions                          │  │
│  │  • Visualization: PyRender or Blender Rendering          │  │
│  └───────────────────────────┬──────────────────────────────┘  │
│                              │                                  │
│  ┌──────────────────────────────────────────────────────────┐  │
│  │              Output Layer (Motion Sequences)             │  │
│  │  • NPY/NPZ Files  • PKL Files  • Visualizations         │  │
│  └──────────────────────────────────────────────────────────┘  │
│                                                                   │
└─────────────────────────────────────────────────────────────────┘
```

### Motion Generation Pipeline Flow

```
Text Input
    ↓
┌─────────────────────────────┐
│ CLIP Text Encoder → Embed   │ (512-dim embedding)
└────────┬────────────────────┘
         ↓
┌─────────────────────────────┐
│ Prepare Diffusion Noise  →  │ (128-dim random latent)
│ Motion History (if autoregr) │
└────────┬────────────────────┘
         ↓
┌─────────────────────────────┐
│ Denoiser Network:           │
│ ┌──────────────────────────┐│
│ │ Timestep Prediction      ││ (Predicts noise residual)
│ │ + Text Conditioning      ││ (Classifier-free guidance)
│ │ (Multiple DDIM steps)    ││
│ └──────────────────────────┘│
└────────┬────────────────────┘ (Repeat 10-50 times)
         ↓
┌─────────────────────────────┐
│ Clean Latent Vector         │ (Denoised 128-dim latent)
└────────┬────────────────────┘
         ↓
┌─────────────────────────────┐
│ MVAE Decoder:               │
│ Latent → Motion Sequence    │
└────────┬────────────────────┘
         ↓
┌─────────────────────────────┐
│ Post-Processing:            │
│ • Rotation Conversions      │
│ • Floor Fixing (optional)   │
│ • Smoothing & Blending      │
└────────┬────────────────────┘
         ↓
┌─────────────────────────────┐
│ SMPL-X Rendering:           │
│ Parameters → 3D Joints      │
└────────┬────────────────────┘
         ↓
Output Motion Sequence
(NPZ/PKL files, Visualization)
```

---

## Core Components

### 1. Diffusion Model & Denoiser

**Location:** `diffusion/`, `mld/train_mld.py`

**Purpose:** Core motion generation engine using diffusion probability models

#### Key Classes:
- **`DenoiserMLP`** / **`DenoiserTransformer`** (`model/mld_denoiser.py`)
  - MLP-based: Simpler, faster inference
  - Transformer-based: Better at long-range dependencies
  - Processes noisy motion latents and text embeddings

- **`GaussianDiffusion`** (`diffusion/gaussian_diffusion.py`)
  - Implements forward diffusion process (adding noise)
  - Reverse diffusion (denoising) with DDIM sampling
  - Supports multiple beta schedules (linear, cosine)
  - Handles loss computation (MSE, KL divergence)

#### Key Parameters:
```python
@dataclass
class DiffusionArgs:
    diffusion_steps: int = 10              # Number of denoising timesteps
    noise_schedule: str = 'cosine'        # Beta schedule: 'linear' or 'cosine'
    sigma_small: bool = True              # Use small sigma at diffusion start
    respacing: str = ''                   # DDIM config: '' (full) or 'ddim10', 'ddim50', etc.

@dataclass
class DenoiserMLPArgs:
    h_dim: int = 512                      # Hidden dimension of MLP blocks
    n_blocks: int = 2                     # Number of MLP blocks (2-8)
    cond_mask_prob: float = 0.1           # Classifier-free guidance mask probability
    clip_dim: int = 512                   # CLIP embedding dimension
    history_shape: tuple = (2, 276)       # Motion history shape [frames, motion_dim]
    noise_shape: tuple = (1, 128)         # Noise/latent shape [frames, latent_dim]

@dataclass
class DenoiserTransformerArgs:
    h_dim: int = 512                      # Hidden dimension
    ff_size: int = 1024                   # Feed-forward layer size
    num_layers: int = 8                   # Number of transformer layers
    num_heads: int = 4                    # Attention heads
    cond_mask_prob: float = 0.1           # Classifier-free guidance masking
    history_shape: tuple = (2, 276)       # Motion history shape (BABEL dataset)
    noise_shape: tuple = (1, 128)         # Noise/latent shape
```

#### Concrete Architecture Details:

**DenoiserMLP Input Processing:**
```
Input Dimensions:
  - Noisy latent:     [batch_size, 1, 128]
  - Text embedding:   [batch_size, 512]
  - Motion history:   [batch_size, 2, 276] → flattened to [batch_size, 552]
  - Timestep:         [batch_size] → embedded to [batch_size, 512]

Input concat:      [batch_size, 512 + 512 + 552 + 128] = [batch_size, 1704]
                     ↓
Linear projection: [batch_size, 512]
                     ↓
MLP blocks (n_blocks times):
  - Dense(512 → 512) + GELU + Dropout(0.1)
  - Dense(512 → 512) + GELU + Dropout(0.1)
                     ↓
Output projection: [batch_size, 128] (noise prediction)
```

**DenoiserTransformer:** Similar structure but uses multi-head attention instead of dense blocks for better sequence modeling.

### 2. Motion Variational Autoencoder (MVAE)

**Location:** `mld/train_mvae.py`, `model/mld_vae.py`

**Implementation Class:** `AutoMldVae` (`model/mld_vae.py`)

**Purpose:** Learns efficient compressed representation of motion sequences

#### Architecture:
- **Encoder:** Motion Sequence (raw joint parameters) → Latent Distribution (μ, σ)
- **Decoder:** Latent Sample → Motion Sequence (reconstructed joints)
- **Latent Dimension:** 128-dimensional vectors
- **Loss Function:** Reconstruction MSE + β·KL_Divergence (VAE regularization)
- **Latent Normalization:** Uses learnable `latent_mean` and `latent_std` for scaling

#### Key Features:
- Handles variable-length sequences through padding/masking
- Outputs 128-dimensional latent vectors for diffusion modeling
- Pre-trained and frozen during denoiser training
- Supports both SMPL-X and SMPL-H body models
- Registers latent statistics (`latent_mean`, `latent_std`) for inference normalization

#### Input/Output Specifications:
```
Input Shape: [batch_size, num_frames, motion_dim]
- BABEL (SMPL-X): motion_dim = 276 (6D rotation [6] + translation [3] + shape params [10]) × frames
- HML3D (SMPL-H): motion_dim differs based on feature extraction method

Output (Latent): [batch_size, 1, 128]
```

#### Loading MVAE in Inference:
```python
checkpoint = torch.load(vae_checkpoint_path, map_location=device)
model_state_dict = checkpoint['model_state_dict']
vae_model.load_state_dict(model_state_dict)
# Important: Register statistics for latent normalization
vae_model.latent_mean = model_state_dict.get('latent_mean', torch.tensor(0))
vae_model.latent_std = model_state_dict.get('latent_std', torch.tensor(1))
vae_model.eval()
```

### 3. Text Encoding & Conditioning

**Location:** `utils/misc_util.py`, CLIP loaded at dataset initialization

**Purpose:** Convert text descriptions to fixed-dimensional embeddings for motion conditioning

#### Important: CLIP is Dataset-Level, Not Standalone
CLIP text encoder is **loaded and managed at the dataset level**, not as a separate utility. This ensures consistent tokenization and device placement.

#### Process:
1. **Dataset Initialization:** CLIP model loaded via `load_and_freeze_clip()` in dataset.__init__
2. **Text Input:** "walk forward slowly" or "person is running"
3. **Text Tokenization:** CLIP tokenizer with context length 77 (standard CLIP)
4. **Embedding:** Transforms text → 512-dimensional fixed embedding
5. **Output:** [batch_size, 512] embeddings passed to denoiser

#### Text Encoding Example:
```python
from utils.misc_util import encode_text

# Use dataset's CLIP model (already loaded and frozen)
text_embedding = encode_text(
    clip_model=dataset.clip_model,  # From dataset
    raw_text=['person is walking', 'jump forward'],  # List of strings
    force_empty_zero=True  # Empty strings → zero embeddings
)
# Output: [2, 512]
```

#### Classifier-Free Guidance:
- Unconditional baseline computed in parallel (mask text embedding during denoiser forward)
- Guidance scale controls text influence during denoising iterations
- Formula: `noise_pred = noise_pred_uncond + scale * (noise_pred_cond - noise_pred_uncond)`
- Standard guidance scale: 5.0 (range: 3.0-7.0, higher = stronger text influence)
- Masking probability during training: `cond_mask_prob=0.1` (10% of batches use no text)

### 4. Motion Primitive System

**Location:** `data_loaders/humanml/data/dataset.py`

**Purpose:** Discretizes long sequences into manageable motion primitives for autoregressive generation

#### Key Concept:
- **Motion Primitive:** ~8-frame motion chunks (at 30fps)
- **Autoregressive Generation:** Generate one primitive at a time, use as history for next
- **Enables:** Long-sequence generation (300+ frames in real-time)

#### Dataset Classes:
- **`PrimitiveSequenceDataset`:** Standard primitive sequences
- **`WeightedPrimitiveSequenceDataset`:** With action frequency weighting
- **`WeightedPrimitiveSequenceDatasetV2`:** Updated weighting strategy

### 5. SMPL-X Body Model Integration

**Location:** `utils/smpl_utils.py`, `config_files/data_paths.py`

**Purpose:** Convert low-level motion parameters to 3D human body representations

#### Supported Body Types:
- **SMPL-X:** Full body with hand articulation (21 joints per hand, 55 total joints)
- **SMPL-H:** Hand and body (22 DoF per hand, 52 total joints)

#### Parameter Format:
- **`global_orient`:** Root rotation (3D angle or 3×3 rotation matrix)
- **`body_pose`:** Per-joint rotations (21 joints × 3D angles)
- **`transl`:** Root translation (3D coordinates)
- **`betas`:** Shape parameters (10-dim, controls body shape)
- **`6D Rotation:** Continuous 6D rotation representation (better for ML)

#### Key Utility Functions:
```python
get_smplx_param_from_6d(primitive_data)    # 6D → 3D rotation conversion
convert_smpl_aa_to_rotmat(smplx_param)     # Axis-angle → rotation matrix
get_new_coordinate(joints)                  # Compute body-centric coordinates
```

---

## Data Pipeline

### Data Structure & Organization

```
data/
├── smplx_lockedhead_20230207/          # Body model parameters
│   └── models_lockedhead/
│       ├── smplh/                      # SMPL-H models
│       │   ├── SMPLH_MALE.pkl
│       │   └── SMPLH_FEMALE.pkl
│       └── smplx/                      # SMPL-X models
│           ├── SMPLX_MALE.npz
│           ├── SMPLX_FEMALE.npz
│           └── SMPLX_NEUTRAL.npz
│
├── amass/                              # AMASS motion dataset
│   ├── babel-teach/                    # BABEL action annotations
│   │   ├── train.json                  # Action labels for training
│   │   └── val.json                    # Action labels for validation
│   ├── smplh_g/                        # SMPL-H motion captures
│   │   ├── ACCAD/
│   │   ├── BioMotionLab_NTroje/
│   │   └── [20+ more datasets]
│   └── smplx_g/                        # SMPL-X motion captures
│       ├── ACCAD/
│       └── [20+ more datasets]
│
├── HumanML3D/                          # HumanML3D dataset
│   ├── HumanML3D/
│   │   ├── new_joint_vecs/             # Motion features
│   │   ├── texts/                      # Text annotations
│   │   └── [other data]
│   └── index.csv                       # Dataset index
│
└── [other directories: traj_test/, inbetween/, optim_interaction/, etc.]
```

### Data Loading Pipeline

#### Step 1: Text-Motion Pairing
```
BABEL JSON: {
    "action": "walk",
    "duration": [0.0, 2.5],
    "motion_id": "ACCAD:...",
    "text": "person is walking forward"
}
```

#### Step 2: Motion Feature Extraction
```python
# Load AMASS/HumanML3D motion data
motion_features = load_motion_features(motion_id)  # Shape: [num_frames, motion_dim]

# Extract parameters for SMPL-X
smplx_params = {
    'gender': 'male',
    'transl': motion[:, :3],           # [num_frames, 3]
    'poses_6d': motion[:, 3:],         # [num_frames, 126] (6D repr)
    'betas': shape_params[:10]         # [10]
}
```

#### Step 3: Motion Sequence Creation
```python
# Segment long sequences into motion primitives
primitive_length = 8  # frames at 30fps
motion_sequence = [
    {'poses_6d': frame_t:t+8, 'transl': transl_t:t+8, ...},
    {'poses_6d': frame_t+8:t+16, 'transl': transl_t+8:t+16, ...},
    ...
]
```

#### Step 4: Batch Processing
```python
# Collate multiple sequences into batch
batch = {
    'motion': torch.tensor([seq1, seq2, ...]),      # [B, T, motion_dim]
    'text': ['walk', 'run', ...],                   # Text prompts
    'text_emb': torch.tensor([emb1, emb2, ...]),   # [B, 512]
    'action_type': ['walk', 'run', ...],           # Action labels
    'motion_length': [50, 45, ...],                # Frame counts
}
```

### Key Dataset Classes

| Class | Purpose | Input | Output |
|-------|---------|-------|--------|
| `HML3dDataset` | HumanML3D dataset loading | motions.npy, texts/ | (motion, text_emb) |
| `WeightedPrimitiveSequenceDataset` | BABEL with primitive segmentation | AMASS data, BABEL labels | (primitive_seq, text, action) |
| `SinglePrimitiveDataset` | Single primitive testing | AMASS data | (single_primitive, text) |

---

## Training Pipeline

### Step 1: MVAE (Motion Autoencoder) Training

**Script:** `mld/train_mvae.py`

**Purpose:** Pre-train motion encoder-decoder for motion representation learning

#### Architecture:
```
Encoder: Motion → Latent Distribution (μ, σ)
Decoder: Latent Sample → Reconstructed Motion
Loss: Reconstruction MSE + β·KL_Divergence
```

#### Training Configuration:
```yaml
batch_size: 64
learning_rate: 1e-4
num_epochs: 100
latent_dim: 128
kl_weight: 0.0001  # β parameter in VAE loss
```

#### Output:
- Pre-trained MVAE checkpoint (**.pt** PyTorch format, not .ckpt)
- Saved to: `mld_fps_clip_repeat_euler/checkpoint_000/model.pt`
- Contains keys: `model_state_dict`, `optimizer_state_dict`, `latent_mean`, `latent_std`

### Step 2: Denoiser (Diffusion Model) Training

**Script:** `mld/train_mld.py`

**Purpose:** Train motion diffusion denoiser conditioned on text embeddings

#### Training Loop:
```
for each epoch:
    for each batch (motion_seq, text_embeddings):
        # Encode motion to latent space
        latent = mvae_encoder(motion_seq)
        
        # Sample random timestep t ∈ [0, T]
        t = randint(0, num_diffusion_steps)
        
        # Add noise to latent (forward diffusion)
        noise = randn_like(latent)
        noisy_latent = sqrt(alpha_bar[t]) * latent + sqrt(1 - alpha_bar[t]) * noise
        
        # Train denoiser to predict noise
        predicted_noise = denoiser(noisy_latent, t, text_emb)
        
        # Compute loss
        loss = MSE(predicted_noise, noise) + classifier_free_guidance_loss
        
        # Backpropagate and update
        optimizer.step()
```

#### Key Training Parameters:
```python
@dataclass
class DiffusionArgs:
    diffusion_steps: int = 10         # Number of DDIM timesteps
    noise_schedule: str = 'cosine'    # Beta schedule
    sigma_small: bool = True          # Use small sigma at start

@dataclass
class DenoiserTransformerArgs:
    h_dim: int = 512                  # Hidden dimension
    ff_size: int = 1024               # Feed-forward size
    num_layers: int = 8               # Transformer layers
    num_heads: int = 4                # Attention heads
    cond_mask_prob: float = 0.1       # Classifier-free guidance masking
```

#### Training Devices:
- **Recommended:** RTX 4090 (24GB VRAM)
- **Batch Size:** 32-64 per GPU
- **Distributed Training:** Multi-GPU DDP supported
- **Precision:** FP16 mixed precision for efficiency

### Step 3: Motion Control Policy Training (Optional)

**Script:** `control/train_reach_location_mld.py`

**Purpose:** Train RL policy for real-time goal reaching with motion generation

#### Environment Loop:
```
for each episode:
    state = initial_standing_pose
    goal = sample_goal_location()
    
    for each timestep:
        action = policy(state, goal)        # Goal reaching action
        motion_primitive = denoiser(        # Generate primitive
            action_embedding,
            previous_motion_history,
            text_prompt='walk'
        )
        state = apply_motion(state, motion_primitive)
        reward = goal_reaching_reward(state, goal)
        
        # Policy gradient update (PPO algorithm)
        policy_loss = -log_prob(action) * advantage + entropy_bonus
        policy_optimizer.step()
```

#### Policy Architecture:
- **Input:** State (joint positions, velocity, goal relative position)
- **Output:** Action (muscle/motion generation parameters)
- **Algorithm:** Proximal Policy Optimization (PPO)
- **Frame Rate:** >300 fps during inference

---

## Inference & Rollout Pipeline

### Rollout Process Overview

**Script:** `mld/rollout_mld.py`

The rollout system generates motion sequences autoregressively by sequentially predicting motion primitives.

#### Initialization:
```python
# Load pre-trained models
mvae = load_mvae(mvae_checkpoint)           # Frozen
denoiser = load_denoiser(denoiser_checkpoint)  # Inference mode
text_encoder = load_clip_encoder()          # Frozen

# Setup diffusion sampler
diffusion = create_gaussian_diffusion(
    diffusion_args.diffusion_steps,
    noise_schedule='cosine'
)
scheduler = create_ddim_schedule_sampler()

# Create classifier-free wrapper for guided generation
guided_denoiser = ClassifierFreeWrapper(denoiser)
```

#### Autoregressive Generation Loop:
```python
def rollout(text_prompt, num_primitives=20):
    """Generate motion sequence autoregressively"""
    
    # Encode text once
    text_embedding = clip_encoder(text_prompt)  # [512]
    
    # Initialize motion history (first primitive)
    motion_history = get_initial_motion()        # Standing pose
    
    full_motion = [motion_history]
    
    for primitive_idx in range(num_primitives):
        # Prepare conditioning
        history_latent = mvae_encoder(motion_history)  # [1, latent_dim]
        
        motion_condition = {
            'text_embedding': text_embedding,
            'motion_history': history_latent,
            'guidance_scale': 5.0,
            'scale': torch.tensor(guidance_scale)
        }
        
        # Diffusion sampling (DDIM)
        noisy_latent = torch.randn(batch_size, latent_dim)
        
        for t in reversed(range(num_timesteps)):
            # Classifier-free guidance
            noise_pred_cond = denoiser(noisy_latent, t, motion_condition)
            motion_condition['uncond'] = True
            noise_pred_uncond = denoiser(noisy_latent, t, motion_condition)
            motion_condition['uncond'] = False
            
            # Guided prediction
            noise_pred = noise_pred_uncond + guidance_scale * (
                noise_pred_cond - noise_pred_uncond
            )
            
            # DDIM update step
            noisy_latent = ddim_step(
                noisy_latent, noise_pred, t, 
                diffusion=diffusion_sampler
            )
        
        # Decode latent to motion
        new_primitive = mvae_decoder(noisy_latent)  # [batch, 8_frames, motion_dim]
        
        # Add to full sequence and update history
        full_motion.append(new_primitive)
        motion_history = new_primitive  # Use as history for next iteration
    
    return torch.cat(full_motion, dim=1)  # [batch, total_frames, motion_dim]
```

#### Sampling Modes:

1. **Guided Sampling (Classifier-Free Guidance)**
   ```python
   guidance_scale = 5.0  # Higher = stronger text influence
   # Formula: noise = uncon_noise + scale * (con_noise - uncon_noise)
   ```

2. **DDIM Sampling**
   ```python
   respacing = 'ddim10'  # Use 10 DDIM steps instead of full diffusion
   # Faster but may sacrifice quality
   ```

3. **Zero Noise Sampling**
   ```python
   zero_noise = 1  # Start denoising from mean instead of random noise
   # More deterministic results
   ```

### Post-Processing

After motion generation, several post-processing steps ensure quality:

```python
def post_process_motion(generated_motion):
    """Apply refinements to generated motion"""
    
    # 1. Convert 6D rotations to 3×3 matrices
    rotation_matrix = rotation_6d_to_matrix(generated_motion['poses_6d'])
    
    # 2. Apply floor contact fixing (optional)
    if fix_floor:
        min_joint = find_lowest_joint(generated_motion['joints_3d'])
        floor_offset = max(0, -min_joint.z)
        generated_motion['transl'][:, 2] += floor_offset
    
    # 3. Smooth velocities (optional)
    generated_motion = smooth_trajectory(generated_motion, kernel_size=3)
    
    # 4. Blend with SMPL-X regression (optional)
    if blend_with_smplx:
        predicted_joints = joints_from_params(generated_motion)
        smplx_joints = smplx_model(generated_motion)
        generated_motion = 0.7 * predicted_joints + 0.3 * smplx_joints
    
    return generated_motion
```

### Output Formats

Generated motions can be exported in multiple formats:

| Format | Purpose | How to Generate |
|--------|---------|-----------------|
| **NPZ** | NumPy compressed array with motion parameters | `export_smpl=1` |
| **PKL** | Python pickle with complete motion dict | `save_format='pkl'` |
| **MP4** | Video rendering with Blender | Use Blender addon |
| **Image Sequence** | PNG frames for visualization | PyRender rendering |

---

## Actual Demo Commands & Usage

### Real-World Command Examples

These commands are directly from `demos/run_demo.sh` and work with actual codebase:

#### 1. Basic Motion Generation Demo
```bash
# Run interactive demo with text-to-motion generation
python -m mld.rollout_demo \
    --denoiser_checkpoint './mld_denoiser/mld_fps_clip_repeat_euler/checkpoint_300000.pt' \
    --batch_size 1 \
    --guidance_param 5.0 \
    --respacing '' \
    --use_predicted_joints 1
```

#### 2. Motion Composition (Multiple Actions)
```bash
# Generate sequence: walk_in_circles (20 primitives) → turn_left (10 primitives) → walk (15 primitives)
python -m mld.rollout_mld \
    --text_prompt 'walk_in_circles*20,turn_left*10,walk*15' \
    --denoiser_checkpoint './mld_denoiser/mld_fps_clip_repeat_euler/checkpoint_300000.pt' \
    --guidance_param 5.0 \
    --export_smpl 1  # Export as NPZ for Blender
```

#### 3. DDIM Fast Sampling (Fewer Steps)
```bash
# Use 10 DDIM steps instead of full diffusion (faster, slight quality trade-off)
python -m mld.rollout_mld \
    --text_prompt 'walk*20' \
    --denoiser_checkpoint './mld_denoiser/mld_fps_clip_repeat_euler/checkpoint_300000.pt' \
    --respacing 'ddim10' \
    --guidance_param 7.0  # Increase guidance for faster sampling
```

#### 4. Deterministic Generation (Zero Noise)
```bash
# Start from latent mean instead of random noise for more deterministic results
python -m mld.rollout_mld \
    --text_prompt 'run*15' \
    --denoiser_checkpoint './mld_denoiser/mld_fps_clip_repeat_euler/checkpoint_300000.pt' \
    --zero_noise 1  # Use zero noise initialization
```

#### 5. In-Betweening (Smooth Transitions)
```bash
# Generate transitions between keyframes
python -m mld.optim_mld \
    --text_prompt 'walk smoothly' \
    --denoiser_checkpoint './mld_denoiser/checkpoint.pt' \
    --mode 'inbetween' \
    --start_frame 0 \
    --end_frame 30
```

---

## Key Applications & Use Cases

### 1. Interactive Online Text-to-Motion Generation

**Script:** `demos/run_demo.sh`

**Features:**
- Real-time motion generation from text input
- Interactive PyRender viewer with keyboard controls
- Live visualization while typing prompts

**Workflow:**
```
┌─────────────────────────┐
│ User Input Text Prompt  │
│ "person is walking"     │
└────────────┬────────────┘
             ↓
┌─────────────────────────┐
│ Encode & Generate       │
│ (0.5-2s latency)        │
└────────────┬────────────┘
             ↓
┌─────────────────────────┐
│ Visualize in PyRender   │
│ • Rotate/Pan/Zoom       │
└─────────────────────────┘
```

**Example Text Prompts:**
- Single verbs: "walk", "run", "dance"
- Phrasal: "walk forward", "turn left"
- Action sequences: "stand, walk, sit"

### 2. Motion Composition

**Script:** `mld/rollout_mld.py` with composition mode

**Input Format:** `action_1*duration_1,action_2*duration_2,...`

**Example:**
```
"walk_in_circles*20,turn_left*10,walk*15"
```

Each number represents duration in motion primitives (8 frames each).

**Execution:**
1. Generate "walk in circles" for 160 frames (20×8)
2. Continue with text context, generate "turn left" for 80 frames
3. Finally generate "walk" for 120 frames
4. Concatenate all primitives into final motion

### 3. Motion In-betweening

**Script:** `mld/optim_mld.py`

**Purpose:** Generate smooth transitions between keyframe poses

**Input:**
- Start keyframe(s) from motion sequence
- End/goal keyframe
- Text description of desired motion
- Duration (number of frames to generate)

**Two Modes:**

**Repeat Mode:**
```
Input Sequence:  [keyframe] [padding] [padding] [goal]
Output:          [generated transition]
Length:          same as input
```

**History Mode:**
```
Input Sequence:  [frame1] [frame2] [frame3] [padding] [goal]
Output:          [generated transition from f1,f2,f3 to goal]
Uses:            Velocity context from first 3 frames
```

### 4. Constrained Motion Synthesis

**Script:** `mld/optim_mld.py` with optimization

**Constraints:**
- Floor contact: Keep feet on floor
- Collision avoidance: Avoid penetrating scene geometry
- Contact points: Hands/feet touch specific locations
- Jerk minimization: Smooth acceleration changes

**Optimization Loop:**
```
Initialize: start_frame + random noise
for iteration in range(max_iterations):
    # Generate motion with diffusion
    motion = denoiser(noisy_latent, timestep, condition)
    
    # Evaluate constraints
    loss = collision_loss + contact_loss + floor_loss + jerk_loss
    
    # Gradient descent on latent space
    latent = latent - learning_rate * ∇loss
    
    # Project back to valid latent space
    latent = clip(latent, mu - 3σ, mu + 3σ)
```

### 5. Goal Reaching with Motion Control

**Scripts:** `control/train_reach_location_mld.py`, `control/test_reach_location_mld.py`

**Concept:** Train an RL policy that controls motion primitives to reach dynamic goals

**Environment:**
- State: Joint positions, velocities, relative goal direction
- Action: Parameters to denoiser for motion primitive generation
- Reward: Negative distance to goal, success bonus
- Curriculum: Gradually increase goal distance and complexity

**Performance:**
- >300 frames per second generation
- Real-time control with dynamic goal updates

### 6. Human-Scene Interaction

**Script:** `mld/optim_scene_mld.py`

**Features:**
- Generate motions respecting 3D scene geometry
- Handle interaction with scene elements (chairs, stairs, etc.)
- Maintain contact constraints (sitting, grasping, etc.)

**Pre-requisites:**
- 3D scene mesh (z-up coordinate system)
- Scene SDF (Signed Distance Field) for collision checking
- Goal joint locations within scene

---

## Configuration System (Extended)

### Hydra Motion Primitive Configs

DARТ uses Hydra for motion primitive configuration. Required configs located in `config_files/config_hydra/`:

#### Key Config Files:
```yaml
# Motion primitive configuration (required for dataset initialization)
config_hydra/motion_primitive/
├── mp_h2_h8_r1.yaml           # Default: history=2, future=8, repeat_factor=1
├── mp_h4_h8_r1.yaml           # Longer history for velocity context
├── mp_h0_h8_r1.yaml           # No history (baseline)
└── [other variants]           # Different frame/primitive configurations
```

#### Example Hydra Config Structure (`mp_h2_h8_r1.yaml`):
```yaml
history_length: 2              # Number of past frames as context
future_length: 8               # Number of future frames to generate per primitive
frame_rate: 30                 # Frames per second
primitive_length: 10           # Total frames = history + future
repeat_factor: 1               # Repetition factor for data augmentation
body_dim: 276                  # Motion feature dimension (BABEL/SMPLX)
```

#### Loading Configs in Code:
```python
import yaml
from omegaconf import OmegaConf

# Option 1: Load YAML directly
with open('./config_files/config_hydra/motion_primitive/mp_h2_h8_r1.yaml') as f:
    cfg = OmegaConf.load(f)

# Option 2: Via dataset (recommended)
from data_loaders.humanml.data.dataset import WeightedPrimitiveSequenceDataset
dataset = WeightedPrimitiveSequenceDataset(
    cfg_path='./config_files/config_hydra/motion_primitive/mp_h2_h8_r1.yaml',
    data_dir='./data/seq_data'
)
```

### Data Paths Configuration

**File:** `config_files/data_paths.py`

Defines all dataset and model paths:
```python
from pathlib import Path

dataset_root_dir = Path('data')
body_model_dir = dataset_root_dir / 'smplx_lockedhead_20230207/models_lockedhead/'
amass_dir = dataset_root_dir / 'amass'
babel_dir = amass_dir / 'babel-teach'
hml3d_dir = dataset_root_dir / 'HumanML3D'
```

**Must be configured before training/inference.** Verify all paths exist:
```bash
# Verify data structure
ls data/smplx_lockedhead_20230207/models_lockedhead/smplx/     # SMPL-X models
ls data/amass/babel-teach/                                     # BABEL labels
ls data/amass/smplh_g/                                         # SMPL-H motion data
```

---

## Integration Guide

### For Developers & Researchers

#### 1. Setting Up Development Environment

```bash
# Clone repository
cd /path/to/DART

# Create conda environment
conda env create -f environment.yml
conda activate DART

# Install DART in development mode
pip install -e .

# Download pre-trained checkpoints and data
# (See main README.md for google drive links)
```

#### 2. Using DART in Custom Python Code

```python
import torch
from mld.rollout_mld import load_mld, load_models
from data_loaders.humanml.data.dataset import WeightedPrimitiveSequenceDataset
from utils.misc_util import encode_text

# Load dataset to get CLIP model and configuration
dataset = WeightedPrimitiveSequenceDataset(
    cfg_path='./config_files/config_hydra/motion_primitive/mp_h2_h8_r1.yaml',
    data_dir='./data/seq_data',
    body_type='smplx'
)

device = 'cuda' if torch.cuda.is_available() else 'cpu'

# Load pre-trained models (.pt format, not .ckpt)
denoiser_checkpoint = './mld_denoiser/mld_fps_clip_repeat_euler/checkpoint_300000.pt'
denoiser_args, denoiser_model, vae_args, vae_model = load_mld(
    denoiser_checkpoint,
    device=device
)

# Text encoding using dataset's CLIP model (loaded at dataset level)
text_embedding = encode_text(dataset.clip_model, 'person is walking forward').to(device)

# Generate motion via rollout
from mld.rollout_mld import rollout
motion = rollout(
    text_prompt='walk*20',  # Format: action*num_primitives
    denoiser_args=denoiser_args,
    denoiser_model=denoiser_model,
    vae_args=vae_args,
    vae_model=vae_model,
    diffusion=diffusion,
    dataset=dataset,
    rollout_args=rollout_args
)
```

#### 3. Custom Dataset Integration

```python
from torch.utils.data import Dataset, DataLoader

class CustomMotionDataset(Dataset):
    def __init__(self, motion_dir, text_file, num_frames=160):
        self.motions = load_motions(motion_dir)
        self.texts = load_texts(text_file)
        self.num_frames = num_frames
    
    def __getitem__(self, idx):
        motion = self.motions[idx]
        text = self.texts[idx]
        
        # Process motion (optional)
        if motion.shape[0] > self.num_frames:
            motion = motion[:self.num_frames]
        
        return {
            'motion': torch.tensor(motion, dtype=torch.float32),
            'text': text,
            'text_emb': encode_text(text)  # Requires CLIP
        }
    
    def __len__(self):
        return len(self.motions)

# Use in training
dataset = CustomMotionDataset('./custom_motions', './custom_texts.txt')
dataloader = DataLoader(dataset, batch_size=32, shuffle=True)
```

#### 4. Training Custom Model

```python
from mld.train_mld import MLDArgs, train_denoiser
import tyro

# Define training configuration
args = tyro.cli(MLDArgs)
args.data_args.dataset = 'custom'
args.data_args.data_dir = './custom_motions'

# Start training
train_denoiser(args)
```

#### 5. Inference API

```python
class MotionGenerator:
    """High-level API for motion generation"""
    
    def __init__(self, checkpoint_path, device='cuda'):
        self.denoiser, self.diffusion = load_mld(checkpoint_path, device)
        self.guided_denoiser = ClassifierFreeWrapper(self.denoiser)
        self.device = device
    
    def generate(self, text_prompt: str, duration_sec: float = 5.0,
                 guidance_scale: float = 5.0) -> torch.Tensor:
        """
        Generate motion from text
        
        Args:
            text_prompt: Description of desired motion
            duration_sec: Duration in seconds (30fps default)
            guidance_scale: Strength of text conditioning
        
        Returns:
            Motion tensor [1, num_frames, motion_dim]
        """
        text_emb = encode_text(text_prompt)
        num_primitives = int(duration_sec * 30 / 8)  # 8 frames per primitive
        
        return self.guided_denoiser.rollout(
            text_embedding=text_emb,
            num_primitives=num_primitives,
            guidance_scale=guidance_scale,
            device=self.device
        )
    
    def generate_sequence(self, prompts: list, duration_secs: list) -> torch.Tensor:
        """Generate sequence of motions from action descriptions"""
        motions = []
        for prompt, duration in zip(prompts, duration_secs):
            motion = self.generate(prompt, duration)
            motions.append(motion)
        return torch.cat(motions, dim=1)
```

### For System Integration

#### 1. Model Serving

```python
# Example with FastAPI for REST API
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

app = FastAPI()
generator = MotionGenerator('./checkpoint.ckpt')

class MotionRequest(BaseModel):
    text: str
    duration: float = 5.0
    guidance_scale: float = 5.0

@app.post("/generate-motion")
async def generate_motion(request: MotionRequest):
    try:
        motion = generator.generate(
            request.text,
            request.duration,
            request.guidance_scale
        )
        return {
            "motion": motion.cpu().numpy().tolist(),
            "shape": motion.shape,
            "duration_sec": request.duration
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
```

#### 2. Pipeline Integration

```python
from PIL import Image
import numpy as np

class MotionPipeline:
    """Full pipeline: text → motion → visualization → video"""
    
    def __init__(self, generator_checkpoint, visualizer_config):
        self.generator = MotionGenerator(generator_checkpoint)
        self.visualizer = MotionVisualizer(visualizer_config)
    
    def text_to_video(self, text_prompt: str, output_path: str):
        """Text → motion → video"""
        
        # Generate motion
        motion = self.generator.generate(text_prompt)
        
        # Render frames
        frames = self.visualizer.render_frames(motion)
        
        # Create video
        self.visualizer.frames_to_video(frames, output_path, fps=30)
        
        return output_path
```

#### 3. Batch Processing

```python
class BatchMotionGenerator:
    """Process multiple requests efficiently"""
    
    def __init__(self, checkpoint_path, batch_size=32):
        self.generator = MotionGenerator(checkpoint_path)
        self.batch_size = batch_size
    
    def generate_batch(self, texts: list) -> list:
        """Generate motions for multiple texts efficiently"""
        
        results = []
        for i in range(0, len(texts), self.batch_size):
            batch_texts = texts[i:i+self.batch_size]
            
            # Pad batch to batch_size
            while len(batch_texts) < self.batch_size:
                batch_texts.append(batch_texts[0])
            
            # Batch generation
            motions = self.generator.generate_batch(batch_texts)
            results.extend(motions[:len(texts[i:i+self.batch_size])])
        
        return results
```

---

## Configuration System

### YAML Configuration Files

DART uses OmegaConf for configuration management.

#### Main Configuration: `configs/config.yaml`

```yaml
# Model Configuration
model:
  mvae_checkpoint: './pretrained_models/mvae.ckpt'
  denoiser_type: 'transformer'  # or 'mlp'
  
  denoiser_args:
    h_dim: 512
    num_layers: 8
    num_heads: 4
    dropout: 0.1
    cond_mask_prob: 0.1

# Diffusion Configuration
diffusion:
  diffusion_steps: 10
  noise_schedule: 'cosine'
  sigma_small: true
  respacing: 'ddim10'  # DDIM sampling

# Data Configuration
data:
  dataset: 'babel'
  batch_size: 32
  num_frames: 160
  frame_rate: 30
  split: 'train'

# Training Configuration
training:
  learning_rate: 1e-4
  num_epochs: 100
  warmup_steps: 1000
  
  optimizer: 'adamw'
  weight_decay: 1e-5
  
  # Checkpointing
  checkpoint_interval: 5000
  best_model_metric: 'val_loss'

# Logging
logging:
  log_interval: 100
  use_wandb: true
  wandb_project: 'dart-motion'
```

#### Asset Paths: `configs/assets.yaml`

```yaml
data_paths:
  body_model_dir: './data/smplx_lockedhead_20230207/models_lockedhead/'
  amass_dir: './data/amass/'
  hml3d_dir: './data/HumanML3D/'
  babel_labels: './data/amass/babel-teach/'

smpl_models:
  smplx: 'smplx'    # Gender-specific models
  smplh: 'smpl'     # SMPL-H fallback
  
body_types:
  default: 'smplx'
  hml3d: 'smplh'    # HML3D uses SMPL-H
```

### Programmatic Configuration

```python
import tyro
from mld.train_mld import MLDArgs

# Parse command-line arguments and load from YAML
args = tyro.cli(MLDArgs, default=MLDArgs())

# Or load from YAML explicitly
import yaml
with open('configs/config.yaml', 'r') as f:
    config_dict = yaml.safe_load(f)
    
args = tyro.extras.from_yaml(MLDArgs, config_dict)

# Modify programmatically
args.denoiser_args.h_dim = 256  # Use smaller model
args.training_args.batch_size = 64
```

---

## File Structure & Key Modules

### Core Modules

```
DART/
├── mld/                    # Motion Latent Diffusion (core)
│   ├── train_mld.py       # Denoiser training script
│   ├── train_mvae.py      # MVAE pre-training
│   ├── rollout_mld.py     # Inference & generation
│   ├── optim_mld.py       # In-betweening & constrained synthesis
│   ├── config.py          # Configuration utilities
│   ├── models/            # Model definitions
│   │   ├── mld_denoiser.py    # DenoiserMLP, DenoiserTransformer
│   │   ├── mld_vae.py         # MVAE architecture
│   │   └── ...
│   ├── callback/          # Training callbacks
│   ├── data/              # Training-specific data handling
│   ├── launch/            # Distributed training scripts
│   ├── tools/             # Utility functions
│   ├── transforms/        # Data transformations
│   └── utils/             # Helper functions
│
├── diffusion/             # Diffusion model components
│   ├── gaussian_diffusion.py  # Core diffusion math
│   ├── respace.py             # DDIM sampling
│   ├── resample.py            # Timestep resampling
│   ├── nn.py                  # Diffusion network utilities
│   ├── losses.py              # Loss computations
│   ├── logger.py              # Training logging
│   └── fp16_util.py           # Mixed precision utilities
│
├── control/               # RL-based motion control
│   ├── train_reach_location_mld.py  # Policy training
│   ├── test_reach_location_mld.py   # Policy evaluation
│   ├── env/                         # RL environments
│   │   └── env_reach_location_mld.py
│   └── policy/                      # Policy architectures
│       └── policy.py
│
├── data_loaders/          # Data loading & processing
│   ├── humanml/
│   │   ├── data/
│   │   │   ├── dataset.py          # HumanML & BABEL datasets
│   │   │   ├── dataset_hml3d.py    # HML3D dataset
│   │   │   └── ...
│   │   ├── scripts/
│   │   └── utils/
│   ├── a2m/               # Action-to-Motion datasets
│   ├── get_data.py        # Data loading interface
│   └── tensors.py         # Batch collation functions
│
├── utils/                 # Utility functions
│   ├── smpl_utils.py      # SMPL-X/SMPL-H utilities
│   ├── misc_util.py       # Text encoding, transformations
│   ├── model_util.py      # Model loading/saving
│   ├── scene_util.py      # 3D scene utilities
│   ├── point_mesh_dist.py # Geometric computations
│   └── ...
│
├── model/                 # Additional models
│   ├── architectures/     # Network architectures
│   ├── losses/            # Custom loss functions
│   ├── metrics/           # Evaluation metrics
│   ├── CFG sampler/       # Classifier-free guidance
│   └── ...
│
├── evaluation/            # Evaluation scripts
│   ├── inbetween.py       # In-betweening evaluation
│   ├── goal_reach.py      # Goal reaching evaluation
│   └── *.sh               # Evaluation demos
│
├── visualize/             # Visualization utilities
│   ├── vis_seq.py         # Main visualization script (PyRender)
│   └── ...
│
├── demos/                 # Demo scripts
│   ├── run_demo.sh        # Interactive generation
│   ├── rollout.sh         # Motion composition
│   ├── inbetween_babel.sh # In-betweening
│   ├── Scene.sh           # Scene interaction
│   ├── goal_reach.sh      # Goal reaching
│   ├── traj.sh            # Trajectory control
│   └── ...
│
├── configs/               # Configuration files
│   ├── config.yaml        # Main training config
│   ├── assets.yaml        # Data path configuration
│   ├── render.yaml        # Visualization config
│   └── config_hydra/      # Hydra configs (if using)
│
├── data/                  # Data storage
│   ├── smplx_lockedhead_20230207/
│   ├── amass/
│   ├── HumanML3D/
│   ├── inbetween/
│   ├── optim_interaction/
│   ├── traj_test/
│   └── ...
│
├── logs/                  # Training logs
├── environment.yml        # Conda environment file
├── README.md              # Original project README
└── topkl.py              # Data conversion script
```

### Key Module Relationships

```
Training Flow:
┌─ train_mvae.py ─────────┐
│  (Pre-train MVAE)       │
└──────────┬──────────────┘
           ↓
┌─ train_mld.py ──────────┐
│  (Train Denoiser)       │
│  Uses: mvae.ckpt        │
└──────────┬──────────────┘
           ↓
┌─ rollout_mld.py ────────┐
│  (Inference)            │
│  Uses: denoiser.ckpt    │
└─────────────────────────┘

Data Pipeline:
data_loaders/humanml/data/dataset.py
           ↓
get_data.py (DataLoader creation)
           ↓
mld/data/ (Batch preprocessing)
           ↓
train_mld.py / train_mvae.py
```

---

## Performance Characteristics

### Computational Requirements

| Component | GPU Memory | CPU | Disk Space |
|-----------|-----------|-----|-----------|
| MVAE | 4 GB | 4 cores | 1 GB |
| Denoiser (MLP) | 6 GB | 4 cores | 0.5 GB |
| Denoiser (Transformer) | 12 GB | 4 cores | 1.5 GB |
| Full Checkpoint | - | - | 3-5 GB |
| BABEL Dataset | - | - | 500 GB |
| HumanML3D Dataset | - | - | 80 GB |

### Inference Speed

| Task | Model | FPS | Latency |
|------|-------|-----|---------|
| Single Primitive (8 frames) | MLP, DDIM10 | 40 | 0.2s |
| Single Primitive (8 frames) | Transformer, DDIM10 | 20 | 0.4s |
| Long Sequence (300 frames) | Autoregressive | 150+ | 2s |
| Goal Reaching Control | RL Policy | 300+ | 0.003s |
| Full Video Rendering | Blender | 5-10 | varies |

### Training Time

| Model | Dataset | GPUs | Duration |
|-------|---------|------|----------|
| MVAE | BABEL (100K samples) | 1× RTX4090 | 12 hours |
| Denoiser | BABEL | 1× RTX4090 | 48 hours |
| Control Policy | Custom goals | 1× RTX4090 | 8-16 hours |

### Memory Usage During Inference

```
Baseline Memory:
- Model parameters: ~200 MB
- CLIP encoder: ~500 MB
- MVAE: ~300 MB
- Denoiser: 500 MB (MLP) - 1.2 GB (Transformer)
- PyRender visualizer: ~400 MB
Total: ~2-3 GB GPU, 4-8 GB CPU
```

---

## Troubleshooting & Common Issues

### Environment Setup Issues

**Problem:** `conda activate DART` doesn't work
```bash
# Solution:
source ~/miniconda3/etc/profile.d/conda.sh  # Initialize conda if not done
conda activate DART
```

**Problem:** CUDA/GPU not detected
```bash
# Check CUDA availability
python -c "import torch; print(torch.cuda.is_available())"

# If False, install correct PyTorch version:
conda install pytorch torchvision torchaudio pytorch-cuda=12.1 -c pytorch -c nvidia
```

### Model Loading Issues

**Problem:** Checkpoint not found or wrong format
```python
# Verify .pt checkpoint exists (not .ckpt)
import os
checkpoint_path = './mld_denoiser/mld_fps_clip_repeat_euler/checkpoint_300000.pt'
assert os.path.exists(checkpoint_path), f"File not found: {checkpoint_path}"

# Check checkpoint contents and load properly
import torch
checkpoint = torch.load(checkpoint_path, map_location='cpu')
print('Keys in checkpoint:', checkpoint.keys())
# Should contain: 'model_state_dict', 'optimizer_state_dict', 'epoch', 'step', etc.

# Load model state dict
model_state_dict = checkpoint['model_state_dict']
model.load_state_dict(model_state_dict)
```

### Data Loading Issues

**Problem:** Dataset shape mismatch
```
Error: Expected size (T, 276) but got (T, 250)
```
Solution: Ensure dataset format matches (BABEL=276, HML3D=different). Check `motion_dim` in config.

### Inference Quality Issues

**Problem:** Generated motion is jittery/unrealistic
- Increase `guidance_scale` (3.0 → 7.0)
- Use more DDIM steps (`respacing='ddim50'`)
- Reduce randomness (`zero_noise=1`)

**Problem:** Text conditioning not working
- Verify text embedding is non-zero
- Check denoiser was trained with `cond_mask_prob > 0`
- Try different text prompts from dataset distribution

---

## References & Additional Resources

### Key Publications
- **DART Paper:** [arxiv.org/abs/2410.05260](https://arxiv.org/abs/2410.05260) (ICLR 2025)
- **Diffusion Models:** Denoising Diffusion Probabilistic Models (Ho et al., 2020)
- **CLIP Text Encoding:** Learning Transferable Visual Models From Natural Language Supervision (Radford et al., 2021)
- **SMPL-X Body Model:** Expressive Body Capture: 3D Hands, Face, and Body from a Single Image (Pavlakos et al., 2019)

### Datasets
- [BABEL Dataset](https://babel.is.tue.mpg.de/) - Action labels for motion captures
- [HumanML3D](https://github.com/EricGuo5513/HumanML3D) - Large-scale motion-text dataset
- [AMASS](https://amass.is.tue.mpg.de/) - Archive of motion capture data
- [SMPL-X](https://smpl-x.is.tue.mpg.de/) - Body model downloads

### Related Projects
- [MDM](https://github.com/GuyTevet/motion-diffusion-model) - Motion diffusion models
- [T2M](https://github.com/EricGuo5513/T2M-GPT) - Text-to-motion with GPT
- [Guided Diffusion](https://github.com/openai/guided-diffusion) - Classifier-free guidance implementation

---

## Appendix: Mathematical Foundations

### Diffusion Process Mathematics

**Forward Diffusion (Adding Noise):**
$$q(x_t | x_0) = \sqrt{\alpha_t} x_0 + \sqrt{1 - \alpha_t} \epsilon$$

where $\alpha_t = \prod_{i=1}^{t} (1 - \beta_i)$ and $\epsilon \sim \mathcal{N}(0, I)$

**Reverse Diffusion (Denoising):**
$$p(x_{t-1} | x_t) = \mathcal{N}(x_{t-1}; \mu_\theta(x_t, t), \sigma_t^2 I)$$

**Objective (Noise Prediction):**
$$\mathcal{L} = \mathbb{E}_{t, x_0, \epsilon} \left[ \|\epsilon - \epsilon_\theta(x_t, t, c)\|^2 \right]$$

### Classifier-Free Guidance

**Unconditioned Prediction:** $\hat{\epsilon}_{uncon}$  
**Conditioned Prediction:** $\hat{\epsilon}_{con}$

**Guided Prediction:**
$$\hat{\epsilon}_{guided} = \hat{\epsilon}_{uncon} + \lambda(\hat{\epsilon}_{con} - \hat{\epsilon}_{uncon})$$

where $\lambda$ is the guidance scale (typical: 3-7)

### Autoregressive Generation

For each timestep $t$:
1. Encode history: $h_t = \text{Encoder}(x_{1:t})$
2. Generate: $x_{t+k} \sim p_\theta(x_{t:t+k} | h_t, c)$
3. Update history: $h_{t+k} = \text{Update}(h_t, x_{t:t+k})$
4. Repeat for next primitive

---

**Document End**

For additional questions or integration support, refer to the original [README.md](README.md) or visit the [project website](https://zkf1997.github.io/DART/).
