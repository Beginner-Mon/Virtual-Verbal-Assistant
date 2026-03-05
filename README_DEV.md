# Virtual Verbal Assistant — Developer Overview

> **Audience**: Developers onboarding to this project. This document gives a full-system overview and then dives deep into the two primary subsystems: **AgenticRAG** and **Text-to-Motion (DART)**.

---

## Table of Contents

1. [System Architecture Overview](#1-system-architecture-overview)
2. [How the Pipeline Works](#2-how-the-pipeline-works)
3. [Subsystem: AgenticRAG](#3-subsystem-agenticrag)
4. [Subsystem: Text-to-Motion (DART)](#4-subsystem-text-to-motion-dart)
5. [Subsystem: SpeechLLm (Overview)](#5-subsystem-speechllm-overview)
6. [Service Integration & API Contract](#6-service-integration--api-contract)
7. [Running the Full Stack](#7-running-the-full-stack)
8. [Repository Map](#8-repository-map)
9. [Environment & Dependencies](#9-environment--dependencies)
10. [Troubleshooting](#10-troubleshooting)

---

## 1. System Architecture Overview

The **Virtual Verbal Assistant** is a multi-service pipeline that takes natural language input, generates an intelligent text response, and simultaneously produces a corresponding human motion animation.

```
┌─────────────────────────────────────────────────────────────┐
│                  Orchestrator  (port 8080)                  │
│                  main_api.py — Unified entry point          │
│                                                             │
│  ┌─────────────────────┐   ┌──────────────────────────────┐ │
│  │    AgenticRAG        │   │  Text-to-Motion (DART)       │ │
│  │    Port 8000         │   │  Port 5001                   │ │
│  │    Windows            │   │  WSL / Linux                 │ │
│  │    Env: firstconda   │   │  Env: DART                   │ │
│  └─────────────────────┘   └──────────────────────────────┘ │
└─────────────────────────────────────────────────────────────┘
          ↑                              ↑
   Handles: text Q&A,           Handles: motion generation
   RAG, memory, document        from text prompt → NPZ file
   retrieval via Gemini          via diffusion model
```

**Three running services required:**

| # | Service | Port | Platform | Conda Env | Entry Point |
|---|---------|------|----------|-----------|-------------|
| 1 | AgenticRAG | 8000 | Windows | `firstconda` | `agenticRAG/agentic_rag_gemini/api_server.py` |
| 2 | DART | 5001 | WSL/Linux | `DART` | `text-to-motion/DART/api_server.py` |
| 3 | Orchestrator | 8080 | Windows | `firstconda` | `agenticRAG/agentic_rag_gemini/main_api.py` |

> **SpeechLLm** (port 5000) is a fourth optional service for voice I/O and emotion detection.

---

## 2. How the Pipeline Works

A user sends a query to the Orchestrator at port 8080. The Orchestrator fans the request out to both AgenticRAG and DART in parallel, then merges the results.

```
User Query: "How do I walk forward?"
    │
    ▼
Orchestrator (POST /answer)
    ├──────► AgenticRAG (POST /query)
    │           │  Gemini LLM + ChromaDB RAG
    │           └── returns: { text_answer, motion_prompt, voice_prompt }
    │
    └──────► DART (POST /generate_motion)
                │  Diffusion model → SMPL-X motion
                └── returns: { motion_file (.npz), num_frames, fps }
    │
    ▼
Combined Response to Frontend:
{
  "text_answer": "To walk forward, shift your weight...",
  "motion": { "file": "data/outputs/motion_<uuid>.npz", "num_frames": 160, "fps": 30 },
  "voice": null,
  "generation_time_ms": 3500.0
}
```

The `motion_prompt` returned by AgenticRAG (e.g., `"walk forward"`) is what gets forwarded to DART for motion generation.

---

## 3. Subsystem: AgenticRAG

> **Location**: `agenticRAG/agentic_rag_gemini/`  
> **Full reference**: [`README_DEVELOPERS.md`](agenticRAG/agentic_rag_gemini/README_DEVELOPERS.md)

### 3.1 What it Does

AgenticRAG is an intelligent conversational system powered by Google Gemini that:
- Routes user queries to the right action (memory retrieval vs. document search vs. plain LLM).
- Maintains persistent per-user conversation memory with semantic search via ChromaDB.
- Retrieves context from uploaded documents (PDF, Word, Images with OCR).
- Returns both a `text_answer` and a `motion_prompt` for downstream DART use.

### 3.2 Internal Architecture

```
User Query (+ optional document uploads)
    │
    ▼
1. ORCHESTRATOR AGENT (gemini-2.5-flash)
   └─ Analyzes query → decides action (retrieve_memory | search_docs | llm_direct)
    │
    ▼
2. MEMORY RETRIEVAL (if needed)
   ├─ EmbeddingService: sentence-transformers/all-MiniLM-L6-v2  (384-dim)
   ├─ VectorStore: ChromaDB PersistentClient
   └─ Retrieves semantically similar past interactions
    │
    ▼
3. DOCUMENT SEARCH (if documents uploaded)
   ├─ DocumentStore: overlapping-chunk search in ChromaDB
   ├─ DocumentLoader: PDF (pypdf + OCR), DOCX, Images (pytesseract)
   └─ Retrieves top-k relevant chunks, deduplicated per document
    │
    ▼
4. RAG PIPELINE (gemini-2.5-flash)
   ├─ Builds prompt: retrieved memory + document chunks + user query
   ├─ Generates text answer via Gemini API
   └─ Returns { text_answer, motion_prompt, voice_prompt }
    │
    ▼
5. MEMORY STORAGE
   └─ Stores this interaction for future retrieval
```

### 3.3 Key Components

| Module | Location | Responsibility |
|--------|----------|---------------|
| `Orchestrator` | `agents/orchestrator.py` | Query routing via `analyze_query()` → `ActionType` enum |
| `RAGPipeline` | `retrieval/rag_pipeline.py` | Context-aware response generation |
| `MemoryManager` | `memory/memory_manager.py` | Store/retrieve conversation history |
| `DocumentStore` | `memory/document_store.py` | Chunked document storage & semantic search |
| `VectorStore` | `memory/vector_store.py` | ChromaDB wrapper (dual-collection) |
| `EmbeddingService` | `memory/embedding_service.py` | Text → 384-dim embedding vectors |
| `GeminiClient` | `utils/gemini_client.py` | OpenAI-compatible Gemini API wrapper |
| `DocumentLoader` | `utils/document_loader.py` | Multi-format document text extraction |
| `api_server.py` | root of `agentic_rag_gemini/` | FastAPI server, exposes `/query`, `/health` |

### 3.4 Vector Database & Chunking

**ChromaDB Collections:**
- `kinetichat_memory` — conversation history and summaries
- `kinetichat_memory_documents` — uploaded document chunks

**Chunking Strategy (`config/config.yaml`):**
```yaml
chunking:
  chunk_size: 1500        # Characters per chunk
  chunk_overlap: 300      # Overlap for context continuity
  min_chunk_size: 300     # Min size to store (else stored whole)
```

**Document Processing Pipeline:**
```
Upload → Text Extraction → Chunking (1500 char, 300 overlap)
       → Embedding (384-dim) → Store in ChromaDB
```

**Search & Deduplication:**
1. Query → 384-dim embedding
2. ChromaDB similarity search (`top_k_documents: 8`)
3. Group chunks by source document
4. Keep top-3 chunks per document (prevents single-doc domination)
5. Sort all by similarity score

**Similarity Calculation** (fixed formula):
```python
similarity = max(0.0, 1.0 - (distance / 2.0))  # Euclidean → cosine-like
```

### 3.5 Configuration (`config/config.yaml`)

```yaml
orchestrator:
  model: "gemini-2.5-flash"
  temperature: 0.1          # Low → deterministic routing decisions

llm:
  model: "gemini-2.5-flash"
  temperature: 0.7          # Higher → more natural responses
  max_tokens: 1000

embedding:
  model: "sentence-transformers/all-MiniLM-L6-v2"
  dimension: 384

rag:
  top_k_documents: 8
  similarity_threshold: 0.3

memory:
  max_items: 100
  relevance_threshold: 0.7
```

### 3.6 File Structure

```
agentic_rag_gemini/
├── api_server.py              # FastAPI REST API  ← START HERE
├── main_api.py                # Orchestrator (multi-service) entry point
├── main.py                    # CLI interactive entry point
├── run_ui.py                  # Streamlit UI launcher
├── ui.py                      # Streamlit web interface
├── config/
│   └── config.yaml            # All tuneable settings
├── agents/
│   └── orchestrator.py        # Query routing logic
├── retrieval/
│   └── rag_pipeline.py        # Core RAG + response generation
├── memory/
│   ├── memory_manager.py      # Conversation memory
│   ├── document_store.py      # Document chunking + search
│   ├── vector_store.py        # ChromaDB wrapper (FIXED similarity)
│   └── embedding_service.py   # Sentence-transformer embeddings
├── utils/
│   ├── gemini_client.py       # Gemini API wrapper
│   ├── document_loader.py     # PDF/Word/Image text extraction
│   ├── validators.py          # Response quality validation
│   ├── prompt_templates.py    # LLM prompt templates
│   └── logger.py              # Logging setup
├── data/
│   └── vector_store/          # Persistent ChromaDB files
└── logs/
    └── agentic_rag.log
```

### 3.7 AgenticRAG API

**Base URL:** `http://localhost:8000`

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/health` | GET | Health check |
| `/query` | POST | Main query endpoint |

**POST `/query` — Request:**
```json
{
  "query": "How do I walk forward?",
  "user_id": "web_user",
  "conversation_history": []
}
```

**POST `/query` — Response:**
```json
{
  "text_answer": "To walk forward, shift your weight...",
  "orchestrator_decision": "search_documents",
  "motion_prompt": "walk forward",
  "voice_prompt": "calm instructional"
}
```

### 3.8 Known Issues & Fixes

| Issue | Cause | Fix |
|-------|-------|-----|
| Documents not persisting | Wrong ChromaDB client | Use `PersistentClient` (already fixed in `vector_store.py`) |
| Negative similarity scores | Wrong distance formula | Fixed to `max(0.0, 1.0 - distance/2)` |
| 404 model errors | Invalid model name | Run `python list_available_models.py` |
| 429 rate limits | Free Gemini tier (20 req/day) | Reduce retries or upgrade API |
| Documents not retrieved | Threshold too high | Lower `similarity_threshold` in `config.yaml` |

---

## 4. Subsystem: Text-to-Motion (DART)

> **Location**: `text-to-motion/DART/`  
> **Full reference**: [`ARCHITECTURE_AND_INTEGRATION.md`](text-to-motion/DART/ARCHITECTURE_AND_INTEGRATION.md)  
> **Paper**: [DART — ICLR 2025 Spotlight](https://arxiv.org/abs/2410.05260)

### 4.1 What it Does

DART (Diffusion-based Autoregressive Real-time Text-driven motion control) converts text descriptions (e.g., `"walk forward"`) into realistic 3D human motion sequences using diffusion models. It runs as a REST API in WSL/Linux.

**Capabilities:**

| Feature | Description |
|---------|-------------|
| Text-to-Motion | Generate motion from text prompts: `"walk"`, `"turn left"` |
| Motion Composition | Chain actions: `"walk_in_circles*20,turn_left*10,walk*15"` |
| Motion In-betweening | Smooth transitions between keyframes |
| Constrained Synthesis | Floor contact, collision avoidance, contact points |
| Real-time Control | RL policy for goal-reaching at >300 FPS |

### 4.2 Architecture Overview

```
Text Input ("walk forward")
    │
    ▼
CLIP Text Encoder (512-dim embedding)
    │
    ▼
Motion Diffusion Denoiser (MLP or Transformer)
  - Input: noisy motion latent [1, 128] + text emb [512] + motion history [2, 276]
  - Iterative DDIM denoising (10–50 steps)
  - Classifier-Free Guidance: scale 5.0
    │
    ▼
MVAE Decoder (Latent → Motion Sequence)
  - Latent [1, 128] → Motion frames [8, 276] (one primitive at 30fps)
    │
    ▼
Post-Processing
  - 6D rotation → 3×3 matrix conversion
  - Floor contact fixing (optional)
  - Smoothing & blending
    │
    ▼
SMPL-X Body Model
  - Parameters → 3D joint positions
    │
    ▼
Output: NPZ / PKL files (SMPL-X format)
```

### 4.3 Core Components

#### 4.3.1 Motion Variational Autoencoder (MVAE)

**Location:** `mld/train_mvae.py`, `model/mld_vae.py` (`AutoMldVae`)

The MVAE is **pre-trained first** and **frozen** during denoiser training. It acts as the compression/decompression layer for motion sequences.

```
Encoder: [T, 276] motion frames → (μ, σ) → sample latent [1, 128]
Decoder: [1, 128] latent        → [8, 276] motion primitive
Loss:     MSE_reconstruction + β · KL_divergence   (β = 0.0001)
```

**Checkpoint format** (`.pt` — NOT `.ckpt`):
```python
checkpoint = {
    'model_state_dict': ...,
    'optimizer_state_dict': ...,
    'latent_mean': ...,    # Required for inference normalization
    'latent_std': ...      # Required for inference normalization
}
```

#### 4.3.2 Diffusion Denoiser

**Location:** `model/mld_denoiser.py`, `diffusion/gaussian_diffusion.py`

Two variants exist:

**DenoiserMLP** (faster):
```
Inputs concatenated: text_emb[512] + timestep_emb[512] + history[552] + latent[128]
                   = [1704]
Linear projection → [512]
MLP Blocks (n=2): Dense(512→512) + GELU + Dropout(0.1) × 2
Output: noise prediction [128]
```

**DenoiserTransformer** (higher quality, more VRAM):
- Same I/O but uses multi-head attention (8 layers, 4 heads)
- Better at long-range motion dependencies

**Training loop (simplified):**
```python
latent = mvae_encoder(motion_seq)           # Encode to latent
t = randint(0, T)                            # Random timestep
noisy = sqrt(alpha_bar[t]) * latent + sqrt(1 - alpha_bar[t]) * noise
pred_noise = denoiser(noisy, t, text_emb)   # Predict noise
loss = MSE(pred_noise, noise)               # Optimize
```

#### 4.3.3 Text Encoding (CLIP)

**Location:** Loaded at dataset level via `utils/misc_util.py`

- Model: `openai/clip-vit-base-patch32`
- Embedding: 512-dimensional fixed vectors
- Loaded once at dataset init → **not standalone**, always via `dataset.clip_model`
- Classifier-Free Guidance: 10% of training batches use zeroed text embedding (`cond_mask_prob=0.1`)

```python
from utils.misc_util import encode_text
text_emb = encode_text(dataset.clip_model, ['walk forward'])  # → [1, 512]
```

#### 4.3.4 Motion Primitive System

- **Primitive** = 8 frames at 30fps (~0.27 seconds of motion)
- Autoregressive generation: each primitive uses the previous as history
- Enables long sequences (300+ frames) by chaining primitives
- Config: `config_files/config_hydra/motion_primitive/mp_h2_h8_r1.yaml`
  - `history_length: 2`, `future_length: 8`, `body_dim: 276`

### 4.4 Autoregressive Inference (Rollout)

**Script:** `mld/rollout_mld.py`

```python
def rollout(text_prompt, num_primitives=20):
    text_emb = clip_encoder(text_prompt)        # Encode once
    motion_history = standing_pose()            # Initial pose

    for i in range(num_primitives):
        # Diffusion sampling (DDIM, 10 steps)
        noisy_latent = torch.randn(1, 128)
        for t in reversed(range(10)):
            noise_cond   = denoiser(noisy_latent, t, text_emb, motion_history)
            noise_uncond = denoiser(noisy_latent, t, zero_emb, motion_history)
            noise        = noise_uncond + 5.0 * (noise_cond - noise_uncond)  # CFG
            noisy_latent = ddim_step(noisy_latent, noise, t)

        new_primitive = mvae_decoder(noisy_latent)  # [8, 276]
        motion_history = new_primitive              # Update history
        full_motion.append(new_primitive)

    return concatenate(full_motion)  # [num_primitives×8, 276]
```

### 4.5 Sampling Modes

| Mode | Flag | Effect |
|------|------|--------|
| Full diffusion | `respacing=''` | Best quality, slower |
| DDIM fast | `respacing='ddim10'` | 10 steps, faster |
| Deterministic | `zero_noise=1` | Start from latent mean |
| Strong guidance | `guidance_scale=7.0` | More text-faithful |

### 4.6 Output Formats

| Format | Content | Use Case |
|--------|---------|---------|
| `.npz` | SMPL-X parameters (poses, transl, betas) | Game engines, Blender |
| `.pkl` | Full Python motion dict | Downstream Python processing |
| `.mp4` | Rendered video | Visualization |

### 4.7 DART API

**Base URL:** `http://localhost:5001` (runs in WSL)

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/health` | GET | Health check |
| `/generate_motion` | POST | Generate motion from text |

**POST `/generate_motion` — Request:**
```json
{
  "text_prompt": "walk forward",
  "num_primitives": 20,
  "guidance_scale": 5.0,
  "num_steps": 10
}
```

**POST `/generate_motion` — Response:**
```json
{
  "motion_file": "data/outputs/motion_<uuid>.npz",
  "num_frames": 160,
  "fps": 30,
  "duration_seconds": 5.333,
  "format": "smpl_x",
  "text_prompt": "walk forward",
  "request_id": "<uuid>"
}
```

### 4.8 DART File Structure

```
text-to-motion/DART/
├── api_server.py              # FastAPI server  ← START HERE
├── mld/                       # Motion Latent Diffusion (core)
│   ├── train_mvae.py          # Step 1: Pre-train MVAE
│   ├── train_mld.py           # Step 2: Train Denoiser
│   ├── rollout_mld.py         # Inference & generation
│   ├── optim_mld.py           # In-betweening & constrained synthesis
│   └── rollout_demo.py        # Interactive CLI demo
├── diffusion/                 # Diffusion math
│   ├── gaussian_diffusion.py  # Forward/reverse diffusion
│   └── respace.py             # DDIM sampling schedules
├── model/                     # Model architectures
│   ├── mld_denoiser.py        # DenoiserMLP, DenoiserTransformer
│   └── mld_vae.py             # AutoMldVae (MVAE)
├── data_loaders/              # Dataset handling
│   └── humanml/data/
│       ├── dataset.py         # BABEL, HumanML3D datasets + CLIP loading
│       └── dataset_hml3d.py   # HML3D specific
├── control/                   # RL-based motion control (optional)
│   ├── train_reach_location_mld.py
│   └── env/env_reach_location_mld.py
├── utils/
│   ├── smpl_utils.py          # SMPL-X/H body model utilities
│   ├── misc_util.py           # encode_text(), transformations
│   └── scene_util.py          # 3D scene utilities
├── config_files/
│   ├── data_paths.py          # All dataset path definitions
│   └── config_hydra/
│       └── motion_primitive/  # Hydra configs (mp_h2_h8_r1.yaml etc.)
├── mld_denoiser/              # Pre-trained denoiser checkpoints
│   └── mld_fps_clip_repeat_euler/
│       └── checkpoint_300000.pt
├── mld_fps_clip_repeat_euler/ # Pre-trained MVAE checkpoint
│   └── checkpoint_000/model.pt
├── data/
│   ├── smplx_lockedhead_20230207/  # SMPL-X body model files
│   ├── amass/                      # AMASS motion captures + BABEL labels
│   ├── HumanML3D/                  # HML3D dataset
│   └── outputs/                    # Generated motion files (.npz)
├── demos/                     # Pre-built demo scripts (.sh)
├── visualize/                 # PyRender visualization
└── environment.yml            # Conda environment (DART env)
```

### 4.9 Training Pipeline (for retraining)

```
Step 1: Train MVAE (motion autoencoder)
  python -m mld.train_mvae
  Output: mld_fps_clip_repeat_euler/checkpoint_000/model.pt
           (~12 hours on RTX 4090)

Step 2: Train Denoiser (diffusion model)
  python -m mld.train_mld
  Requires: MVAE checkpoint from step 1
  Output: mld_denoiser/mld_fps_clip_repeat_euler/checkpoint_300000.pt
           (~48 hours on RTX 4090)

Step 3: (Optional) Train RL Control Policy
  python -m control.train_reach_location_mld
```

### 4.10 Computational Requirements

| Component | GPU Memory | Disk |
|-----------|-----------|------|
| MVAE | 4 GB | 1 GB |
| Denoiser (MLP) | 6 GB | 0.5 GB |
| Denoiser (Transformer) | 12 GB | 1.5 GB |
| CLIP encoder | 0.5 GB RAM | — |
| BABEL dataset | — | ~500 GB |
| HumanML3D dataset | — | ~80 GB |

**Inference speed:**
- Single 8-frame primitive (MLP, DDIM10): ~0.2s
- 300-frame sequence (autoregressive): ~2s
- Goal-reaching RL control: >300 FPS

---

## 5. Subsystem: SpeechLLm (Overview)

> **Location**: `SpeechLLm/`  
> *(Not the primary focus of this document — see SpeechLLm/README.md for full details)*

SpeechLLm handles voice I/O and emotion-aware dialogue using a **local Small Language Model**.

```
Voice / Text Input
    ↓
Speech-to-Text (Whisper / wav2vec2)
    ↓
Emotion Detection (audio/text)
    ↓
Context Formatting (JSON)
    ↓
Local SLM Reasoning (Phi-3 / Mistral / Gemma via Ollama)
    ↓
Text-to-Speech + Avatar control
```

**Key modules:** `src/core/orchestrator.py` (pipeline loop), `src/stages/` (STT, LLM, TTS, emotion stages), `src/services/` (Whisper, Phi-3 clients)

**Port:** 5000 (optional — Orchestrator degrades gracefully if unavailable)

---

## 6. Service Integration & API Contract

### 6.1 Orchestrator (main_api.py)

The Orchestrator at port 8080 is the **single entry point** for frontends. It:
1. Receives `POST /answer` from frontend
2. Calls AgenticRAG `/query` → gets `text_answer` + `motion_prompt`
3. Calls DART `/generate_motion` using the `motion_prompt`
4. Merges and returns a unified response

### 6.2 Data Flow Contract

```
Frontend → Orchestrator:
  POST /answer  { query, user_id, conversation_history }

Orchestrator → AgenticRAG:
  POST /query   { query, user_id, conversation_history }

AgenticRAG → Orchestrator:
  { text_answer, motion_prompt, voice_prompt, orchestrator_decision }

Orchestrator → DART (using motion_prompt):
  POST /generate_motion  { text_prompt, num_primitives, guidance_scale, num_steps }

DART → Orchestrator:
  { motion_file, num_frames, fps, duration_seconds }

Orchestrator → Frontend:
  { text_answer, voice, motion: { file, num_frames, fps }, generation_time_ms, errors }
```

### 6.3 Motion Prompt Convention

AgenticRAG generates `motion_prompt` in **DART-compatible format**:
- Simple: `"walk forward"`, `"turn left"`, `"stand"`
- Composed: `"walk_in_circles*20,turn_left*10,walk*15"` (action*num_primitives)

---

## 7. Running the Full Stack

### 7.1 Prerequisites

- Windows with WSL installed
- Conda environments:
  - `firstconda` (Windows) — for AgenticRAG + Orchestrator
  - `DART` (WSL) — for DART
- `GEMINI_API_KEY` set in `agenticRAG/agentic_rag_gemini/.env`
- DART model checkpoints downloaded (see `text-to-motion/DART/README.md`)

### 7.2 Start Order

> Always start AgenticRAG and DART **before** the Orchestrator.

**Terminal 1 — AgenticRAG (Windows):**
```powershell
conda activate firstconda
cd agenticRAG\agentic_rag_gemini
python api_server.py
# Serves on http://localhost:8000
```

**Terminal 2 — DART (via WSL):**
```powershell
wsl -e bash -ic "conda activate DART && cd /mnt/d/Project_A/Virtual-Verbal-Assistant/text-to-motion/DART && python api_server.py"
# Serves on http://localhost:5001
```

**Terminal 3 — Orchestrator (Windows):**
```powershell
conda activate firstconda
cd agenticRAG\agentic_rag_gemini
python main_api.py
# Serves on http://localhost:8080
```

### 7.3 Health Checks

```powershell
# AgenticRAG
Invoke-WebRequest -Uri "http://localhost:8000/health" -UseBasicParsing | Select-Object -ExpandProperty Content

# DART
Invoke-WebRequest -Uri "http://localhost:5001/health" -UseBasicParsing | Select-Object -ExpandProperty Content

# Orchestrator
Invoke-WebRequest -Uri "http://localhost:8080/health" -UseBasicParsing | Select-Object -ExpandProperty Content
```

### 7.4 Quick Test via Orchestrator

```powershell
$body = '{"query":"How do I walk forward?","user_id":"test","conversation_history":[]}'
Invoke-WebRequest -Uri "http://localhost:8080/answer" -Method POST `
  -ContentType "application/json" -Body $body -UseBasicParsing `
  | Select-Object -ExpandProperty Content
```

### 7.5 Test UI

Open `test-ui/index.html` in a browser for a visual test interface.

---

## 8. Repository Map

```
Virtual-Verbal-Assistant/
├── README_DEV.md                      # ← This file
├── HOW_TO_RUN.md                      # Startup guide (quick reference)
│
├── agenticRAG/
│   └── agentic_rag_gemini/            # AgenticRAG service (port 8000)
│       ├── api_server.py              # REST API entry point
│       ├── main_api.py                # Orchestrator entry point (port 8080)
│       ├── config/config.yaml         # All AgenticRAG configuration
│       ├── agents/orchestrator.py     # Query routing
│       ├── retrieval/rag_pipeline.py  # RAG + LLM response generation
│       ├── memory/                    # Vector store, document store, embeddings
│       └── utils/                     # Gemini client, document loader, etc.
│
├── text-to-motion/
│   ├── DART/                          # DART service (port 5001, WSL)
│   │   ├── api_server.py              # REST API entry point
│   │   ├── mld/                       # Motion Latent Diffusion core
│   │   ├── diffusion/                 # Gaussian diffusion implementation
│   │   ├── model/                     # Denoiser + VAE architectures
│   │   ├── data_loaders/              # Dataset classes
│   │   ├── utils/                     # SMPL utilities, text encoding
│   │   └── data/outputs/              # Generated motion .npz files
│   └── motion-diffusion-model/        # Alternative/supplementary MDM repo
│
├── SpeechLLm/                         # Speech I/O + emotion (port 5000, optional)
│   ├── api_server.py
│   ├── src/                           # Pipeline stages (STT, LLM, TTS, emotion)
│   └── configs/                       # Model paths and sample rates
│
└── test-ui/
    └── index.html                     # Browser-based test interface
```

---

## 9. Environment & Dependencies

### AgenticRAG (`firstconda` on Windows)

**Key packages:**
```
google-generativeai       # Gemini API
chromadb                  # Vector database (PersistentClient)
sentence-transformers     # all-MiniLM-L6-v2 embeddings
fastapi + uvicorn         # API server
pypdf + python-docx       # Document loading
pytesseract + pillow      # OCR for images/scanned PDFs
streamlit                 # Web UI (optional)
```

### DART (`DART` conda env on WSL/Linux)

**Key packages:**
```
torch + torchvision       # Deep learning (CUDA required)
transformers              # CLIP text encoder
smplx                     # SMPL-X body model
pyrender                  # Motion visualization
omegaconf + hydra         # Configuration management
tyro                      # CLI argument parsing
fastapi + uvicorn         # API server
numpy + scipy             # Numerical computation
```

**Setup:**
```bash
# Inside WSL
conda env create -f text-to-motion/DART/environment.yml
conda activate DART
pip install transformers  # If CLIP warning appears
```

### Verify DART GPU:
```bash
python -c "import torch; print(torch.cuda.is_available())"  # Must be True
```

---

## 10. Troubleshooting

### AgenticRAG Issues

| Problem | Cause | Solution |
|---------|-------|----------|
| `GEMINI_API_KEY not set` | Missing `.env` | Copy `.env.example` → `.env`, add key |
| `404 model not found` | Invalid model name | `python list_available_models.py` |
| `429 rate limited` | Free Gemini tier | Wait 1–2 min or upgrade API quota |
| Documents not retrieved | Threshold too high | Lower `similarity_threshold` in `config.yaml` |
| Circular import on startup | `response_templates.py` imports from `api_server.py` | Define `MotionPrompt`/`VoicePrompt` locally in `response_templates.py` |

### DART Issues

| Problem | Cause | Solution |
|---------|-------|----------|
| DART conda env not found | Env is in WSL, not Windows | Always run via `wsl -e bash -ic "conda activate DART && ..."` |
| CLIP placeholder warning | `transformers` not installed | `conda activate DART && pip install transformers` |
| CUDA not available | Wrong PyTorch build | `conda install pytorch torchvision pytorch-cuda=12.1 -c pytorch -c nvidia` |
| Checkpoint not found | Wrong file path/format | Checkpoints are `.pt` (not `.ckpt`). Verify path in `api_server.py` |
| Shape mismatch `(T, 250)` | Wrong dataset | BABEL uses `motion_dim=276`, HML3D is different. Check config |
| Jittery motion | Low guidance / few steps | Increase `guidance_scale` (→7.0) or use `respacing='ddim50'` |

### Port Conflicts

```powershell
# Find and kill process on a port
Get-NetTCPConnection -LocalPort 8000 -ErrorAction SilentlyContinue | Select-Object OwningProcess
taskkill /PID <PID> /F
```

---

## Related Documentation

- [`HOW_TO_RUN.md`](HOW_TO_RUN.md) — Quick startup reference
- [`agenticRAG/agentic_rag_gemini/README_DEVELOPERS.md`](agenticRAG/agentic_rag_gemini/README_DEVELOPERS.md) — Deep AgenticRAG documentation
- [`text-to-motion/DART/ARCHITECTURE_AND_INTEGRATION.md`](text-to-motion/DART/ARCHITECTURE_AND_INTEGRATION.md) — Deep DART documentation
- [`SpeechLLm/README.md`](SpeechLLm/README.md) — SpeechLLm documentation
