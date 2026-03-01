# How to Run the Virtual Verbal Assistant Pipeline

## Overview

This project runs **3 servers** that work together:

```
┌──────────────────────────────────────────────────────┐
│  Orchestrator (port 8080) — Unified entry point      │
│  ┌────────────────┐  ┌─────────────────────────────┐ │
│  │ AgenticRAG     │  │ DART (Text-to-Motion)       │ │
│  │ Port 8000      │  │ Port 5001                   │ │
│  │ Windows        │  │ WSL / Linux                 │ │
│  │ Env: firstconda│  │ Env: DART                   │ │
│  └────────────────┘  └─────────────────────────────┘ │
└──────────────────────────────────────────────────────┘
```

---

## Prerequisites

- **Windows** with WSL installed
- **Conda** installed on both Windows and WSL
- Conda environments: `firstconda` (Windows), `DART` (WSL)

---

## Step 1: Start AgenticRAG (Windows Terminal 1)

### Command
```powershell
conda activate firstconda
cd agenticRAG\agentic_rag_gemini
python api_server.py
```

### Expected Output
```
INFO Logging initialized: level=INFO, file=logs/agentic_rag.log
INFO Initializing AgenticRAG API...
INFO Initializing ResponseTemplateGenerator
INFO AgenticRAG API initialized successfully
INFO:     Uvicorn running on http://0.0.0.0:8000 (Press CTRL+C to quit)
INFO:     Application startup complete.
```

### Test Endpoint

**Health check:**
```powershell
Invoke-WebRequest -Uri "http://localhost:8000/health" -UseBasicParsing | Select-Object -ExpandProperty Content
```
Expected: `{"status":"healthy","service":"agenticrag"}`

**Query test:**
```powershell
$body = '{"query":"How do I walk?","user_id":"test","conversation_history":[]}'
Invoke-WebRequest -Uri "http://localhost:8000/query" -Method POST -ContentType "application/json" -Body $body -UseBasicParsing | Select-Object -ExpandProperty Content
```
Expected: JSON with `text_answer`, `orchestrator_decision`, `motion_prompt`, `voice_prompt`

---

## Step 2: Start DART (Windows Terminal 2 → WSL)

### Command
```powershell
wsl -e bash -ic "conda activate DART && cd /mnt/d/'Swin documents'/Virtual-Verbal-Assistant/text-to-motion/DART && python api_server.py"
```

### Expected Output
```
INFO:__main__:Starting DART Motion REST API server on port 5001...
INFO:__main__:Initializing DART Motion API...
INFO:__main__:Initializing DART models on device: cuda
INFO:__main__:Loading DART models...
INFO:__main__:✓ Denoiser checkpoint: mld_denoiser/checkpoint_300000.pt
INFO:__main__:✓ VAE checkpoint: mld_fps_clip_repeat_euler/checkpoint_000/model.pt
INFO:__main__:✓ Models loaded successfully
INFO:__main__:DART Motion API initialized successfully
INFO:     Uvicorn running on http://0.0.0.0:5001 (Press CTRL+C to quit)
INFO:     Application startup complete.
```

### Test Endpoint

**Health check:**
```powershell
Invoke-WebRequest -Uri "http://localhost:5001/health" -UseBasicParsing | Select-Object -ExpandProperty Content
```
Expected: `{"status":"healthy","service":"dart"}`

**Motion generation test:**
```powershell
$body = '{"text_prompt":"walk forward","num_primitives":20,"guidance_scale":5.0,"num_steps":10}'
Invoke-WebRequest -Uri "http://localhost:5001/generate_motion" -Method POST -ContentType "application/json" -Body $body -UseBasicParsing | Select-Object -ExpandProperty Content
```
Expected:
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

---

## Step 3: Start Orchestrator (Windows Terminal 3)

> ⚠️ **Start this AFTER** AgenticRAG and DART are running.

### Command
```powershell
conda activate firstconda
cd agenticRAG\agentic_rag_gemini
python main_api.py
```

### Expected Output
```
INFO Starting Unified Multi-Service Pipeline API on port 8080...

Make sure you have these services running:
  1. SpeechLLm:      python SpeechLLm/api_server.py         (port 5000)
  2. DART:           python text-to-motion/DART/api_server.py (port 5001, Linux)
  3. AgenticRAG:     python agenticRAG/agentic_rag_gemini/api_server.py (port 8000)

Frontend can call: POST http://localhost:8080/answer

INFO:     Uvicorn running on http://0.0.0.0:8080 (Press CTRL+C to quit)
INFO:     Application startup complete.
```

### Test Endpoint

**Health check:**
```powershell
Invoke-WebRequest -Uri "http://localhost:8080/health" -UseBasicParsing | Select-Object -ExpandProperty Content
```
Expected: `{"status":"healthy","service":"unified-pipeline","orchestrator":"ready"}`

**Full pipeline test:**
```powershell
$body = '{"query":"How do I walk?","user_id":"test","conversation_history":[]}'
Invoke-WebRequest -Uri "http://localhost:8080/answer" -Method POST -ContentType "application/json" -Body $body -UseBasicParsing | Select-Object -ExpandProperty Content
```
Expected:
```json
{
  "text_answer": "To walk forward...",
  "voice": null,
  "motion": {
    "file": "data/outputs/motion_<uuid>.npz",
    "num_frames": 160,
    "fps": 30
  },
  "generation_time_ms": 3500.0,
  "errors": null
}
```

---

## Quick Reference

| # | Service | Port | Env | Platform | Command |
|---|---------|------|-----|----------|---------|
| 1 | AgenticRAG | 8000 | `firstconda` | Windows | `python api_server.py` |
| 2 | DART | 5001 | `DART` | WSL | `python api_server.py` |
| 3 | Orchestrator | 8080 | `firstconda` | Windows | `python main_api.py` |

## Test UI

Open `test-ui/index.html` in your browser to interactively test all endpoints with a visual interface.

---

## Troubleshooting

### Port already in use
```powershell
Get-NetTCPConnection -LocalPort 8000 -ErrorAction SilentlyContinue | Select-Object OwningProcess
taskkill /PID <PID> /F
```

### Circular import error in AgenticRAG
Ensure `agents/response_templates.py` does NOT import from `api_server.py`. The models (`MotionPrompt`, `VoicePrompt`) should be defined locally in `response_templates.py`.

### DART conda env not found
The DART env lives in **WSL**, not Windows. Always run DART via:
```powershell
wsl -e bash -ic "conda activate DART && ..."
```

### CLIP placeholder warning in DART
Install transformers in WSL DART env:
```bash
# Inside WSL
conda activate DART
pip install transformers
```
