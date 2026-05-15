---
title: "System Overview"
description: "High-level architecture, service topology, and data flow."
tags:
  - system
  - architecture
  - overview
  - ports
  - services
  - agentic-rag
  - dart
  - speechllm
---

# System Overview

## Architecture Diagram

```
                         ┌──────────────────────┐
                         │   ECA Official UI     │
                         │     (Port 3000)       │
                         └──────────┬───────────┘
                                    │
                         ┌──────────▼───────────┐
                         │   Unified Gateway     │
                         │     (Port 8000)       │
                         │  /process_query       │
                         │  /tasks/{id}          │
                         │  /download/{file}     │
                         └──────────┬───────────┘
                                    │
               ┌────────────────────┼────────────────────┐
               │                    │                    │
    ┌──────────▼──────────┐  ┌──────▼──────┐  ┌─────────▼─────────┐
    │   AgenticRAG        │  │    DART     │  │    SpeechLLm      │
    │   Orchestrator      │  │  (WSL)     │  │    (Optional)     │
    │   (Port 8080)       │  │ (Port 5001) │  │    (Port 5000)    │
    │                     │  │             │  │                   │
    │  • Gemini LLM       │  │ • Diffusion │  │  • Whisper STT    │
    │  • ChromaDB Memory  │  │ • CLIP      │  │  • Emotion Det.   │
    │  • Document RAG     │  │ • SMPL-X    │  │  • TTS Output     │
    │  • Clinical Safety  │  │ • 30fps     │  │  • Ollama SLM     │
    └─────────────────────┘  └─────────────┘  └───────────────────┘
```

## Services & Ports

| Service | Port | Binding | Purpose |
|---------|------|---------|---------|
| **AgenticRAG API** | 8000 | `0.0.0.0` | Public gateway — `/query`, `/health` |
| **Orchestrator** | 8080 | `127.0.0.1` | Internal — fanout to AgenticRAG/DART/SpeechLLm |
| **DART** | 5001 | `127.0.0.1` | Internal — motion synthesis (WSL) |
| **SpeechLLm** | 5000 | `127.0.0.1` | Optional — voice I/O |
| **ECA UI** | 3000 | `0.0.0.0` | Default frontend |
| **Streamlit UI** | 8501 | `0.0.0.0` | Alternate chat interface |
| **ChromaDB** | 8100 | Docker | Vector store |
| **Redis** | 6379 | `127.0.0.1` | Cache + broker |

## Query Flow

1. User sends query to **Unified Gateway** (`POST /process_query`)
2. Gateway forwards to **AgenticRAG** (`POST /query`) — returns text first
3. If motion requested, gateway calls **DART** (`POST /generate`) in background
4. UI polls `/tasks/{task_id}` for `progress_stage`
5. Final response includes `text_answer`, `exercises`, `motion`, optional `audio_url`

## Key Design Principles

- **Single Origin**: UI never talks to ports 5001/8080 directly; everything goes through 8000.
- **Async Enrichment**: Text returned immediately; motion/TTS generated in background.
- **Polling Contract**: `queued → rag_query → motion_generation → tts → completed`

## Related Notes

- [[ARCHITECTURE_REVIEW]] — Detailed architectural critique and proposed refactor
- [[agentic_rag_refactor]] — May 2026 refactoring notes
- [[api_contract]] — Full API schema
- [[dart_architecture]] — DART internals
- [[setup_guide]] — How to run the stack

---

#system #architecture #overview #ports #services #agentic-rag #dart #speechllm
