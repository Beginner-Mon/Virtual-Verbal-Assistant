---
title: "Setup Guide"
description: "Prerequisites, environments, one-command launch, and first health check."
tags:
  - setup
  - install
  - conda
  - docker
  - redis
  - ffmpeg
  - quickstart
---

# Setup Guide

## Prerequisites

| Tool | Notes |
|------|-------|
| Windows + WSL2 | DART runs inside WSL |
| Conda | `firstconda` (Windows), `DART` (WSL), `tts` (SpeechLLm) |
| Docker Desktop | `docker compose up -d chromadb` |
| Redis | `redis-server` in PATH |
| ffmpeg | In PATH for motion video rendering |
| GEMINI_API_KEY | Set in `agenticRAG/agentic_rag_gemini/.env` |

## One-Command Launch

```powershell
# 1. Start ChromaDB
docker compose up -d chromadb

# 2. Launch full stack (API + Orchestrator + Celery + DART + Chat UI)
conda activate firstconda
python run_stack.py
```

Default frontend: `http://localhost:3000`

## Health Check

```powershell
curl http://localhost:8000/health
```

Returns per-service status for Redis, ChromaDB, Celery, DART, and Orchestrator.

## Quick API Test

```powershell
curl -X POST http://localhost:8000/query `
  -H "Content-Type: application/json" `
  -d '{"query": "Show me neck pain exercises", "user_id": "guest"}'
```

## Remote Access (Ngrok)

```powershell
# UI tunnel
ngrok http 3000

# API tunnel (if needed separately)
ngrok http 8000
```

Open UI tunnel URL with `?api_base=<api-tunnel>`.

## Conda Environments

| Env | Purpose | Entry Point |
|-----|---------|-------------|
| `firstconda` | AgenticRAG + Orchestrator | `agenticRAG/agentic_rag_gemini/api_server.py` (8000) |
| `DART` | Motion synthesis | `text-to-motion/DART/api_server.py` (5001) |
| `tts` | SpeechLLm | `SpeechLLm/api_server.py` (5000) |

## Stop

```powershell
# In run_stack.py terminal: Ctrl+C
docker compose down
```

## Related Notes

- [[system_overview]] — Architecture and ports
- [[troubleshooting]] — If setup fails
- [[api_contract]] — Test the running API

---

#setup #install #conda #docker #redis #ffmpeg #quickstart
