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
| GEMINI_API_KEYS | Set in `agenticRAG/.env` (mẫu: `agenticRAG/.env.example`) |

## One-Command Launch

> ⛔ **ĐÃ XOÁ (10/08/2026).** Mục này từng hướng dẫn `docker compose up -d chromadb`
> + `python run_stack.py` + frontend :3000. Cả ba đã biến mất cùng
> `agentic_rag_gemini` — `run_stack.py` và `start_server.ps1` không còn tồn tại.
>
> **Hướng dẫn đang dùng: [`scripts/QUICKSTART.md`](../../scripts/QUICKSTART.md).**

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

- [[system-overview]] — Architecture and ports
- [[troubleshooting]] — If setup fails
- [[api-contract]] — Test the running API

---

#setup #install #conda #docker #redis #ffmpeg #quickstart
