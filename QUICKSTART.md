# QUICKSTART

## Prerequisites

| Tool | Notes |
|------|-------|
| Windows + WSL2 | DART runs inside WSL |
| Conda | `firstconda` env on Windows, `DART` env in WSL |
| Docker Desktop | For ChromaDB (`docker compose up -d chromadb`) |
| Redis | `redis-server` in PATH |
| ffmpeg | In PATH (for motion video rendering) |

## Ports

| Service | Port | Binding |
|---------|:----:|---------|
| **AgenticRAG API** | 8000 | `0.0.0.0` (public gateway) |
| **ECA Official UI 2.0** | 3000 | `0.0.0.0` (default frontend) |
| **Streamlit Chat UI** | 8501 | `0.0.0.0` |
| Orchestrator | 8080 | `127.0.0.1` (internal) |
| DART | 5001 | `127.0.0.1` (internal) |
| ChromaDB | 8100 | Docker |
| Redis | 6379 | `127.0.0.1` |

## Port 8000 Gateway Architecture

- Official UI always calls Port 8000 (`/process_query`, `/tasks/{task_id}`)
- Motion artifacts are also served through Port 8000 (`/download/{file}` or `/static/...`), never direct Port 5001 links
- This enables a single Ngrok tunnel for end-to-end remote usage

Unified polling contract:

- `POST /process_query` -> `{task_id, status, progress_stage, result, error}`
- `GET /tasks/{task_id}` -> same schema with progressive `progress_stage` updates (`queued -> motion_generation -> ... -> completed`)

## Start

```powershell
# 1. Start ChromaDB
docker compose up -d chromadb

# 2. Launch full stack (API + Orchestrator + Celery + DART + Chat UI)
conda activate firstconda

h
# Default frontend (Official UI)
# http://localhost:3000
```

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

**Note:** Do not install ngrok via `pip` (it only installs a Python wrapper). You need the system binary.

**1. Install & Authenticate:**
```powershell
Invoke-WebRequest -Uri "https://bin.equinox.io/c/bNyj1mQVY4c/ngrok-v3-stable-windows-amd64.zip" -OutFile "ngrok.zip"
Expand-Archive -Path "ngrok.zip" -DestinationPath "$env:USERPROFILE\ngrok-cli" -Force
[Environment]::SetEnvironmentVariable("PATH", "$env:USERPROFILE\ngrok-cli;" + [Environment]::GetEnvironmentVariable("PATH", "User"), "User")
$env:PATH = "$env:USERPROFILE\ngrok-cli;" + $env:PATH
ngrok config add-authtoken <your-token-from-dashboard.ngrok.com>
```

**2. Start Tunnel** (make sure `run_stack.py` is running first):
```powershell
# Expose Official ECA UI static page (single tunnel)
ngrok http 3000
```

For Official ECA UI remote access, open the 3000 tunnel URL with the 8000 API base query parameter:

`https://<eca-ui-tunnel>.ngrok-free.app/?api_base=https://<gateway-8000-tunnel>.ngrok-free.app`

If you also need to expose the API directly, run a second tunnel for 8000:

```powershell
ngrok http 8000
ngrok http 3000
```

This preserves single-origin API usage from the UI without exposing internal ports 5001/8080.

Streamlit parity UI can be exposed separately if needed (`ngrok http 8501`). Keep 5001 internal.

## Troubleshooting

```powershell
docker compose ps                                    # ChromaDB running?
python check_ports.py --ports 5001 6379 8000 8080    # Port conflicts?
Get-Command ffmpeg                                   # ffmpeg in PATH?
```

## Stop

Press `Ctrl+C` in the `run_stack.py` terminal, then `docker compose down`.

## Docs

- [README_DEV.md](README_DEV.md) — Architecture overview
- [README_DEVELOPERS.md](agenticRAG/agentic_rag_gemini/README_DEVELOPERS.md) — AgenticRAG internals
