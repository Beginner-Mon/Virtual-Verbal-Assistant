<p align="center">
  <h1 align="center">🧠 Virtual Verbal Assistant</h1>
  <p align="center">
    <strong>An AI-powered multimodal assistant that understands natural language, generates intelligent responses, produces realistic 3D human motion, and speaks — all in one unified pipeline.</strong>
  </p>
  <p align="center">
    <a href="QUICKSTART.md"><img src="https://img.shields.io/badge/⚡_Quickstart-blue?style=for-the-badge" alt="Quickstart"></a>
    <a href="README_DEV.md"><img src="https://img.shields.io/badge/📖_Developer_Docs-4B32C3?style=for-the-badge" alt="Developer Docs"></a>
    <a href="agenticRAG/agentic_rag_gemini/README_DEVELOPERS.md"><img src="https://img.shields.io/badge/🤖_AgenticRAG-00B4D8?style=for-the-badge" alt="AgenticRAG"></a>
  </p>
</p>

---

## ✨ What is This?

Virtual Verbal Assistant is a multi-service research platform that combines:

- **🤖 AgenticRAG** — A double-RAG conversational AI powered by Google Gemini with persistent memory, document understanding, and clinical knowledge retrieval.
- **🏃 Text-to-Motion (DART)** — A diffusion-based motion synthesis engine that generates realistic 3D human animations from natural language descriptions. ([ICLR 2025 Spotlight](https://arxiv.org/abs/2410.05260))
- **🗣️ SpeechLLm** — Voice I/O with emotion-aware dialogue using local Small Language Models (Whisper + Phi-3/Mistral via Ollama).
- **💬 ECA UI 2.0** — A modern chat interface with real-time motion visualization, TTS audio playback, and exercise cards.

> **Example:** Ask *"My neck is stiff from coding all day"* → get clinically-safe exercise advice + corresponding 3D motion animation + spoken audio response.

---

## 🏗️ Architecture

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

---

## 🚀 Quick Start

### Prerequisites

| Requirement | Notes |
|---|---|
| **Windows + WSL2** | DART runs inside WSL with CUDA |
| **Conda** | `firstconda` (Windows), `DART` (WSL), `tts` (SpeechLLm) |
| **Redis** | Via Docker or system install |
| **ffmpeg** | In PATH for motion video rendering |
| **GEMINI_API_KEY** | Set in `agenticRAG/agentic_rag_gemini/.env` |

### One-Command Launch

```powershell
conda activate firstconda
python run_stack.py
```

This starts **all services** automatically: Redis, Celery, AgenticRAG API, Orchestrator, DART (WSL), SpeechLLm, ECA UI, and Streamlit UI.

### Access Points

| Interface | URL | Description |
|---|:---:|---|
| **ECA Official UI** | [localhost:3000](http://localhost:3000) | Default frontend — chat + motion viewer |
| **Streamlit UI** | [localhost:8501](http://localhost:8501) | Alternate chat interface |
| **API Gateway** | [localhost:8000](http://localhost:8000) | Unified REST API for all clients |

### Verify

```powershell
curl http://localhost:8000/health
```

---

## 🔌 API Reference

All browser and client traffic routes through the **unified gateway** on Port 8000.

### Endpoints

| Endpoint | Method | Purpose |
|---|---|---|
| `/process_query` | `POST` | Submit an async query task |
| `/tasks/{task_id}` | `GET` | Poll task progress and results |
| `/download/{file}` | `GET` | Proxy DART motion artifacts |
| `/history/{user_id}` | `GET` | Retrieve chat history |
| `/health` | `GET` | Per-service health status |

### Task Lifecycle

```json
{
  "task_id": "abc123",
  "status": "processing | completed | failed",
  "progress_stage": "queued → rag_query → motion_generation → tts → completed",
  "result": {
    "text_answer": "...",
    "exercises": [{"name": "Chin tuck"}, {"name": "Shoulder roll"}],
    "motion": {"motion_file_url": "/download/motion_abc123.npz", "fps": 30},
    "audio_url": "/static/tts_abc123.wav"
  },
  "error": null
}
```

> **Design principle:** Internal service ports (`5001`, `8080`) are never exposed to clients. Motion URLs from DART are rewritten to gateway-safe paths through Port 8000.

---

## 🌐 Remote Access (Ngrok)

For remote demonstrations, tunnel the UI and API:

```powershell
# Terminal 1 — UI tunnel
ngrok http 3000

# Terminal 2 — API tunnel
ngrok http 8000
```

Then open:

```
https://<ui-tunnel>.ngrok-free.app/?api_base=https://<api-tunnel>.ngrok-free.app
```

> Do not expose internal ports `5001` or `8080` directly.

---

## 📁 Project Structure

```
Virtual-Verbal-Assistant/
├── run_stack.py                        # One-command stack launcher
├── README.md                          # ← You are here
├── README_DEV.md                      # Full developer & architecture docs
├── QUICKSTART.md                      # Quick reference startup guide
│
├── agenticRAG/agentic_rag_gemini/     # 🤖 AgenticRAG + Orchestrator
│   ├── api_server.py                  #    REST API (Port 8000)
│   ├── main_api.py                    #    Orchestrator (Port 8080)
│   ├── agents/                        #    Query routing (Gemini)
│   ├── retrieval/                     #    RAG pipeline
│   ├── memory/                        #    ChromaDB vector store
│   └── config/config.yaml             #    All tuneable settings
│
├── text-to-motion/DART/               # 🏃 DART motion synthesis (WSL)
│   ├── api_server.py                  #    REST API (Port 5001)
│   ├── mld/                           #    Motion Latent Diffusion
│   ├── diffusion/                     #    Gaussian diffusion
│   ├── model/                         #    Denoiser + VAE
│   └── data/outputs/                  #    Generated .npz files
│
├── SpeechLLm/                         # 🗣️ Voice I/O + Emotion (Port 5000)
│   ├── api_server.py
│   └── src/                           #    STT, LLM, TTS, emotion stages
│
├── ECA_UI/                            # 💬 Official Web Interface (Port 3000)
│   ├── index.html                     #    Main dashboard
│   └── api.js                         #    API client
│
└── test-ui/                           # 🧪 Developer test interface
    └── index.html
```

---

## 📚 Documentation

| Document | Description |
|---|---|
| [QUICKSTART.md](QUICKSTART.md) | Prerequisites, ports, one-command startup |
| [README_DEV.md](README_DEV.md) | Full architecture, subsystem deep-dives, troubleshooting |
| [AgenticRAG Developers](agenticRAG/agentic_rag_gemini/README_DEVELOPERS.md) | Internal AgenticRAG architecture |
| [DART Architecture](text-to-motion/DART/ARCHITECTURE_AND_INTEGRATION.md) | Motion synthesis internals & integration |
| [SpeechLLm](SpeechLLm/README.md) | Voice pipeline documentation |

---

## 🛠️ Troubleshooting

```powershell
# Check if all ports are free
python check_ports.py --ports 3000 5001 6379 8000 8080

# Verify ffmpeg is available
Get-Command ffmpeg

# Check ChromaDB container
docker compose ps

# Kill a stuck port
Get-NetTCPConnection -LocalPort 8000 | Select-Object OwningProcess
taskkill /PID <PID> /F
```

For detailed troubleshooting, see [README_DEV.md § Troubleshooting](README_DEV.md#10-troubleshooting).

---

<p align="center"><sub>Built with Gemini · DART · ChromaDB · SMPL-X · Whisper</sub></p>
