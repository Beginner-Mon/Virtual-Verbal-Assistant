# Virtual Verbal Assistant (SpeechSLM)

This project implements a multimodal virtual assistant that accepts
either **speech or text input**, detects the user's **emotion**, formats
the interaction into a structured JSON representation, and generates
responses using a **Small Language Model (SLM)** running locally.

The assistant then outputs both **text and speech**, and can optionally
control a virtual avatar.

---

## 🎯 Features

- Speech or text input
- Speech-to-Text (Whisper / wav2vec2)
- Emotion recognition from text/audio
- Context formatting using JSON
- Local SLM inference (Phi / Mistral / Gemma via Ollama)
- Text-to-Speech output
- Modular system architecture

---

## 🧠 System Pipeline

Voice / Text Input
↓
Speech-to-Text
↓
Emotion Detection
↓
Context Formatter (JSON)
↓
SLM Reasoning Engine
↓
Text-to-Speech + Avatar

---

## 📁 Project Structure

SpeechLLM/
├── src/
│   ├── core/                   # The "Brain" and State Logic
│   │   ├── orchestrator.py      # Main loop; moves data between STT -> LLM -> TTS
│   │   ├── state_machine.py     # Tracks IDLE, LISTENING, THINKING, SPEAKING
│   │   └── events.py            # Signals for "Interrupt" or "Speech Detected"
│   │
│   ├── stages/                  # Pipeline Steps (The "What")
│   │   ├── stt_stage.py         # Takes audio buffer -> returns text
│   │   ├── llm_stage.py         # Takes text + emotion -> returns Phi-3 response
│   │   ├── tts_stage.py         # Takes text -> returns audio stream
│   │   └── emotion_stage.py     # Analyzes text/audio for emotional metadata
│   │
│   ├── services/                # Model Wrappers (The "How")
│   │   ├── phi3_client.py       # Specific implementation for Phi-3 inference
│   │   ├── whisper_client.py    # Faster-Whisper or OpenAI Whisper setup
│   │   └── voice_driver.py      # Low-level audio device management (PyAudio)
│   │
│   ├── context/                 # Memory & Formatting
│   │   ├── memory_manager.py    # Handles conversation history (Short-term/Long-term)
│   │   └── prompt_templates.py  # Stores system prompts and JSON schemas
│   │
│   ├── ui/                      # Visuals
│   │   ├── avatar_controller.py # Sends blendshapes/visemes to the 3D model
│   │   └── web_server.py        # FastAPI/WebSocket for the frontend dashboard
│   │
│   └── utils/                   # Helpers
│       ├── logger.py            # Custom logging to console and file
│       └── audio_tools.py       # VAD (Voice Activity Detection) and Chunking
│
├── configs/                     # System Settings
│   ├── base.yaml                # Default parameters (Sample rates, etc.)
│   └── models.yaml              # Paths to local .bin or .onnx files
│
├── models/                      # Weights (Git-ignored)
│   ├── phi3/                    # Phi-3-3.8B files
│   └── whisper/                 # Whisper weights
│
├── data/                        # Persistent Data
│   ├── logs/                    # Runtime logs for debugging
│   └── temp_audio/              # Cache for temporary .wav processing
│
├── .env                         # Environment variables (API keys)
├── requirements.txt             # Python dependencies
├── main.py                      # Application Entry Point
└── README.md                    # Setup and usage guide


---

## ⚙️ Setup

1. Create Conda Environment

```bash
conda env create -f environment.yml
conda activate speechslm

2. Install Ollama (for SLM)
Download from:

https://ollama.com

Pull a model:

ollama pull phi3
or:

ollama pull mistral
3. Run
python main.py
🧪 Example Output
User (speech): I feel stressed today.
Emotion: anxious (0.81)

Model: I'm sorry you're feeling stressed. Want to talk about what's causing it?

---

## 📊 Project Goals

Demonstrate multimodal interaction

Explore emotion-aware dialogue systems

Compare SLM performance vs large LLMs

Evaluate latency and accuracy

Control avatar expressions

--- 

📄 License

Educational use only.


---