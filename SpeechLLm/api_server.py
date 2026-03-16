"""
REST API: Frontend → Coqui TTS → Audio File
- Accepts raw text
- Cleans formatting artifacts
- Detects language
- Returns audio + metadata
"""

import time
import re
from pathlib import Path
from typing import Optional

from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import uvicorn
import yaml
from langdetect import detect

from src.services.coqui_client import CoquiClient


# =========================
# App Init
# =========================
app = FastAPI(title="SpeechLLM Coqui TTS API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # set frontend URL in production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# =========================
# Request Schema
# =========================
class TTSRequest(BaseModel):
    text: str
    user_id: Optional[str] = None
    emotion: Optional[str] = None


# =========================
# Helpers
# =========================
def load_yaml(path: str):
    with open(path, "r") as f:
        return yaml.safe_load(f)


def clean_text_for_tts(text: str) -> str:
    """
    Clean formatting artifacts while preserving meaning.
    Safe for multilingual text.
    """

    # Replace newlines with natural spacing
    text = text.replace("\\n", " ")
    text = text.replace("\n", " ")

    # Remove bullet symbols
    text = re.sub(r"[*•]+", "", text)

    # Remove markdown-style bullet dashes
    text = re.sub(r"\s-\s", " ", text)

    # Normalize whitespace
    text = re.sub(r"\s+", " ", text)

    return text.strip()


def detect_language_safe(text: str) -> str:
    try:
        return detect(text)
    except Exception:
        return "unknown"


# =========================
# Services Init
# =========================
model_config = load_yaml("configs/models.yaml")
coqui_client = CoquiClient(model_config["coqui"])

audio_dir = Path(model_config["coqui"].get("output_dir", "data/temp_audio"))
audio_dir.mkdir(parents=True, exist_ok=True)


# =========================
# Routes
# =========================
@app.post("/synthesize")
async def synthesize(req: TTSRequest):
    """
    Receive text → Clean → Detect language → TTS → Return metadata
    """
    try:
        if not req.text.strip():
            raise HTTPException(400, "Text cannot be empty")

        # Clean text
        clean_text = clean_text_for_tts(req.text)

        # Detect language
        language = detect_language_safe(clean_text)

        # Run TTS with timing
        start_time = time.time()
        audio_path = coqui_client.synthesize(clean_text)
        tts_time = time.time() - start_time

        filename = Path(audio_path).name

        return {
            "message": "Synthesis complete",
            "audio_file": filename,
            "text_original": req.text,
            "text_spoken": clean_text,
            "language": language,
            "tts_time_sec": round(tts_time, 3),
            "emotion": req.emotion,
            "request_id": req.user_id,
        }

    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(500, f"TTS failed: {str(e)}")


@app.get("/audio/{filename}")
async def get_audio(filename: str):
    """
    Serve generated WAV file
    """
    path = audio_dir / filename

    if not path.exists():
        raise HTTPException(404, "Audio file not found")

    return FileResponse(path, media_type="audio/wav")


@app.get("/health")
async def health():
    """
    Health check endpoint
    """
    return {"status": "ok"}


# =========================
# Run
# =========================
if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=5000)