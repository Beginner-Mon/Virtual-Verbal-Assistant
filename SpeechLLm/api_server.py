import time
import re
from pathlib import Path
from typing import Optional
import asyncio

from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import uvicorn
import yaml

from src.services.elevenlabs_client import ElevenLabsClient


# =========================
# App Init
# =========================
app = FastAPI(title="SpeechLLM ElevenLabs TTS API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # ⚠️ set frontend URL in production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# =========================
# Request Schemas
# =========================
class VoicePrompt(BaseModel):
    text: str
    emotion: Optional[str] = None


class TTSRequest(BaseModel):
    # Simple mode (dashboard)
    text: Optional[str] = None
    emotion: Optional[str] = None

    # LLM mode (pipeline)
    voice_prompt: Optional[VoicePrompt] = None

    language: Optional[str] = "en"
    user_id: Optional[str] = None


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
    text = text.replace("\\n", " ")
    text = text.replace("\n", " ")
    text = re.sub(r"[*•]+", "", text)
    text = re.sub(r"\s-\s", " ", text)
    text = re.sub(r"\s+", " ", text)
    return text.strip()


# =========================
# Services Init
# =========================
model_config = load_yaml("configs/models.yaml")
elevenlabs_client = ElevenLabsClient(**model_config["elevenlabs"])

audio_dir = Path(model_config["elevenlabs"].get("output_dir", "data/temp_audio"))
audio_dir.mkdir(parents=True, exist_ok=True)


# =========================
# Routes
# =========================
@app.post("/synthesize")
async def synthesize(req: TTSRequest):
    """
    Receive request → Extract text → Clean → TTS → Return metadata
    """
    try:
        # -----------------------------
        # Extract text + emotion
        # -----------------------------
        if req.voice_prompt:
            text = req.voice_prompt.text.strip()
            emotion = req.voice_prompt.emotion
        else:
            text = (req.text or "").strip()
            emotion = req.emotion

        emotion = emotion or "neutral"
        language = req.language or "en"

        if not text:
            raise HTTPException(status_code=400, detail="Voice text cannot be empty")

        clean_text = clean_text_for_tts(text)

        # -----------------------------
        # Run TTS with timing (async-safe)
        # -----------------------------
        start_time = time.time()

        # Run in executor to prevent blocking the event loop
        audio_path = await asyncio.get_event_loop().run_in_executor(
            None,
            elevenlabs_client.synthesize,
            clean_text
        )

        tts_time = time.time() - start_time
        filename = Path(audio_path).name

        return {
            "message": "Synthesis complete",
            "audio_file": filename,
            "language": language,
            "emotion": emotion,
            "tts_time_sec": round(tts_time, 3),
            "request_id": req.user_id,
        }

    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"TTS failed: {str(e)}")


@app.get("/audio/{filename}")
async def get_audio(filename: str):
    """
    Serve generated audio file (MP3 for ElevenLabs, WAV for Coqui)
    """
    path = audio_dir / filename

    if not path.exists():
        raise HTTPException(status_code=404, detail="Audio file not found")

    # Detect media type from file extension
    media_type = "audio/mpeg" if filename.endswith(".mp3") else "audio/wav"
    return FileResponse(path, media_type=media_type)


@app.get("/health")
async def health():
    return {"status": "ok"}


# =========================
# Run
# =========================
if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=5000)