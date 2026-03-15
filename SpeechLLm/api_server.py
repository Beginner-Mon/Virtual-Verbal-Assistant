from pathlib import Path
from typing import Optional
import re

from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import uvicorn
import yaml

from src.services.coqui_client import CoquiClient


# =========================
# App Init
# =========================
app = FastAPI(title="Coqui TTS API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # or ["http://127.0.0.1:5500"]
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
def load_yaml(path):
    with open(path, "r") as f:
        return yaml.safe_load(f)


def clean_text_for_tts(text: str) -> str:
    """
    Make text sound natural when spoken
    """
    # Convert escaped newlines to pauses
    text = text.replace("\\n", ". ")
    text = text.replace("\n", ". ")

    # Remove bullet symbols
    text = re.sub(r"[*•\-]+", "", text)

    # Remove list numbers like "1."
    text = re.sub(r"\b\d+\.\s*", "", text)

    # Collapse whitespace
    text = re.sub(r"\s+", " ", text)

    return text.strip()


# =========================
# Services Init
# =========================
model_config = load_yaml("configs/models.yaml")
coqui_client = CoquiClient(model_config["coqui"])

audio_dir = Path("data/temp_audio")
audio_dir.mkdir(parents=True, exist_ok=True)


# =========================
# Routes
# =========================
@app.post("/synthesize")
async def synthesize(req: TTSRequest):
    try:
        if not req.text.strip():
            raise HTTPException(400, "Text cannot be empty")

        clean_text = clean_text_for_tts(req.text)

        audio_path = coqui_client.synthesize(clean_text)
        filename = Path(audio_path).name

        return {
            "message": "Synthesis complete",
            "audio_file": filename,
            "text_original": req.text,
            "text_spoken": clean_text,
            "emotion": req.emotion,
            "request_id": req.user_id,
        }

    except Exception as e:
        raise HTTPException(500, str(e))


@app.get("/audio/{filename}")
async def get_audio(filename: str):
    path = audio_dir / filename

    if not path.exists():
        raise HTTPException(404, "File not found")

    return FileResponse(path, media_type="audio/wav")


@app.get("/health")
async def health():
    return {"status": "ok"}


# =========================
# Run
# =========================
if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=5000)
