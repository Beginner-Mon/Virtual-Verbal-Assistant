"""REST API server for SpeechLLm.

This module exposes the SpeechLLm system as a FastAPI REST service,
allowing external systems to request voice synthesis from text prompts.
"""

import logging
import os
import uuid
from typing import Optional, Dict, Any
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
import uvicorn
import yaml

from src.services.phi3_client import Phi3Client
from src.services.elevenlabs_client import ElevenLabsClient
from src.services.coqui_client import CoquiClient
from src.services.tts_router import TTSRouter
from src.context.memory_manager import MemoryManager
from src.stages.llm_stage import LLMStage
from src.stages.emotion_stage import EmotionStage
from src.stages.tts_stage import TTSStage
from src.core.state_machine import StateMachine, AssistantState

# ===========================
# Setup Logging
# ===========================

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


# ===========================
# Request/Response Models
# ===========================


class SynthesizeRequest(BaseModel):
    """Request to synthesize speech."""

    text: str = Field(..., description="Text to synthesize")
    user_id: str = Field(default="default", description="User identifier")
    emotion: Optional[str] = Field(
        None, description="Requested emotion (happy, sad, angry, calm, neutral)"
    )


class SynthesizeResponse(BaseModel):
    """Response from speech synthesis."""

    audio_file: str = Field(..., description="Path to generated audio file")
    text: str = Field(..., description="Text that was synthesized")
    duration_seconds: float = Field(..., description="Duration of audio in seconds")
    emotion: Optional[str] = Field(None, description="Emotion used for synthesis")
    request_id: str = Field(..., description="Unique request identifier")


# ===========================
# Load Configuration
# ===========================


def load_yaml(path: str) -> Dict[str, Any]:
    """Load YAML configuration file."""
    with open(path, "r") as f:
        return yaml.safe_load(f)


# ===========================
# SpeechLLm API Class
# ===========================


class SpeechLLmAPI:
    """SpeechLLm API wrapper."""

    def __init__(self):
        """Initialize all components."""
        logger.info("Initializing SpeechLLm API...")

        try:
            # Load configuration
            self.base_config = load_yaml("configs/base.yaml")
            self.model_config = load_yaml("configs/models.yaml")

            # Initialize core clients
            self.phi3_client = Phi3Client(self.model_config["phi3"])
            logger.info("✓ Phi3 client initialized")

            # Initialize memory
            self.memory_manager = MemoryManager()
            logger.info("✓ Memory manager initialized")

            # Initialize stages
            self.emotion_stage = EmotionStage()
            self.llm_stage = LLMStage(
                llm_client=self.phi3_client,
                memory_manager=self.memory_manager,
                config=self.model_config["phi3"],
            )
            logger.info("✓ LLM stage initialized")

            # Initialize TTS (use ElevenLabs by default)
            eleven_api_key = os.getenv("ELEVENLABS_API_KEY")
            if not eleven_api_key:
                logger.warning("ELEVENLABS_API_KEY not found, using Coqui fallback only")
                self.tts_router = None
                self.coqui_client = CoquiClient(self.model_config["coqui"])
            else:
                eleven_client = ElevenLabsClient(
                    api_key=eleven_api_key,
                    voice_id=self.model_config["tts"]["voice_id"],
                )
                coqui_client = CoquiClient(self.model_config["coqui"])
                self.tts_router = TTSRouter(
                    eleven_client=eleven_client,
                    coqui_client=coqui_client,
                )
                self.coqui_client = coqui_client
                logger.info("✓ TTS router initialized (ElevenLabs + Coqui)")

            # Initialize TTS stage
            self.tts_stage = TTSStage(tts_router=self.tts_router or self.coqui_client)
            logger.info("✓ TTS stage initialized")

            # Create audio output directory if it doesn't exist
            self.audio_output_dir = Path("data/temp_audio")
            self.audio_output_dir.mkdir(parents=True, exist_ok=True)

            logger.info("SpeechLLm API initialized successfully")

        except Exception as e:
            logger.error(f"Failed to initialize SpeechLLm API: {e}")
            raise

    def synthesize(
        self,
        text: str,
        user_id: str = "default",
        emotion: Optional[str] = None,
    ) -> SynthesizeResponse:
        """Synthesize speech from text.

        Args:
            text: Text to synthesize
            user_id: User identifier
            emotion: Optional emotion context

        Returns:
            SynthesizeResponse with audio file path and metadata
        """
        request_id = str(uuid.uuid4())
        logger.info(f"[{request_id}] Synthesizing speech: {text[:100]}...")

        try:
            # Detect emotion if not provided
            if not emotion:
                detected = self.emotion_stage.process(text)
                emotion = detected.get("emotion", "neutral")
                logger.info(f"[{request_id}] Detected emotion: {emotion}")

            # Generate LLM response based on text (can be enhanced with context)
            llm_response = self.llm_stage.process(text, emotion_context=emotion)
            logger.info(f"[{request_id}] LLM response generated: {llm_response[:100]}...")

            # Synthesize speech
            # Generate audio file
            audio_file_name = f"synthesis_{request_id}.wav"
            audio_file_path = self.audio_output_dir / audio_file_name

            # Use TTS to generate audio
            if self.tts_router:
                self.tts_router.synthesize(
                    text=llm_response,
                    output_path=str(audio_file_path),
                    emotion=emotion,
                )
            else:
                # Fallback to Coqui
                self.coqui_client.synthesize(
                    text=llm_response,
                    output_path=str(audio_file_path),
                )

            # Estimate duration (rough heuristic: 150 words per minute)
            word_count = len(llm_response.split())
            duration_seconds = max(2.0, word_count * 0.4)

            response = SynthesizeResponse(
                audio_file=str(audio_file_path),
                text=llm_response,
                duration_seconds=duration_seconds,
                emotion=emotion,
                request_id=request_id,
            )

            logger.info(f"[{request_id}] Synthesis complete: {audio_file_path}")
            return response

        except Exception as e:
            logger.error(f"[{request_id}] Error during synthesis: {e}")
            raise HTTPException(status_code=500, detail=str(e))


# ===========================
# FastAPI Application
# ===========================

api_instance: Optional[SpeechLLmAPI] = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    """FastAPI lifespan context manager for startup/shutdown."""
    global api_instance
    logger.info("Starting SpeechLLm API server...")
    api_instance = SpeechLLmAPI()
    yield
    logger.info("Shutting down SpeechLLm API server...")


app = FastAPI(
    title="SpeechLLm API",
    description="REST API for multimodal speech and LLM synthesis",
    version="1.0.0",
    lifespan=lifespan,
)


# ===========================
# API Routes
# ===========================


@app.post("/synthesize", response_model=SynthesizeResponse, summary="Synthesize speech from text")
async def synthesize(request: SynthesizeRequest) -> SynthesizeResponse:
    """Synthesize speech from text.

    Args:
        request: Synthesis request with text and optional emotion

    Returns:
        SynthesizeResponse with audio file and metadata
    """
    if api_instance is None:
        raise HTTPException(status_code=500, detail="API not initialized")

    response = api_instance.synthesize(
        text=request.text,
        user_id=request.user_id,
        emotion=request.emotion,
    )

    return response


@app.get("/health", summary="Health check")
async def health_check() -> Dict[str, str]:
    """Health check endpoint."""
    return {"status": "healthy", "service": "speechllm"}


@app.get("/info", summary="Get service info")
async def get_info() -> Dict[str, Any]:
    """Get service information."""
    return {
        "service": "SpeechLLm API",
        "version": "1.0.0",
        "description": "REST API for multimodal speech and LLM synthesis",
        "endpoints": {
            "POST /synthesize": "Synthesize speech from text",
            "GET /health": "Health check",
            "GET /info": "Service information",
        },
    }


# ===========================
# Main
# ===========================

if __name__ == "__main__":
    logger.info("Starting SpeechLLm REST API server on port 5000...")
    uvicorn.run(
        app,
        host="0.0.0.0",
        port=5000,
        log_level="info",
    )
