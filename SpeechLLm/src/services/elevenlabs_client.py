import os
import time
import uuid
from pathlib import Path
from typing import Optional
from dotenv import load_dotenv
from elevenlabs.client import ElevenLabs
import requests


load_dotenv()


class ElevenLabsClient:
    """
    Production-ready ElevenLabs TTS client.

    Features:
    - Streaming audio
    - Timeout protection
    - Duration estimation
    - Clean error handling
    - Designed for motion sync
    """

    def __init__(
        self,
        voice_id: str = "EXAVITQu4vr4xnSDxMaL",
        model_id: str = "eleven_multilingual_v2",
        timeout: int = 20,
        output_dir: str = "data/temp_audio",
        stability: float = 0.6,
        similarity_boost: float = 0.7,
        use_speaker_boost: bool = True
    ):
        api_key = os.getenv("ELEVENLABS_API_KEY")

        if not api_key:
            raise ValueError("ELEVENLABS_API_KEY not found in environment variables.")

        self.client = ElevenLabs(api_key=api_key)
        self.voice_id = voice_id
        self.model_id = model_id
        self.timeout = timeout
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        self.voice_settings = {
            "stability": stability,
            "similarity_boost": similarity_boost,
            "use_speaker_boost": use_speaker_boost
        }

    def synthesize(
        self,
        text: str
    ) -> str:
        """
        Stream TTS audio and save to file.
        Returns: path to generated MP3 file
        """

        if not text or not text.strip():
            raise ValueError("Text for synthesis cannot be empty.")

        start_time = time.time()

        try:
            audio_stream = self.client.text_to_speech.convert(
                voice_id=self.voice_id,
                model_id=self.model_id,
                text=text,
                voice_settings=self.voice_settings
            )

            audio_buffer = bytearray()

            for chunk in audio_stream:

                # Timeout protection
                if time.time() - start_time > self.timeout:
                    raise TimeoutError("ElevenLabs streaming timed out.")

                if chunk:
                    audio_buffer.extend(chunk)

            audio_bytes = bytes(audio_buffer)

            if not audio_bytes:
                raise RuntimeError("Received empty audio from ElevenLabs.")

            # Save to file
            file_id = uuid.uuid4().hex
            output_path = self.output_dir / f"elevenlabs_{file_id}.mp3"
            
            with open(output_path, "wb") as f:
                f.write(audio_bytes)

            return str(output_path)

        except TimeoutError:
            raise

        except requests.exceptions.RequestException as e:
            print("\n[ElevenLabs REQUEST ERROR]")
            print(e)
            raise

        except Exception as e:
            print("\n[ElevenLabs UNKNOWN ERROR]")
            print(type(e))
            print(e)
            raise

    # -------------------------------------------------------
    # Utility: Estimate MP3 Duration (Lightweight)
    # -------------------------------------------------------

    def _estimate_duration(self, audio_bytes: bytes) -> float:
        """
        Estimate duration assuming ~128kbps MP3.
        This is lightweight and avoids heavy audio libraries.
        """

        bitrate = 128000  # bits per second
        file_size_bits = len(audio_bytes) * 8

        duration_seconds = file_size_bits / bitrate

        return round(duration_seconds, 2)