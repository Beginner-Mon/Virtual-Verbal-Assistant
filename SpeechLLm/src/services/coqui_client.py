from pathlib import Path
from TTS.api import TTS
import uuid
import os


class CoquiClient:
    """
    Local neural TTS using Coqui.
    Fully offline.
    Designed to integrate with TTSRouter.
    """

    def __init__(self, config: dict):

        self.model_name = config.get(
            "model_name",
            "tts_models/en/ljspeech/tacotron2-DDC"
        )

        self.output_dir = Path(config.get("output_dir", "data/temp_audio"))
        self.output_dir.mkdir(parents=True, exist_ok=True)

        env_gpu = os.getenv("COQUI_USE_GPU")
        if env_gpu is not None:
            self.use_gpu = env_gpu.lower() == "true"
        else:
            self.use_gpu = config.get("use_gpu", False)

        self.default_speaker = config.get("speaker", None)

        print(f"[Coqui] Loading model: {self.model_name}")
        self.tts = TTS(model_name=self.model_name, gpu=self.use_gpu)
        print("[Coqui] Model loaded successfully.")

        if hasattr(self.tts, "speakers") and self.tts.speakers:
            self.available_speakers = self.tts.speakers
        else:
            self.available_speakers = None

        if self.available_speakers:
            print("[Coqui] Available speakers:")
            for i, spk in enumerate(self.available_speakers):
                print(f"{i}: {spk}")

    # -----------------------------
    # Public Methods
    # -----------------------------

    def get_available_speakers(self):
        return self.available_speakers or []

    def synthesize(self, text: str, speaker: str = None) -> str:

        if not text or not text.strip():
            raise ValueError("Text for TTS cannot be empty.")

        file_id = uuid.uuid4().hex
        output_path = self.output_dir / f"coqui_{file_id}.wav"

        selected_speaker = speaker or self.default_speaker

        # If model is multi-speaker and no speaker provided
        if self.available_speakers:
            if not selected_speaker:
                print("[Coqui] No speaker provided. Using first available speaker.")
                selected_speaker = self.available_speakers[0]

            if selected_speaker not in self.available_speakers:
                raise ValueError(f"Invalid speaker: {selected_speaker}")

        try:
            self.tts.tts_to_file(
                text=text,
                file_path=str(output_path),
                speaker=selected_speaker
            )
        except Exception as e:
            raise RuntimeError(f"[Coqui ERROR] {e}")

        return str(output_path)