from pathlib import Path
from TTS.api import TTS
import uuid
import os
import torch


class CoquiClient:
    """
    Local neural TTS using Coqui (Fully Offline)
    Supports multi-language routing
    """

    def __init__(self, config: dict):

        # -----------------------------
        # Language → Model Map
        # -----------------------------
        # You can expand this anytime
        self.language_models = config.get("language_models", {
            "en": "tts_models/en/ljspeech/tacotron2-DDC",
        })

        # Output directory
        self.output_dir = Path(config.get("output_dir", "data/temp_audio"))
        self.output_dir.mkdir(parents=True, exist_ok=True)

        # -----------------------------
        # GPU Auto Detection
        # -----------------------------
        env_gpu = os.getenv("COQUI_USE_GPU")

        if env_gpu is not None:
            self.use_gpu = env_gpu.lower() == "true"
        elif "use_gpu" in config:
            self.use_gpu = config["use_gpu"]
        else:
            self.use_gpu = torch.cuda.is_available()

        device_name = "GPU" if self.use_gpu else "CPU"
        print(f"[Coqui] Using device: {device_name}")

        # Default speaker
        self.default_speaker = config.get("speaker", None)

        # -----------------------------
        # Lazy Model Cache
        # -----------------------------
        self.loaded_models = {}          # language → TTS instance
        self.speakers_map = {}           # language → speaker list

        print("[Coqui] Client initialized (lazy model loading).")

    # -----------------------------
    # Internal
    # -----------------------------
    def _load_model_for_language(self, language: str):

        if language in self.loaded_models:
            return self.loaded_models[language]

        if language not in self.language_models:
            raise ValueError(f"Unsupported language: {language}")

        model_name = self.language_models[language]
        print(f"[Coqui] Loading model for '{language}': {model_name}")

        tts = TTS(model_name=model_name, gpu=self.use_gpu)
        self.loaded_models[language] = tts

        # Detect speakers
        if hasattr(tts, "speakers") and tts.speakers:
            self.speakers_map[language] = tts.speakers
            print(f"[Coqui] Speakers ({language}):")
            for i, spk in enumerate(tts.speakers):
                print(f"  {i}: {spk}")
        else:
            self.speakers_map[language] = []

        print(f"[Coqui] Model '{language}' ready.")
        return tts

    # -----------------------------
    # Public Methods
    # -----------------------------
    def get_available_speakers(self, language: str):
        return self.speakers_map.get(language, [])

    def synthesize(
        self,
        text: str,
        language: str = "en",
        speaker: str = None,
        emotion: str = None
    ) -> str:

        if not text or not text.strip():
            raise ValueError("Text for TTS cannot be empty.")

        # NOTE: Coqui models used here don't expose an "emotion" control.
        # We accept it to keep the API stable and ignore it for now.
        _ = emotion

        tts = self._load_model_for_language(language)

        file_id = uuid.uuid4().hex
        output_path = self.output_dir / f"coqui_{language}_{file_id}.wav"

        available_speakers = self.speakers_map.get(language, [])
        selected_speaker = speaker or self.default_speaker

        # -----------------------------
        # Speaker Handling
        # -----------------------------
        if available_speakers:
            if not selected_speaker:
                selected_speaker = available_speakers[0]
                print(f"[Coqui] Using default speaker: {selected_speaker}")

            if selected_speaker not in available_speakers:
                raise ValueError(f"Invalid speaker: {selected_speaker}")

        # -----------------------------
        # Generate Speech
        # -----------------------------
        try:
            tts.tts_to_file(
                text=text,
                file_path=str(output_path),
                speaker=selected_speaker
            )
        except Exception as e:
            raise RuntimeError(f"[Coqui ERROR] {e}")

        return str(output_path)
