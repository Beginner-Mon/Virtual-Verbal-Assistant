import time
import uuid
from pathlib import Path


class VieNeuClient:
    """
    Local TTS using VieNeu-TTS-v2-Turbo (GGUF + ONNX codec).
    CPU-only, Vietnamese-English bilingual, zero-shot voice cloning.
    """

    def __init__(self, config: dict):
        self.mode = config.get("mode", "turbo")
        self.model_path = config.get("model_path", None)
        self.device = config.get("device", "cpu")
        self.output_dir = Path(config.get("output_dir", "data/temp_audio"))
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self._tts = None

        src = self.model_path or "HF hub (auto-download)"
        print(f"[VieNeu] Initialized (mode={self.mode}, device={self.device}, model={src}).")

    def _load_model(self):
        if self._tts is not None:
            return self._tts

        from vieneu import Vieneu

        print("[VieNeu] Loading model (GGUF + ONNX codec)...")
        load_start = time.time()

        kwargs = {"mode": self.mode, "device": self.device}
        if self.model_path:
            kwargs["backbone_repo"] = self.model_path

        self._tts = Vieneu(**kwargs)

        print(f"[VieNeu] Model loaded in {time.time() - load_start:.1f}s")
        return self._tts

    def synthesize(self, text: str, language: str = "vi") -> str:
        if not text.strip():
            raise ValueError("Text for TTS cannot be empty.")

        tts = self._load_model()

        infer_start = time.time()
        audio = tts.infer(text=text)
        infer_time = time.time() - infer_start

        file_id = uuid.uuid4().hex
        output_path = self.output_dir / f"vieneu_{language}_{file_id}.wav"
        tts.save(audio, str(output_path))

        print(f"[VieNeu] Synthesized {len(text)} chars in {infer_time:.2f}s "
              f"-> {output_path}")

        return str(output_path)
