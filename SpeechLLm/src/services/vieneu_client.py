import time
import uuid
from pathlib import Path
from typing import Optional


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
        self.default_voice_path = config.get("voice_path", None)
        self._tts = None
        self._voice_cache: dict = {}
        self._default_voice = None

        src = self.model_path or "HF hub (auto-download)"
        voice_info = self.default_voice_path or "preset"
        print(f"[VieNeu] Initialized (mode={self.mode}, device={self.device}, "
              f"model={src}, voice={voice_info}).")

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

        # Pre-load default voice if configured
        if self.default_voice_path and Path(self.default_voice_path).exists():
            print(f"[VieNeu] Encoding default voice: {self.default_voice_path}")
            self._default_voice = self._tts.encode_reference(self.default_voice_path)
            print("[VieNeu] Default voice ready.")

        print(f"[VieNeu] Model loaded in {time.time() - load_start:.1f}s")
        return self._tts

    def _resolve_voice(self, voice_path: Optional[str] = None):
        if voice_path:
            if voice_path not in self._voice_cache:
                tts = self._load_model()
                print(f"[VieNeu] Encoding voice: {voice_path}")
                v_start = time.time()
                self._voice_cache[voice_path] = tts.encode_reference(voice_path)
                print(f"[VieNeu] Voice encoded in {time.time() - v_start:.1f}s")
            return self._voice_cache[voice_path]
        if self._default_voice is not None:
            return self._default_voice
        return None

    def synthesize(self, text: str, language: str = "vi",
                   voice_path: Optional[str] = None) -> str:
        if not text.strip():
            raise ValueError("Text for TTS cannot be empty.")

        tts = self._load_model()
        voice = self._resolve_voice(voice_path)

        infer_start = time.time()
        audio = tts.infer(text=text, voice=voice)
        infer_time = time.time() - infer_start

        file_id = uuid.uuid4().hex
        output_path = self.output_dir / f"vieneu_{language}_{file_id}.wav"
        tts.save(audio, str(output_path))

        voice_tag = "custom" if voice_path or self.default_voice_path else "preset"
        print(f"[VieNeu] Synthesized {len(text)} chars in {infer_time:.2f}s "
              f"(voice={voice_tag}) -> {output_path}")

        return str(output_path)
