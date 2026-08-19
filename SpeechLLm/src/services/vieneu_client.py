import logging
import time
import uuid
from pathlib import Path
from typing import Optional

logger = logging.getLogger("speechllm.vieneu")

# SpeechLLm/ — src/services/vieneu_client.py -> services -> src -> SpeechLLm
_ROOT = Path(__file__).resolve().parents[2]


def _resolve_ref(voice_path: str) -> Path:
    """Resolve a caller-supplied reference path against SpeechLLm's own root.

    The caller is the LangGraph service on another process (and, deployed, on
    another host); it sends a name like `voices/anne_vi.wav` and cannot see this
    filesystem. Anchoring to `_ROOT` rather than the CWD means the answer does
    not change depending on where uvicorn was started from.
    """
    p = Path(voice_path)
    return p if p.is_absolute() else (_ROOT / p)


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
        if self.default_voice_path:
            default_ref = _resolve_ref(self.default_voice_path)
            if default_ref.is_file():
                print(f"[VieNeu] Encoding default voice: {default_ref}")
                self._default_voice = self._tts.encode_reference(str(default_ref))
                print("[VieNeu] Default voice ready.")
            else:
                logger.warning(
                    "default_voice_missing configured=%s resolved=%s -> VieNeu preset",
                    self.default_voice_path, default_ref,
                )

        print(f"[VieNeu] Model loaded in {time.time() - load_start:.1f}s")
        return self._tts

    def _resolve_voice(self, voice_path: Optional[str] = None):
        """Encode the requested reference voice, or fall back to the preset.

        A missing .wav must NOT raise: characters appear in the UI catalog long
        before anyone records a voice for them, and a character with no
        recording still has to be able to speak. This is also the only place in
        the system that can answer "does that file exist?" — the caller builds
        the name on another process, and in the deployed layout another host.
        """
        if voice_path:
            ref = _resolve_ref(voice_path)
            key = str(ref)
            if key in self._voice_cache:
                return self._voice_cache[key]
            if ref.is_file():
                tts = self._load_model()
                print(f"[VieNeu] Encoding voice: {ref}")
                v_start = time.time()
                self._voice_cache[key] = tts.encode_reference(str(ref))
                print(f"[VieNeu] Voice encoded in {time.time() - v_start:.1f}s")
                return self._voice_cache[key]
            # WARNING with the absolute path, not a debug line: the symptom of
            # this branch is audio in the wrong voice, which by ear is indis-
            # tinguishable from "right voice, mediocre model". Without this
            # there is nothing to search for.
            logger.warning(
                "voice_reference_missing requested=%s resolved=%s -> falling back to %s",
                voice_path, ref,
                "configured default voice" if self._default_voice is not None
                else "VieNeu preset",
            )
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

        # Reports what was ACTUALLY used, not what was asked for. The old
        # version said "custom" whenever a voice_path was passed, which is
        # exactly the case where a missing file silently downgrades to preset.
        voice_tag = "cloned" if voice is not None else "preset"
        print(f"[VieNeu] Synthesized {len(text)} chars in {infer_time:.2f}s "
              f"(voice={voice_tag}, requested={voice_path or '-'}, lang={language}) "
              f"-> {output_path}")

        return str(output_path)
