from pathlib import Path
from TTS.api import TTS
import uuid
import os
import torch
import wave
from concurrent.futures import ThreadPoolExecutor


class CoquiClient:
    """
    Local neural TTS using Coqui (Fully Offline)
    Optimized for speed with chunking + parallel synthesis
    """

    def __init__(self, config: dict):

        self.language_models = config.get("language_models", {
            "en": "tts_models/en/ljspeech/tacotron2-DDC",
        })

        self.output_dir = Path(config.get("output_dir", "data/temp_audio"))
        self.output_dir.mkdir(parents=True, exist_ok=True)

        # -----------------------------
        # Safe Device Detection
        # -----------------------------
        env_gpu = os.getenv("COQUI_USE_GPU")

        if env_gpu is not None:
            self.use_gpu = env_gpu.lower() == "true"
        else:
            self.use_gpu = torch.cuda.is_available()

        # Final safety check
        if self.use_gpu and not torch.cuda.is_available():
            print("[Coqui] CUDA requested but not available → Falling back to CPU")
            self.use_gpu = False

        device_name = "GPU" if self.use_gpu else "CPU"
        print(f"[Coqui] Using device: {device_name}")

        self.default_speaker = config.get("speaker", None)

        self.loaded_models = {}
        self.speakers_map = {}

        print("[Coqui] Client initialized (lazy model loading).")

    # -----------------------------
    # Utilities
    # -----------------------------
    def _smart_chunks(self, text: str, max_chars=400):
        chunks = []
        while len(text) > max_chars:
            split = text.rfind(".", 0, max_chars)
            if split == -1:
                split = max_chars
            chunks.append(text[:split + 1])
            text = text[split + 1:]
        chunks.append(text)
        return chunks

    def _merge_wavs(self, wav_files, output_path):
        with wave.open(output_path, 'wb') as out:
            for i, wav in enumerate(wav_files):
                with wave.open(wav, 'rb') as w:
                    if i == 0:
                        out.setparams(w.getparams())
                    out.writeframes(w.readframes(w.getnframes()))

    # -----------------------------
    # Model Loading
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

        if hasattr(tts, "speakers") and tts.speakers:
            self.speakers_map[language] = tts.speakers
        else:
            self.speakers_map[language] = []

        print(f"[Coqui] Model '{language}' ready.")
        return tts

    # -----------------------------
    # Public API
    # -----------------------------
    def get_available_speakers(self, language: str):
        return self.speakers_map.get(language, [])

    def synthesize(self, text: str, language="en", speaker=None) -> str:

        if not text.strip():
            raise ValueError("Text for TTS cannot be empty.")

        tts = self._load_model_for_language(language)

        available_speakers = self.speakers_map.get(language, [])
        selected_speaker = speaker or self.default_speaker

        if available_speakers and not selected_speaker:
            selected_speaker = available_speakers[0]

        # -----------------------------
        # Chunk + Parallel TTS
        # -----------------------------
        chunks = self._smart_chunks(text)

        def tts_chunk(chunk_text):
            fid = uuid.uuid4().hex
            part_path = self.output_dir / f"part_{fid}.wav"
            tts.tts_to_file(
                text=chunk_text,
                file_path=str(part_path),
                speaker=selected_speaker
            )
            return str(part_path)

        with ThreadPoolExecutor() as ex:
            part_files = list(ex.map(tts_chunk, chunks))

        # -----------------------------
        # Merge Parts
        # -----------------------------
        final_id = uuid.uuid4().hex
        final_path = self.output_dir / f"coqui_{language}_{final_id}.wav"
        self._merge_wavs(part_files, str(final_path))

        # cleanup temp parts
        for p in part_files:
            os.remove(p)

        return str(final_path)