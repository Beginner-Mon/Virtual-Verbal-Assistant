from faster_whisper import WhisperModel


class WhisperClient:

    def __init__(self, config: dict):
        model_size = config.get("model_size", "base")
        device = config.get("device", "cpu")
        compute_type = config.get("compute_type", "int8")

        self.model = WhisperModel(
            model_size,
            device=device,
            compute_type=compute_type
        )

    def transcribe(self, audio_path: str) -> str:
        segments, _ = self.model.transcribe(audio_path)

        transcript = ""
        for segment in segments:
            transcript += segment.text

        return transcript.strip()