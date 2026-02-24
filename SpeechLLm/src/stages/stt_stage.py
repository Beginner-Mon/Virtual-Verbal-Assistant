from src.services.whisper_client import WhisperClient


class STTStage:

    def __init__(self, whisper_client: WhisperClient):
        self.client = whisper_client

    def process(self, audio_path: str) -> str:
        transcript = self.client.transcribe(audio_path)
        return transcript.strip()