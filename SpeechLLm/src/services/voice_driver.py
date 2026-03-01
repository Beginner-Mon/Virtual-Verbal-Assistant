import sounddevice as sd
import soundfile as sf
from pathlib import Path
import uuid


class VoiceDriver:

    def __init__(self, config: dict):
        self.sample_rate = config.get("sample_rate", 16000)
        self.channels = config.get("channels", 1)
        self.default_duration = config.get("duration", 5)

        self.temp_dir = Path("data/temp_audio")
        self.temp_dir.mkdir(parents=True, exist_ok=True)

    def record(self, duration: int = None) -> str:
        duration = duration or self.default_duration

        recording = sd.rec(
            int(duration * self.sample_rate),
            samplerate=self.sample_rate,
            channels=self.channels
        )
        sd.wait()

        file_id = uuid.uuid4().hex
        file_path = self.temp_dir / f"{file_id}.wav"

        sf.write(file_path, recording, self.sample_rate)

        return str(file_path)

    def play_audio(self, audio_path: str):
        data, samplerate = sf.read(audio_path)
        sd.play(data, samplerate)
        sd.wait()