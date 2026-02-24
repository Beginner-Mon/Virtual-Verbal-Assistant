import yaml
from pathlib import Path

from src.services.phi3_client import Phi3Client
from src.services.whisper_client import WhisperClient
from src.services.voice_driver import VoiceDriver

from src.context.memory_manager import MemoryManager

from src.stages.llm_stage import LLMStage
from src.stages.stt_stage import STTStage
from src.stages.tts_stage import TTSStage
from src.stages.emotion_stage import EmotionStage

from streaming.audio_stream_buffer import AudioStreamBuffer
from streaming.interrupt_controller import InterruptController

from src.core.orchestrator import Orchestrator


def load_yaml(path):
    with open(path, "r") as f:
        return yaml.safe_load(f)


if __name__ == "__main__":

    base_config = load_yaml("configs/base.yaml")
    model_config = load_yaml("configs/models.yaml")

    phi3_client = Phi3Client(model_config["phi3"])
    whisper_client = WhisperClient(model_config["whisper"])
    voice_driver = VoiceDriver(base_config["audio"])

    memory = MemoryManager()

    llm_stage = LLMStage(phi3_client, memory)
    stt_stage = STTStage(whisper_client)
    tts_stage = TTSStage(base_config["tts"])
    emotion_stage = EmotionStage()

    audio_buffer = AudioStreamBuffer()
    interrupt_controller = InterruptController()

    orchestrator = Orchestrator(
        stt_stage=stt_stage,
        emotion_stage=emotion_stage,
        llm_stage=llm_stage,
        tts_stage=tts_stage,
        voice_driver=voice_driver,
        audio_buffer=audio_buffer,
        interrupt_controller=interrupt_controller,
    )

    print("Assistant Ready.")
    print("Type 'voice' for microphone mode, 'text' for typing, 'exit' to quit.")

    while True:
        mode = input("Mode: ").strip().lower()

        if mode == "exit":
            break

        elif mode == "text":
            user_text = input("You: ")
            response = orchestrator.handle_text_input(user_text)
            print("Assistant:", response)

        elif mode == "voice":
            print("Recording...")
            audio_path = voice_driver.record()

            print("Transcribing...")
            transcript = stt_stage.process(audio_path)

            print("You:", transcript)

            print("Processing...")
            response = orchestrator.handle_text_input(transcript)

            print("Assistant:", response)

        else:
            print("Invalid mode.")