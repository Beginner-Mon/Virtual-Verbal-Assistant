import os
import yaml
from dotenv import load_dotenv

# Clients
from src.services.phi3_client import Phi3Client
from src.services.whisper_client import WhisperClient
from src.services.voice_driver import VoiceDriver
from src.services.elevenlabs_client import ElevenLabsClient
from src.services.coqui_client import CoquiClient
from src.services.tts_router import TTSRouter

# Core / Stages
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

    # =========================
    # Load environment variables
    # =========================
    load_dotenv()

    # =========================
    # Load configs
    # =========================
    base_config = load_yaml("configs/base.yaml")
    model_config = load_yaml("configs/models.yaml")

    # =========================
    # Initialize Core Clients
    # =========================
    phi3_client = Phi3Client(model_config["phi3"])
    whisper_client = WhisperClient(model_config["whisper"])
    voice_driver = VoiceDriver(base_config["audio"])

    # =========================
    # Initialize TTS Clients
    # =========================
    eleven_api_key = os.getenv("ELEVENLABS_API_KEY")

    if not eleven_api_key:
        raise ValueError("ELEVENLABS_API_KEY not found in .env")

    eleven_client = ElevenLabsClient(
        api_key=eleven_api_key,
        voice_id=model_config["tts"]["voice_id"]
    )

    # =========================
    # Initialize Coqui (temporary for speaker listing)
    # =========================
    temp_coqui = CoquiClient(model_config["coqui"])

    print("\nAvailable Coqui Speakers:")
    speakers = temp_coqui.get_available_speakers()

    for i, spk in enumerate(speakers):
        print(f"{i}: {spk}")

    while True:
        try:
            choice = int(input("Select speaker number: "))
            if 0 <= choice < len(speakers):
                selected_speaker = speakers[choice]
                break
            else:
                print("Invalid selection.")
        except ValueError:
            print("Enter a valid number.")

    print(f"Selected Speaker: {selected_speaker}\n")

    # Recreate Coqui client with selected speaker
    coqui_config = model_config["coqui"]
    coqui_config["speaker"] = selected_speaker

    coqui_client = CoquiClient(coqui_config)

    tts_router = TTSRouter(
        eleven_client=eleven_client,
        coqui_client=coqui_client
    )

    # =========================
    # Memory + Stages
    # =========================
    memory = MemoryManager()

    llm_stage = LLMStage(phi3_client, memory)
    stt_stage = STTStage(whisper_client)
    tts_stage = TTSStage(tts_router)
    emotion_stage = EmotionStage()

    # =========================
    # Streaming + Control
    # =========================
    audio_buffer = AudioStreamBuffer()
    interrupt_controller = InterruptController()

    # =========================
    # Orchestrator
    # =========================
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

    # =========================
    # Main Loop
    # =========================
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