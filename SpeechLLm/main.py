import json
import os
import yaml
from dotenv import load_dotenv

from src.services.elevenlabs_client import ElevenLabsClient
from src.services.voice_driver import VoiceDriver


# =========================
# Helpers
# =========================
def load_yaml(path):
    with open(path, "r") as f:
        return yaml.safe_load(f)


def load_script(json_path):
    with open(json_path, "r", encoding="utf-8") as f:
        return json.load(f)


def extract_voice_fields(script: dict):
    voice_prompt = script.get("voice_prompt", {})
    text = voice_prompt.get("text", "").strip()
    emotion = voice_prompt.get("emotion") or "neutral"
    language = script.get("language", "en")
    return text, emotion, language


# =========================
# Main
# =========================
if __name__ == "__main__":

    load_dotenv()

    base_config = load_yaml("configs/base.yaml")
    model_config = load_yaml("configs/models.yaml")

    # Init audio driver
    voice_driver = VoiceDriver(base_config["audio"])

    # =========================
    # Init ElevenLabs TTS Client
    # =========================
    elevenlabs_client = ElevenLabsClient(**model_config["elevenlabs"])

    # =========================
    # Load Script
    # =========================
    script = load_script("data/scripts.json")
    text_to_read, emotion, language = extract_voice_fields(script)

    if not text_to_read:
        print("No readable text found in JSON.")
        exit()

    print("\n===== READING TEXT =====\n")
    print(text_to_read)
    print(f"\nLanguage: {language}")
    print(f"Emotion: {emotion}")
    print("\n========================\n")

    # =========================
    # TTS + Play (AUTO RUN)
    # =========================
    try:
        audio_path = coqui_client.synthesize(
            text=text_to_read,
            language=language,
            emotion=emotion
        )

        if audio_path and os.path.exists(audio_path):
            os.system(f'start "" "{audio_path}"')  # Windows playback
        else:
            print("Audio file not generated.")

    except Exception as e:
        print("TTS Error:", e)elevenlabs_client.synthesize(
            text=text_to_read