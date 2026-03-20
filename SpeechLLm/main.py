import json
import os
import yaml
from dotenv import load_dotenv

from src.services.coqui_client import CoquiClient
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
    # Init Coqui (temp for speaker list)
    # =========================
    temp_coqui = CoquiClient(model_config["coqui"])
    speakers = temp_coqui.get_available_speakers()

    # =========================
    # Speaker Selection
    # =========================
    if speakers:
        print("\nAvailable Speakers:")
        for i, spk in enumerate(speakers):
            print(f"{i}: {spk}")

        while True:
            try:
                choice = int(input("Select speaker number: "))
                if 0 <= choice < len(speakers):
                    selected_speaker = speakers[choice]
                    break
                print("Invalid selection.")
            except ValueError:
                print("Enter a valid number.")
    else:
        selected_speaker = None

    print(f"\nSelected Speaker: {selected_speaker}")

    # Recreate client with selected speaker
    coqui_config = model_config["coqui"]
    coqui_config["speaker"] = selected_speaker
    coqui_client = CoquiClient(coqui_config)

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
        print("TTS Error:", e)