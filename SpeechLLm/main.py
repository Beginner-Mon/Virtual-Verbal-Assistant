import json
import os
import re
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


def extract_text_to_read(script: dict) -> str:
    voice_prompt = script.get("voice_prompt", {})
    text_blob = voice_prompt.get("text")

    if not text_blob:
        return ""

    # Case 1 — already dict
    if isinstance(text_blob, dict):
        return text_blob.get("text_answer", "").strip()

    # Case 2 — JSON string
    if isinstance(text_blob, str):
        try:
            inner = json.loads(text_blob)
            return inner.get("text_answer", "").strip()

        except json.JSONDecodeError:
            match = re.search(r'"text_answer"\s*:\s*"(.*)"', text_blob, re.DOTALL)
            if match:
                text = match.group(1)
            else:
                text = text_blob

            return (
                text.replace("\\n", "\n")
                    .replace('\\"', '"')
                    .strip()
            )

    return ""


# =========================
# Main
# =========================
if __name__ == "__main__":

    load_dotenv()

    base_config = load_yaml("configs/base.yaml")
    model_config = load_yaml("configs/models.yaml")

    # Init audio (kept in case you use recording later)
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
    text_to_read = extract_text_to_read(script)

    if not text_to_read:
        print("No readable text found in JSON.")
        exit()

    print("\n===== READING TEXT =====\n")
    print(text_to_read)
    print("\n========================\n")

    # =========================
    # TTS + Play (AUTO RUN)
    # =========================
    try:
        audio_path = coqui_client.synthesize(text_to_read)

        if audio_path and os.path.exists(audio_path):
            # Windows audio playback
            os.system(f'start "" "{audio_path}"')
        else:
            print("Audio file not generated.")

    except Exception as e:
        print("TTS Error:", e)