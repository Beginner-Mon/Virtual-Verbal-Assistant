import requests

class ElevenLabsClient:
    def __init__(self, api_key: str, voice_id: str = "EXAVITQu4vr4xnSDxMaL"):
        self.api_key = api_key
        self.voice_id = voice_id
        self.base_url = "https://api.elevenlabs.io/v1/text-to-speech"

    def synthesize(self, text: str, output_path: str = None) -> bytes:
        url = f"{self.base_url}/{self.voice_id}"

        headers = {
            "xi-api-key": self.api_key,
            "Content-Type": "application/json"
        }

        payload = {
            "text": text,
            "model_id": "eleven_multilingual_v2",  # good default
            "voice_settings": {
                "stability": 0.5,
                "similarity_boost": 0.8
            }
        }

        try:
            response = requests.post(url, json=payload, headers=headers)

            if response.status_code != 200:
                raise Exception(f"ElevenLabs API error: {response.text}")

            audio_bytes = response.content

            if output_path:
                with open(output_path, "wb") as f:
                    f.write(audio_bytes)

            return audio_bytes

        except Exception as e:
            print(f"[ElevenLabs ERROR] {e}")
            raise