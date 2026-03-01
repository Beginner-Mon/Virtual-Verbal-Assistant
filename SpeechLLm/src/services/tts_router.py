class TTSRouter:
    def __init__(self, eleven_client=None, coqui_client=None):
        self.eleven = eleven_client
        self.coqui = coqui_client

    # -----------------------------
    # ADD THIS METHOD
    # -----------------------------
    def get_available_speakers(self):
        if self.coqui and hasattr(self.coqui, "get_available_speakers"):
            return self.coqui.get_available_speakers()
        return []

    # -----------------------------
    # Existing synthesize method
    # -----------------------------
    def synthesize(self, text: str, output_path: str = None, prefer="eleven"):

        if prefer == "eleven" and self.eleven:
            try:
                print("[TTS] Using ElevenLabs...")
                return self.eleven.synthesize(text, output_path)
            except Exception:
                print("[TTS] ElevenLabs failed, switching to fallback...")

        if self.coqui:
            try:
                print("[TTS] Using Coqui TTS...")
                return self.coqui.synthesize(text)
            except Exception:
                print("[TTS] Coqui also failed.")

        raise Exception("All TTS providers failed.")