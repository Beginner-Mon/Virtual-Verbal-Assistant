class TTSRouter:
    """Routes TTS requests to VieNeu (primary, with extensible fallback)."""

    def __init__(self, vieneu_client=None):
        self.vieneu = vieneu_client
        self.last_provider = None

    def synthesize(self, text: str, language: str = "vi") -> str:
        if self.vieneu:
            audio_path = self.vieneu.synthesize(text, language=language)
            self.last_provider = "vieneu"
            return audio_path

        raise RuntimeError("No TTS provider available (VieNeu not configured).")
