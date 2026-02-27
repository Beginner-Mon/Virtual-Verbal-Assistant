class TTSStage:
    def __init__(self, tts_router, speaker: str = None, prefer="eleven"):
        self.tts_router = tts_router
        self.speaker = speaker
        self.prefer = prefer

        # Optional: show available speakers if using Coqui
        speakers = self.tts_router.get_available_speakers()
        if speakers:
            print("Available speakers:")
            for i, spk in enumerate(speakers):
                print(f"{i}: {spk}")

    def process(self, text: str):
        return self.tts_router.synthesize(
            text=text,
            prefer=self.prefer
        )