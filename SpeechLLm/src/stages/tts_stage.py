import pyttsx3


class TTSStage:

    def __init__(self, rate: int = 180, volume: float = 1.0):
        self.engine = pyttsx3.init()
        self.engine.setProperty("rate", rate)
        self.engine.setProperty("volume", volume)

    def process(self, text: str) -> None:
        self.engine.say(text)
        self.engine.runAndWait()