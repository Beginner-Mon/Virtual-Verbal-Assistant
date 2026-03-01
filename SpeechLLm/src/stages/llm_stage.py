from typing import Optional
from src.services.phi3_client import Phi3Client
from src.context.memory_manager import MemoryManager
from src.context.prompt_templates import get_system_prompt
from src.stages.emotion_stage import EmotionStage


class LLMStage:

    def __init__(
        self,
        phi3_client: Phi3Client,
        memory_manager: MemoryManager,
    ):
        self.client = phi3_client
        self.memory = memory_manager
        self.emotion_stage = EmotionStage()

    def process(
        self,
        user_text: str,
        emotion: Optional[str] = None
    ) -> str:

        detected_emotion = emotion
        if detected_emotion is None:
            emotion_data = self.emotion_stage.detect(user_text)
            detected_emotion = emotion_data["emotion"]

        system_prompt = get_system_prompt()
        history = self.memory.get_conversation()

        user_payload = user_text
        if detected_emotion and detected_emotion != "neutral":
            user_payload = f"[User emotion: {detected_emotion}] {user_text}"

        messages = []
        messages.append({"role": "system", "content": system_prompt})
        messages.extend(history)
        messages.append({"role": "user", "content": user_payload})

        response = self.client.generate(messages)

        self.memory.add_message("user", user_text)
        self.memory.add_message("assistant", response)

        return response 