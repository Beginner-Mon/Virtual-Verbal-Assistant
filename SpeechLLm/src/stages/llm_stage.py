import json
import re
from typing import Optional, Dict, Any
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

    def _safe_json_parse(self, raw_text: str) -> Optional[Dict[str, Any]]:
        """
        Safely extract JSON from model output.
        Handles markdown ```json blocks without altering logic.
        """

        if not raw_text:
            return None

        cleaned = raw_text.strip()

        # Extract JSON inside ```json ... ```
        match = re.search(r"```json\s*(\{.*?\})\s*```", cleaned, re.DOTALL)

        if match:
            cleaned = match.group(1)
        else:
            # Remove generic triple backticks if present
            cleaned = cleaned.replace("```", "").strip()

        try:
            return json.loads(cleaned)
        except Exception:
            return None

    def process(
        self,
        user_text: str,
        emotion: Optional[str] = None
    ) -> Dict[str, Any]:

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

        raw_response = self.client.generate(messages)

        # -------------------------
        # Structured parsing (improved, logic unchanged)
        # -------------------------
        structured_response = self._safe_json_parse(raw_response)

        if not structured_response or "intent" not in structured_response:
            structured_response = {
                "intent": "speech",
                "text": raw_response
            }

        # Save raw assistant text to memory for conversational continuity
        self.memory.add_message("user", user_text)
        self.memory.add_message("assistant", raw_response)

        return structured_response