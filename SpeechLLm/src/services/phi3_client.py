from ollama import chat
from typing import List, Dict, Generator


class Phi3Client:
    """
    Wrapper for Phi-3 3.8B model via Ollama.
    """

    def __init__(self, config: dict):
        """
        Expected config format:
        {
            "model_name": "phi3:3.8b",
            "temperature": 0.7
        }
        """

        self.model_name = config.get("model_name", "phi3:3.8b")
        self.temperature = config.get("temperature", 0.7)

        if not isinstance(self.model_name, str):
            raise ValueError("model_name must be a string")

    # --------------------------------------------------
    # Standard (non-streaming) generation
    # --------------------------------------------------
    def generate(self, messages: List[Dict[str, str]]) -> str:

        response = chat(
            model=self.model_name,
            messages=messages,
            options={
                "temperature": self.temperature,
            }
        )

        # Ollama returns dict-style response
        return response["message"]["content"]

    # --------------------------------------------------
    # Streaming generation
    # --------------------------------------------------
    def stream_generate(
        self,
        messages: List[Dict[str, str]],
    ) -> Generator[str, None, None]:

        stream = chat(
            model=self.model_name,
            messages=messages,
            stream=True,
            options={
                "temperature": self.temperature,
            }
        )

        for chunk in stream:
            if "message" in chunk:
                yield chunk["message"]["content"]