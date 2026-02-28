from typing import List, Dict


class MemoryManager:
    """
    Manages short-term conversation memory.
    Stores chat history in OpenAI/Ollama message format.
    """

    def __init__(self, max_history: int = 10):
        self.max_history = max_history
        self.conversation: List[Dict[str, str]] = []

    def get_conversation(self) -> List[Dict[str, str]]:
        """
        Returns conversation history.
        """
        return self.conversation

    def add_message(self, role: str, content: str) -> None:
        """
        Adds a message to memory.
        Keeps only the latest N messages.
        """
        self.conversation.append({
            "role": role,
            "content": content
        })

        # Trim history if too long
        if len(self.conversation) > self.max_history:
            self.conversation = self.conversation[-self.max_history:]
