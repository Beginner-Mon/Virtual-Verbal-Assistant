from enum import Enum, auto
import threading


class AssistantState(Enum):
    IDLE = auto()
    LISTENING = auto()
    THINKING = auto()
    SPEAKING = auto()


class StateMachine:
    def __init__(self):
        self._state = AssistantState.IDLE
        self._lock = threading.Lock()

    def get_state(self) -> AssistantState:
        with self._lock:
            return self._state

    def set_state(self, new_state: AssistantState):
        with self._lock:
            self._state = new_state

    def is_idle(self) -> bool:
        return self.get_state() == AssistantState.IDLE

    def is_listening(self) -> bool:
        return self.get_state() == AssistantState.LISTENING

    def is_thinking(self) -> bool:
        return self.get_state() == AssistantState.THINKING

    def is_speaking(self) -> bool:
        return self.get_state() == AssistantState.SPEAKING