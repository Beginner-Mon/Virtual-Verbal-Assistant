import threading


class InterruptController:
    def __init__(self):
        self._interrupted = False
        self._lock = threading.Lock()

    def trigger(self):
        with self._lock:
            self._interrupted = True

    def reset(self):
        with self._lock:
            self._interrupted = False

    def is_interrupted(self) -> bool:
        with self._lock:
            return self._interrupted