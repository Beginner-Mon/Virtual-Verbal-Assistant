import queue
import threading
from typing import Optional


class AudioStreamBuffer:
    def __init__(self):
        self.buffer = queue.Queue()
        self.active = False
        self.lock = threading.Lock()

    def start(self):
        with self.lock:
            self.active = True

    def stop(self):
        with self.lock:
            self.active = False

        self.clear()

    def add_chunk(self, audio_chunk):
        if self.active:
            self.buffer.put(audio_chunk)

    def get_chunk(self, timeout: Optional[float] = 0.1):
        if not self.active:
            return None

        try:
            return self.buffer.get(timeout=timeout)
        except queue.Empty:
            return None

    def clear(self):
        while not self.buffer.empty():
            self.buffer.get_nowait()