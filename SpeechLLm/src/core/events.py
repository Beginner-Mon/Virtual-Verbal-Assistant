import queue


class EventBus:
    def __init__(self):
        self._queue = queue.Queue()

    def emit(self, event):
        self._queue.put(event)

    def get(self):
        try:
            return self._queue.get_nowait()
        except queue.Empty:
            return None