from typing import Generator


class TokenStreamer:
    def __init__(self, llm_client):
        self.client = llm_client

    def stream(self, messages) -> Generator[str, None, None]:
        buffer = ""

        for token in self.client.stream(messages):
            if not token:
                continue

            buffer += token

            if " " in buffer:
                parts = buffer.split(" ")
                for word in parts[:-1]:
                    yield word + " "
                buffer = parts[-1]

        if buffer:
            yield buffer