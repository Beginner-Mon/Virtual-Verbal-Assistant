# from ollama import Client
# from typing import List, Dict, Generator


# class Phi3Client:
#     """
#     Wrapper for Phi-3 3.8B model via Ollama.
#     """

#     def __init__(self, config: dict):

#         self.model_name = config.get("model_name", "phi3:3.8b")
#         self.temperature = config.get("temperature", 0.7)

#         if not isinstance(self.model_name, str):
#             raise ValueError("model_name must be a string")

#         # 🔥 Explicit client with correct host
#         self.client = Client(host="http://127.0.0.1:11434")

#     # --------------------------------------------------
#     # Standard (non-streaming) generation
#     # --------------------------------------------------
#     def generate(self, messages: List[Dict[str, str]]) -> str:

#         response = self.client.chat(
#             model=self.model_name,
#             messages=messages,
#             options={
#                 "temperature": self.temperature,
#             }
#         )

#         return response["message"]["content"]

#     # --------------------------------------------------
#     # Streaming generation
#     # --------------------------------------------------
#     def stream_generate(
#         self,
#         messages: List[Dict[str, str]],
#     ) -> Generator[str, None, None]:

#         stream = self.client.chat(
#             model=self.model_name,
#             messages=messages,
#             stream=True,
#             options={
#                 "temperature": self.temperature,
#             }
#         )

#         for chunk in stream:
#             if "message" in chunk:
#                 yield chunk["message"]["content"]