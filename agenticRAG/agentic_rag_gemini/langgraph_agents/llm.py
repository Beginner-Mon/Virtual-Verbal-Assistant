"""Thin factory for LangChain ChatModels.

Centralizes model selection per node role. Replaces llm_gateway.py.
All nodes import from here instead of init-ing ChatModels ad-hoc.

Provider: **DeepSeek** (OpenAI-compatible API).

Env:
  DEEPSEEK_API_KEY   — required for live LLM calls
  DEEPSEEK_BASE_URL  — optional override (default https://api.deepseek.com)
  DEEPSEEK_MODEL     — optional override (default deepseek-v4-pro)

A .env file at agenticRAG/agentic_rag_gemini/.env is auto-loaded if python-dotenv
is installed.
"""

import os
from functools import lru_cache
from pathlib import Path

from langchain_openai import ChatOpenAI


def _load_dotenv_once() -> None:
    """Best-effort load of agenticRAG/agentic_rag_gemini/.env. No-op if dotenv missing."""
    try:
        from dotenv import load_dotenv
    except ImportError:
        return
    env_path = Path(__file__).resolve().parents[1] / ".env"
    if env_path.exists():
        load_dotenv(env_path, override=False)


_load_dotenv_once()


_DEFAULT_BASE_URL = "https://api.deepseek.com"
_DEFAULT_MODEL = "deepseek-v4-pro"

# Temperature per role kept as-is; DeepSeek honors the OpenAI temperature param.
_DEFAULT_TEMPS = {
    "planner": 0.0,
    "synthesizer": 0.7,
    "conversation": 0.7,
    "retriever": 0.0,
}


def _resolve_api_key() -> str | None:
    return os.getenv("DEEPSEEK_API_KEY")


@lru_cache(maxsize=8)
def get_chat_model(role: str, *, temperature: float | None = None):
    """Return a LangChain ChatModel for the given node role.

    Cached per (role, temperature). Build via ChatOpenAI pointing at the DeepSeek
    OpenAI-compatible endpoint.
    """
    model_name = os.getenv("DEEPSEEK_MODEL", _DEFAULT_MODEL)
    temp = _DEFAULT_TEMPS.get(role, 0.7) if temperature is None else temperature
    api_key = _resolve_api_key()
    base_url = os.getenv("DEEPSEEK_BASE_URL", _DEFAULT_BASE_URL)
    return ChatOpenAI(
        model=model_name,
        temperature=temp,
        api_key=api_key,
        base_url=base_url,
    )
