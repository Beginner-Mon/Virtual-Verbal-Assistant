"""LangChain LLM provider with multi-key rotation support.

Replaces the custom GeminiClient / GeminiClientWrapper with LangChain's
ChatGoogleGenerativeAI, while preserving API key rotation via APIKeyManager.

Usage:
    from utils.llm_provider import get_llm, get_llm_with_fallback

    llm = get_llm()                         # default model
    llm = get_llm(model="gemini-2.5-flash") # specific model
    response = llm.invoke("Hello")          # LangChain interface
"""

import logging
from typing import Optional

from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.language_models import BaseChatModel

from utils.api_key_manager import get_api_key_manager, APIKeyManager
from utils.logger import get_logger

logger = get_logger(__name__)

# ---------------------------------------------------------------------------
# Singleton state
# ---------------------------------------------------------------------------

_key_manager: Optional[APIKeyManager] = None
_current_llm: Optional[ChatGoogleGenerativeAI] = None
_current_model: Optional[str] = None


def _get_key_manager() -> APIKeyManager:
    """Get or create the global API key manager singleton."""
    global _key_manager
    if _key_manager is None:
        _key_manager = get_api_key_manager()
    return _key_manager


# ---------------------------------------------------------------------------
# Factory
# ---------------------------------------------------------------------------


def get_llm(
    model: str = "gemini-2.5-flash",
    temperature: float = 0.7,
    max_tokens: int = 2048,
    streaming: bool = False,
) -> ChatGoogleGenerativeAI:
    """Create a ChatGoogleGenerativeAI instance using the current API key.

    Re-uses the same instance if the model + api_key haven't changed.
    Temperature and max_tokens are set at creation but can be overridden
    per-call by the shim layer (via invoke kwargs).

    Args:
        model:       Gemini model name (e.g. "gemini-2.5-flash").
        temperature: Sampling temperature (0.0–2.0).
        max_tokens:  Maximum output tokens.
        streaming:   Whether to enable streaming mode.

    Returns:
        A ready-to-use ChatGoogleGenerativeAI instance.
    """
    global _current_llm, _current_model

    km = _get_key_manager()
    api_key = km.get_current_key()

    if not api_key:
        raise ValueError(
            "No Gemini API key found. "
            "Set GEMINI_API_KEYS or GEMINI_API_KEY in environment."
        )

    # Reuse if model + key haven't changed (temp/max_tokens don't force recreation)
    if (
        _current_llm is not None
        and _current_model == model
    ):
        # Compare actual key values (google_api_key is SecretStr)
        current_key_val = getattr(_current_llm.google_api_key, 'get_secret_value', lambda: _current_llm.google_api_key)()
        if current_key_val == api_key:
            return _current_llm

    llm = ChatGoogleGenerativeAI(
        model=model,
        google_api_key=api_key,
        temperature=temperature,
        max_output_tokens=max_tokens,
        streaming=streaming,
        convert_system_message_to_human=True,
    )

    _current_llm = llm
    _current_model = model

    key_idx = km.get_current_key_index() + 1
    total = km.get_total_keys()
    logger.info(
        "LLM provider: initialized ChatGoogleGenerativeAI "
        "(model=%s, key=%d/%d)",
        model, key_idx, total,
    )
    return llm


def rotate_and_get_llm(
    model: str = "gemini-2.5-flash",
    temperature: float = 0.7,
    max_tokens: int = 2048,
    streaming: bool = False,
) -> Optional[ChatGoogleGenerativeAI]:
    """Rotate to the next API key and return a fresh LLM instance.

    Called by higher-level code when a 429 / quota error is encountered.
    Returns None if all keys are exhausted.

    Args:
        model:       Gemini model name.
        temperature: Sampling temperature.
        max_tokens:  Maximum output tokens.
        streaming:   Whether to enable streaming.

    Returns:
        A new ChatGoogleGenerativeAI with the rotated key, or None.
    """
    global _current_llm, _current_model

    km = _get_key_manager()

    if not km.rotate_to_next_key():
        logger.error("LLM provider: all API keys exhausted")
        km.set_last_error("All API keys exhausted after rotation attempts")
        return None

    # Force recreation
    _current_llm = None
    _current_model = None

    logger.info(
        "LLM provider: rotated to key %d/%d",
        km.get_current_key_index() + 1,
        km.get_total_keys(),
    )

    return get_llm(
        model=model,
        temperature=temperature,
        max_tokens=max_tokens,
        streaming=streaming,
    )


def invoke_with_retry(
    llm: ChatGoogleGenerativeAI,
    messages,
    *,
    model: str = "gemini-2.5-flash",
    temperature: float = 0.7,
    max_tokens: int = 2048,
    max_retries: int = 0,
):
    """Invoke the LLM with automatic key rotation on quota errors.

    Args:
        llm:         The ChatGoogleGenerativeAI instance to start with.
        messages:    LangChain messages list or prompt value.
        model:       Model name for recreating on rotation.
        temperature: Temperature for recreating on rotation.
        max_tokens:  Max tokens for recreating on rotation.
        max_retries: Max rotation attempts (0 = use all available keys).

    Returns:
        The LLM response (AIMessage).

    Raises:
        RuntimeError: If all keys are exhausted.
    """
    km = _get_key_manager()
    attempts = max_retries or (km.get_total_keys() * km._MAX_CYCLES)

    last_error = None
    current_llm = llm

    for attempt in range(attempts):
        try:
            result = current_llm.invoke(messages)
            km.reset_success()
            return result
        except Exception as e:
            error_str = str(e)
            is_quota = any(
                kw in error_str.lower()
                for kw in ("429", "quota", "exceeded", "rate limit", "resource_exhausted")
            )

            if not is_quota:
                raise

            last_error = e
            logger.warning(
                "LLM quota error (attempt %d/%d): %s",
                attempt + 1, attempts, error_str[:120],
            )

            rotated = rotate_and_get_llm(
                model=model,
                temperature=temperature,
                max_tokens=max_tokens,
            )
            if rotated is None:
                break
            current_llm = rotated

    raise RuntimeError(
        f"All API keys exhausted after {attempts} attempts. "
        f"Last error: {last_error}"
    )


def get_key_status() -> dict:
    """Get current API key manager status.

    Returns:
        Dict with key rotation status information.
    """
    return _get_key_manager().get_status()
"""LangChain LLM provider — backwards-compatible shim.

To ease the migration, this module also re-exports a ``GeminiClientWrapper``
class that wraps LangChain's ChatGoogleGenerativeAI with the old
``client.chat.completions.create(...)`` interface so existing code can be
migrated file-by-file.
"""


# ---------------------------------------------------------------------------
# Backwards-compatible GeminiClientWrapper shim
# ---------------------------------------------------------------------------


class _ShimMessage:
    """Mimics OpenAI-style message object."""
    __slots__ = ("content",)

    def __init__(self, text: str):
        self.content = text


class _ShimChoice:
    """Mimics OpenAI-style choice object."""
    __slots__ = ("message",)

    def __init__(self, text: str):
        self.message = _ShimMessage(text)


class _ShimResponse:
    """Mimics OpenAI-style response object."""
    __slots__ = ("choices",)

    def __init__(self, text: str):
        self.choices = [_ShimChoice(text)]


class _ShimCompletions:
    """Mimics ``client.chat.completions.create(...)``."""

    def __init__(self, get_llm_fn):
        self._get_llm = get_llm_fn

    def create(
        self,
        model: str = "gemini-2.5-flash",
        messages=None,
        temperature: float = 0.7,
        max_tokens: int = 1000,
        response_format=None,
        stream: bool = False,
        **kwargs,
    ):
        from langchain_core.messages import HumanMessage, AIMessage, SystemMessage

        lc_messages = []
        for msg in (messages or []):
            role = msg.get("role", "user")
            content = msg.get("content", "")
            if role == "system":
                lc_messages.append(SystemMessage(content=content))
            elif role == "assistant":
                lc_messages.append(AIMessage(content=content))
            else:
                lc_messages.append(HumanMessage(content=content))

        # Reuse the cached LLM — only model + api_key trigger recreation.
        # Temperature and max_tokens are applied per-call.
        llm = self._get_llm(model=model)

        # Apply per-call overrides without recreating the instance
        llm.temperature = temperature
        llm.max_output_tokens = max_tokens

        if stream:
            def _stream_gen():
                for chunk in llm.stream(lc_messages):
                    text = chunk.content if hasattr(chunk, "content") else str(chunk)
                    yield text
            return _stream_gen()

        result = invoke_with_retry(
            llm, lc_messages,
            model=model,
            temperature=temperature,
            max_tokens=max_tokens,
        )
        text = result.content if hasattr(result, "content") else str(result)
        return _ShimResponse(text)


class _ShimChat:
    """Mimics ``client.chat``."""

    def __init__(self, get_llm_fn):
        self.completions = _ShimCompletions(get_llm_fn)


class GeminiClientWrapper:
    """Drop-in replacement for the old GeminiClientWrapper.

    Provides the same ``client.chat.completions.create(...)`` interface but
    delegates to LangChain's ChatGoogleGenerativeAI under the hood.

    This shim exists so that files can be migrated incrementally.  New code
    should use ``get_llm()`` directly.
    """

    def __init__(self, api_key: str = None):
        self.chat = _ShimChat(get_llm)

    def get_key_status(self) -> dict:
        return get_key_status()
