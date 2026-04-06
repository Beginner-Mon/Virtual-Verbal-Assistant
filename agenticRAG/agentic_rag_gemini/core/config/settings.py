import os
from dataclasses import dataclass

from utils.logger import get_logger

logger = get_logger(__name__)


def _env_int(name: str, default: int) -> int:
    value = os.getenv(name)
    if value is None:
        return default
    try:
        return int(value)
    except ValueError:
        logger.warning(f"Invalid integer for {name}={value!r}; using default {default}")
        return default


def _env_float(name: str, default: float) -> float:
    value = os.getenv(name)
    if value is None:
        return default
    try:
        return float(value)
    except ValueError:
        logger.warning(f"Invalid float for {name}={value!r}; using default {default}")
        return default


def _env_bool(name: str, default: bool) -> bool:
    value = os.getenv(name)
    if value is None:
        return default
    return str(value).strip().lower() in {"1", "true", "yes", "on"}


@dataclass(frozen=True)
class MainAPISettings:
    agentic_rag_url: str
    dart_url: str
    tts_url: str
    main_api_host: str
    main_api_port: int
    async_enrichment: bool
    include_debug: bool
    motion_default_duration_seconds: float
    downstream_timeout: float
    downstream_session_timeout: float


_SETTINGS: MainAPISettings | None = None


def get_main_api_settings() -> MainAPISettings:
    global _SETTINGS
    if _SETTINGS is not None:
        return _SETTINGS

    agentic_rag_host = os.getenv("AGENTIC_RAG_HOST", "localhost")
    agentic_rag_port = _env_int("AGENTIC_RAG_PORT", 8000)
    dart_host = os.getenv("DART_HOST", "localhost")
    dart_port = _env_int("DART_PORT", 5001)
    tts_host = os.getenv("TTS_HOST", "localhost")
    tts_port = _env_int("TTS_PORT", 5000)

    _SETTINGS = MainAPISettings(
        agentic_rag_url=os.getenv("AGENTIC_RAG_URL", f"http://{agentic_rag_host}:{agentic_rag_port}"),
        dart_url=os.getenv("DART_URL", f"http://{dart_host}:{dart_port}"),
        tts_url=os.getenv("TTS_URL", f"http://{tts_host}:{tts_port}"),
        main_api_host=os.getenv("MAIN_API_HOST", "0.0.0.0"),
        main_api_port=_env_int("MAIN_API_PORT", 8080),
        async_enrichment=_env_bool("MAIN_API_ASYNC_ENRICHMENT", True),
        include_debug=_env_bool("MAIN_API_INCLUDE_DEBUG", True),
        motion_default_duration_seconds=_env_float("MOTION_DEFAULT_DURATION_SECONDS", 5.33),
        downstream_timeout=_env_float("DOWNSTREAM_TIMEOUT", 90.0),
        downstream_session_timeout=_env_float("DOWNSTREAM_SESSION_TIMEOUT", 15.0),
    )
    return _SETTINGS
