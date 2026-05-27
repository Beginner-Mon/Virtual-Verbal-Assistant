"""Shared exceptions for LangGraph service clients."""


class ServiceUnavailableError(Exception):
    """Raised when a downstream service (Kimodo/VieNeu-TTS) is unreachable
    or its circuit breaker is open."""

    def __init__(self, service_name: str, reason: str = ""):
        self.service_name = service_name
        self.reason = reason
        super().__init__(f"{service_name} unavailable: {reason}")
