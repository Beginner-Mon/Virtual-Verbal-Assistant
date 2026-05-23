"""Async HTTP client for VieNeu-TTS speech synthesis service."""

from __future__ import annotations

from pathlib import Path
from typing import Optional

import httpx

from core.circuit_breaker import CircuitBreaker
from langgraph_agents.services.exceptions import ServiceUnavailableError


def _load_vieneu_config() -> dict:
    import yaml
    config_path = Path(__file__).resolve().parents[5] / "config" / "langgraph.yaml"
    if config_path.exists():
        with open(config_path, "r", encoding="utf-8") as f:
            return yaml.safe_load(f).get("langgraph", {}).get("services", {}).get("vieneu_tts", {})
    return {}


class VieNeuTTSClient:
    """Async HTTP client for VieNeu-TTS speech synthesis REST API."""

    def __init__(
        self,
        base_url: str = "http://localhost:5000",
        endpoint: str = "/synthesize",
        timeout: float = 15,
        circuit_breaker_cfg: dict = None,
    ):
        self.base_url = base_url.rstrip("/")
        self.endpoint = endpoint
        self.timeout = timeout
        breaker_cfg = circuit_breaker_cfg or {}
        self._breaker = CircuitBreaker(
            name="vieneu_tts",
            failure_threshold=breaker_cfg.get("failure_threshold", 3),
            cool_down_seconds=breaker_cfg.get("cool_down_seconds", 60),
        )

    def is_healthy(self) -> bool:
        return self._breaker.state != "open"

    @property
    def circuit_state(self) -> str:
        return self._breaker.state

    async def synthesize(self, text: str, voice_path: str = None) -> dict:
        """POST /synthesize → return audio result dict."""
        if not self._breaker.allow():
            raise ServiceUnavailableError("vieneu_tts", "circuit breaker open")

        url = f"{self.base_url}{self.endpoint}"
        payload = {"text": text}
        if voice_path:
            payload["voice_path"] = voice_path

        try:
            async with httpx.AsyncClient(timeout=self.timeout) as client:
                resp = await client.post(url, json=payload)
                resp.raise_for_status()
                data = resp.json()
                self._breaker.record_success()
                return data
        except (httpx.TimeoutException, httpx.ConnectError, httpx.HTTPStatusError) as exc:
            self._breaker.record_failure()
            raise ServiceUnavailableError("vieneu_tts", str(exc)) from exc
        except Exception as exc:
            self._breaker.record_failure()
            raise ServiceUnavailableError("vieneu_tts", str(exc)) from exc

    def synthesize_sync(self, text: str, voice_path: str = None) -> dict:
        """Synchronous wrapper for Celery tasks."""
        if not self._breaker.allow():
            raise ServiceUnavailableError("vieneu_tts", "circuit breaker open")

        url = f"{self.base_url}{self.endpoint}"
        payload = {"text": text}
        if voice_path:
            payload["voice_path"] = voice_path

        try:
            resp = httpx.post(url, json=payload, timeout=self.timeout)
            resp.raise_for_status()
            data = resp.json()
            self._breaker.record_success()
            return data
        except (httpx.TimeoutException, httpx.ConnectError, httpx.HTTPStatusError) as exc:
            self._breaker.record_failure()
            raise ServiceUnavailableError("vieneu_tts", str(exc)) from exc
        except Exception as exc:
            self._breaker.record_failure()
            raise ServiceUnavailableError("vieneu_tts", str(exc)) from exc


# Module-level singleton
_client: Optional[VieNeuTTSClient] = None


def get_vieneu_tts_client() -> VieNeuTTSClient:
    global _client
    if _client is None:
        cfg = _load_vieneu_config()
        _client = VieNeuTTSClient(
            base_url=cfg.get("url", "http://localhost:5000"),
            endpoint=cfg.get("endpoint", "/synthesize"),
            timeout=cfg.get("timeout", 15),
            circuit_breaker_cfg=cfg.get("circuit_breaker", {}),
        )
    return _client
