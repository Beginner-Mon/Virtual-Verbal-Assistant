"""Async HTTP client for Kimodo motion generation service."""

from __future__ import annotations

from pathlib import Path
from typing import Optional

import httpx

from core.circuit_breaker import CircuitBreaker
from langgraph_agents.services.exceptions import ServiceUnavailableError


def _load_kimodo_config() -> dict:
    import yaml
    config_path = Path(__file__).resolve().parents[5] / "config" / "langgraph.yaml"
    if config_path.exists():
        with open(config_path, "r", encoding="utf-8") as f:
            return yaml.safe_load(f).get("langgraph", {}).get("services", {}).get("kimodo", {})
    return {}


class KimodoClient:
    """Async HTTP client for Kimodo motion generation REST API."""

    def __init__(
        self,
        base_url: str = "http://localhost:5001",
        endpoint: str = "/generate",
        timeout: float = 30,
        circuit_breaker_cfg: dict = None,
    ):
        self.base_url = base_url.rstrip("/")
        self.endpoint = endpoint
        self.timeout = timeout
        breaker_cfg = circuit_breaker_cfg or {}
        self._breaker = CircuitBreaker(
            name="kimodo",
            failure_threshold=breaker_cfg.get("failure_threshold", 3),
            cool_down_seconds=breaker_cfg.get("cool_down_seconds", 60),
        )

    def is_healthy(self) -> bool:
        return self._breaker.state != "open"

    @property
    def circuit_state(self) -> str:
        return self._breaker.state

    async def generate(
        self,
        prompt: str,
        constraints: Optional[list] = None,
        duration_seconds: float = 3.0,
    ) -> dict:
        """POST /generate → return motion result dict."""
        if not self._breaker.allow():
            raise ServiceUnavailableError("kimodo", "circuit breaker open")

        url = f"{self.base_url}{self.endpoint}"
        payload = {
            "prompt": prompt,
            "constraints": constraints or [],
            "duration_seconds": duration_seconds,
        }

        try:
            async with httpx.AsyncClient(timeout=self.timeout) as client:
                resp = await client.post(url, json=payload)
                resp.raise_for_status()
                data = resp.json()
                self._breaker.record_success()
                return data
        except (httpx.TimeoutException, httpx.ConnectError, httpx.HTTPStatusError) as exc:
            self._breaker.record_failure()
            raise ServiceUnavailableError("kimodo", str(exc)) from exc
        except Exception as exc:
            self._breaker.record_failure()
            raise ServiceUnavailableError("kimodo", str(exc)) from exc

    def generate_sync(
        self,
        prompt: str,
        constraints: Optional[list] = None,
        duration_seconds: float = 3.0,
    ) -> dict:
        """Synchronous wrapper for Celery tasks."""
        if not self._breaker.allow():
            raise ServiceUnavailableError("kimodo", "circuit breaker open")

        url = f"{self.base_url}{self.endpoint}"
        payload = {
            "prompt": prompt,
            "constraints": constraints or [],
            "duration_seconds": duration_seconds,
        }

        try:
            resp = httpx.post(url, json=payload, timeout=self.timeout)
            resp.raise_for_status()
            data = resp.json()
            self._breaker.record_success()
            return data
        except (httpx.TimeoutException, httpx.ConnectError, httpx.HTTPStatusError) as exc:
            self._breaker.record_failure()
            raise ServiceUnavailableError("kimodo", str(exc)) from exc
        except Exception as exc:
            self._breaker.record_failure()
            raise ServiceUnavailableError("kimodo", str(exc)) from exc


# Module-level singleton
_client: Optional[KimodoClient] = None


def get_kimodo_client() -> KimodoClient:
    global _client
    if _client is None:
        cfg = _load_kimodo_config()
        _client = KimodoClient(
            base_url=cfg.get("url", "http://localhost:5001"),
            endpoint=cfg.get("endpoint", "/generate"),
            timeout=cfg.get("timeout", 30),
            circuit_breaker_cfg=cfg.get("circuit_breaker", {}),
        )
    return _client
