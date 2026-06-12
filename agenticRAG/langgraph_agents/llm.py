"""Thin factory for LangChain ChatModels + circuit breaker wrapping.

Centralizes model selection per node role. Replaces llm_gateway.py.
All nodes import from here instead of init-ing ChatModels ad-hoc.

Provider: **DeepSeek** (OpenAI-compatible API).

Env:
  DEEPSEEK_API_KEY        — required for live LLM calls
  DEEPSEEK_BASE_URL       — optional override (default https://api.deepseek.com)
  DEEPSEEK_MODEL          — heavy model, used by synthesizer (default deepseek-v4-pro)
  DEEPSEEK_MODEL_FAST     — fast model, used by planner/retriever/conversation
                            (default deepseek-v4-flash)

Per-role model strategy (L4 optimization):
  - synthesizer needs reasoning quality → heavy model
  - planner does intent classification + JSON output → fast model is enough
  - retriever_agent picks tools per plan → fast model is enough
  - conversation styles greeting/clarify → fast model is enough
  - health_check / others → fast model

This cuts 2-3 LLM call latencies (planner, retriever rounds, greeting) by 50-70%
while preserving clinical content quality.

A .env file at agenticRAG/agentic_rag_gemini/.env is auto-loaded if python-dotenv
is installed.
"""

import asyncio
import logging
import os
from functools import lru_cache
from pathlib import Path
from typing import Any

from langchain_openai import ChatOpenAI
from langgraph_agents.core.circuit_breaker import CircuitBreaker

logger = logging.getLogger("langgraph.llm")


def _load_dotenv_once() -> None:
    """Best-effort load of .env in new parent, legacy nested, or repo-root paths."""
    try:
        from dotenv import load_dotenv
    except ImportError:
        return
    
    # Try current parent directory (agenticRAG/.env)
    env_path = Path(__file__).resolve().parents[1] / ".env"
    if env_path.exists():
        load_dotenv(env_path, override=False)
        return
        
    # Try legacy nested directory (agenticRAG/agentic_rag_gemini/.env)
    legacy_path = Path(__file__).resolve().parents[1] / "agentic_rag_gemini" / ".env"
    if legacy_path.exists():
        load_dotenv(legacy_path, override=False)
        return

    # Try repository root directory (Virtual-Verbal-Assistant/.env)
    root_path = Path(__file__).resolve().parents[2] / ".env"
    if root_path.exists():
        load_dotenv(root_path, override=False)


_load_dotenv_once()


_DEFAULT_BASE_URL = "https://api.deepseek.com"
_DEFAULT_HEAVY_MODEL = "deepseek-v4-pro"
_DEFAULT_FAST_MODEL = "deepseek-v4-flash"

# Temperature per role kept as-is; DeepSeek honors the OpenAI temperature param.
_DEFAULT_TEMPS = {
    "planner": 0.0,
    "synthesizer": 0.7,
    "conversation": 0.7,
    "retriever": 0.0,
    "health_check": 0.0,
}

# Roles that use the HEAVY model (reasoning-intensive). All others use FAST.
_HEAVY_ROLES = {"synthesizer"}


def _model_for_role(role: str) -> str:
    """Resolve model name per role. Heavy roles use DEEPSEEK_MODEL, others use
    DEEPSEEK_MODEL_FAST. Both can be env-overridden."""
    if role in _HEAVY_ROLES:
        return os.getenv("DEEPSEEK_MODEL", _DEFAULT_HEAVY_MODEL)
    return os.getenv("DEEPSEEK_MODEL_FAST", _DEFAULT_FAST_MODEL)


def _resolve_api_key() -> str | None:
    return os.getenv("DEEPSEEK_API_KEY")


# ── Circuit breaker wrappers (one per role) ──────────────────────────

# Each role gets its own breaker so a flaky planner doesn't trip conversation.
_cb_map: dict[str, CircuitBreaker] = {}


def _get_breaker(role: str) -> CircuitBreaker:
    if role not in _cb_map:
        _cb_map[role] = CircuitBreaker(
            name=f"llm:{role}",
            failure_threshold=3,
            cool_down_seconds=30.0,
        )
    return _cb_map[role]


class CircuitBreakerOpenError(RuntimeError):
    """Raised when a call is rejected because the circuit breaker is open."""


def _apply_circuit_breaker(model: ChatOpenAI, role: str) -> ChatOpenAI:
    """Patch ainvoke/astream on a ChatOpenAI instance with circuit breaker guards.

    Monkey-patches the instance so that .with_structured_output() and .bind_tools()
    (which return new Runnables that internally call model.ainvoke/astream)
    still go through the breaker.

    Returns the same instance (mutated in place).
    """
    breaker = _get_breaker(role)
    _raw_ainvoke = model.ainvoke
    _raw_astream = model.astream

    async def guarded_ainvoke(*args, **kwargs):
        if not breaker.allow():
            raise CircuitBreakerOpenError(
                f"LLM circuit breaker open for '{breaker.name}' — "
                f"too many consecutive failures, cooling down for {breaker.cool_down_seconds}s"
            )
        try:
            result = await _raw_ainvoke(*args, **kwargs)
            breaker.record_success()
            return result
        except Exception:
            breaker.record_failure()
            raise

    async def guarded_astream(*args, **kwargs):
        if not breaker.allow():
            raise CircuitBreakerOpenError(
                f"LLM circuit breaker open for '{breaker.name}' — "
                f"too many consecutive failures, cooling down for {breaker.cool_down_seconds}s"
            )
        try:
            async for chunk in _raw_astream(*args, **kwargs):
                yield chunk
            breaker.record_success()
        except Exception:
            breaker.record_failure()
            raise

    # Patch instance methods via object.__setattr__ — bypasses Pydantic v2
    # field validation on ChatOpenAI which would reject direct assignment.
    object.__setattr__(model, "ainvoke", guarded_ainvoke)
    object.__setattr__(model, "astream", guarded_astream)
    object.__setattr__(model, "_raw_ainvoke", _raw_ainvoke)
    object.__setattr__(model, "_breaker", breaker)

    return model


@lru_cache(maxsize=8)
def get_chat_model(role: str, *, temperature: float | None = None):
    """Return a LangChain ChatModel for the given node role, guarded by circuit breaker.

    Cached per (role, temperature). Build via ChatOpenAI pointing at the DeepSeek
    OpenAI-compatible endpoint. Patches ainvoke/astream with per-role circuit breaker.
    """
    model_name = _model_for_role(role)
    temp = _DEFAULT_TEMPS.get(role, 0.7) if temperature is None else temperature
    logger.info("llm_init", extra={
        "role": role, "model": model_name, "temperature": temp,
    })
    api_key = _resolve_api_key()
    base_url = os.getenv("DEEPSEEK_BASE_URL", _DEFAULT_BASE_URL)
    model = ChatOpenAI(
        model=model_name,
        temperature=temp,
        api_key=api_key,
        base_url=base_url,
        streaming=True,
    )
    return _apply_circuit_breaker(model, role)


def get_breaker_snapshot(role: str) -> dict | None:
    """Return circuit breaker state for a role, or None if not yet created."""
    return _cb_map[role].snapshot() if role in _cb_map else None
