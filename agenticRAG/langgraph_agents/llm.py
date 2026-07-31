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
  GEMINI_API_KEYS         — optional, comma-separated. First key is used to build a
                            one-shot Gemini fallback model (get_fallback_chat_model)
                            for planner/synthesizer when the primary DeepSeek call
                            fails or times out. Fallback unavailable if unset.

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
import threading
import time
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

# Roles that use the HEAVY model (reasoning-intensive). Empty — synthesizer (the last
# role still on deepseek-v4-pro) moved to the fast model after a live head-to-head A/B
# test: deepseek-v4-flash was faster on EVERY sample across 5 diverse scenarios (exercise
# recommendation, safety-critical red-flag, out-of-scope refuse) — 1.5-4x, with pro's worst
# case still slower than flash's worst case — and no observed quality/safety gap (checked
# by hand incl. the red-flag scenario). Kept as a set (not deleted) so a role can move back
# to heavy with a one-line change if a real quality regression shows up later.
_HEAVY_ROLES: set[str] = set()

# Roles whose output can legitimately run long (clinical answers with citations) and so
# need a generous max_tokens/timeout ceiling REGARDLESS of model tier. Kept separate from
# _HEAVY_ROLES so switching a role's model doesn't silently also shrink its output budget —
# synthesizer's live test samples ranged up to ~480 words (~700+ tokens for Vietnamese
# text), which would risk truncation under the FAST budget (512 tokens).
_LONG_OUTPUT_ROLES = {"synthesizer"}

# Per-role request timeout (seconds). Bounds the previously-unbounded httpx/OS default
# (fix #2). _LONG_OUTPUT_ROLES gets the larger ceiling as headroom, even on the fast model.
_TIMEOUT_HEAVY = 35.0
_TIMEOUT_FAST = 20.0

# Cheap built-in retry for transient network blips only — kept low (1) so a
# slow-but-succeeding call isn't retried 3x and made worse (fix #2).
_MAX_RETRIES = 1

# Output token ceiling per role (fix #3). _LONG_OUTPUT_ROLES needs headroom for longer
# clinical answers with citations; other roles (planner JSON / retriever tool-call
# decisions) rarely produce long output — bounded as a safety net too.
_MAX_TOKENS_HEAVY = 1024
_MAX_TOKENS_FAST = 512

# Gemini fallback model per role class (fix #2).
# FAST tier verified live 21/07: gemini-3.1-flash-lite REJECTED — its response content
# comes back as a list of content blocks ([{"type": "text", "text": ..., "extras": {...}}])
# via langchain_google_genai==4.2.3, not a plain string; production code (e.g.
# `final += content` in synthesizer.py) assumes string content and would crash on it —
# exactly when the fallback is needed most. gemini-2.5-flash tested clean (plain string,
# ~6s, correct Vietnamese output) — used instead. Revisit 3.1-flash-lite if the
# langchain_google_genai dependency is upgraded and re-verified.
_FALLBACK_MODEL_HEAVY = "gemini-2.5-pro"
_FALLBACK_MODEL_FAST = "gemini-2.5-flash"


def _model_for_role(role: str) -> str:
    """Resolve model name per role. Heavy roles use DEEPSEEK_MODEL, others use
    DEEPSEEK_MODEL_FAST. Both can be env-overridden."""
    if role in _HEAVY_ROLES:
        return os.getenv("DEEPSEEK_MODEL", _DEFAULT_HEAVY_MODEL)
    return os.getenv("DEEPSEEK_MODEL_FAST", _DEFAULT_FAST_MODEL)


def _timeout_for_role(role: str) -> float:
    return _TIMEOUT_HEAVY if role in _HEAVY_ROLES or role in _LONG_OUTPUT_ROLES else _TIMEOUT_FAST


def _max_tokens_for_role(role: str) -> int:
    return _MAX_TOKENS_HEAVY if role in _HEAVY_ROLES or role in _LONG_OUTPUT_ROLES else _MAX_TOKENS_FAST


def _fallback_model_for_role(role: str) -> str:
    return _FALLBACK_MODEL_HEAVY if role in _HEAVY_ROLES else _FALLBACK_MODEL_FAST


def _resolve_api_key() -> str | None:
    return os.getenv("DEEPSEEK_API_KEY")


def _resolve_first_gemini_key() -> str | None:
    """Return the first key from comma-separated GEMINI_API_KEYS, or None."""
    raw = os.getenv("GEMINI_API_KEYS", "")
    keys = [k.strip() for k in raw.split(",") if k.strip()]
    return keys[0] if keys else None


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
    timeout = _timeout_for_role(role)
    max_tokens = _max_tokens_for_role(role)
    logger.info("llm_init", extra={
        "role": role, "model": model_name, "temperature": temp,
        "timeout": timeout, "max_retries": _MAX_RETRIES, "max_tokens": max_tokens,
    })
    api_key = _resolve_api_key()
    if not api_key:
        logger.warning("deepseek_key_missing", extra={
            "role": role, "action": "using_gemini_fallback",
        })
        return get_fallback_chat_model(role)
    base_url = os.getenv("DEEPSEEK_BASE_URL", _DEFAULT_BASE_URL)
    model = ChatOpenAI(
        model=model_name,
        temperature=temp,
        api_key=api_key,
        base_url=base_url,
        streaming=True,
        timeout=timeout,
        max_retries=_MAX_RETRIES,
        max_tokens=max_tokens,
    )
    return _apply_circuit_breaker(model, role)


@lru_cache(maxsize=8)
def get_fallback_chat_model(role: str):
    """Return a one-shot Gemini fallback ChatModel for the given role, or None.

    Used by planner/synthesizer when the primary DeepSeek call fails/times out
    (fix #2). Reads the FIRST key from comma-separated GEMINI_API_KEYS. Returns
    None (not an exception) when no key is configured — fallback is simply
    unavailable, not a startup error. No circuit breaker wrapping: this is
    already a last-resort, one-shot call.
    """
    api_key = _resolve_first_gemini_key()
    if not api_key:
        return None

    from langchain_google_genai import ChatGoogleGenerativeAI

    model_name = _fallback_model_for_role(role)
    temp = _DEFAULT_TEMPS.get(role, 0.7)
    logger.info("llm_fallback_init", extra={"role": role, "model": model_name})
    return ChatGoogleGenerativeAI(
        model=model_name,
        temperature=temp,
        google_api_key=api_key,
        timeout=_timeout_for_role(role),
        max_output_tokens=_max_tokens_for_role(role),
        # max_retries=0: without this, google-genai's SDK default retries 429/5xx
        # with exponential backoff internally (observed live: 5 retries, ~36s of
        # backoff before giving up) — defeating the point of a fast "one-shot"
        # fallback. The caller (planner/synthesizer) already falls through to the
        # existing safe error path on failure; a slow-to-fail fallback only makes
        # the worst case worse (primary timeout + this retry storm, stacked).
        max_retries=0,
    )


def get_breaker_snapshot(role: str) -> dict | None:
    """Return circuit breaker state for a role, or None if not yet created."""
    return _cb_map[role].snapshot() if role in _cb_map else None


# ── Prompt-cache telemetry (fix #1) ──────────────────────────────────

# DeepSeek's OpenAI-compatible API extension fields (response_metadata.token_usage),
# with generic cache-read/write field names as a fallback for usage_metadata.
_CACHE_HIT_KEYS = ("prompt_cache_hit_tokens", "cache_read_input_tokens", "cache_read")
_CACHE_MISS_KEYS = ("prompt_cache_miss_tokens", "cache_creation_input_tokens")


def extract_cache_tokens(ai_msg: Any) -> tuple[int | None, int | None]:
    """Best-effort (cache_hit_tokens, cache_miss_tokens) from an AIMessage.

    Checks ``response_metadata["token_usage"]`` first (DeepSeek's extension
    fields, populated on ``ainvoke`` / the final aggregated stream chunk),
    then falls back to ``usage_metadata`` (LangChain's provider-agnostic
    field). Never raises — returns (None, None) if telemetry is absent,
    which is an acceptable/informative outcome, not a bug (see fix #1 spec).
    """
    def _first(d: dict, keys: tuple[str, ...]) -> int | None:
        for k in keys:
            v = d.get(k)
            if v is not None:
                return v
        return None

    if ai_msg is None:
        return None, None

    try:
        token_usage = (getattr(ai_msg, "response_metadata", None) or {}).get("token_usage") or {}
    except Exception:
        token_usage = {}
    hit = _first(token_usage, _CACHE_HIT_KEYS)
    miss = _first(token_usage, _CACHE_MISS_KEYS)

    if hit is None and miss is None:
        try:
            usage_meta = getattr(ai_msg, "usage_metadata", None) or {}
            details = usage_meta.get("input_token_details") or {}
        except Exception:
            details = {}
        hit = _first(details, _CACHE_HIT_KEYS)
        miss = _first(details, _CACHE_MISS_KEYS)

    return hit, miss


# ── Gemini explicit context caching (forward-looking; not yet active) ──────
#
# Unlike DeepSeek, Gemini does NOT auto-cache repeated prompt prefixes — this
# was verified live (same 1300-token static planner prompt sent 3x back to
# back via ChatGoogleGenerativeAI: cache_read stayed 0 every time). Gemini
# caching only works if you EXPLICITLY create a CachedContent resource (with a
# TTL) and reference it via `cached_content=` on later calls — and, critically,
# you then CANNOT send a separate system_instruction; it must live in the
# cache itself.
#
# Only planner's Gemini fallback uses this: planner's system prompt is 100%
# static (same reasoning as its ~91% DeepSeek cache-hit rate — see fix #1),
# so caching it can genuinely pay off across repeated fallback calls.
# synthesizer's system prompt embeds per-request evidence and is NOT reused
# call-to-call, so caching it would cost a create-cache round-trip for a
# cache that's immediately stale — not implemented there, on purpose.
#
# Split into two functions so the cache NEVER adds latency to the fallback's
# critical path (that would repeat the exact "fallback slower than no
# fallback" mistake fixed earlier for the Gemini retry-storm bug):
#   - get_warm_gemini_cache(): synchronous, in-memory lookup ONLY, no network
#     call, safe to call inline during a live fallback attempt.
#   - warm_gemini_cache(): the one that actually calls the API to create (or
#     reuse if still valid) a cache. Deliberately NOT invoked from inside the
#     planner fallback path — call it explicitly (e.g. a startup hook or a
#     periodic task) once Gemini is a viable, billed provider.
#
# CURRENTLY INERT on the project's shared free-tier Gemini key: creating any
# cache fails immediately with 429 TotalCachedContentStorageTokensPerModelFreeTier
# limit=0 (verified live) — a tier/billing limit, not a prompt-size or code
# issue. warm_gemini_cache() catches this (and any other failure) and returns
# None; get_warm_gemini_cache() then simply finds nothing warmed, and callers
# fall through to the existing uncached fallback — zero behavior change until
# a paid-tier (or user-supplied) key actually allows a cache to be created.

_GEMINI_CACHE_TTL_SECONDS = 3600  # re-created automatically ~1min before expiry
_gemini_cache_lock = threading.Lock()
_gemini_cache_registry: dict[tuple[str, str], tuple[str, float]] = {}  # (role, model) -> (cache_name, expires_at)


def get_warm_gemini_cache(role: str, model_name: str) -> str | None:
    """Return an already-warm Gemini cache name for (role, model_name), or None.

    Pure in-memory lookup — no network call, cannot fail, safe to call from a
    live fallback's critical path.
    """
    with _gemini_cache_lock:
        entry = _gemini_cache_registry.get((role, model_name))
    if entry and entry[1] > time.time():
        return entry[0]
    return None


async def warm_gemini_cache(role: str, model_name: str, system_prompt: str) -> str | None:
    """Create (or reuse, if still valid) a Gemini CachedContent for this role's
    static system prompt. Makes a real API call — call this explicitly/eagerly
    (NOT from inside a live fallback attempt), e.g. at startup or on a timer,
    once Gemini caching is actually usable (paid tier / user-supplied key).

    Never raises. Returns None on any failure (including the current
    zero-quota free tier) — that is an expected, non-fatal outcome.
    """
    warm = get_warm_gemini_cache(role, model_name)
    if warm is not None:
        return warm

    api_key = _resolve_first_gemini_key()
    if not api_key:
        return None

    try:
        from langchain_core.messages import SystemMessage
        from langchain_google_genai import ChatGoogleGenerativeAI, create_context_cache

        base_model = ChatGoogleGenerativeAI(
            model=model_name, temperature=_DEFAULT_TEMPS.get(role, 0.7),
            google_api_key=api_key, timeout=_timeout_for_role(role),
        )
        cache_name = await asyncio.to_thread(
            create_context_cache,
            base_model,
            [SystemMessage(content=system_prompt)],
            ttl=f"{_GEMINI_CACHE_TTL_SECONDS}s",
        )
        with _gemini_cache_lock:
            _gemini_cache_registry[(role, model_name)] = (
                cache_name, time.time() + _GEMINI_CACHE_TTL_SECONDS - 60,
            )
        logger.info("gemini_cache_warmed", extra={"role": role, "model": model_name})
        return cache_name
    except Exception as exc:
        logger.warning("gemini_cache_unavailable", extra={
            "role": role, "model": model_name, "error": str(exc),
        })
        return None


def get_gemini_cached_chat_model(role: str) -> "Any | None":
    """Return a Gemini ChatModel with `cached_content` bound (constructor-time,
    not per-call — more reliable through `.with_structured_output()` wrapping),
    IF a warm cache already exists for this role. Returns None otherwise —
    callers must fall back to the regular uncached fallback model + full
    system message; do NOT send a system message alongside this model
    (the cache already carries it, and the API rejects both being set).
    """
    model_name = _fallback_model_for_role(role)
    cache_name = get_warm_gemini_cache(role, model_name)
    if cache_name is None:
        return None

    api_key = _resolve_first_gemini_key()
    if not api_key:
        return None

    from langchain_google_genai import ChatGoogleGenerativeAI
    return ChatGoogleGenerativeAI(
        model=model_name, temperature=_DEFAULT_TEMPS.get(role, 0.7),
        google_api_key=api_key, timeout=_timeout_for_role(role),
        max_output_tokens=_max_tokens_for_role(role), max_retries=0,
        cached_content=cache_name,
    )
