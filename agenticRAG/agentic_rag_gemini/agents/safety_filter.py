"""Safety Filter — Pure Regex/Keyword-based Red Flag Screening.

Zero LLM calls. Runs in <1ms. Checks for:
- Medical emergencies (chest pain, stroke, fracture...)
- Violence / self-harm
- Sexual content
- Illegal activity

If NONE matched → SAFE.  If ANY matched → UNSAFE with reason.
"""

import os
import re
import time
from typing import Dict, Any, Optional

from utils.logger import get_logger

logger = get_logger(__name__)


# ── Blocklists ────────────────────────────────────────────────────────────
# Each tuple: (compiled regex pattern, human-readable category)

_MEDICAL_EMERGENCY_KEYWORDS = [
    "chest pain", "heart attack", "can't breathe", "cannot breathe",
    "difficulty breathing", "loss of bowel", "loss of bladder",
    "stroke", "numbness spreading", "severe pain", "worsening pain",
    "fracture", "dislocation", "paralysis", "seizure", "convulsion",
    "blood clot", "internal bleeding", "anaphylaxis",
]

_VIOLENCE_KEYWORDS = [
    "self-harm", "hurt myself", "kill myself", "suicide", "overdose",
    "want to die", "end my life", "cut myself", "harm others",
    "attack someone", "murder", "assault",
]

_SEXUAL_KEYWORDS = [
    "sexual abuse", "rape", "molest", "pornography",
    "child exploitation", "trafficking",
]

_ILLEGAL_KEYWORDS = [
    "how to make a bomb", "how to make drugs", "buy illegal",
    "hack into", "steal identity", "counterfeit",
]

# Pre-compile a single master regex for maximum speed
_ALL_PATTERNS: list[tuple[re.Pattern, str]] = []
for _kw in _MEDICAL_EMERGENCY_KEYWORDS:
    _ALL_PATTERNS.append((re.compile(re.escape(_kw), re.IGNORECASE), f"Medical emergency: {_kw}"))
for _kw in _VIOLENCE_KEYWORDS:
    _ALL_PATTERNS.append((re.compile(re.escape(_kw), re.IGNORECASE), f"Violence/self-harm: {_kw}"))
for _kw in _SEXUAL_KEYWORDS:
    _ALL_PATTERNS.append((re.compile(re.escape(_kw), re.IGNORECASE), f"Inappropriate content: {_kw}"))
for _kw in _ILLEGAL_KEYWORDS:
    _ALL_PATTERNS.append((re.compile(re.escape(_kw), re.IGNORECASE), f"Illegal activity: {_kw}"))


class SafetyFilter:
    """Pure keyword/regex safety screener. No LLM calls."""

    def __init__(self, **kwargs):
        """Initialize SafetyFilter.

        Accepts **kwargs for backward compatibility (old callers may pass
        model_name=, timeout=, etc.).  All are ignored since we no longer
        use an SLM.
        """
        self.cache_ttl_seconds = max(0, int(os.getenv("SAFETY_CACHE_TTL_SECONDS", "300") or 300))
        self._result_cache: Dict[str, Dict[str, Any]] = {}
        logger.info("SafetyFilter initialized (pure regex, 0 LLM calls)")

    # ── Cache helpers (unchanged) ─────────────────────────────────────────

    def _cache_get(self, query: str) -> Optional[Dict[str, Any]]:
        if self.cache_ttl_seconds <= 0:
            return None
        key = (query or "").strip().lower()
        cached = self._result_cache.get(key)
        if not cached:
            return None
        age = time.time() - float(cached.get("ts", 0))
        if age > self.cache_ttl_seconds:
            self._result_cache.pop(key, None)
            return None
        return {
            "is_safe": bool(cached.get("is_safe", True)),
            "reason": str(cached.get("reason", "Safe")),
        }

    def _cache_set(self, query: str, result: Dict[str, Any]) -> None:
        if self.cache_ttl_seconds <= 0:
            return
        key = (query or "").strip().lower()
        self._result_cache[key] = {
            "ts": time.time(),
            "is_safe": bool(result.get("is_safe", True)),
            "reason": str(result.get("reason", "Safe")),
        }

    # ── Public API ────────────────────────────────────────────────────────

    def check_query_safety(self, query: str) -> Dict[str, Any]:
        """Check if a query contains unsafe content.

        Runs entirely via compiled regex — no network calls, no LLM.

        Args:
            query: The user's query.

        Returns:
            Dict with ``is_safe`` (bool) and ``reason`` (str).
        """
        cached = self._cache_get(query)
        if cached is not None:
            return cached

        q = (query or "").strip()
        if not q:
            result = {"is_safe": True, "reason": "Safe"}
            self._cache_set(query, result)
            return result

        # Scan against all compiled patterns
        for pattern, reason in _ALL_PATTERNS:
            if pattern.search(q):
                logger.warning("[SafetyFilter] 🚨 Blocked: %s", reason)
                result = {
                    "is_safe": False,
                    "reason": f"⚠️ {reason}. Please seek appropriate help.",
                }
                self._cache_set(query, result)
                return result

        # Nothing matched → safe
        result = {"is_safe": True, "reason": "Safe"}
        self._cache_set(query, result)
        return result
