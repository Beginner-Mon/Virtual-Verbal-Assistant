"""Safety Filter — Local SLM-based Red Flag Screening.

Uses Qwen2.5-3B via Ollama to detect medical red flags before processing.
"""

import json
import os
import time
from typing import Dict, Any, Optional

from utils.logger import get_logger
from utils.ollama_client import OllamaClient

logger = get_logger(__name__)

# Prompt designed for fast, accurate red flag detection by a 3B model.
SAFETY_PROMPT = """You are a medical safety screener for a physical therapy application.
Your ONLY job is to detect RED FLAG symptoms in the user's query that require immediate emergency medical attention or a doctor's visit.

Red flag symptoms include:
- Severe, unremitting, or worsening pain
- Chest pain radiating to the arm, jaw, or back
- Sudden numbness, tingling, or severe weakness (especially after trauma)
- Sudden loss of bowel/bladder control
- Suspected fractures or dislocations
- High fever accompanied by joint pain or stiffness

Reply purely in JSON format with exactly two keys:
1. "is_safe": true if safe (no red flags), false if unsafe (red flags detected).
2. "reason": A brief explanation of why it is unsafe, or "Safe" if it is safe.

User query: "{query}"

JSON format only:"""

class SafetyFilter:
    """SLM-based medical safety bounds checking."""
    
    def __init__(self, model_name: str = "qwen:0.5b", timeout: int = 5):
        """Initialize SafetyFilter.
        
        Args:
            model_name: Name of the local SLM to use (must be fast and local).
            timeout: Timeout in seconds for the safety check.
        """
        self.model_name = model_name
        self.ollama_client = OllamaClient(model_name)
        # Override the default 30s timeout to be much faster since this is a gate
        self.ollama_client.timeout = timeout
        self.fast_safe_bypass = str(os.getenv("SAFETY_FAST_SAFE_BYPASS", "true")).strip().lower() in {
            "1", "true", "yes", "on"
        }
        self.cache_ttl_seconds = max(0, int(os.getenv("SAFETY_CACHE_TTL_SECONDS", "300") or 300))
        self._result_cache: Dict[str, Dict[str, Any]] = {}
        
        logger.info(f"SafetyFilter initialized with SLM: {model_name}")

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

    def _looks_high_risk(self, query: str) -> bool:
        q = (query or "").lower()
        high_risk_markers = [
            "chest pain", "heart attack", "can't breathe", "cannot breathe",
            "loss of bowel", "loss of bladder", "stroke", "numbness spreading",
            "severe pain", "worsening pain", "fracture", "dislocation",
            "self-harm", "hurt myself", "kill myself", "suicide", "overdose",
        ]
        return any(marker in q for marker in high_risk_markers)

    def _looks_low_risk_fitness_query(self, query: str) -> bool:
        q = (query or "").lower()
        if not q:
            return True
        if self._looks_high_risk(q):
            return False
        safe_markers = [
            "exercise", "workout", "stretch", "squat", "plank", "walk", "run",
            "warm-up", "mobility", "posture", "motivate", "habit", "hydration",
            "water", "routine", "show me", "demonstrate", "visualize", "animation",
            "motion",
        ]
        conversation_markers = ["hi", "hello", "thanks", "thank you", "good morning", "good evening"]
        return any(marker in q for marker in safe_markers) or q.strip() in conversation_markers

    def check_query_safety(self, query: str) -> Dict[str, Any]:
        """Check if a query contains medical red flags.
        
        Args:
            query: The user's query.
            
        Returns:
            Dict containing:
                - is_safe (bool): True if safe, False if red flags detected.
                - reason (str): Explanation for rejection, or "Safe".
        """
        cached = self._cache_get(query)
        if cached is not None:
            return cached

        # 1. Rule-based Fast-Path
        fast_check = query.lower()
        strict_keywords = [
            "chest pain", "heart attack", "can't breathe", "cannot breathe",
            "loss of bowel", "loss of bladder", "stroke", "numbness spreading"
        ]
        
        for word in strict_keywords:
            if word in fast_check:
                logger.warning(f"[SafetyFilter] 🚨 Fast-Path Reject: '{word}' detected.")
                result = {
                    "is_safe": False, 
                    "reason": f"System detected critical keyword: {word}. Please seek emergency medical help immediately."
                }
                self._cache_set(query, result)
                return result

        # 1b. Skip expensive SLM call when query clearly looks low-risk.
        if self.fast_safe_bypass and self._looks_low_risk_fitness_query(query):
            result = {"is_safe": True, "reason": "Safe (Fast bypass)"}
            self._cache_set(query, result)
            return result

        # 2. SLM (Local LLM) Semantic Check
        prompt = SAFETY_PROMPT.format(query=query)
        
        try:
            logger.info(f"[SafetyFilter] Calling SLM {self.model_name} for Red Flag check...")
            raw_response = self.ollama_client.generate(
                prompt=prompt,
                format="json",
                temperature=0.1,  # Highly deterministic
                max_tokens=64,    # Tiny budget for fast response
            )
            
            # Parse response
            try:
                result = json.loads(raw_response)
                is_safe = bool(result.get("is_safe", True))
                reason = result.get("reason", "Safe")
            except json.JSONDecodeError:
                # If SLM fails to output valid JSON, fail open (safe) but log it
                logger.error(f"[SafetyFilter] Failed to parse SLM output: {raw_response[:100]}")
                parsed_result = {"is_safe": True, "reason": "Safe (Parse error)"}
                self._cache_set(query, parsed_result)
                return parsed_result
                
            if not is_safe:
                logger.warning(f"[SafetyFilter] 🚨 SLM Reject: {reason}")

            slm_result = {
                "is_safe": is_safe,
                "reason": reason
            }
            self._cache_set(query, slm_result)
            return slm_result
            
        except Exception as exc:
            logger.error(f"[SafetyFilter] Timeout or Error during generation: {exc}")
            # In case of SLM crash/timeout, we "fail open" to avoid blocking standard processing,
            # since RAG prompt itself has medical disclaimers.
            error_result = {"is_safe": True, "reason": "Safe (SLM Error/Timeout)"}
            self._cache_set(query, error_result)
            return error_result
