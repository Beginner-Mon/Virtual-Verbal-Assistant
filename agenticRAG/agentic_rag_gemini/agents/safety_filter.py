"""Safety Filter — Local SLM-based Red Flag Screening.

Uses Qwen2.5-3B via Ollama to detect medical red flags before processing.
"""

import json
from typing import Dict, Any

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
        
        logger.info(f"SafetyFilter initialized with SLM: {model_name}")

    def check_query_safety(self, query: str) -> Dict[str, Any]:
        """Check if a query contains medical red flags.
        
        Args:
            query: The user's query.
            
        Returns:
            Dict containing:
                - is_safe (bool): True if safe, False if red flags detected.
                - reason (str): Explanation for rejection, or "Safe".
        """
        # 1. Rule-based Fast-Path
        fast_check = query.lower()
        strict_keywords = [
            "chest pain", "heart attack", "can't breathe", "cannot breathe",
            "loss of bowel", "loss of bladder", "stroke", "numbness spreading"
        ]
        
        for word in strict_keywords:
            if word in fast_check:
                logger.warning(f"[SafetyFilter] 🚨 Fast-Path Reject: '{word}' detected.")
                return {
                    "is_safe": False, 
                    "reason": f"System detected critical keyword: {word}. Please seek emergency medical help immediately."
                }

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
                return {"is_safe": True, "reason": "Safe (Parse error)"}
                
            if not is_safe:
                logger.warning(f"[SafetyFilter] 🚨 SLM Reject: {reason}")
                
            return {
                "is_safe": is_safe,
                "reason": reason
            }
            
        except Exception as exc:
            logger.error(f"[SafetyFilter] Timeout or Error during generation: {exc}")
            # In case of SLM crash/timeout, we "fail open" to avoid blocking standard processing,
            # since RAG prompt itself has medical disclaimers.
            return {"is_safe": True, "reason": "Safe (SLM Error/Timeout)"}
