"""Local Orchestrator — Qwen2.5-3B based query analysis and agent routing.

This module implements a local orchestrator that uses Qwen2.5-3B via Ollama
to analyze user queries, extract entities, and route to appropriate agents.
"""

# Per-intent max-token budget for the Ollama routing call.
# Smaller budgets = faster generation for lightweight intents.
# NOTE: Since intent isn't known *before* the call, these are applied
# based on the *previous* turn's intent (or use DEFAULT_TOKEN_BUDGET for
# the first call). Intent-aware budgets take effect from the second request.
_INTENT_TOKEN_BUDGETS: dict = {
    "greeting":                 64,
    "followup_question":        64,
    "resume_conversation":      64,
    "conversation":             64,   # canonical alias
    "ask_exercise_info":       128,
    "knowledge_query":         128,   # canonical alias
    "general_fitness_question": 128,
    "visualize_motion":        192,   # needs extra fields (exercise_name, motion_type)
    "unknown":                 128,
}
_DEFAULT_TOKEN_BUDGET: int = 128

import json
import logging
import os
from typing import Dict, Any, List, Optional

from config import get_config
from utils.logger import get_logger
from utils.ollama_client import OllamaClient
from prompts.orchestrator_prompts import ORCHESTRATOR_PROMPT


logger = get_logger(__name__)


class LocalOrchestrator:
    """Local Qwen2.5-3B orchestrator for agent routing."""
    
    def __init__(self, model_name: str = "qwen:0.5b"):
        """Initialize local orchestrator.
        
        Args:
            model_name: Name of the local model to use
        """
        self.model_name = model_name
        self.ollama_client = OllamaClient(model_name)
        self.prompt_template = ORCHESTRATOR_PROMPT
        
        # Configuration
        self.temperature = getattr(get_config().local_orchestrator, 'temperature', 0.1)
        self.max_tokens = getattr(get_config().local_orchestrator, 'max_tokens', 120)
        self.confidence_threshold = getattr(get_config().local_orchestrator, 'confidence_threshold', 0.7)
        # Tracks the intent returned on the previous turn; used to select
        # the appropriate token budget for the next Ollama call.
        self._last_intent: str | None = None
        
        logger.info(f"LocalOrchestrator initialized with model: {model_name}")
        
        # Check Ollama connection
        if not self.ollama_client.check_connection():
            logger.warning("Ollama service is not available")
    
    def analyze_query(self, query: str, conversation_history=None, detected_exercise=None) -> Dict[str, Any]:
        """Analyze query and return routing decision.
        
        Args:
            query: User query text
            conversation_history: Previous conversation turns
            detected_exercise: Exercise name detected by ExerciseDetector (optional)
            
        Returns:
            Routing decision dictionary with intent, exercise, agents, etc.
        """
        try:
            # Build prompt with query and context
            prompt = self._build_prompt(query, conversation_history, detected_exercise)
            logger.debug(f"[LocalOrchestrator] Prompt length: {len(prompt)} chars")

            # Use a smaller token budget when the previous intent was lightweight.
            # For the first call (no prior intent) we fall back to self.max_tokens.
            prev_intent = getattr(self, "_last_intent", None)
            token_budget = _INTENT_TOKEN_BUDGETS.get(prev_intent, self.max_tokens)

            # Call local model
            logger.info(
                f"[LocalOrchestrator] Calling Ollama | model={self.model_name} "
                f"timeout={self.ollama_client.timeout}s max_tokens={token_budget} "
                f"(prev_intent={prev_intent!r})"
            )
            response = self.ollama_client.generate(
                prompt=prompt,
                format="json",
                temperature=self.temperature,
                max_tokens=token_budget,
            )
            
            logger.debug(f"[LocalOrchestrator] Raw response length: {len(response)} chars, first 200: {response[:200]}")
            
            # Parse and validate JSON response
            decision = self._parse_response(response)
            
            # Validate confidence
            if decision['confidence'] < self.confidence_threshold:
                logger.warning(
                    f"[LocalOrchestrator] Low confidence routing: {decision['confidence']:.2f} "
                    f"(threshold={self.confidence_threshold}) — intent={decision['intent']}"
                )
            
            # Cache intent for next-call token-budget optimisation
            self._last_intent = decision.get("intent", "unknown")

            logger.info(
                f"[LocalOrchestrator] ✓ Routing decision: intent={decision['intent']}, "
                f"exercise={decision.get('exercise')}, confidence={decision['confidence']:.2f}, "
                f"agents={decision['agents']}"
            )
            return decision
            
        except Exception as e:
            logger.error(
                f"[LocalOrchestrator] ✗ Query analysis failed: {type(e).__name__}: {e}",
                exc_info=True
            )
            return self._get_fallback_response()
    
    def _build_prompt(self, query: str, conversation_history=None, detected_exercise=None) -> str:
        """Build prompt for the local model.
        
        Args:
            query: User query
            conversation_history: Previous conversation context
            detected_exercise: Exercise name detected by ExerciseDetector (optional)
            
        Returns:
            Complete prompt string
        """
        prompt = self.prompt_template + "\n\n"
        
        # Add detected exercise context if available
        if detected_exercise:
            prompt += f"DETECTED EXERCISE: {detected_exercise}\n"
            prompt += f"NOTE: The exercise '{detected_exercise}' has been detected in user query. "
            prompt += f"Use this exercise name instead of trying to infer it again.\n\n"
        
        # Add conversation context if available
        if conversation_history:
            prompt += "CONVERSATION HISTORY:\n"
            for turn in conversation_history[-3:]:  # Last 3 turns
                prompt += f"{turn.get('role', 'user')}: {turn.get('content', '')}\n"
            prompt += "\n"
        
        # Add current query
        prompt += f"User: {query}\n\n"
        prompt += "Return ONLY JSON:"
        
        return prompt
    
    def _parse_response(self, response: str) -> Dict[str, Any]:
        """Parse and validate JSON response from model.
        
        Args:
            response: Raw response text from model
            
        Returns:
            Parsed and validated decision dictionary
        """
        try:
            # Clean response (remove markdown code blocks if present)
            response = response.strip()
            if response.startswith('```json'):
                response = response[7:]
            if response.endswith('```'):
                response = response[:-3]
            response = response.strip()
            
            logger.debug(f"[LocalOrchestrator._parse_response] Cleaned response: {response}")
            
            # Parse JSON
            try:
                parsed = json.loads(response)
                logger.debug(f"[LocalOrchestrator._parse_response] ✓ JSON parsed successfully")
            except json.JSONDecodeError as je:
                logger.error(
                    f"[LocalOrchestrator._parse_response] ✗ JSON parse failed at position {je.pos}: {je.msg}"
                )
                logger.debug(f"Raw response that failed: {response}")
                return self._get_fallback_response()
            
            # Validate required fields
            required_fields = [
                "intent", "exercise", "agents", 
                "needs_motion", "needs_retrieval", "needs_web_search", "confidence"
            ]
            
            missing = [f for f in required_fields if f not in parsed]
            if missing:
                logger.error(
                    f"[LocalOrchestrator._parse_response] ✗ Missing required fields: {missing}"
                )
                return self._get_fallback_response()
            
            # Validate intent
            valid_intents = [
                "ask_exercise_info", "visualize_motion", "greeting", 
                "followup_question", "resume_conversation", "general_fitness_question", "unknown"
            ]
            
            if parsed["intent"] not in valid_intents:
                logger.warning(
                    f"[LocalOrchestrator._parse_response] ✗ Invalid intent '{parsed['intent']}' "
                    f"— valid: {valid_intents}"
                )
                return self._get_fallback_response()
            
            # Validate agents list
            valid_agents = ["retrieval_agent", "motion_agent", "web_search_agent", "memory_agent"]
            for agent in parsed["agents"]:
                if agent not in valid_agents:
                    logger.warning(
                        f"[LocalOrchestrator._parse_response] ✗ Invalid agent '{agent}' "
                        f"— valid: {valid_agents}"
                    )
                    return self._get_fallback_response()
            
            # Validate confidence
            confidence = parsed["confidence"]
            if not isinstance(confidence, (int, float)) or not (0.0 <= confidence <= 1.0):
                logger.warning(
                    f"[LocalOrchestrator._parse_response] ✗ Invalid confidence {confidence} "
                    f"— must be float in [0.0, 1.0]"
                )
                return self._get_fallback_response()
            
            logger.debug(f"[LocalOrchestrator._parse_response] ✓ All validation passed")
            return parsed
            
        except json.JSONDecodeError as e:
            logger.error(f"[LocalOrchestrator._parse_response] ✗ JSON parsing failed: {e}")
            logger.debug(f"Raw response: {response}")
            return self._get_fallback_response()
    
    def _get_fallback_response(self) -> Dict[str, Any]:
        """Return safe fallback response when analysis fails."""
        logger.info(
            "[LocalOrchestrator._get_fallback_response] "
            "Returning fallback 'unknown' decision. Check logs above for failure reason."
        )
        return {
            "intent": "unknown",
            "exercise": None,
            "agents": ["retrieval_agent"],
            "needs_motion": False,
            "needs_retrieval": True,
            "needs_web_search": False,
            "confidence": 0.1
        }
    
    def warmup(self) -> None:
        """Pre-load the Ollama model into memory.

        Fires a minimal dummy inference so the Qwen model is fully loaded
        before the first real user request arrives.  Called once during
        AgenticRAG API startup.  Failure is non-fatal — a warning is logged
        so the operator knows the model is cold.
        """
        import time
        logger.info(f"[LocalOrchestrator] Warming up model '{self.model_name}'...")
        t0 = time.time()
        try:
            self.ollama_client.generate(
                prompt="hi",
                format=None,
                temperature=0.0,
                max_tokens=1,
            )
            elapsed = time.time() - t0
            logger.info(
                f"[LocalOrchestrator] ✓ Warm-up done in {elapsed:.2f}s "
                f"— model is ready."
            )
        except Exception as exc:
            elapsed = time.time() - t0
            logger.warning(
                f"[LocalOrchestrator] ✗ Warm-up failed after {elapsed:.2f}s: {exc}. "
                f"First real request may experience cold-start latency."
            )

    def health_check(self) -> Dict[str, Any]:
        """Check orchestrator health status.

        Returns:
            Health status dictionary
        """
        ollama_available = self.ollama_client.check_connection()
        available_models = self.ollama_client.list_models()
        model_available = self.model_name in available_models

        return {
            "orchestrator": "local",
            "model": self.model_name,
            "ollama_available": ollama_available,
            "model_available": model_available,
            "available_models": available_models,
            "status": "healthy" if ollama_available and model_available else "unhealthy",
        }
