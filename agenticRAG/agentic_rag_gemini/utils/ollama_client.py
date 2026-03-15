"""Ollama Client — Local model interface wrapper.

This module provides a clean interface for interacting with Ollama local models,
specifically designed for the Qwen2.5-3B orchestrator.
"""

import json
import logging
from typing import Optional, Dict, Any

import requests
from config import get_config
from utils.logger import get_logger


logger = get_logger(__name__)


class OllamaClient:
    """Wrapper for Ollama local model interface."""
    
    def __init__(self, model_name: str = "qwen:0.5b"):
        """Initialize Ollama client.
        
        Args:
            model_name: Name of the model to use in Ollama
        """
        self.model_name = model_name
        self.base_url = getattr(get_config().ollama, 'base_url', 'http://localhost:11434')
        self.timeout = getattr(get_config().ollama, 'timeout', 30)
        
    def generate(self, prompt: str, format: str = None, temperature: float = None, max_tokens: int = None) -> str:
        """Generate response from local model.
        
        Args:
            prompt: Input prompt for the model
            format: Response format (e.g., 'json')
            temperature: Sampling temperature
            max_tokens: Maximum tokens to generate
            
        Returns:
            Generated response text
        """
        try:
            # Build request payload
            payload = {
                "model": self.model_name,
                "prompt": prompt,
                "stream": False
            }
            
            # Add optional parameters
            if format:
                payload["format"] = format
            if temperature is not None:
                payload["temperature"] = temperature
            if max_tokens is not None:
                payload["options"] = {"num_predict": max_tokens}
            
            logger.info(f"[OllamaClient] Calling {self.base_url}/api/generate")
            logger.debug(f"[OllamaClient] model={self.model_name}, timeout={self.timeout}s, format={format}, max_tokens={max_tokens}")
            logger.debug(f"[OllamaClient] prompt length={len(prompt)} chars")
            
            response = requests.post(
                f"{self.base_url}/api/generate",
                json=payload,
                timeout=self.timeout
            )
            
            response.raise_for_status()
            result = response.json()
            
            logger.info(f"[OllamaClient] ✓ Response received: {len(result.get('response', ''))} chars")
            logger.debug(f"[OllamaClient] Response preview: {result.get('response', '')[:200]}")
            
            if "response" in result:
                return result["response"]
            else:
                logger.error(f"[OllamaClient] ✗ Unexpected response format (missing 'response' key): {list(result.keys())}")
                return self._get_fallback_response()
                
        except requests.exceptions.ConnectionError as e:
            logger.error(f"[OllamaClient] ✗ Connection failed to {self.base_url}: {e}")
            return self._get_fallback_response()
        except requests.exceptions.Timeout as e:
            logger.error(
                f"[OllamaClient] ✗ Request timeout after {self.timeout}s "
                f"(model may be loading or slow). Consider using API orchestrator."
            )
            return self._get_fallback_response()
        except requests.exceptions.HTTPError as e:
            logger.error(f"[OllamaClient] ✗ HTTP error {response.status_code}: {e}")
            logger.debug(f"Response text: {response.text}")
            return self._get_fallback_response()
        except Exception as e:
            logger.error(f"[OllamaClient] ✗ Unexpected error: {type(e).__name__}: {e}", exc_info=True)
            return self._get_fallback_response()
    
    def _get_fallback_response(self) -> str:
        """Return safe fallback response when model fails."""
        return json.dumps({
            "intent": "unknown",
            "exercise": None,
            "agents": ["retrieval_agent"],
            "needs_motion": False,
            "needs_retrieval": True,
            "needs_web_search": False,
            "confidence": 0.1
        })
    
    def check_connection(self) -> bool:
        """Check if Ollama service is available.
        
        Returns:
            True if Ollama is responding, False otherwise
        """
        try:
            response = requests.get(f"{self.base_url}/api/tags", timeout=5)
            return response.status_code == 200
        except Exception as e:
            logger.warning(f"Ollama health check failed: {e}")
            return False
    
    def list_models(self) -> list:
        """List available models in Ollama.
        
        Returns:
            List of available model names
        """
        try:
            response = requests.get(f"{self.base_url}/api/tags", timeout=10)
            response.raise_for_status()
            data = response.json()
            return [model["name"] for model in data.get("models", [])]
        except Exception as e:
            logger.error(f"Failed to list Ollama models: {e}")
            return []
