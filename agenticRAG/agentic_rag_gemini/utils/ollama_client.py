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
        
    def generate(self, prompt: str, format: str = None, temperature: float = None, max_tokens: int = None, stream: bool = False) -> Any:
        """Generate response from local model.
        
        Args:
            prompt: Input prompt for the model
            format: Response format (e.g., 'json')
            temperature: Sampling temperature
            max_tokens: Maximum tokens to generate
            stream: Whether to stream the response
            
        Returns:
            Generated response text (str) if stream=False,
            else a generator yielding chunks (Generator[str, None, None])
        """
        try:
            # Build request payload
            payload = {
                "model": self.model_name,
                "prompt": prompt,
                "stream": stream,
                "keep_alive": "10m"
            }
            
            # Add optional parameters
            if format:
                payload["format"] = format
            if temperature is not None:
                payload["temperature"] = temperature
            if max_tokens is not None:
                payload["options"] = {"num_predict": max_tokens}
            
            logger.info(f"[OllamaClient] Calling {self.base_url}/api/generate (stream={stream})")
            
            if stream:
                return self._stream_generate(payload)

            response = requests.post(
                f"{self.base_url}/api/generate",
                json=payload,
                timeout=self.timeout
            )
            
            response.raise_for_status()
            result = response.json()
            
            logger.info(f"[OllamaClient] ✓ Response received: {len(result.get('response', ''))} chars")
            
            if "response" in result:
                return result["response"]
            else:
                logger.error(f"[OllamaClient] ✗ Unexpected response format (missing 'response' key): {list(result.keys())}")
                return self._get_fallback_response()
                
        except Exception as e:
            logger.error(f"[OllamaClient] ✗ Error: {type(e).__name__}: {e}", exc_info=True)
            if stream:
                # Re-raise or yield error? Yielding is safer for generators.
                def error_gen(): yield f"[ERROR] {str(e)}"
                return error_gen()
            return self._get_fallback_response()

    def _stream_generate(self, payload: Dict[str, Any]):
        """Internal generator for streaming responses."""
        try:
            response = requests.post(
                f"{self.base_url}/api/generate",
                json=payload,
                timeout=self.timeout,
                stream=True
            )
            response.raise_for_status()
            
            for line in response.iter_lines():
                if line:
                    chunk = json.loads(line)
                    if "response" in chunk:
                        yield chunk["response"]
                    if chunk.get("done"):
                        break
        except Exception as e:
            logger.error(f"[OllamaClient] ✗ Stream error: {e}")
            yield f"\n[STREAM_ERROR] {str(e)}"
    
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
