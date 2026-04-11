"""LLM Action Extractor - Highly intelligent verb extraction using Gemini-2.5-Flash."""

import logging
from typing import Optional

from utils.logger import get_logger
from utils.ollama_client import OllamaClient

logger = get_logger(__name__)

class LLMActionExtractor:
    """Intelligently extract the target physical action using a zero-shot local LLM."""

    def __init__(self):
        # We use qwen:0.5b because it's already in memory and blazing fast
        self.client = OllamaClient(model_name="qwen:0.5b")
        self.client.timeout = 5  # aggressive timeout for fast fallback
        logger.info("LLMActionExtractor initialized with local OllamaClient (qwen:0.5b).")

    def extract_core_action(self, text: str) -> str:
        """Extract the exact exercise noun or verb and nothing else."""
        if not text:
            return ""
            
        text = text.strip()
        
        # If it's incredibly short (1-2 words) AND doesn't have common conversational prefixes,
        # it's probably already clean. We just return it directly.
        words = text.split()
        if len(words) <= 2 and words[0].lower() not in ("show", "how", "what", "is", "a", "an", "the"):
            return text.lower()
            
        system_prompt = (
            "You are an extraction engine. Extract ONLY the core physical exercise name or action from the query.\n"
            "Respond ONLY with a single exercise name or verb in lowercase (no punctuation).\n"
            "Examples:\n"
            "Query: show me how to do shoulder rolls -> shoulder rolls\n"
            "Query: demonstrate a cartwheel please -> cartwheel\n"
            "Query: show me how to jump -> jump\n"
            "Query: i want to see a push up -> push up\n"
            "Query: my back hurts what should I do -> stretch\n"
            f"Query: {text} ->"
        )
        
        try:
            # We use format="json" if required, but prompt is simple enough for raw text
            response = self.client.generate(
                prompt=system_prompt,
                temperature=0.0,
                max_tokens=10,
                stream=False
            )
            
            # Ollama fallback typically returns JSON string if it crashed locally on the old codebase
            # since we just send text, it's a string
            extracted = str(response).strip().lower()
            
            # Failsafe: if the LLM hallucinated and returned a full sentence
            if len(extracted.split()) > 5 or "{" in extracted:
                logger.warning(f"LLM verb extraction failed or hallucinated: {extracted[:30]}...")
                return text
                
            logger.info(f"LLMActionExtractor parsed '{text}' -> '{extracted}' (qwen:0.5b)")
            return extracted
            
        except Exception as e:
            logger.warning(f"LLM extraction local API failed ({e}). Falling back to raw text.")
            return text

# Singleton instance
_extractor_instance: Optional[LLMActionExtractor] = None

def get_llm_extractor() -> LLMActionExtractor:
    global _extractor_instance
    if _extractor_instance is None:
        _extractor_instance = LLMActionExtractor()
    return _extractor_instance

def extract_action(text: str) -> str:
    """Convenience function to extract core action via LLM."""
    extractor = get_llm_extractor()
    return extractor.extract_core_action(text)
