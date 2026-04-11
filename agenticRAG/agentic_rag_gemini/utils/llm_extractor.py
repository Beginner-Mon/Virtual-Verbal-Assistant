"""LLM Action Extractor - Highly intelligent verb extraction using Gemini-2.5-Flash."""

import logging
from typing import Optional

from utils.logger import get_logger
from utils.gemini_client import GeminiClientWrapper

logger = get_logger(__name__)

class LLMActionExtractor:
    """Intelligently extract the target physical action using a zero-shot LLM prompt."""

    def __init__(self):
        self.client = GeminiClientWrapper()
        logger.info("LLMActionExtractor initialized globally with GeminiClientWrapper.")

    def extract_core_action(self, text: str) -> str:
        """Extract the exact exercise noun or verb and nothing else."""
        if not text:
            return ""
            
        text = text.strip()
        
        # If it's incredibly short (1-2 words), it's probably already clean.
        if len(text.split()) <= 2:
            return text.lower()
            
        system_prompt = (
            "You are an exercise extraction engine. The user will provide a conversational query asking to demonstrate or perform a physical exercise, or describing a physical issue.\n"
            "Your job is to identify the core physical exercise name.\n"
            "If the query explicitly mentions an exercise or action (e.g., 'jump', 'push up'), extract exactly that action.\n"
            "If the query is vague, mentions a symptom, or asks for an exercise recommendation without naming one (e.g., 'exercise to alleviate pain'), you MUST generate a single, highly relevant exercise verb (e.g., 'stretch', 'run', 'yoga').\n"
            "Respond ONLY with a single exercise name or verb in lowercase (no punctuation, no conversational filler).\n"
            "Examples:\n"
            "- 'show me how to do shoulder rolls' -> 'shoulder rolls'\n"
            "- 'demonstrate a cartwheel please' -> 'cartwheel'\n"
            "- 'show mw to jump' -> 'jump'\n"
            "- 'i want to see a push up' -> 'push up'\n"
            "- 'is there any exercise to help me alleviate the pain' -> 'stretch'\n"
            "- 'my back hurts what should I do' -> 'stretch'\n"
            "- 'I want to lose weight' -> 'run'\n"
        )
        
        try:
            response = self.client.chat.completions.create(
                model="gemini-2.5-flash",
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": text}
                ],
                temperature=0.1, # Tiny bit of creativity to prevent blockages
                max_tokens=100,
            )
            
            extracted = response.choices[0].message.content.strip().lower()
            
            # Failsafe: if the LLM hallucinated and returned a full sentence
            if len(extracted.split()) > 5:
                logger.warning(f"LLM verb extraction failed (result too long): {extracted[:30]}...")
                return text
                
            logger.info(f"LLMActionExtractor parsed '{text}' -> '{extracted}'")
            return extracted
            
        except Exception as e:
            logger.error(f"LLM extraction API failed: {e}")
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
