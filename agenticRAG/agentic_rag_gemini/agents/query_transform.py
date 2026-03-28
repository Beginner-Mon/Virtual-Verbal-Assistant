"""Transformation Engine — Query Expansion and HyDE (Hypothetical Document Embeddings).

Uses Gemini cloud LLM to expand informal queries into clinical terminology
and hallucinate a technical description of the required motion for HumanML3D retrieval.
"""

import json
from typing import Dict, Any

from utils.logger import get_logger
from utils.gemini_client import GeminiClientWrapper
from utils.cache_service import CacheService

logger = get_logger(__name__)

TRANSFORMATION_PROMPT = """You are an expert physical therapist and movement scientist.
The user has provided an informal query or symptom regarding a physical movement or exercise.
Your task is to transform this query to bridge the gap between their informal language and formal clinical/kinematic datasets.

Output MUST be valid JSON with exactly two fields:
1. "expanded_query": A formal, clinical description of the issue or the recommended exercise. Include proper medical terms (e.g., knee pain -> patellofemoral pain, nhảy cóc -> jump squat / plyometric bounding). (Used to search Clinical KB).
2. "hyde_document": A highly detailed, anatomical description of the physical MOTION needed to perform the recommended exercise. Describe joint angles, muscle engagement, and kinematic trajectories as if describing a HumanML3D motion capture sequence. This will be embedded to search the motion database.

User query: "{query}"

JSON output structure:
{{"expanded_query": "...", "hyde_document": "..."}}"""

class QueryTransformer:
    """Handles expanding user queries and generating HyDE documents."""

    def __init__(self, use_cache: bool = True):
        self.client = GeminiClientWrapper()
        self.cache = CacheService() if use_cache else None
        self.model = "gemini-2.5-flash"
        
    def transform_query(self, query: str) -> Dict[str, str]:
        """Transform user query into expanded clinical query and HyDE document.
        
        Args:
            query: The raw user query.
            
        Returns:
            Dict containing 'expanded_query' and 'hyde_document'.
            Falls back to the original query if LLM fails.
        """
        # 1. Check Cache
        if self.cache:
            cached = self.cache.get_semantic_transformation(query)
            if cached:
                logger.info(f"[QueryTransformer] ⚡ Semantic Cache Hit for: '{query[:20]}...'")
                return cached
                
        prompt = TRANSFORMATION_PROMPT.format(query=query)
        
        try:
            logger.info(f"[QueryTransformer] Calling Gemini ({self.model}) for Transformation...")
            response = self.client.chat.completions.create(
                model=self.model,
                messages=[{"role": "user", "content": prompt}],
                temperature=0.2, # low temperature for clinical accuracy
                max_tokens=600,
                response_format={"type": "json_object"}
            )
            
            content = response.choices[0].message.content.strip()
            
            # Handle potential markdown fencing from Gemini despite JSON instruction
            if content.startswith("```json"):
                content = content[7:-3]
            elif content.startswith("```"):
                content = content[3:-3]
                
            result = json.loads(content)
            
            transformed = {
                "expanded_query": result.get("expanded_query", query).strip(),
                "hyde_document": result.get("hyde_document", query).strip()
            }
            
            # Simple validation to ensure fields aren't completely empty
            if not transformed["expanded_query"]:
                transformed["expanded_query"] = query
            if not transformed["hyde_document"]:
                transformed["hyde_document"] = query
            
            logger.info(f"[QueryTransformer] ✓ Transformation successful. HyDE length: {len(transformed['hyde_document'])}")
            
            if self.cache:
                self.cache.set_semantic_transformation(query, transformed)
                
            return transformed
            
        except json.JSONDecodeError as e:
            logger.error(f"[QueryTransformer] JSON parse error: {e}. Raw content: {content[:100]}...")
        except Exception as e:
            logger.error(f"[QueryTransformer] Failed to transform query: {type(e).__name__} - {e}", exc_info=True)
            
        # Fallback
        logger.warning("[QueryTransformer] Falling back to raw query for both fields.")
        return {
            "expanded_query": query,
            "hyde_document": query
        }
