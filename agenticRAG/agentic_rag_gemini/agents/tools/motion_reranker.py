"""LLM reranker + Semantic Bridge for motion candidates.

- select_best():     Pick the best candidate from Top-K (legacy, kept for compat)
- rewrite_prompt():  Semantic Bridge — rewrite user query in HumanML3D vocabulary
"""

import json
import re
from typing import List, Optional

from agents.tools.motion_candidate_retriever import MotionCandidate
from utils.cache_service import CacheService
from utils.gemini_client import GeminiClientWrapper
from utils.logger import get_logger

logger = get_logger(__name__)

# Maximum words for the rewritten caption (Slow Path)
MAX_CAPTION_WORDS = 20


class MotionReranker:
    def __init__(self) -> None:
        self.client = GeminiClientWrapper()
        self._cache = CacheService()

    # ── Semantic Bridge (Slow Path) ─────────────────────────────

    def rewrite_prompt(
        self,
        user_query: str,
        reference_captions: List[str],
        timeout_seconds: int = 8,
    ) -> str:
        """Rewrite user query into a HumanML3D-style motion caption.

        Uses reference captions from the knowledge base as style examples
        so the T2M model receives vocabulary it was trained on.

        Args:
            user_query:          The raw user query (e.g. "show me a burpee")
            reference_captions:  Top-K captions retrieved from ChromaDB
            timeout_seconds:     LLM call timeout

        Returns:
            A concise motion caption in HumanML3D linguistic style,
            max 20 words.  Falls back to the best reference caption
            on any failure.
        """
        if not reference_captions:
            return user_query

        # ── Redis cache check ───────────────────────────────────
        cached = self._cache.get_motion_prompt(user_query)
        if cached:
            logger.info(
                "Semantic Bridge CACHE HIT: '%s' → '%s'",
                user_query[:40], cached[:40],
            )
            return cached

        refs_block = "\n".join(
            f'  {i + 1}. "{c}"' for i, c in enumerate(reference_captions)
        )

        prompt = (
            "You are a motion-description translator. "
            "Given reference motion captions and a user request, "
            "rewrite the request as a single motion caption that matches "
            "the linguistic style of the references.\n\n"
            "RULES:\n"
            "- Output ONLY the rewritten caption, nothing else.\n"
            "- Hard limit: maximum 20 words.\n"
            "- Strictly describe body movements: joints, limbs, spatial orientation.\n"
            "- Use simple, physical action verbs (walks, bends, raises, lowers, steps).\n"
            "- DO NOT use exercise names (e.g., deadlift, burpee, push-up, squat).\n"
            "- Maintain the linguistic style of the HumanML3D reference captions.\n\n"
            f"Reference captions:\n{refs_block}\n\n"
            f'User request: "{user_query}"\n\n'
            "Rewritten caption:"
        )

        try:
            resp = self.client.chat.completions.create(
                model="gemini-2.5-flash",
                messages=[
                    {"role": "user", "content": prompt},
                ],
                temperature=0.2,
                max_tokens=100,
                timeout=timeout_seconds,
                stream=False,
            )
            content = (
                resp.choices[0].message.content if resp and resp.choices else ""
            )
            content = content.strip().strip('"').strip("'").strip()

            # Enforce 25-word limit
            words = content.split()
            if len(words) > MAX_CAPTION_WORDS:
                content = " ".join(words[:MAX_CAPTION_WORDS])

            if content:
                logger.info(
                    "Semantic Bridge: '%s' → '%s'",
                    user_query[:60],
                    content[:80],
                )
                # ── Store in Redis (7-day TTL) ──────────────────
                self._cache.set_motion_prompt(user_query, content)
                return content

        except Exception as exc:
            logger.warning("Semantic Bridge failed, using best reference: %s", exc)

        # Fallback: return the first (best) reference caption
        return reference_captions[0]

    # ── Legacy: select best candidate index ─────────────────────

    def select_best(
        self,
        user_query: str,
        candidates: List[MotionCandidate],
        timeout_seconds: int = 8,
    ) -> Optional[MotionCandidate]:
        """Select the best candidate from Top-K (legacy method)."""
        if not candidates:
            return None

        candidates_blob = "\n".join(
            [
                f"{idx + 1}. id={c.candidate_id}; description={c.text_description}"
                for idx, c in enumerate(candidates)
            ]
        )

        prompt = (
            'You are ranking motion candidates. Return ONLY JSON: {"selected_index": <1-based index>}\n'
            f"User query: {user_query}\n"
            "Candidates:\n"
            f"{candidates_blob}\n"
        )

        try:
            resp = self.client.chat.completions.create(
                model="gemini-2.5-flash",
                messages=[
                    {"role": "system", "content": "Select the best candidate index only."},
                    {"role": "user", "content": prompt},
                ],
                temperature=0.0,
                max_tokens=64,
                timeout=timeout_seconds,
                stream=False,
            )
            content = resp.choices[0].message.content if resp and resp.choices else ""
            content = content.strip()

            try:
                parsed = json.loads(content)
                idx = int(parsed.get("selected_index", 0)) - 1
            except Exception:
                m = re.search(r"(\d+)", content)
                idx = int(m.group(1)) - 1 if m else -1

            if idx < 0 or idx >= len(candidates):
                return None
            return candidates[idx]
        except Exception as exc:
            logger.warning("Reranker failed, will fallback to embedding top-1: %s", exc)
            return None
