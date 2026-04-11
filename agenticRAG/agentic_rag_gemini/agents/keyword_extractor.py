"""Keyword Extractor — Zero-LLM motion prompt generator.

Replaces the Gemini-heavy SemanticBridge with a simple regex + dictionary
lookup to extract exercise verbs from user queries and format them into
DART-compatible motion prompts.

Pipeline position:
    User Query → KeywordExtractor.extract() → "a person performs squat" → DART /generate

No LLM calls. No API quota usage. Sub-millisecond latency.
"""

import re
from typing import Optional

from utils.logger import get_logger

logger = get_logger(__name__)

# ── Verb map: user phrase → canonical motion verb ─────────────────
# Sorted by longest-first at lookup time to ensure multi-word phrases
# match before single words (e.g. "chin tuck" before "chin").
VERB_MAP = {
    # Neck / Head
    "chin tuck": "chin tuck",
    "chin tucks": "chin tuck",
    "neck rotation": "neck rotation",
    "neck stretch": "neck stretch",
    "neck extension": "neck extension",
    "neck flexion": "neck flexion",
    "head tilt": "head tilt",
    "head turn": "head turn",

    # Shoulder / Upper body
    "shoulder shrug": "shoulder shrug",
    "shoulder roll": "shoulder roll",
    "shoulder stretch": "shoulder stretch",
    "shoulder press": "shoulder press",
    "arm raise": "arm raise",
    "lateral raise": "lateral raise",
    "arm circle": "arm circle",
    "overhead press": "overhead press",

    # Push / Pull
    "push up": "push up",
    "pushup": "push up",
    "push-up": "push up",
    "push ups": "push up",
    "pushups": "push up",
    "pull up": "pull up",
    "pullup": "pull up",
    "pull-up": "pull up",

    # Core
    "plank": "plank",
    "side plank": "side plank",
    "crunch": "crunch",
    "crunches": "crunch",
    "sit up": "sit up",
    "situp": "sit up",
    "sit-up": "sit up",

    # Legs
    "squat": "squat",
    "squats": "squat",
    "deep squat": "deep squat",
    "bodyweight squat": "squat",
    "lunge": "lunge",
    "lunges": "lunge",
    "walking lunge": "walking lunge",
    "reverse lunge": "reverse lunge",
    "leg raise": "leg raise",
    "leg extension": "leg extension",
    "calf raise": "calf raise",
    "wall sit": "wall sit",

    # Compound / Explosive
    "jumping jack": "jumping jacks",
    "jumping jacks": "jumping jacks",
    "burpee": "burpee",
    "burpees": "burpee",
    "box jump": "box jump",
    "box jumps": "box jump",
    "mountain climber": "mountain climber",
    "mountain climbers": "mountain climber",

    # Weight training
    "deadlift": "deadlift",
    "deadlifts": "deadlift",
    "bench press": "bench press",
    "bicep curl": "bicep curl",
    "bicep curls": "bicep curl",
    "tricep extension": "tricep extension",
    "row": "row",
    "rows": "row",
    "kettlebell swing": "kettlebell swing",
    "kettlebell swings": "kettlebell swing",

    # Cardio / Locomotion
    "walk": "walk",
    "walking": "walk",
    "run": "run",
    "running": "run",
    "jog": "jog",
    "jogging": "jog",
    "jump": "jump",
    "kick": "kick",
    "kicking": "kick",
    "step": "step",

    # Flexibility / Rehab
    "stretch": "stretch",
    "stretching": "stretch",
    "bend": "bend",
    "twist": "twist",
    "rotation": "rotation",
    "cartwheel": "cartwheel",

    # Vietnamese common terms → English canonical
    "gập bụng": "crunch",
    "chống đẩy": "push up",
    "ngồi xổm": "squat",
    "nhảy dây": "jump",
    "chạy bộ": "run",
}

# Pre-sorted by phrase length (longest first) for greedy matching
_SORTED_VERBS = sorted(VERB_MAP.items(), key=lambda x: -len(x[0]))


class KeywordExtractor:
    """Extract exercise verbs from queries using dictionary lookup. Zero LLM calls."""

    DEFAULT_REPS = 20  # ~5 seconds at 160 frames

    def __init__(self, exercise_detector=None):
        """Initialize with optional ExerciseDetector for fuzzy fallback.

        Args:
            exercise_detector: Optional ExerciseDetector instance.
                              If provided, used as a fallback when dictionary
                              lookup fails (still zero LLM calls — it uses rapidfuzz).
        """
        self._exercise_detector = exercise_detector
        logger.info(
            "KeywordExtractor initialized with %d verb mappings (fuzzy_fallback=%s)",
            len(VERB_MAP),
            exercise_detector is not None,
        )

    def extract(self, query: str) -> Optional[str]:
        """Extract primary exercise verb from query.

        Args:
            query: Raw user query string.

        Returns:
            Canonical verb string (e.g. "squat", "chin tuck"), or None.
        """
        if not query:
            return None

        q = query.lower().strip()

        # 1) Dictionary lookup (longest match first)
        for phrase, verb in _SORTED_VERBS:
            if phrase in q:
                logger.debug("KeywordExtractor: dict match '%s' → '%s'", phrase, verb)
                return verb

        # 2) Fuzzy fallback via ExerciseDetector (still zero LLM)
        if self._exercise_detector is not None:
            try:
                detected = self._exercise_detector.detect_exercise(query)
                if detected:
                    logger.debug(
                        "KeywordExtractor: fuzzy fallback '%s' → '%s'",
                        query[:40], detected,
                    )
                    return detected
            except Exception:
                pass

        logger.debug("KeywordExtractor: no verb found in '%s'", query[:40])
        return None

    def to_motion_prompt(self, verb: str, reps: int = None) -> str:
        """Format verb into a DART-compatible motion description.

        Args:
            verb: Canonical exercise verb (e.g. "squat").
            reps: Frame count hint. Defaults to DEFAULT_REPS (20 ≈ 5s).

        Returns:
            Motion prompt string like "a person performs a squat".
        """
        r = reps or self.DEFAULT_REPS
        prompt = f"a person performs {verb}"
        logger.debug("KeywordExtractor: motion prompt = '%s' (reps=%d)", prompt, r)
        return prompt

    def extract_and_prompt(self, query: str) -> Optional[str]:
        """Convenience: extract verb + build motion prompt in one call.

        Returns None if no verb is found.
        """
        verb = self.extract(query)
        if verb:
            return self.to_motion_prompt(verb)
        return None
