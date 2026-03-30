"""Response template generator for AgenticRAG.

This module parses orchestrator decisions and generates structured prompts
for downstream services (motion generation and voice synthesis).
"""

import logging
import re
from typing import Optional, Dict, Any

from utils.logger import get_logger

logger = get_logger(__name__)





class ResponseTemplateGenerator:
    """Generate structured prompts from orchestrator decisions."""

    # Motion primitive database
    MOTION_PRIMITIVES = {
        "walk": {"num_primitives": 10, "description": "Walking forward"},
        "run": {"num_primitives": 10, "description": "Running forward"},
        "jog": {"num_primitives": 10, "description": "Jogging forward"},
        "stand": {"num_primitives": 10, "description": "Standing idle"},
        "sit": {"num_primitives": 10, "description": "Sitting down"},
        "jump": {"num_primitives": 10, "description": "Jumping in place"},
        "turn_left": {"num_primitives": 10, "description": "Turning left"},
        "turn_right": {"num_primitives": 10, "description": "Turning right"},
        "wave": {"num_primitives": 10, "description": "Waving hand"},
        "raise_arm": {"num_primitives": 10, "description": "Raising arm"},
        "lower_arm": {"num_primitives": 10, "description": "Lowering arm"},
        "stretch": {"num_primitives": 10, "description": "Stretching"},
        "dance": {"num_primitives": 10, "description": "Dancing"},
        "kick": {"num_primitives": 10, "description": "Kicking"},
        "punch": {"num_primitives": 10, "description": "Punching"},
    }

    # Emotion mappings for voice
    EMOTION_KEYWORDS = {
        "happy": [
            "happy",
            "joyful",
            "cheerful",
            "excited",
            "delighted",
            "pleased",
            "great",
            "wonderful",
            "fantastic",
        ],
        "sad": [
            "sad",
            "unhappy",
            "depressed",
            "disappointed",
            "down",
            "sorry",
            "terrible",
            "awful",
        ],
        "angry": ["angry", "furious", "mad", "irritated", "annoyed", "upset"],
        "calm": ["calm", "peaceful", "serene", "relaxed", "chill"],
        "surprised": ["surprised", "amazed", "shocked", "wow", "incredible"],
    }

    def __init__(self):
        """Initialize template generator."""
        logger.info("Initializing ResponseTemplateGenerator")

    def generate_motion_prompt(
        self,
        query: str,
        action_plan: Dict[str, Any],
        response: str,
    ) -> Optional[Dict[str, Any]]:
        """Generate a motion prompt from query and action plan.

        Returns a plain dict so api_server.py can construct its own
        typed MotionPrompt, avoiding cross-module Pydantic type mismatches.
        """
        try:
            # Extract motion type from action plan parameters
            parameters = action_plan.get("parameters", {})
            motion_type = parameters.get("motion_type")

            if not motion_type:
                # Try to extract from query using keyword matching
                motion_type = self._extract_motion_type_from_query(query)

            if not motion_type:
                logger.info("No motion type detected in query")
                return None

            # Estimate duration from primitive metadata while keeping a plain prompt.
            # DART expects 16 frames per primitive (chunk). 10 primitives = 160 frames = 5.33 seconds
            primitive_count = self.MOTION_PRIMITIVES.get(motion_type, {}).get("num_primitives", 10)
            duration_seconds = max(2.0, (primitive_count * 16.0) / 30.0)
            num_frames = int(round(duration_seconds * 30.0))

            motion_prompt = {
                "description": f"Motion for: {query}",
                "duration_seconds": duration_seconds,
                "num_frames": num_frames,
                "fps": 30,
            }

            logger.info(f"Generated motion prompt with duration_seconds={duration_seconds:.2f}")
            return motion_prompt

        except Exception as e:
            logger.error(f"Error generating motion prompt: {e}")
            return None

    def generate_voice_prompt(
        self,
        text: str,
        query: str,
        user_id: str,
        action_plan: Dict[str, Any],
    ) -> Optional[Dict[str, Any]]:
        """Generate a voice prompt for TTS synthesis.

        Returns a plain dict so api_server.py can construct its own
        typed VoicePrompt, avoiding cross-module Pydantic type mismatches.
        """
        try:
            # Detect emotion from text and query
            emotion = self._detect_emotion(text, query)

            # Estimate duration: roughly 150 words per minute = 0.4 seconds per word
            word_count = len(text.split())
            estimated_duration = max(2.0, word_count * 0.4)

            voice_prompt = {
                "text": text,
                "emotion": emotion,
                "duration_estimate_seconds": estimated_duration,
            }

            logger.info(f"Generated voice prompt with emotion: {emotion}")
            return voice_prompt

        except Exception as e:
            logger.error(f"Error generating voice prompt: {e}")
            return None

    def _extract_motion_type_from_query(self, query: str) -> Optional[str]:
        """Extract motion type from user query using keyword matching.

        Args:
            query: User query text

        Returns:
            Motion type identifier or None
        """
        query_lower = query.lower()

        # Check for direct motion type matches
        for motion_type in self.MOTION_PRIMITIVES.keys():
            if motion_type in query_lower:
                return motion_type

        # Check for composite patterns (e.g., "walk and turn")
        if "walk" in query_lower:
            if "turn" in query_lower:
                return "walk_turn"
            elif "run" in query_lower:
                return "walk_run"

        return None

    def _generate_primitive_sequence(self, motion_type: str, query: str) -> str:
        """Generate a plain action sequence string (no repetition syntax).

        Args:
            motion_type: Primary motion type
            query: Original query for context

        Returns:
            Comma-separated action sequence string
        """
        sequence = []
        query_lower = query.lower()

        # Handle basic motion types
        if motion_type in self.MOTION_PRIMITIVES:
            sequence.append(motion_type)

        # Handle composite motions
        elif motion_type == "walk_turn":
            sequence.append("walk")
            sequence.append("turn left")
            sequence.append("walk")

        elif motion_type == "walk_run":
            sequence.append("walk")
            sequence.append("run")

        else:
            # Default to walk if motion type not recognized
            sequence.append("walk")

        # Add secondary actions if detected in query
        if "turn" in query_lower and "turn" not in motion_type:
            sequence.append("turn left")

        if "wave" in query_lower and "wave" not in motion_type:
            sequence.append("wave")

        if "jump" in query_lower and "jump" not in motion_type:
            sequence.append("jump")

        return ",".join(sequence)

    def _detect_emotion(self, text: str, query: str) -> Optional[str]:
        """Detect emotion from text and query.

        Args:
            text: Response text
            query: Original query

        Returns:
            Emotion identifier or None
        """
        combined = (text + " " + query).lower()

        # Check for emotion keywords
        for emotion, keywords in self.EMOTION_KEYWORDS.items():
            for keyword in keywords:
                if keyword in combined:
                    return emotion

        # Default to neutral
        return None

    def update_motion_primitive_database(
        self, motion_type: str, num_primitives: int, description: str
    ) -> None:
        """Update or add a motion primitive type.

        Args:
            motion_type: Motion type identifier
            num_primitives: Number of primitives for this motion
            description: Description of motion
        """
        self.MOTION_PRIMITIVES[motion_type] = {
            "num_primitives": num_primitives,
            "description": description,
        }
        logger.info(f"Updated motion primitive: {motion_type}")
