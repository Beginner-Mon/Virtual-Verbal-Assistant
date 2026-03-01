"""Response template generator for AgenticRAG.

This module parses orchestrator decisions and generates structured prompts
for downstream services (motion generation and voice synthesis).
"""

import logging
import re
from typing import Optional, Dict, Any

from pydantic import BaseModel, Field
from utils.logger import get_logger

logger = get_logger(__name__)


# Define models locally to avoid circular import with api_server.py
class MotionPrompt(BaseModel):
    """Motion generation prompt."""

    description: str = Field(..., description="Natural language motion description")
    primitive_sequence: str = Field(
        ...,
        description='Primitive action sequence (e.g., "walk*20,turn_left*10")',
    )
    num_frames: int = Field(160, description="Number of frames to generate")
    fps: int = Field(30, description="Frames per second")


class VoicePrompt(BaseModel):
    """Voice synthesis prompt."""

    text: str = Field(..., description="Text to synthesize")
    emotion: Optional[str] = Field(None, description="Detected or requested emotion")
    duration_estimate_seconds: float = Field(5.0, description="Estimated audio duration")



class ResponseTemplateGenerator:
    """Generate structured prompts from orchestrator decisions."""

    # Motion primitive database
    MOTION_PRIMITIVES = {
        "walk": {"num_primitives": 20, "description": "Walking forward"},
        "run": {"num_primitives": 15, "description": "Running forward"},
        "jog": {"num_primitives": 18, "description": "Jogging forward"},
        "stand": {"num_primitives": 5, "description": "Standing idle"},
        "sit": {"num_primitives": 8, "description": "Sitting down"},
        "jump": {"num_primitives": 6, "description": "Jumping in place"},
        "turn_left": {"num_primitives": 8, "description": "Turning left"},
        "turn_right": {"num_primitives": 8, "description": "Turning right"},
        "wave": {"num_primitives": 6, "description": "Waving hand"},
        "raise_arm": {"num_primitives": 5, "description": "Raising arm"},
        "lower_arm": {"num_primitives": 5, "description": "Lowering arm"},
        "stretch": {"num_primitives": 10, "description": "Stretching"},
        "dance": {"num_primitives": 25, "description": "Dancing"},
        "kick": {"num_primitives": 8, "description": "Kicking"},
        "punch": {"num_primitives": 6, "description": "Punching"},
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
    ) -> Optional[MotionPrompt]:
        """Generate a motion prompt from query and action plan.

        Args:
            query: Original user query
            action_plan: Orchestrator action plan
            response: Generated response text

        Returns:
            MotionPrompt or None if motion generation not applicable
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

            # Get primitive sequence
            primitive_sequence = self._generate_primitive_sequence(motion_type, query)

            # Calculate frame count: 8 frames per primitive at 30 fps
            num_primitives = len(primitive_sequence.split(","))
            num_frames = num_primitives * 8

            motion_prompt = MotionPrompt(
                description=f"Motion for: {query}",
                primitive_sequence=primitive_sequence,
                num_frames=num_frames,
                fps=30,
            )

            logger.info(f"Generated motion prompt: {primitive_sequence}")
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
    ) -> Optional[VoicePrompt]:
        """Generate a voice prompt for TTS synthesis.

        Args:
            text: Text to synthesize
            query: Original user query
            user_id: User identifier
            action_plan: Orchestrator action plan

        Returns:
            VoicePrompt for voice synthesis
        """
        try:
            # Detect emotion from text and query
            emotion = self._detect_emotion(text, query)

            # Estimate duration: roughly 150 words per minute = 0.4 seconds per word
            word_count = len(text.split())
            estimated_duration = max(2.0, word_count * 0.4)

            voice_prompt = VoicePrompt(
                text=text,
                emotion=emotion,
                duration_estimate_seconds=estimated_duration,
            )

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
        """Generate a primitive sequence string for DART motion generation.

        Format: "action1*num1,action2*num2,..."

        Args:
            motion_type: Primary motion type
            query: Original query for context

        Returns:
            Primitive sequence string
        """
        sequence = []
        query_lower = query.lower()

        # Handle basic motion types
        if motion_type in self.MOTION_PRIMITIVES:
            base_primitives = self.MOTION_PRIMITIVES[motion_type]["num_primitives"]
            sequence.append(f"{motion_type}*{base_primitives}")

        # Handle composite motions
        elif motion_type == "walk_turn":
            sequence.append("walk*15")
            sequence.append("turn_left*8")
            sequence.append("walk*15")

        elif motion_type == "walk_run":
            sequence.append("walk*10")
            sequence.append("run*12")

        else:
            # Default to walk if motion type not recognized
            sequence.append("walk*20")

        # Add secondary actions if detected in query
        if "turn" in query_lower and "turn" not in motion_type:
            sequence.append("turn_left*5")

        if "wave" in query_lower and "wave" not in motion_type:
            sequence.append("wave*4")

        if "jump" in query_lower and "jump" not in motion_type:
            sequence.append("jump*3")

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
