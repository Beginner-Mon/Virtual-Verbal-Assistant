"""Exercise Detector - Hybrid Entity Extraction using Dictionary + Fuzzy Matching.

This module provides fast and accurate exercise detection using dictionary lookup
combined with fuzzy matching, reducing the workload on the small LLM orchestrator.
"""

import os
import re
import logging
from typing import Optional, List, Set
from rapidfuzz import fuzz, process

from utils.logger import get_logger

logger = get_logger(__name__)


class ExerciseDetector:
    """Fast exercise detection using dictionary lookup and fuzzy matching."""
    
    def __init__(self, similarity_threshold: float = 80.0):
        """Initialize the exercise detector.
        
        Args:
            similarity_threshold: Minimum similarity score (0-100) for exercise detection
        """
        self.similarity_threshold = similarity_threshold
        self.exercise_list: List[str] = []
        self.exercise_set: Set[str] = set()
        
        # Load exercises from documents.txt
        self._load_exercises()
        
        logger.info(f"ExerciseDetector initialized with {len(self.exercise_list)} exercises")
        logger.info(f"Similarity threshold: {self.similarity_threshold}%")
    
    def _load_exercises(self) -> None:
        """Load exercise names from documents.txt."""
        try:
            # Path to documents.txt
            documents_path = os.path.join(
                os.path.dirname(os.path.dirname(__file__)), 
                "data", "knowledge_base",
                "documents.txt"
            )
            
            if not os.path.exists(documents_path):
                logger.warning(f"documents.txt not found at {documents_path}")
                self._load_fallback_exercises()
                return
            
            # Read and parse documents.txt
            with open(documents_path, 'r', encoding='utf-8') as f:
                content = f.read()
            
            # Extract exercise names using improved patterns
            exercises = self._extract_exercises_from_text(content)
            
            # Add common exercise variations
            exercises.extend(self._get_common_variations())
            
            # Clean and deduplicate
            self.exercise_list = sorted(list(set(exercises)))
            self.exercise_set = set(self.exercise_list)
            
            logger.info(f"Loaded {len(self.exercise_list)} exercises from documents.txt")
            
        except Exception as e:
            logger.error(f"Failed to load exercises from documents.txt: {e}")
            self._load_fallback_exercises()
    
    def _extract_exercises_from_text(self, content: str) -> List[str]:
        """Extract exercise names from document content."""
        exercises = []
        
        # Improved patterns for exercise extraction
        patterns = [
            r'(?:how to do|how to perform|show me how to|teach me how to|demonstrate)\s+([a-zA-Z\s-]+?)(?:\s+exercise|\s+workout|\s+movement)?[\.!?]',
            r'(?:exercise|workout|movement):\s*([a-zA-Z\s-]+?)[\.!?]',
            r'^([a-zA-Z\s-]+?)(?:\s+exercise|\s+workout|\s+movement)[\.!?]',
            r'([a-zA-Z\s-]+?)\s+(?:exercise|workout|movement)[\.!?]',
            r'perform\s+([a-zA-Z\s-]+?)[\.!?]',
            r'demonstrate\s+([a-zA-Z\s-]+?)[\.!?]',
        ]
        
        found_exercises = set()
        
        for pattern in patterns:
            matches = re.findall(pattern, content, re.IGNORECASE | re.MULTILINE)
            for match in matches:
                exercise = self._clean_exercise_name(match.strip())
                if exercise and self._is_valid_exercise(exercise):
                    found_exercises.add(exercise)
        
        exercises.extend(list(found_exercises))
        return exercises
    
    def _clean_exercise_name(self, name: str) -> Optional[str]:
        """Clean and normalize exercise name."""
        if not name:
            return None
        
        # Convert to lowercase and strip
        name = name.lower().strip()
        
        # Remove common prefixes/suffixes
        prefixes_to_remove = ['a ', 'an ', 'the ', 'your ', 'my ', 'our ']
        for prefix in prefixes_to_remove:
            if name.startswith(prefix):
                name = name[len(prefix):]
        
        # Remove trailing punctuation
        name = re.sub(r'[\.!?]+$', '', name)
        
        # Remove extra spaces
        name = re.sub(r'\s+', ' ', name)
        
        return name if len(name) >= 3 else None
    
    def _is_valid_exercise(self, name: str) -> bool:
        """Check if exercise name is valid."""
        words = name.split()
        
        # Basic validation
        if (len(words) < 1 or len(words) > 4 or  # Reasonable word count
            len(name) < 3 or len(name) > 30 or   # Reasonable length
            name.startswith(('-', '--', '...', '___')) or  # Bad starts
            any(char in name for char in ['{', '}', '[', ']', '<', '>', '|', '\\']) or  # Special chars
            name.endswith(('type', 'style', 'form', 'way', 'method', 'technique'))):  # Generic words
            return False
        
        return True
    
    def _get_common_variations(self) -> List[str]:
        """Get common exercise variations and synonyms."""
        return [
            # Basic exercises with variations
            "push up", "pushup", "push-ups",
            "squat", "squats", "bodyweight squat",
            "plank", "planks", "front plank", "side plank",
            "crunch", "crunches", "abdominal crunch", "sit up", "situps",
            "burpee", "burpees", "burpee exercise",
            "lunges", "lunge", "walking lunge", "reverse lunge",
            "jumping jacks", "jumping jack",
            "chin tuck", "chin tucks", "neck stretch",
            "shoulder roll", "shoulder rolls", "shoulder stretch",
            
            # Weight training
            "deadlift", "deadlifts", "deadlift exercise",
            "bench press", "bench press exercise",
            "bicep curl", "bicep curls", "biceps curl",
            "tricep extension", "tricep extensions",
            
            # Modern exercises
            "kettlebell swing", "kettlebell swings", "kbell swing",
            "turkish getup", "turkish get up", "get up",
            "box jump", "box jumps", "plyometric box jump",
            
            # Cardio
            "running", "run", "jogging", "jog",
            "cycling", "bike", "biking",
            "swimming", "swim",
        ]
    
    def _load_fallback_exercises(self) -> None:
        """Load fallback exercise list if documents.txt is not available."""
        self.exercise_list = [
            "push up", "squat", "plank", "jumping jacks", "chin tuck",
            "shoulder roll", "lunges", "burpees", "deadlift", "bench press",
            "crunch", "kettlebell swing", "turkish getup", "box jump",
            "bicep curls", "tricep extension", "running", "cycling"
        ]
        self.exercise_set = set(self.exercise_list)
        logger.warning("Using fallback exercise list")
    
    def detect_exercise(self, query: str) -> Optional[str]:
        """Detect exercise name from user query using fuzzy matching.
        
        Args:
            query: User query string
            
        Returns:
            Detected exercise name or None if no exercise found
        """
        if not query or not self.exercise_list:
            return None
        
        # Preprocess query
        query_clean = query.lower().strip()
        
        # First try exact matches in exercise set (highest priority)
        for exercise in self.exercise_set:
            if exercise in query_clean:
                logger.debug(f"Exact match found: '{exercise}' in query")
                return exercise
        
        # Debug: Show what we're working with
        logger.debug(f"Query clean: '{query_clean}'")
        logger.debug(f"Exercise set contains: {list(self.exercise_set)[:10]}...")  # Show first 10
        
        # Try fuzzy matching for partial matches
        # Extract potential exercise phrases from query
        potential_phrases = self._extract_potential_phrases(query_clean)
        
        best_match = None
        best_score = 0
        
        for phrase in potential_phrases:
            # Use rapidfuzz for fuzzy matching
            result = process.extractOne(
                phrase, 
                self.exercise_list,
                scorer=fuzz.token_set_ratio,
                score_cutoff=self.similarity_threshold
            )
            
            if result and result[1] > best_score:
                best_match = result[0]
                best_score = result[1]
        
        if best_match:
            logger.debug(f"Fuzzy match found: '{best_match}' with score {best_score:.1f}%")
            return best_match
        
        logger.debug("No exercise detected in query")
        return None
    
    def _extract_potential_phrases(self, query: str) -> List[str]:
        """Extract potential exercise phrases from query."""
        phrases = []
        
        # Common exercise-related patterns
        patterns = [
            r'(?:how to do|how to perform|show me how to|teach me how to|demonstrate)\s+([a-z\s-]+)',
            r'(?:do|perform|execute|try|practice)\s+([a-z\s-]+)',
            r'([a-z\s-]+)\s+(?:exercise|workout|movement)',
            r'([a-z\s-]+)\s+(?:properly|correctly|right|form)',
        ]
        
        for pattern in patterns:
            matches = re.findall(pattern, query, re.IGNORECASE)
            for match in matches:
                phrase = match.strip()
                if 2 <= len(phrase.split()) <= 4:  # Reasonable phrase length
                    phrases.append(phrase)
        
        # Also add individual words and 2-word combinations
        words = query.split()
        
        # Single words (might be exercise names)
        for word in words:
            if len(word) >= 3:  # Minimum length
                phrases.append(word)
        
        # Two-word combinations
        for i in range(len(words) - 1):
            phrase = f"{words[i]} {words[i+1]}"
            if len(phrase) <= 20:  # Reasonable length
                phrases.append(phrase)
        
        # Three-word combinations
        for i in range(len(words) - 2):
            phrase = f"{words[i]} {words[i+1]} {words[i+2]}"
            if len(phrase) <= 25:  # Reasonable length
                phrases.append(phrase)
        
        # Remove duplicates and return
        return list(set(phrases))
    
    def get_exercise_count(self) -> int:
        """Get the number of loaded exercises."""
        return len(self.exercise_list)
    
    def get_sample_exercises(self, count: int = 10) -> List[str]:
        """Get sample exercises from the loaded list."""
        return self.exercise_list[:count]
    
    def is_exercise_known(self, exercise: str) -> bool:
        """Check if an exercise is in the known exercise list."""
        return exercise.lower().strip() in self.exercise_set


# Singleton instance for performance
_detector_instance: Optional[ExerciseDetector] = None


def get_exercise_detector(similarity_threshold: float = 80.0) -> ExerciseDetector:
    """Get singleton exercise detector instance.
    
    Args:
        similarity_threshold: Similarity threshold for fuzzy matching
        
    Returns:
        ExerciseDetector instance
    """
    global _detector_instance
    
    if _detector_instance is None:
        _detector_instance = ExerciseDetector(similarity_threshold)
    
    return _detector_instance


def detect_exercise(query: str, similarity_threshold: float = 80.0) -> Optional[str]:
    """Convenience function to detect exercise from query.
    
    Args:
        query: User query string
        similarity_threshold: Similarity threshold for fuzzy matching
        
    Returns:
        Detected exercise name or None
    """
    detector = get_exercise_detector(similarity_threshold)
    return detector.detect_exercise(query)
