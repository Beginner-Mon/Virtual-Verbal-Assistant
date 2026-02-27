"""Response validators for ensuring safe and appropriate outputs."""

from typing import List, Dict, Any
import re

from config import get_config
from utils.logger import get_logger


logger = get_logger(__name__)


class ResponseValidator:
    """Validates LLM responses for safety, relevance, and quality."""
    
    def __init__(self, config=None):
        """Initialize validator.
        
        Args:
            config: Configuration object. If None, loads from default config.
        """
        self.config = config or get_config().validation
        
        # Unsafe keywords that indicate medical claims
        self.unsafe_keywords = self.config.unsafe_keywords or [
            "diagnosis", "diagnose", "treatment plan", "prescription",
            "medication", "medicine", "drug", "surgery", "surgical",
            "disease", "disorder", "condition requires", "you have",
            "medical condition", "clinical", "therapy session"
        ]
        
        logger.info("Initialized ResponseValidator")
    
    def validate_response(
        self,
        response: str,
        query: str = None
    ) -> Dict[str, Any]:
        """Validate a response for safety and quality.
        
        Args:
            response: Response text to validate
            query: Original query (optional, for relevance check)
            
        Returns:
            Dictionary with validation results
        """
        issues = []
        
        # Check 1: Safety
        if self.config.enable_safety_check:
            safety_issues = self._check_safety(response)
            issues.extend(safety_issues)
        
        # Check 2: Length
        length_issues = self._check_length(response)
        issues.extend(length_issues)
        
        # Check 3: Relevance (if query provided)
        if query and self.config.enable_relevance_check:
            relevance_issues = self._check_relevance(response, query)
            issues.extend(relevance_issues)
        
        # Determine if valid
        is_valid = len(issues) == 0
        
        result = {
            "is_valid": is_valid,
            "issues": issues,
            "checks": {
                "safety": self.config.enable_safety_check,
                "length": True,
                "relevance": self.config.enable_relevance_check and query is not None
            }
        }
        
        if not is_valid:
            logger.warning(f"Response validation failed: {issues}")
        
        return result
    
    def _check_safety(self, response: str) -> List[str]:
        """Check response for safety concerns.
        
        Args:
            response: Response text
            
        Returns:
            List of safety issues found
        """
        issues = []
        response_lower = response.lower()
        
        # Check for unsafe keywords
        found_keywords = []
        for keyword in self.unsafe_keywords:
            if keyword.lower() in response_lower:
                found_keywords.append(keyword)
        
        if found_keywords:
            issues.append(
                f"Response contains medical/clinical language: {', '.join(found_keywords[:3])}"
            )
        
        # Check for prescriptive language
        prescriptive_patterns = [
            r"you must\b",
            r"you should definitely\b",
            r"this will cure\b",
            r"guaranteed to\b",
            r"proven to treat\b"
        ]
        
        for pattern in prescriptive_patterns:
            if re.search(pattern, response_lower):
                issues.append(f"Response uses overly prescriptive language: '{pattern}'")
                break
        
        # Check for diagnosis language
        diagnosis_patterns = [
            r"you (have|suffer from|are experiencing) [a-z]+ (disease|disorder|condition)",
            r"this (is|could be) [a-z]+ (disease|disorder|syndrome)",
            r"symptoms of [a-z]+ (disease|disorder)"
        ]
        
        for pattern in diagnosis_patterns:
            if re.search(pattern, response_lower):
                issues.append("Response appears to provide medical diagnosis")
                break
        
        return issues
    
    def _check_length(self, response: str) -> List[str]:
        """Check response length is appropriate.
        
        Args:
            response: Response text
            
        Returns:
            List of length issues found
        """
        issues = []
        
        length = len(response)
        
        if length < self.config.min_response_length:
            issues.append(
                f"Response too short ({length} chars, minimum {self.config.min_response_length})"
            )
        
        if length > self.config.max_response_length:
            issues.append(
                f"Response too long ({length} chars, maximum {self.config.max_response_length})"
            )
        
        # Check for empty or whitespace-only response
        if not response.strip():
            issues.append("Response is empty or whitespace only")
        
        return issues
    
    def _check_relevance(self, response: str, query: str) -> List[str]:
        """Check if response is relevant to query.
        
        Args:
            response: Response text
            query: Original query
            
        Returns:
            List of relevance issues found
        """
        issues = []
        
        # Simple keyword overlap check
        query_words = set(query.lower().split())
        response_words = set(response.lower().split())
        
        # Remove common stop words
        stop_words = {'the', 'a', 'an', 'and', 'or', 'but', 'in', 'on', 'at', 'to', 'for',
                      'of', 'with', 'by', 'from', 'as', 'is', 'was', 'are', 'were', 'been',
                      'be', 'have', 'has', 'had', 'do', 'does', 'did', 'will', 'would',
                      'could', 'should', 'may', 'might', 'can', 'i', 'you', 'we', 'they'}
        
        query_content_words = query_words - stop_words
        response_content_words = response_words - stop_words
        
        # Calculate overlap
        if query_content_words:
            overlap = len(query_content_words & response_content_words)
            overlap_ratio = overlap / len(query_content_words)
            
            # Flag if very low overlap (< 20%)
            if overlap_ratio < 0.2:
                issues.append(
                    f"Low relevance: response may not address query (overlap: {overlap_ratio:.1%})"
                )
        
        # Check for generic fallback responses
        generic_phrases = [
            "i'm having trouble",
            "could you rephrase",
            "i don't understand",
            "technical difficulties",
            "try again later"
        ]
        
        response_lower = response.lower()
        if any(phrase in response_lower for phrase in generic_phrases):
            # This might be a fallback, which is okay for error handling
            pass
        
        return issues
    
    def is_safe_for_user(self, response: str) -> bool:
        """Quick check if response is safe for user.
        
        Args:
            response: Response text
            
        Returns:
            True if safe, False if potentially unsafe
        """
        safety_issues = self._check_safety(response)
        return len(safety_issues) == 0
    
    def sanitize_response(self, response: str) -> str:
        """Attempt to sanitize a response by removing unsafe content.
        
        Args:
            response: Original response
            
        Returns:
            Sanitized response
        """
        # This is a simple implementation
        # In production, you'd want more sophisticated sanitization
        
        sanitized = response
        
        # Remove sentences with unsafe keywords
        sentences = response.split('. ')
        safe_sentences = []
        
        for sentence in sentences:
            sentence_lower = sentence.lower()
            contains_unsafe = any(
                keyword.lower() in sentence_lower
                for keyword in self.unsafe_keywords
            )
            
            if not contains_unsafe:
                safe_sentences.append(sentence)
        
        if safe_sentences:
            sanitized = '. '.join(safe_sentences)
            if not sanitized.endswith('.'):
                sanitized += '.'
        else:
            # All sentences were unsafe, return generic message
            sanitized = "I want to provide helpful guidance. Could you clarify what specific advice you're looking for?"
        
        return sanitized


if __name__ == "__main__":
    # Test the validator
    validator = ResponseValidator()
    
    # Test 1: Safe response
    safe_response = "Try gentle neck stretches by slowly tilting your head to each side. Hold for 15-20 seconds. Remember to move slowly and stop if you feel pain."
    result1 = validator.validate_response(safe_response, "What can I do for neck pain?")
    print("Test 1 - Safe response:", result1)
    
    # Test 2: Unsafe response (medical diagnosis)
    unsafe_response = "You have a herniated disc. This requires surgery and prescription medication."
    result2 = validator.validate_response(unsafe_response, "What's wrong with my back?")
    print("\nTest 2 - Unsafe response:", result2)
    
    # Test 3: Too short response
    short_response = "Stretch."
    result3 = validator.validate_response(short_response, "How to relieve pain?")
    print("\nTest 3 - Too short:", result3)
