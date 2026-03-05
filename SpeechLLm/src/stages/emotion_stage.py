from typing import Dict
import re


class EmotionStage:
    """
    Improved heuristic-based emotion detector.
    Adds:
    - Weighted keywords
    - Phrase matching
    - Negation handling
    - Intensity boost
    """

    def __init__(self):
        # Weighted keywords (word/phrase -> weight)
        self.emotion_keywords = {
            "happy": {
                "happy": 2,
                "great": 2,
                "awesome": 3,
                "amazing": 3,
                "fantastic": 3,
                "glad": 2,
                "love": 2
            },
            "sad": {
                "sad": 2,
                "down": 2,
                "unhappy": 2,
                "depressed": 4,
                "upset": 2,
                "hurt": 2,
                "cry": 3,
                "lonely": 3,
                "nothing works": 3,
                "failed": 2
            },
            "angry": {
                "angry": 3,
                "mad": 2,
                "furious": 4,
                "annoyed": 2,
                "hate": 3,
                "stupid": 2,
                "ridiculous": 2
            },
            "frustrated": {
                "frustrated": 3,
                "tired of": 3,
                "sick of": 3,
                "why does this": 2,
                "doesn't work": 3,
                "can't fix": 2
            },
            "anxious": {
                "worried": 2,
                "nervous": 2,
                "anxious": 3,
                "scared": 3,
                "afraid": 3,
                "stress": 2,
                "overthinking": 2
            },
            "excited": {
                "excited": 3,
                "can't wait": 3,
                "so ready": 2,
                "thrilled": 3,
                "let's go": 2
            }
        }

        self.negations = ["not", "never", "no", "don't", "doesn't", "didn't", "isn't", "wasn't"]

    def _contains_negation(self, text: str, keyword: str) -> bool:
        """
        Check if keyword appears near a negation (simple window-based check).
        """
        pattern = rf"(?:{'|'.join(self.negations)})\s+{re.escape(keyword)}"
        return re.search(pattern, text) is not None

    def _intensity_boost(self, original_text: str) -> float:
        """
        Detect intensity via punctuation or caps.
        """
        boost = 1.0

        if "!!!" in original_text:
            boost += 0.5

        # If many uppercase words → intensity
        words = original_text.split()
        caps_words = [w for w in words if w.isupper() and len(w) > 2]
        if len(caps_words) >= 2:
            boost += 0.5

        return boost

    def process(self, text: str) -> Dict:
        original_text = text
        text = text.lower()

        scores = {emotion: 0.0 for emotion in self.emotion_keywords}

        for emotion, keywords in self.emotion_keywords.items():
            for keyword, weight in keywords.items():
                # Phrase or word boundary match
                if re.search(rf"\b{re.escape(keyword)}\b", text):
                    if self._contains_negation(text, keyword):
                        # Negation flips weight slightly negative
                        scores[emotion] -= weight * 0.8
                    else:
                        scores[emotion] += weight

        # Apply intensity boost
        boost = self._intensity_boost(original_text)
        for emotion in scores:
            scores[emotion] *= boost

        # Determine top emotion
        detected_emotion = "neutral"
        confidence = 0.0

        positive_scores = {k: v for k, v in scores.items() if v > 0}

        if positive_scores:
            detected_emotion = max(positive_scores, key=positive_scores.get)
            total_score = sum(positive_scores.values())
            confidence = positive_scores[detected_emotion] / total_score

        return {
            "emotion": detected_emotion,
            "confidence": round(confidence, 2),
            "scores": {k: round(v, 2) for k, v in scores.items()}
        }