from typing import Dict
import re


class EmotionStage:

    def __init__(self):
        self.emotion_keywords = {
            "happy": [
                "happy", "great", "good", "awesome",
                "amazing", "fantastic", "glad", "love"
            ],
            "sad": [
                "sad", "down", "unhappy", "depressed",
                "upset", "hurt", "cry", "lonely"
            ],
            "angry": [
                "angry", "mad", "furious", "annoyed",
                "hate", "ridiculous", "stupid"
            ],
            "frustrated": [
                "frustrated", "tired of", "sick of",
                "why does this", "doesn't work"
            ],
            "anxious": [
                "worried", "nervous", "anxious",
                "scared", "afraid", "stress"
            ],
            "excited": [
                "excited", "can't wait", "so ready",
                "thrilled"
            ]
        }

    def process(self, text: str) -> Dict:
        text = text.lower()

        scores = {emotion: 0 for emotion in self.emotion_keywords}

        for emotion, keywords in self.emotion_keywords.items():
            for keyword in keywords:
                if re.search(rf"\b{re.escape(keyword)}\b", text):
                    scores[emotion] += 1

        detected_emotion = "neutral"
        confidence = 0.0

        if any(scores.values()):
            detected_emotion = max(scores, key=scores.get)
            confidence = scores[detected_emotion] / sum(scores.values())

        return {
            "emotion": detected_emotion,
            "confidence": round(confidence, 2)
        }