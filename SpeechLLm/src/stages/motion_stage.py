from src.utils.action_normalizer import ActionNormalizer


class MotionStage:
    def __init__(self):
        self.normalizer = ActionNormalizer()

    def process(self, action_text: str):
        normalized = self.normalizer.normalize(action_text)

        # For now just return normalized result
        return {
            "normalized_action": normalized
        }