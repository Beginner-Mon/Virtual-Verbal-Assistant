# def normalize(action: dict) -> dict:
#     """
#     Normalize structured LLM action output into
#     deterministic numeric parameters for DART.
#     """

#     # Canonical allowed actions
#     allowed_actions = {
#         "wave",
#         "jump",
#         "clap",
#         "walk",
#         "point",
#         "idle"
#     }

#     # Allowed body parts
#     allowed_body_parts = {
#         "left_hand",
#         "right_hand",
#         "both_hands",
#         "head",
#         "full_body"
#     }

#     # Speed mapping
#     speed_map = {
#         "slow": 0.3,
#         "medium": 0.6,
#         "fast": 0.9
#     }

#     # Emotion intensity mapping
#     emotion_map = {
#         "neutral": 0.5,
#         "happy": 0.8,
#         "sad": 0.4,
#         "angry": 0.9
#     }

#     # ---- Sanitize inputs ----
#     action_type = str(action.get("type", "idle")).lower().strip()
#     body_part = str(action.get("body_part", "full_body")).lower().strip()
#     speed = str(action.get("speed", "medium")).lower().strip()
#     emotion = str(action.get("emotion", "neutral")).lower().strip()

#     # ---- Validate action ----
#     if action_type not in allowed_actions:
#         action_type = "idle"

#     # ---- Validate body part ----
#     if body_part not in allowed_body_parts:
#         body_part = "full_body"

#     # ---- Map qualitative → numeric ----
#     speed_value = speed_map.get(speed, 0.6)
#     emotion_intensity = emotion_map.get(emotion, 0.5)

#     normalized = {
#         "action_type": action_type,
#         "body_part": body_part,
#         "speed_value": speed_value,
#         "emotion_intensity": emotion_intensity
#     }

#     return normalized