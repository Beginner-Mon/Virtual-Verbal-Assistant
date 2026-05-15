import unittest

from utils.entity_tags import decode_entity_tags, encode_entity_tags, extract_entity_tags


class TestEntityTags(unittest.TestCase):
    def test_extract_entity_tags(self) -> None:
        tags = extract_entity_tags("I have neck pain and need mobility exercises for posture")
        self.assertIn("neck", tags)
        self.assertIn("pain_relief", tags)
        self.assertIn("mobility", tags)
        self.assertIn("posture", tags)

    def test_encode_decode_roundtrip(self) -> None:
        raw = ["neck", "mobility", "neck", "posture"]
        encoded = encode_entity_tags(raw)
        decoded = decode_entity_tags(encoded)
        self.assertEqual(decoded, ["neck", "mobility", "posture"])

    def test_decode_accepts_list(self) -> None:
        decoded = decode_entity_tags(["back", "rehab", "back"])
        self.assertEqual(decoded, ["back", "rehab"])


if __name__ == "__main__":
    unittest.main()
