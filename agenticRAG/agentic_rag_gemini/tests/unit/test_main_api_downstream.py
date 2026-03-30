import unittest

from services.main_api_downstream import detect_query_language, resolve_motion_duration_seconds


class TestMainApiDownstream(unittest.TestCase):
    def test_detect_query_language(self) -> None:
        self.assertEqual(detect_query_language("Hello, how are you?"), "en")
        self.assertEqual(detect_query_language("xin chao bạn, cảm ơn"), "vi")
        self.assertEqual(detect_query_language("こんにちは"), "jp")

    def test_resolve_motion_duration_seconds_bounds(self) -> None:
        self.assertEqual(resolve_motion_duration_seconds({}, 12.0), 12.0)
        self.assertEqual(resolve_motion_duration_seconds({"duration_seconds": 0}, 12.0), 1.0)
        self.assertEqual(resolve_motion_duration_seconds({"duration_seconds": 999}, 12.0), 120.0)


if __name__ == "__main__":
    unittest.main()
