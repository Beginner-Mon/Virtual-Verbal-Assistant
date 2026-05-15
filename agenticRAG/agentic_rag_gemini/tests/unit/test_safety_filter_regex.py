import statistics
import time
import unittest
import pathlib
import importlib.util


def _load_safety_filter_class():
    project_root = pathlib.Path(__file__).resolve().parents[2]
    module_path = project_root / "agents" / "safety_filter.py"
    spec = importlib.util.spec_from_file_location("isolated_safety_filter", module_path)
    module = importlib.util.module_from_spec(spec)
    assert spec and spec.loader
    spec.loader.exec_module(module)
    return module.SafetyFilter


SafetyFilter = _load_safety_filter_class()


class TestSafetyFilterRegex(unittest.TestCase):
    def setUp(self) -> None:
        self.filter = SafetyFilter()

    def test_blocks_medical_emergency(self) -> None:
        result = self.filter.check_query_safety("I have severe chest pain and cannot breathe")
        self.assertFalse(result["is_safe"])
        self.assertIn("Medical emergency", result["reason"])

    def test_blocks_violence_self_harm(self) -> None:
        result = self.filter.check_query_safety("I want to hurt myself")
        self.assertFalse(result["is_safe"])
        self.assertIn("Violence/self-harm", result["reason"])

    def test_allows_normal_rehab_query(self) -> None:
        result = self.filter.check_query_safety("What are good neck mobility exercises for desk workers?")
        self.assertTrue(result["is_safe"])
        self.assertEqual(result["reason"], "Safe")

    def test_regex_path_is_low_latency(self) -> None:
        samples = [
            "show me squat form",
            "how to improve hip mobility",
            "suggest exercises for posture",
            "what is lumbar stabilization",
            "can you explain hamstring stretch",
        ]
        durations_ms = []
        for q in samples:
            start = time.perf_counter()
            self.filter.check_query_safety(q)
            durations_ms.append((time.perf_counter() - start) * 1000)

        median_ms = statistics.median(durations_ms)
        self.assertLess(median_ms, 10.0)


if __name__ == "__main__":
    unittest.main()
