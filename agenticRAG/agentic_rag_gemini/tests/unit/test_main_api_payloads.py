import unittest

from schemas.main_api import AnswerResponse
from services.main_api_payloads import answer_to_query_payload, query_to_task_payload, to_progress_stage


class TestMainApiPayloads(unittest.TestCase):
    def test_to_progress_stage_mappings(self) -> None:
        self.assertEqual(to_progress_stage("motion_generation", "processing"), "motion_generation")
        self.assertEqual(to_progress_stage("voice_synthesis", "processing"), "text_ready")
        self.assertEqual(to_progress_stage("anything", "completed"), "completed")
        self.assertEqual(to_progress_stage(None, "failed"), "failed")

    def test_answer_to_query_payload_processing_has_motion_job(self) -> None:
        answer = AnswerResponse(
            request_id="abc123",
            status="processing",
            pending_services=["tts"],
            language="en",
            selected_strategy="visualize_motion",
            progress_stage="motion_generation",
            text_answer="Do this exercise",
            exercises=[{"name": "squat", "reason": "legs"}],
            motion=None,
            tts=None,
            generation_time_ms=123.4,
            errors=None,
            debug={"trace": "ok"},
        )

        payload = answer_to_query_payload(answer, query="help", user_id="u1")
        self.assertEqual(payload["query"], "help")
        self.assertEqual(payload["user_id"], "u1")
        self.assertEqual(payload["language"], "en")
        self.assertIn("motion_job", payload)
        self.assertEqual(payload["motion_job"]["job_id"], "abc123")

    def test_query_to_task_payload_failed_when_errors_and_no_text(self) -> None:
        answer = AnswerResponse(
            request_id="abc123",
            status="processing",
            pending_services=[],
            language="en",
            selected_strategy="unknown",
            progress_stage="queued",
            text_answer="",
            exercises=[],
            motion=None,
            tts=None,
            generation_time_ms=10.0,
            errors={"agenticrag": "boom"},
            debug=None,
        )

        task_payload = query_to_task_payload("task1", answer, query="q", user_id="u")
        self.assertEqual(task_payload.task_id, "task1")
        self.assertEqual(task_payload.status, "failed")
        self.assertEqual(task_payload.progress_stage, "failed")


if __name__ == "__main__":
    unittest.main()
