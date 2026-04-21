"""
Unit tests for agenticRAG Pydantic schemas.

Target: schemas/main_api.py
No I/O, no mocking — pure model validation.
"""

import pytest
from pydantic import ValidationError

from schemas.main_api import (
    AnswerRequest,
    AnswerResponse,
    ConversationTurn,
    MotionJobStatus,
    MotionMetadata,
    QueryRequestCompat,
    TTSMetadata,
    UnifiedTaskResponseCompat,
)


# ============================================================================
# AnswerRequest
# ============================================================================

class TestAnswerRequest:
    """Tests for the main /answer request schema."""

    def test_minimal_valid_request(self):
        req = AnswerRequest(query="What exercises help with back pain?")
        assert req.query == "What exercises help with back pain?"
        assert req.user_id == "default"
        assert req.motion_format == "glb"
        assert req.session_id is None
        assert req.conversation_history is None

    def test_all_fields_populated(self):
        turns = [
            ConversationTurn(role="user", content="Hello"),
            ConversationTurn(role="assistant", content="Hi there!"),
        ]
        req = AnswerRequest(
            query="Show me a squat",
            user_id="user_42",
            session_id="sess_xyz",
            motion_format="npz",
            conversation_history=turns,
        )
        assert req.user_id == "user_42"
        assert req.session_id == "sess_xyz"
        assert req.motion_format == "npz"
        assert len(req.conversation_history) == 2

    def test_requires_query(self):
        with pytest.raises(ValidationError):
            AnswerRequest()

    def test_motion_format_only_glb_or_npz(self):
        with pytest.raises(ValidationError):
            AnswerRequest(query="test", motion_format="mp4")

    def test_motion_format_glb_default(self):
        req = AnswerRequest(query="test")
        assert req.motion_format == "glb"

    def test_motion_format_npz_accepted(self):
        req = AnswerRequest(query="test", motion_format="npz")
        assert req.motion_format == "npz"


# ============================================================================
# AnswerResponse
# ============================================================================

class TestAnswerResponse:
    """Tests for the unified answer response schema."""

    def test_minimal_valid_response(self):
        resp = AnswerResponse(
            request_id="abc123",
            status="completed",
            text_answer="Do neck stretches.",
            generation_time_ms=42.0,
        )
        assert resp.request_id == "abc123"
        assert resp.status == "completed"
        assert resp.language == "other"
        assert resp.selected_strategy == "unknown"
        assert resp.progress_stage == "completed"
        assert resp.exercises == []
        assert resp.pending_services == []
        assert resp.motion is None
        assert resp.tts is None
        assert resp.errors is None
        assert resp.debug is None

    def test_all_optional_fields(self):
        motion = MotionMetadata(
            motion_file_url="http://dart/download/motion.glb",
            num_frames=60,
            fps=30,
            duration_seconds=2.0,
            text_prompt="walk forward",
        )
        tts = TTSMetadata(
            audio_file="audio_abc.mp3",
            audio_url="/audio/audio_abc.mp3",
            text="Hello world",
            emotion="neutral",
        )
        resp = AnswerResponse(
            request_id="xyz",
            status="processing",
            pending_services=["dart", "tts"],
            language="vi",
            selected_strategy="visualize_motion",
            progress_stage="motion_generation",
            text_answer="Try squats.",
            exercises=[{"name": "squat", "reason": "legs"}],
            motion=motion,
            tts=tts,
            generation_time_ms=500.0,
            errors={"dart": "timeout"},
            debug={"trace": "ok"},
        )
        assert resp.language == "vi"
        assert resp.motion.num_frames == 60
        assert resp.tts.audio_file == "audio_abc.mp3"
        assert resp.errors == {"dart": "timeout"}
        assert len(resp.pending_services) == 2

    def test_requires_request_id_and_text_answer(self):
        with pytest.raises(ValidationError):
            AnswerResponse(status="completed", generation_time_ms=1.0)

    def test_default_exercises_is_empty_list(self):
        resp = AnswerResponse(
            request_id="a",
            status="completed",
            text_answer="t",
            generation_time_ms=1.0,
        )
        assert resp.exercises == []


# ============================================================================
# MotionMetadata
# ============================================================================

class TestMotionMetadata:
    """Tests for the motion output metadata schema."""

    def test_valid_construction(self):
        m = MotionMetadata(
            motion_file_url="http://dart:5001/download/motion_abc.glb",
            num_frames=120,
            fps=30,
            duration_seconds=4.0,
            text_prompt="do a squat",
        )
        assert m.num_frames == 120
        assert m.fps == 30
        assert m.duration_seconds == 4.0
        assert "motion_abc.glb" in m.motion_file_url

    def test_requires_all_fields(self):
        with pytest.raises(ValidationError):
            MotionMetadata(motion_file_url="http://dart/file.glb")


# ============================================================================
# TTSMetadata
# ============================================================================

class TestTTSMetadata:
    """Tests for the TTS output metadata schema."""

    def test_valid_construction(self):
        t = TTSMetadata(
            audio_file="test.mp3",
            audio_url="/audio/test.mp3",
            text="Hello",
            emotion="happy",
        )
        assert t.audio_file == "test.mp3"
        assert t.emotion == "happy"

    def test_emotion_optional(self):
        t = TTSMetadata(
            audio_file="test.mp3",
            audio_url="/audio/test.mp3",
            text="Hello",
        )
        assert t.emotion is None


# ============================================================================
# MotionJobStatus
# ============================================================================

class TestMotionJobStatus:
    """Tests for the async motion job status schema."""

    def test_minimal_job(self):
        j = MotionJobStatus(job_id="job_1", status="queued")
        assert j.job_id == "job_1"
        assert j.status == "queued"
        assert j.motion_file_url is None
        assert j.video_url is None
        assert j.error is None

    def test_completed_job(self):
        j = MotionJobStatus(
            job_id="job_2",
            status="completed",
            motion_file_url="http://dart/motion.glb",
            video_url="http://dart/video.mp4",
            stage="rendering",
            timings_ms={"generation": 1200.0, "rendering": 800.0},
        )
        assert j.status == "completed"
        assert j.timings_ms["generation"] == 1200.0

    def test_failed_job_with_error(self):
        j = MotionJobStatus(
            job_id="job_3",
            status="failed",
            error="CUDA out of memory",
        )
        assert j.error == "CUDA out of memory"


# ============================================================================
# QueryRequestCompat
# ============================================================================

class TestQueryRequestCompat:
    """Tests for the compatibility request schema (8000-style clients)."""

    def test_defaults(self):
        req = QueryRequestCompat(query="help me")
        assert req.user_id == "default"
        assert req.stream is False
        assert req.motion_job_enabled is True
        assert req.session_id is None
        assert req.conversation_history is None
        assert req.motion_duration_seconds is None

    def test_requires_query(self):
        with pytest.raises(ValidationError):
            QueryRequestCompat()

    def test_with_conversation_history(self):
        turns = [ConversationTurn(role="user", content="hi")]
        req = QueryRequestCompat(query="test", conversation_history=turns)
        assert len(req.conversation_history) == 1


# ============================================================================
# ConversationTurn
# ============================================================================

class TestConversationTurn:
    """Tests for the conversation turn schema."""

    def test_valid_turn(self):
        t = ConversationTurn(role="user", content="Hello")
        assert t.role == "user"
        assert t.content == "Hello"

    def test_requires_both_fields(self):
        with pytest.raises(ValidationError):
            ConversationTurn(role="user")

        with pytest.raises(ValidationError):
            ConversationTurn(content="Hello")


# ============================================================================
# UnifiedTaskResponseCompat
# ============================================================================

class TestUnifiedTaskResponseCompat:
    """Tests for the task envelope compatibility schema."""

    def test_minimal_task(self):
        t = UnifiedTaskResponseCompat(
            task_id="task_1",
            status="processing",
            progress_stage="queued",
        )
        assert t.task_id == "task_1"
        assert t.result is None
        assert t.error is None

    def test_completed_task_with_result(self):
        t = UnifiedTaskResponseCompat(
            task_id="task_2",
            status="completed",
            progress_stage="completed",
            result={"text_answer": "Do squats.", "exercises": []},
        )
        assert t.result["text_answer"] == "Do squats."

    def test_failed_task_with_error(self):
        t = UnifiedTaskResponseCompat(
            task_id="task_3",
            status="failed",
            progress_stage="failed",
            error="AgenticRAG unavailable",
        )
        assert t.error == "AgenticRAG unavailable"
