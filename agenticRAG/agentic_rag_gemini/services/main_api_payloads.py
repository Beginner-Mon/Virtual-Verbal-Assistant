from typing import Any, Dict, Optional

from schemas.main_api import AnswerResponse, UnifiedTaskResponseCompat


def model_to_dict(value: Any) -> Dict[str, Any]:
    if value is None:
        return {}
    if isinstance(value, dict):
        return value
    if hasattr(value, "model_dump"):
        return value.model_dump()
    if hasattr(value, "dict"):
        return value.dict()
    return dict(value)


def extract_motion_file_name(motion_file_url: Optional[str]) -> Optional[str]:
    if not motion_file_url:
        return None
    base = motion_file_url.split("?", 1)[0].rstrip("/")
    if not base:
        return None
    return base.split("/")[-1] or None


def to_progress_stage(stage: Optional[str], status: str) -> str:
    stage_value = (stage or "").strip().lower()
    status_value = (status or "").strip().lower()
    if status_value == "failed":
        return "failed"
    if status_value == "completed":
        return "completed"
    if stage_value in {"queued", "text_ready", "motion_generation", "completed", "failed"}:
        return stage_value
    if stage_value in {"rag_processing", "voice_synthesis"}:
        return "text_ready"
    return "queued"


def answer_to_query_payload(answer: AnswerResponse, query: str, user_id: str) -> Dict[str, Any]:
    motion_payload = None
    if answer.motion:
        motion_file_url = answer.motion.motion_file_url
        motion_payload = {
            "motion_file": extract_motion_file_name(motion_file_url),
            "motion_file_url": motion_file_url,
            "frames": answer.motion.num_frames,
            "fps": answer.motion.fps,
            "duration_seconds": answer.motion.duration_seconds,
            "text_prompt": answer.motion.text_prompt,
        }

    motion_job_payload = None
    if answer.status == "processing":
        motion_job_payload = {
            "job_id": answer.request_id,
            "status": "queued" if answer.progress_stage != "completed" else "completed",
            "motion_file_url": motion_payload.get("motion_file_url") if motion_payload else None,
            "video_url": None,
            "error": None,
        }

    error_text = None
    if answer.errors:
        error_text = "; ".join(f"{k}: {v}" for k, v in answer.errors.items())

    return {
        "query": query,
        "user_id": user_id,
        "language": answer.language,
        "text_answer": answer.text_answer,
        "exercises": answer.exercises,
        "exercise_motion_prompt": answer.motion.text_prompt if answer.motion else None,
        "motion": motion_payload,
        "motion_job": motion_job_payload,
        "orchestrator_decision": {
            "action": "full_pipeline",
            "intent": answer.selected_strategy,
            "confidence": 1.0,
            "language": answer.language,
            "reasoning": "Main API orchestrated downstream services",
            "parameters": {"progress_stage": answer.progress_stage, "request_id": answer.request_id},
        },
        "motion_prompt": {
            "description": answer.motion.text_prompt,
            "duration_seconds": answer.motion.duration_seconds,
        }
        if answer.motion
        else None,
        "voice_prompt": None,
        "metadata": {
            "request_id": answer.request_id,
            "status": answer.status,
            "pending_services": answer.pending_services,
            "progress_stage": answer.progress_stage,
            "tts": model_to_dict(answer.tts) if answer.tts else None,
            "debug": answer.debug,
        },
        "pipeline_trace": {"path": "main_api_full_pipeline", "errors": answer.errors or {}},
        "performance": {"total_ms": answer.generation_time_ms},
        "errors": answer.errors,
        "error": error_text,
    }


def query_to_task_payload(task_id: str, answer: AnswerResponse, query: str, user_id: str) -> UnifiedTaskResponseCompat:
    status_value = "failed" if answer.errors and not answer.text_answer else answer.status
    progress_stage = to_progress_stage(answer.progress_stage, status_value)
    result = answer_to_query_payload(answer, query, user_id)
    error_text = result.get("error")
    return UnifiedTaskResponseCompat(
        task_id=task_id,
        status=status_value,
        progress_stage=progress_stage,
        result=result,
        error=error_text,
    )
