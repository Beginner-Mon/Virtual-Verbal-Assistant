"""Utility helpers for payload normalization and URL rewriting."""

from enum import Enum
from typing import Any, Dict, Optional

from fastapi import Request


def _normalize_enums(value: Any) -> Any:
    """Recursively normalize enums and model-like objects into JSON-safe values."""
    if isinstance(value, Enum):
        return value.value
    if hasattr(value, "model_dump"):
        try:
            return _normalize_enums(value.model_dump())
        except Exception:
            pass
    if hasattr(value, "dict"):
        try:
            return _normalize_enums(value.dict())
        except Exception:
            pass
    if isinstance(value, dict):
        return {str(k): _normalize_enums(v) for k, v in value.items()}
    if isinstance(value, list):
        return [_normalize_enums(v) for v in value]
    return value


def _model_to_dict(value: Any) -> Dict[str, Any]:
    """Convert pydantic model (v1/v2) or mapping payload to a dict."""
    if value is None:
        return {}
    if isinstance(value, dict):
        return value
    if hasattr(value, "model_dump"):
        return value.model_dump()
    if hasattr(value, "dict"):
        return value.dict()
    return dict(value)


def _get_request_base_url(request: Request) -> str:
    """Build base URL using reverse-proxy headers so Ngrok URLs remain valid."""
    proto = request.headers.get("x-forwarded-proto") or request.url.scheme
    host = request.headers.get("x-forwarded-host") or request.headers.get("host") or request.url.netloc
    if not host:
        return str(request.base_url).rstrip("/")
    return f"{proto}://{host}".rstrip("/")


def _to_absolute_url(url_or_path: Optional[str], request: Request) -> Optional[str]:
    if not url_or_path:
        return None
    if url_or_path.startswith("http://") or url_or_path.startswith("https://"):
        return url_or_path
    base = _get_request_base_url(request)
    path = url_or_path if url_or_path.startswith("/") else f"/{url_or_path}"
    return f"{base}{path}"


def _to_dart_download_url(motion_ref: Optional[str], request: Request) -> Optional[str]:
    """Build absolute DART download URLs for sync motion payloads."""
    if not motion_ref:
        return None

    if motion_ref.startswith("http://") or motion_ref.startswith("https://"):
        return motion_ref

    # Force all motion artifact URLs through the 8000 gateway.
    base = _get_request_base_url(request)

    if motion_ref.startswith("/static/"):
        path = motion_ref
    elif motion_ref.startswith("/"):
        path = motion_ref
    elif motion_ref.startswith("download/"):
        path = f"/{motion_ref}"
    else:
        # Legacy payloads only include the artifact filename.
        path = f"/download/{motion_ref}"

    return f"{base}{path}"


def _build_unified_result(query_payload: Dict[str, Any], request: Request) -> Dict[str, Any]:
    """Project QueryResponse into the stable result payload used by Official UI."""
    orchestrator = _model_to_dict(query_payload.get("orchestrator_decision") or {})
    motion = _model_to_dict(query_payload.get("motion") or {})
    motion_job = _model_to_dict(query_payload.get("motion_job") or {})
    performance = _model_to_dict(query_payload.get("performance") or {})
    pipeline_trace = _model_to_dict(query_payload.get("pipeline_trace") or {})

    motion_file_url = None

    if isinstance(motion, dict):
        motion_file_url = motion.get("motion_file_url")

    if not motion_file_url and isinstance(motion_job, dict):
        motion_file_url = motion_job.get("motion_file_url") or motion_job.get("video_url")

    if isinstance(motion, dict) and motion_file_url:
        motion_file_url = _to_dart_download_url(motion_file_url, request)

    if not motion_file_url and isinstance(motion, dict):
        # Legacy best-effort path where only motion_file exists.
        motion_file = motion.get("motion_file")
        motion_file_url = _to_dart_download_url(motion_file, request)

    return {
        "query": query_payload.get("query", ""),
        "user_id": query_payload.get("user_id", "guest"),
        "language": query_payload.get("language", "other"),
        "text_answer": query_payload.get("text_answer", ""),
        "exercises": query_payload.get("exercises", []),
        "exercise_motion_prompt": query_payload.get("exercise_motion_prompt"),
        "motion_duration_seconds": (
            (query_payload.get("metadata") or {}).get("motion_duration_seconds")
            if isinstance(query_payload.get("metadata"), dict)
            else None
        ),
        "orchestrator": {
            "action": orchestrator.get("action", "unknown"),
            "intent": orchestrator.get("intent", "unknown"),
            "confidence": orchestrator.get("confidence", 0.0),
        },
        "motion_file_url": _to_absolute_url(motion_file_url, request),
        "motion": motion,
        "motion_job": motion_job,
        "performance": performance,
        "pipeline_trace": pipeline_trace,
    }
