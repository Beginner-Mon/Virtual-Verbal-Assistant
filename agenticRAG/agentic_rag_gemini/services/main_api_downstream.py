import re
from typing import Any, Dict, List, Literal, Optional

import httpx

from schemas.main_api import MotionMetadata, TTSMetadata
from utils.logger import get_logger

logger = get_logger(__name__)


def normalize_motion_description(prompt: str) -> str:
    normalized = (prompt or "").strip()
    if not normalized:
        return normalized

    # Legacy cleanup: convert "squat*12" or multi-action "a*5,b*3" into plain text.
    cleaned = re.sub(r"\*\s*\d+", "", normalized)
    cleaned = re.sub(r"\s*,\s*", ", ", cleaned)
    cleaned = re.sub(r"\s+", " ", cleaned).strip(" ,")
    return cleaned


def resolve_motion_duration_seconds(rag_data: Dict[str, Any], default_duration_seconds: float) -> float:
    raw = rag_data.get("duration_seconds")
    if raw is None and isinstance(rag_data.get("motion_prompt"), dict):
        mp = rag_data.get("motion_prompt") or {}
        raw = mp.get("duration_seconds") or mp.get("duration_estimate_seconds")
    try:
        value = float(raw) if raw is not None else default_duration_seconds
    except (TypeError, ValueError):
        value = default_duration_seconds
    return max(1.0, min(value, 120.0))


def detect_query_language(query: str) -> str:
    """Detect query language and map to en | vi | jp | other."""
    text = (query or "").strip()
    if not text:
        return "other"

    if re.search(r"[\u3040-\u30ff\u31f0-\u31ff\u4e00-\u9fff]", text):
        return "jp"

    vi_chars_pattern = r"[ăâđêôơưĂÂĐÊÔƠƯáàảãạắằẳẵặấầẩẫậéèẻẽẹếềểễệíìỉĩịóòỏõọốồổỗộớờởỡợúùủũụứừửữựýỳỷỹỵ]"
    if re.search(vi_chars_pattern, text):
        return "vi"

    lowered = f" {text.lower()} "
    vi_keywords = (
        " bạn ", " tôi ", " chúng tôi ", " xin chào ", " cảm ơn ", " bài tập ",
        " đau ", " không ", " giúp ", " như thế nào ", " thế nào ",
    )
    if any(k in lowered for k in vi_keywords):
        return "vi"

    if re.search(r"[a-zA-Z]", text):
        if not re.search(r"[\u0400-\u04FF\u0600-\u06FF\u0590-\u05FF\u0900-\u097F\u0E00-\u0E7F]", text):
            return "en"

    return "other"


async def call_agenticrag(
    client: httpx.AsyncClient,
    base_url: str,
    query: str,
    user_id: str,
    conversation_history: Optional[List[Dict[str, str]]],
    session_id: Optional[str] = None,
) -> Dict[str, Any]:
    """POST to AgenticRAG /query and return the parsed JSON body."""
    payload: Dict[str, Any] = {
        "query": query,
        "user_id": user_id,
    }
    if session_id:
        payload["session_id"] = session_id
    elif conversation_history:
        payload["conversation_history"] = conversation_history

    logger.info(f"[AgenticRAG] -> POST {base_url}/query  query={query[:80]}...")
    resp = await client.post(f"{base_url}/query", json=payload)
    resp.raise_for_status()
    data = resp.json()
    logger.info(f"[AgenticRAG] <- {resp.status_code} OK")
    return data


def build_motion_from_agenticrag(rag_data: Dict[str, Any], dart_url: str) -> Optional[MotionMetadata]:
    """Map AgenticRAG motion payload to unified MotionMetadata when available."""
    motion = rag_data.get("motion")
    if not isinstance(motion, dict):
        return None

    motion_file = motion.get("motion_file")
    if not motion_file:
        return None

    frames = int(motion.get("frames", 0) or 0)
    fps = int(motion.get("fps", 30) or 30)
    duration_seconds = round(frames / fps, 2) if frames > 0 and fps > 0 else 0.0
    text_prompt = rag_data.get("exercise_motion_prompt") or rag_data.get("query") or ""

    return MotionMetadata(
        motion_file_url=f"{dart_url}/download/{motion_file}",
        num_frames=frames,
        fps=fps,
        duration_seconds=duration_seconds,
        text_prompt=text_prompt,
    )


async def generate_motion_from_dart(
    client: httpx.AsyncClient,
    dart_url: str,
    motion_prompt: str,
    duration_seconds: float,
    motion_format: Literal["glb", "npz"],
    rag_data: Dict[str, Any],
    semantic_bridge_prompt: Optional[str] = None,
) -> MotionMetadata:
    """Generate motion by calling DART API directly.

    When a semantic_bridge_prompt is provided (from the parallel Semantic
    Bridge), it takes priority over the normalized motion_prompt because it
    has already been translated into HumanML3D-compatible vocabulary.
    """
    normalized_prompt = normalize_motion_description(motion_prompt)

    # Priority: semantic_bridge_prompt > normalized_prompt
    effective_prompt = semantic_bridge_prompt.strip() if semantic_bridge_prompt else normalized_prompt
    if semantic_bridge_prompt:
        logger.info(
            "[DART] Using semantic_bridge_prompt: '%s' (overriding normalized: '%s')",
            effective_prompt[:60],
            normalized_prompt[:40],
        )

    resolved_motion_format = rag_data.get("motion_format") if rag_data.get("motion_format") in {"glb", "npz"} else motion_format
    dart_body: Dict[str, Any] = {
        "text_prompt": effective_prompt,
        "duration_seconds": duration_seconds,
        "output_format": resolved_motion_format,
        "guidance_scale": 5.0,
        "num_steps": 10,
        "gender": "female",
    }
    if "respacing" in rag_data:
        dart_body["respacing"] = rag_data["respacing"]
    if "seed" in rag_data:
        dart_body["seed"] = rag_data["seed"]

    resp = await client.post(f"{dart_url}/generate", json=dart_body)
    resp.raise_for_status()
    dart_data = resp.json()
    motion_file_url = f"{dart_url}{dart_data['motion_file_url']}" if dart_data.get("motion_file_url") else ""
    return MotionMetadata(
        motion_file_url=motion_file_url,
        num_frames=dart_data.get("num_frames", 0),
        fps=dart_data.get("fps", 30),
        duration_seconds=dart_data.get("duration_seconds", 0.0),
        text_prompt=dart_data.get("text_prompt", effective_prompt),
    )


async def generate_tts(
    client: httpx.AsyncClient,
    tts_url: str,
    text_answer: str,
    user_id: str,
) -> TTSMetadata:
    """Generate TTS from SpeechLLM."""
    tts_payload = {"text": text_answer, "user_id": user_id}
    tts_resp = await client.post(f"{tts_url}/synthesize", json=tts_payload)
    tts_resp.raise_for_status()
    tts_data = tts_resp.json()
    audio_file = tts_data.get("audio_file", "")
    # Use relative path so it routes through whatever domain the frontend used (e.g. Ngrok proxy)
    audio_url = f"/audio/{audio_file}" if audio_file else ""
    return TTSMetadata(
        audio_file=audio_file,
        audio_url=audio_url,
        text=tts_data.get("text", text_answer),
        emotion=tts_data.get("emotion", None),
    )
