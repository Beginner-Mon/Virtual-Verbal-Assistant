"""Grader node — M.3 tag-driven, rule-based, deterministic.

Decisions encoded:
  D6:   Tag 2 loại: safety → template cứng (no retry); quality → retry max 1
  D7:   TAG_RULES = 1 dict nguồn-chân-lý + startup assertion
  D8:   required_outputs=[] → grader skip (handled by routing, D15)
  D31:  Rule-based = lưới chặn "quên hẳn", NOT judge intensity
  D32:  Safety warning = ĐẦU (synthesizer writes, grader verifies);
        Unverified disclaimer = CUỐI (grader appends)
  D33:  Danger detection = PLANNER (1 place); grader only checks presence

Grader scope (D31 — honest about limits):
  - CAN catch: absent marker ("quên hẳn" cảnh báo → has_referral=false → fail)
  - CANNOT catch: intensity ("có nhắc bác sĩ nhưng quá nhẹ" → regex sees "bác sĩ" → pass)
  - Intensity is persona prompt's job; grader is the coarse net.
"""

from __future__ import annotations

import re
import time
from typing import Callable

from langchain_core.messages import ToolMessage
from langchain_core.runnables import RunnableConfig

from langgraph_agents.state import AgentState
from langgraph_agents.nodes._persona_loader import get_persona
from langgraph_agents.shared.logging import get_logger

logger = get_logger("langgraph.grader")


# ═══════════════════════════════════════════════════════════════════════════
# TAG_RULES — single source of truth (D7)
# ═══════════════════════════════════════════════════════════════════════════

# ── Rule check functions (heuristic marker-based, NOT LLM — D31) ─────────

def _has_danger_warning(text: str) -> bool:
    """Check for danger/red-flag warning markers."""
    patterns = [
        r"(?:nguy hiểm|nghiêm trọng|khẩn cấp|dấu hiệu)",
        r"(?:ngừng|dừng)\s*(?:tập|ngay|lập tức)",
        r"(?:đi khám|gặp bác sĩ|chuyên gia y tế|cấp cứu)",
        r"(?:chest pain|danger|warning|emergency|seek medical)",
        r"⚠",
    ]
    return any(re.search(p, text, re.IGNORECASE) for p in patterns)


def _has_referral(text: str) -> bool:
    """Check for referral/consultation recommendation."""
    patterns = [
        r"(?:khuyên|nên|hãy)\s*(?:bạn\s*)?(?:đi khám|gặp|hỏi|tham khảo)\s*(?:ý kiến\s*)?(?:bác sĩ|chuyên gia|bác sĩ chuyên khoa|chuyên viên y tế)",
        r"(?:consult|see|visit|refer)\s*(?:a\s*)?(?:doctor|physician|specialist|medical professional)",
        r"(?:không thể|không đủ)\s*(?:chẩn đoán|kê đơn|điều trị)",
    ]
    return any(re.search(p, text, re.IGNORECASE) for p in patterns)


def _has_disclaimer(text: str) -> bool:
    """Check for wellness scope disclaimer."""
    patterns = [
        r"(?:tư vấn|hướng dẫn)\s*(?:wellness|sức khỏe|thể chất)",
        r"(?:không thay thế|không phải là)\s*(?:cho\s*(?:việc\s*)?)?(?:thăm\s*)?(?:khám|chẩn đoán|điều trị)\s*(?:lâm sàng|y tế|y khoa)",
        r"(?:tham khảo|chỉ mang tính)\s*(?:tham khảo|giáo dục)",
        r"(?:wellness|educational|informational)\s*(?:advice|purpose)",
        r"(?:không phải|không thể)\s*(?:thay thế|coi là)\s*(?:lời khuyên y tế|chẩn đoán)",
    ]
    return any(re.search(p, text, re.IGNORECASE) for p in patterns)


def _has_sets_reps_frequency(text: str) -> bool:
    """Check for exercise protocol: sets + reps + frequency."""
    has_sets_reps = bool(re.search(
        r"\d+\s*(?:lần|hiệp|reps?|sets?|lần lặp)",
        text, re.IGNORECASE,
    ))
    has_frequency = bool(re.search(
        r"(?:\d+\s*(?:lần|buổi|ngày|tuần|times?|days?|week))|"
        r"(?:mỗi ngày|hàng ngày|hằng ngày|mỗi tuần|hàng tuần|daily|weekly)",
        text, re.IGNORECASE,
    ))
    return has_sets_reps and has_frequency


def _has_ordered_steps(text: str) -> bool:
    """Check for ≥2 ordered/enumerated steps."""
    # Match numbered steps anywhere: "1. Step", "1) Step", "Bước 1", "step 1"
    # Use non-multiline pattern to catch steps on the same line
    numbered = len(re.findall(r"(?:^|\s|\.\s*)\d+[\.\)]\s+\S", text))
    if numbered >= 2:
        return True

    # Bullet points (need at least 2)
    bullets = len(re.findall(r"(?m)^\s*[-–—•]\s+\S", text))
    if bullets >= 2:
        return True

    # Explicit step markers: "bước 1", "step 1"
    step_markers = len(re.findall(r"(?:bước|step)\s*\d", text, re.IGNORECASE))
    if step_markers >= 2:
        return True

    # Sequential transition words (at least 2 distinct)
    seq_count = sum(
        1 for p in [r"đầu tiên", r"trước tiên", r"sau đó", r"tiếp theo", r"cuối cùng"]
        if re.search(p, text, re.IGNORECASE)
    )
    return seq_count >= 2


def _has_contraindication(text: str) -> bool:
    """Check for contraindication/warning section."""
    patterns = [
        r"(?:chống chỉ định|không nên|tránh|thận trọng|không áp dụng|không tập)",
        r"(?:không dành cho|không phù hợp)\s*(?:người|ai|trường hợp)",
        r"(?:contraindication|precaution|warning|do not|cannot|avoid|caution)",
        r"(?:nếu bạn|nếu có|người bị)\s*(?:đau|viêm|thoát vị|loãng xương|tim|huyết áp)",
    ]
    return any(re.search(p, text, re.IGNORECASE) for p in patterns)


def _has_source(text: str) -> bool:
    """Check for citation/source reference."""
    patterns = [
        r"(?:nguồn|theo|trích|dẫn|tài liệu|tham khảo|source|reference|according to)",
        r"\[[\d,]+\]",                          # [1], [1,2,3]
        r"\([\w\s]+\s*\d{4}\)",                 # (Author 2024)
        r"(?:https?://|www\.)",                  # URL
    ]
    return any(re.search(p, text, re.IGNORECASE) for p in patterns)


def _has_motion_fields(text: str) -> bool:
    """Check for motion description: movement + joints mentioned."""
    has_movement = bool(re.search(
        r"(?:động tác|cử động|chuyển động|giơ|nâng|gập|duỗi|xoay|nghiêng|"
        r"movement|motion|raise|lower|bend|extend|rotate|tilt)",
        text, re.IGNORECASE,
    ))
    has_joints = bool(re.search(
        r"(?:khớp|vai|khuỷu|cổ tay|hông|đầu gối|cổ chân|cột sống|lưng|"
        r"joint|shoulder|elbow|wrist|hip|knee|ankle|spine)",
        text, re.IGNORECASE,
    ))
    return has_movement and has_joints


# ── TAG_RULES — single source of truth (D7) ──────────────────────────────

TAG_RULES: dict[str, tuple[str, Callable[[str], bool], str]] = {
    # ── SAFETY (thiếu → chèn TEMPLATE CỨNG, KHÔNG retry — D6) ──
    "red_flag_screen": (
        "safety",
        _has_danger_warning,
        # Template cứng (appended when missing):
        "**Cảnh báo quan trọng:** Triệu chứng bạn mô tả có thể là dấu hiệu "
        "của một tình trạng nghiêm trọng. Bạn nên NGỪNG tập luyện ngay và đi "
        "khám bác sĩ chuyên khoa để được chẩn đoán chính xác.",
    ),
    "referral_advice": (
        "safety",
        _has_referral,
        "**Lưu ý:** Với câu hỏi này, tôi khuyên bạn nên tham khảo ý kiến "
        "bác sĩ hoặc chuyên gia y tế. Tôi chỉ có thể cung cấp thông tin tham "
        "khảo về wellness, không thay thế chẩn đoán lâm sàng.",
    ),
    "scope_disclaimer": (
        "safety",
        _has_disclaimer,
        "*Thông tin này chỉ mang tính tham khảo về wellness và không thay "
        "thế cho việc khám và chẩn đoán y tế chuyên nghiệp.*",
    ),

    # ── QUALITY (thiếu → retry max 1 — D6) ──
    "exercise_protocol": (
        "quality",
        _has_sets_reps_frequency,
        "Missing sets/reps/frequency. Include specific numbers: "
        "\"3 hiệp × 10 lần, 2-3 lần/tuần\".",
    ),
    "exercise_steps": (
        "quality",
        _has_ordered_steps,
        "Missing ordered steps. Provide ≥2 numbered steps or bullet points "
        "describing the movement sequence.",
    ),
    "contraindication": (
        "quality",
        _has_contraindication,
        "Missing contraindications. List conditions where this exercise "
        "should NOT be done (pain, inflammation, specific conditions).",
    ),
    "evidence_citation": (
        "quality",
        _has_source,
        "Missing source citation. Include document title, URL, or reference "
        "for the information provided.",
    ),
    "motion_descriptor": (
        "quality",
        _has_motion_fields,
        "Missing motion description. Describe the movement AND the joints "
        "involved (e.g. \"giơ tay phải lên — khớp vai và khuỷu\").",
    ),
}

# ── Default safety templates (fallback when persona doesn't define them) ──

# Derived from TAG_RULES rather than restated. These three strings used to be a
# verbatim second copy of the safety templates above, which is a standing
# invitation to edit one and not the other — and that is exactly what nearly
# happened while stripping the emoji out of both.
DEFAULT_SAFETY_TEMPLATES: dict[str, str] = {
    tag: rule[2] for tag, rule in TAG_RULES.items() if rule[0] == "safety"
}

# Startup assertion: ensure planner vocabulary ⊆ TAG_RULES (D7)
# Called once at module import — catches drift between planner and grader.
_PLANNER_TAGS = frozenset({
    "red_flag_screen", "referral_advice", "scope_disclaimer",
    "exercise_protocol", "exercise_steps", "contraindication",
    "evidence_citation", "motion_descriptor",
})
assert _PLANNER_TAGS == set(TAG_RULES.keys()), (
    f"TAG_RULES drift detected! "
    f"planner_tags - TAG_RULES = {_PLANNER_TAGS - set(TAG_RULES.keys())}, "
    f"TAG_RULES - planner_tags = {set(TAG_RULES.keys()) - _PLANNER_TAGS}"
)

# ── Warning message for pass_with_warning ─────────────────────────────────
_UNAUTHORIZED_DISCLAIMER = (
    "*Thông tin này chưa được kiểm chứng bởi chuyên gia y tế. "
    "Vui lòng tham khảo ý kiến bác sĩ trước khi áp dụng.*"
)


def get_safety_text(tag: str, persona_id: str) -> str:
    """Get safety template for tag, persona-customized if available.

    Falls back to DEFAULT_SAFETY_TEMPLATES if the persona hasn't
    defined its own safety_templates, or if the tag is missing.
    """
    persona = get_persona(persona_id)
    custom = persona.get("safety_templates", {}).get(tag)
    return custom or DEFAULT_SAFETY_TEMPLATES.get(tag, "")


# ── Grader logic ─────────────────────────────────────────────────────────

def _grade_tags(final_answer: str, required_outputs: list[str]) -> dict:
    """Run tag-driven checks against final_answer.

    Returns:
        {result: "pass"|"retry"|"pass_with_warning",
         feedback: str|None,
         safety_missing: list[str],    # safety tags that failed
         quality_missing: list[str]}   # quality tags that failed
    """
    if not final_answer or not final_answer.strip():
        return {
            "result": "retry",
            "feedback": "Empty response. Must produce an answer.",
            "safety_missing": [],
            "quality_missing": [],
        }

    safety_missing = []
    quality_missing = []

    for tag in required_outputs:
        if tag not in TAG_RULES:
            logger.warning("unknown_tag_in_grader", extra={"tag": tag})
            continue

        kind, rule_fn, _ = TAG_RULES[tag]
        if not rule_fn(final_answer):
            if kind == "safety":
                safety_missing.append(tag)
            else:
                quality_missing.append(tag)

    # Safety failures → template cứng (D6: no retry for safety)
    if safety_missing:
        return {
            "result": "pass_with_warning",  # We'll inject templates
            "feedback": None,
            "safety_missing": safety_missing,
            "quality_missing": quality_missing,
        }

    # Quality failures → retry (max 1, D6)
    if quality_missing:
        feedback_parts = []
        for tag in quality_missing:
            _, _, fb = TAG_RULES[tag]
            feedback_parts.append(f"[{tag}] {fb}")
        return {
            "result": "retry",
            "feedback": " ".join(feedback_parts),
            "safety_missing": [],
            "quality_missing": quality_missing,
        }

    # All tags pass
    return {
        "result": "pass",
        "feedback": None,
        "safety_missing": [],
        "quality_missing": [],
    }


# ── Node ─────────────────────────────────────────────────────────────────

async def grader_node(state: AgentState, config: RunnableConfig) -> dict:
    """Tag-driven grader node — M.3.

    Reads: required_outputs (tags), final_answer, persona_id (from config)
    Logic:
      - required_outputs=[] → skipped by routing (D8), this node never called
      - safety tag missing → inject persona-aware template (NO retry — D6)
      - quality tag missing → retry max 1 (D6)
      - retry_count >= 1 + still failing quality → pass_with_warning
      - D32: unverified disclaimer appended at END on pass_with_warning
    """
    t0 = time.perf_counter()
    required_outputs = state.get("required_outputs", [])
    final_answer = state.get("final_answer", "")
    retry_count = state.get("retry_count", 0)
    persona_id = config["configurable"].get("persona_id", "eca_default")

    # Safety: required_outputs=[] should never reach grader (D8 + D15 routing)
    if not required_outputs:
        logger.info("node_skip", extra={
            "node": "grader", "reason": "no_tags",
        })
        return {"grader_result": "pass"}

    result = _grade_tags(final_answer, required_outputs)

    elapsed_ms = round((time.perf_counter() - t0) * 1000)

    # ── PASS ──────────────────────────────────────────────────────────
    if result["result"] == "pass":
        logger.info("node_complete", extra={
            "node": "grader", "result": "pass",
            "elapsed_ms": elapsed_ms, "tags": required_outputs,
        })
        return {"grader_result": "pass"}

    # ── SAFETY MISSING → inject template cứng (D6: NO retry) ─────────
    if result["safety_missing"]:
        # Inject persona-aware safety templates at START of answer (D32)
        templates = []
        for tag in result["safety_missing"]:
            templates.append(get_safety_text(tag, persona_id))

        safety_block = "\n\n---\n\n".join(templates)
        modified_answer = f"{safety_block}\n\n---\n\n{final_answer}"

        # Also append unverified disclaimer if quality also failed (D32: CUỐI)
        disclaimer = ""
        if result["quality_missing"]:
            disclaimer = f"\n\n---\n\n{_UNAUTHORIZED_DISCLAIMER}"
            modified_answer += disclaimer

        logger.warning("node_complete", extra={
            "node": "grader", "result": "pass_with_warning",
            "elapsed_ms": elapsed_ms,
            "safety_missing": result["safety_missing"],
            "quality_missing": result["quality_missing"],
        })
        return {
            "grader_result": "pass_with_warning",
            "final_answer": modified_answer,
        }

    # ── QUALITY MISSING → retry (max 1) ───────────────────────────────
    if retry_count == 0:
        logger.info("node_complete", extra={
            "node": "grader", "result": "retry",
            "elapsed_ms": elapsed_ms,
            "quality_missing": result["quality_missing"],
        })
        return {
            "grader_result": "retry",
            "retry_count": 1,
            "grader_feedback": result["feedback"],
        }

    # ── RETRY EXHAUSTED → pass with warning (D32: disclaimer cuối) ────
    logger.warning("node_complete", extra={
        "node": "grader", "result": "pass_with_warning",
        "elapsed_ms": elapsed_ms,
        "quality_missing": result["quality_missing"],
        "retry_count": retry_count,
    })
    modified_answer = f"{final_answer}\n\n---\n\n{_UNAUTHORIZED_DISCLAIMER}"
    return {
        "grader_result": "pass_with_warning",
        "final_answer": modified_answer,
    }
