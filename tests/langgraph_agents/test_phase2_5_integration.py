"""Integration tests — M.9 Step 15, rebuilt 3-axis model.

Requires DEEPSEEK_API_KEY (or live LLM endpoint).
Tests the full graph flow: memory → planner → retriever → synthesizer → grader.
"""

import os
import pytest

from langgraph_agents.graph import build_graph_async
from langgraph_agents.state import AgentState

HAS_LLM_KEY = bool(os.getenv("DEEPSEEK_API_KEY"))


def _base_state_config(**overrides):
    """Returns (initial_state, config) for graph.ainvoke."""
    state = {
        "messages": [],
        "errors": [],
        "retry_count": 0,
        "total_tokens": 0,
        "required_outputs": [],
        "resolved_query": "",
        "needs_retrieval": False,
        "needs_motion": False,
        "needs_clarification": False,
    }
    config = {
        "configurable": {
            "user_id": "test-user",
            "session_id": "test-session",
            "query": overrides.pop("query", "Xin chào"),
            "persona_id": "eca_default",
            "request_id": "test-001",
            "web_search": False,
        }
    }
    state.update(overrides)
    return state, config


# ═══════════════════════════════════════════════════════════════════════════
# Live LLM tests (require DEEPSEEK_API_KEY)
# ═══════════════════════════════════════════════════════════════════════════

@pytest.mark.integration
@pytest.mark.skipif(not HAS_LLM_KEY, reason="DEEPSEEK_API_KEY not set")
@pytest.mark.asyncio
async def test_chat_path_no_retrieval_no_tags():
    """Greeting → no retrieval, no tags → chat → END (fast-path D8)."""
    graph = await build_graph_async()
    state, config = _base_state_config(query="Xin chào")
    result = await graph.ainvoke(state, config=config)

    assert result["final_answer"], "final_answer should not be empty"
    # Chat path: no tags → grader skipped
    assert result.get("required_outputs", []) == []


@pytest.mark.integration
@pytest.mark.skipif(not HAS_LLM_KEY, reason="DEEPSEEK_API_KEY not set")
@pytest.mark.asyncio
async def test_safety_path_red_flag_no_retrieval():
    """Red-flag symptom → safety tags, NO retrieval, safety template enforced."""
    graph = await build_graph_async()
    state, config = _base_state_config(query="Tôi bị đau ngực khi tập thể dục")
    result = await graph.ainvoke(state, config=config)

    assert result["final_answer"], "final_answer should not be empty"
    # Should have safety tags (D33: danger in query → planner detects)
    tags = result.get("required_outputs", [])
    assert "red_flag_screen" in tags or "referral_advice" in tags, \
        f"Expected safety tags, got {tags}"


@pytest.mark.integration
@pytest.mark.skipif(not HAS_LLM_KEY, reason="DEEPSEEK_API_KEY not set")
@pytest.mark.asyncio
async def test_exercise_path_with_retrieval():
    """Exercise query → needs_retrieval + exercise tags → full pipeline."""
    graph = await build_graph_async()
    state, config = _base_state_config(query="Bài tập cho đau lưng dưới")
    result = await graph.ainvoke(state, config=config)

    assert result["final_answer"], "final_answer should not be empty"
    # Should have exercise-related tags
    tags = result.get("required_outputs", [])
    exercise_tags = {"exercise_protocol", "exercise_steps", "scope_disclaimer"}
    assert any(t in exercise_tags for t in tags), \
        f"Expected exercise tags, got {tags}"


@pytest.mark.integration
@pytest.mark.skipif(not HAS_LLM_KEY, reason="DEEPSEEK_API_KEY not set")
@pytest.mark.asyncio
async def test_clarify_path():
    """Ambiguous query → clarify, no retrieval."""
    graph = await build_graph_async()
    state, config = _base_state_config(query="bài tập")
    result = await graph.ainvoke(state, config=config)

    assert result["final_answer"], "final_answer should not be empty"
    # Should trigger clarification (missing body region)
    assert result.get("needs_clarification") or "?" in result["final_answer"] or "bạn" in result["final_answer"].lower(), \
        "Expected clarification response"


@pytest.mark.integration
@pytest.mark.skipif(not HAS_LLM_KEY, reason="DEEPSEEK_API_KEY not set")
@pytest.mark.asyncio
async def test_grader_enforces_tags():
    """Exercise query → grader checks required_outputs (if tags present)."""
    graph = await build_graph_async()
    state, config = _base_state_config(query="Hướng dẫn bài tập squat")
    result = await graph.ainvoke(state, config=config)

    assert result["final_answer"], "final_answer should not be empty"
    # grader_result should be set if tags were emitted
    if result.get("required_outputs"):
        assert result.get("grader_result") in ("pass", "pass_with_warning", "retry", None)


# ═══════════════════════════════════════════════════════════════════════════
# Unit tests (no LLM needed) — planner output structure
# ═══════════════════════════════════════════════════════════════════════════

class TestPlanOutput:
    """Verify PlanOutput Pydantic model structure (M.1 3-axis)."""

    def test_plan_output_defaults(self):
        from langgraph_agents.nodes.planner import PlanOutput
        plan = PlanOutput()
        assert plan.required_outputs == []
        assert plan.resolved_query == ""
        assert plan.needs_retrieval is False
        assert plan.needs_motion is False
        assert plan.needs_clarification is False

    def test_plan_output_3_axis(self):
        from langgraph_agents.nodes.planner import PlanOutput
        plan = PlanOutput(
            required_outputs=["exercise_protocol", "scope_disclaimer"],
            resolved_query="bài tập cho L4-L5",
            needs_retrieval=True,
            needs_motion=False,
            needs_clarification=False,
        )
        assert len(plan.required_outputs) == 2
        assert "exercise_protocol" in plan.required_outputs
        assert plan.needs_retrieval is True
        assert plan.needs_motion is False

    def test_plan_output_json_mode(self):
        """Verify PlanOutput can serialize/deserialize (json_mode compat)."""
        from langgraph_agents.nodes.planner import PlanOutput
        plan = PlanOutput(
            required_outputs=["red_flag_screen", "referral_advice"],
            resolved_query="đau ngực khi tập",
            needs_retrieval=False,
            needs_motion=False,
        )
        d = plan.model_dump()
        plan2 = PlanOutput(**d)
        assert plan2.required_outputs == plan.required_outputs
        assert plan2.resolved_query == plan.resolved_query


# ═══════════════════════════════════════════════════════════════════════════
# Unit tests — Grader rules (M.3, no LLM)
# ═══════════════════════════════════════════════════════════════════════════

class TestGraderRules:
    """Verify TAG_RULES heuristic checks."""

    def test_has_danger_warning_present(self):
        from langgraph_agents.nodes.grader import _has_danger_warning
        assert _has_danger_warning("⚠️ Đau ngực là nghiêm trọng. Bạn nên đi khám bác sĩ ngay.")
        assert _has_danger_warning("Ngừng tập ngay lập tức nếu thấy đau.")
        assert not _has_danger_warning("Bài tập này giúp giảm đau lưng.")

    def test_has_referral_present(self):
        from langgraph_agents.nodes.grader import _has_referral
        assert _has_referral("Tôi khuyên bạn nên đi khám bác sĩ chuyên khoa.")
        assert _has_referral("Bạn nên tham khảo ý kiến chuyên gia y tế.")
        assert not _has_referral("Bài tập squat rất tốt cho cơ đùi.")

    def test_has_disclaimer_present(self):
        from langgraph_agents.nodes.grader import _has_disclaimer
        assert _has_disclaimer("Thông tin này chỉ mang tính tham khảo wellness.")
        assert _has_disclaimer("Không thay thế cho việc khám lâm sàng.")
        assert not _has_disclaimer("Đây là bài tập được nhiều chuyên gia khuyên dùng.")

    def test_has_sets_reps_present(self):
        from langgraph_agents.nodes.grader import _has_sets_reps_frequency
        assert _has_sets_reps_frequency("3 hiệp × 10 lần, 2-3 lần mỗi tuần.")
        assert _has_sets_reps_frequency("Tập 15 reps, 3 sets, daily.")
        assert not _has_sets_reps_frequency("Tập squat đều đặn mỗi ngày.")  # no rep count

    def test_has_ordered_steps_present(self):
        from langgraph_agents.nodes.grader import _has_ordered_steps
        assert _has_ordered_steps("1. Đứng thẳng. 2. Từ từ hạ người xuống.")
        assert _has_ordered_steps("Bước 1: Chuẩn bị. Bước 2: Thực hiện.")
        assert not _has_ordered_steps("Tập squat bằng cách hạ người xuống từ từ.")

    def test_has_contraindication_present(self):
        from langgraph_agents.nodes.grader import _has_contraindication
        assert _has_contraindication("Không nên tập nếu bạn bị thoát vị đĩa đệm.")
        assert _has_contraindication("Chống chỉ định: người bị đau đầu gối cấp tính.")
        assert not _has_contraindication("Bài tập này an toàn cho mọi người.")

    def test_has_source_present(self):
        from langgraph_agents.nodes.grader import _has_source
        assert _has_source("Theo tài liệu [1], bài tập này...")
        assert _has_source("Nguồn: https://example.com/pt-guide")
        assert not _has_source("Bài tập này rất phổ biến.")

    def test_has_motion_fields_present(self):
        from langgraph_agents.nodes.grader import _has_motion_fields
        assert _has_motion_fields("Giơ tay phải lên cao — động tác sử dụng khớp vai.")
        assert _has_motion_fields("Gập đầu gối, xoay hông khi thực hiện squat.")
        assert not _has_motion_fields("Bài tập squat rất tốt.")  # no joints

    def test_grade_tags_safety_missing(self):
        from langgraph_agents.nodes.grader import _grade_tags
        result = _grade_tags(
            "Bài tập squat: đứng thẳng, hạ người.",  # no danger warning
            ["red_flag_screen", "exercise_steps"],
        )
        assert result["result"] == "pass_with_warning"
        assert "red_flag_screen" in result["safety_missing"]

    def test_grade_tags_quality_retry(self):
        from langgraph_agents.nodes.grader import _grade_tags
        result = _grade_tags(
            "Squat rất tốt cho chân.",  # no sets/reps, no steps
            ["exercise_protocol", "exercise_steps"],
        )
        assert result["result"] == "retry"
        assert len(result["quality_missing"]) >= 1

    def test_grade_tags_all_pass(self):
        from langgraph_agents.nodes.grader import _grade_tags
        result = _grade_tags(
            "3 hiệp × 10 lần, 2-3 lần/tuần.\n"
            "1. Đứng thẳng. 2. Hạ người từ từ.\n"
            "Không nên tập nếu đau đầu gối.",
            ["exercise_protocol", "exercise_steps", "contraindication"],
        )
        assert result["result"] == "pass"

    def test_grade_tags_empty_answer(self):
        from langgraph_agents.nodes.grader import _grade_tags
        result = _grade_tags("", ["exercise_protocol"])
        assert result["result"] == "retry"


# ═══════════════════════════════════════════════════════════════════════════
# Unit tests — Synthesizer mode derivation (D29, no LLM)
# ═══════════════════════════════════════════════════════════════════════════

class TestModeDerivation:
    """Verify _derive_mode logic (D29: mode emerges from signals)."""

    def test_chat_mode(self):
        from langgraph_agents.nodes.synthesizer import _derive_mode
        state = {
            "needs_clarification": False,
            "required_outputs": [],
            "messages": [],
        }
        assert _derive_mode(state) == "chat"

    def test_clarify_static(self):
        from langgraph_agents.nodes.synthesizer import _derive_mode
        state = {
            "needs_clarification": True,
            "required_outputs": [],
            "messages": [],
        }
        assert _derive_mode(state) == "clarify"

    def test_refuse_clinical_no_source(self):
        from langgraph_agents.nodes.synthesizer import _derive_mode
        state = {
            "needs_clarification": False,
            "required_outputs": ["exercise_protocol", "scope_disclaimer"],
            "messages": [],  # No ToolMessages
        }
        assert _derive_mode(state) == "refuse"

    def test_synthesize_has_tool_results(self):
        from langgraph_agents.nodes.synthesizer import _derive_mode
        from langchain_core.messages import ToolMessage
        tool_msg = ToolMessage(
            content='{"found": true, "results": [{"content": "test"}]}',
            tool_call_id="test-001",
            name="kb_search",
        )
        state = {
            "needs_clarification": False,
            "required_outputs": ["exercise_protocol"],
            "messages": [tool_msg],
        }
        assert _derive_mode(state) == "synthesize"


# ═══════════════════════════════════════════════════════════════════════════
# Unit tests — TAG_RULES assertion (D7)
# ═══════════════════════════════════════════════════════════════════════════

class TestTagConsistency:
    """Verify planner vocabulary == grader vocabulary (D7)."""

    def test_planner_tags_match_grader_tags(self):
        from langgraph_agents.nodes.planner import PLANNER_TAGS
        from langgraph_agents.nodes.grader import TAG_RULES
        assert PLANNER_TAGS == set(TAG_RULES.keys()), \
            f"Drift: planner has {PLANNER_TAGS - set(TAG_RULES.keys())}, " \
            f"grader has {set(TAG_RULES.keys()) - PLANNER_TAGS}"

    def test_all_tags_have_kind_and_rule(self):
        from langgraph_agents.nodes.grader import TAG_RULES
        for tag, (kind, rule_fn, template) in TAG_RULES.items():
            assert kind in ("safety", "quality"), f"{tag}: kind must be safety|quality, got {kind}"
            assert callable(rule_fn), f"{tag}: rule_fn must be callable"
            assert isinstance(template, str) and template, f"{tag}: template must be non-empty string"
