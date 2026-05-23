"""Tests for grader node (Task 2.5.8)."""

import pytest

from langgraph_agents.nodes.grader import _check_rules
from langchain_core.messages import ToolMessage


def _mk_state(**overrides):
    base = {"reasoning_output": "", "intent": "conversation", "messages": []}
    base.update(overrides)
    return base


@pytest.mark.unit
def test_grader_empty_reasoning_fails():
    passed, feedback = _check_rules(_mk_state(reasoning_output=""))
    assert passed is False
    assert feedback


@pytest.mark.unit
def test_grader_exercise_with_markers_passes():
    passed, feedback = _check_rules(_mk_state(
        reasoning_output="Bridge: 3 hiệp 10 lần. Thực hiện từ từ.",
        intent="exercise_recommendation",
    ))
    assert passed is True
    assert feedback is None


@pytest.mark.unit
def test_grader_exercise_no_markers_fails():
    passed, feedback = _check_rules(_mk_state(
        reasoning_output="Bạn nên tập thể dục nhẹ nhàng.",
        intent="exercise_recommendation",
    ))
    assert passed is False
    assert "guidance" in feedback.lower() or "rep" in feedback.lower() or "step" in feedback.lower()


@pytest.mark.unit
def test_grader_exercise_bullet_only_fails():
    """A lone bullet must NOT be accepted as guidance (regression of C4)."""
    passed, feedback = _check_rules(_mk_state(
        reasoning_output="- Hãy thư giãn và đi bộ nhẹ mỗi sáng.",
        intent="exercise_recommendation",
    ))
    assert passed is False, "Bullet without rep/set/step counts must fail"


@pytest.mark.unit
def test_grader_exercise_enumerated_steps_passes():
    """Two or more enumerated steps count as guidance."""
    passed, _ = _check_rules(_mk_state(
        reasoning_output="1. Nằm ngửa.\n2. Co gối, nâng hông.\n3. Giữ 5 giây.",
        intent="exercise_recommendation",
    ))
    assert passed is True


@pytest.mark.unit
def test_grader_exercise_single_step_fails():
    """A single numbered line is not enough."""
    passed, _ = _check_rules(_mk_state(
        reasoning_output="1. Tập thường xuyên.",
        intent="exercise_recommendation",
    ))
    assert passed is False


@pytest.mark.unit
def test_grader_exercise_count_keyword_passes():
    """'3 hiệp 10 lần' style passes via count keyword."""
    passed, _ = _check_rules(_mk_state(
        reasoning_output="Bridge: 3 hiệp 10 lần. Thực hiện chậm rãi.",
        intent="exercise_recommendation",
    ))
    assert passed is True


@pytest.mark.unit
def test_grader_visualize_motion_no_tool_fails():
    passed, feedback = _check_rules(_mk_state(
        reasoning_output="Chuyển động cầu vai.",
        intent="visualize_motion",
        messages=[],
    ))
    assert passed is False
    assert "generate_motion" in feedback


@pytest.mark.unit
def test_grader_visualize_motion_with_tool_passes():
    passed, feedback = _check_rules(_mk_state(
        reasoning_output="Động tác bridge.",
        intent="visualize_motion",
        messages=[ToolMessage(content="motion data", name="generate_motion", tool_call_id="1")],
    ))
    assert passed is True


@pytest.mark.unit
@pytest.mark.asyncio
async def test_grader_retry_then_pass_with_warning():
    from langgraph_agents.nodes.grader import grader_node

    # First call: empty reasoning → retry
    from langgraph_agents.state import AgentState
    state1: AgentState = {
        "messages": [],
        "errors": [],
        "retry_count": 0,
        "reasoning_output": "",
        "intent": "knowledge_query",
        "final_answer": "",
    }
    result1 = await grader_node(state1)
    assert result1["grader_result"] == "retry"
    assert result1["retry_count"] == 1

    # Second call: still empty → pass_with_warning
    state1["retry_count"] = 1
    result2 = await grader_node(state1)
    assert result2["grader_result"] == "pass_with_warning"
    assert result2["grader_warning"]
