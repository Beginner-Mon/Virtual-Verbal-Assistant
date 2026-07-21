"""Planner unit tests — 3-axis model (M.1), PlanOutput, TAG_RULES.
"""

import pytest
from unittest.mock import AsyncMock, patch

from langgraph_agents.nodes.planner import PlanOutput, PLANNER_TAGS
from langgraph_agents.state import AgentState


@pytest.mark.unit
class TestPlanOutput:
    """Verify PlanOutput Pydantic model structure (M.1 3-axis)."""

    def test_defaults(self):
        plan = PlanOutput()
        assert plan.required_outputs == []
        assert plan.resolved_query == ""
        assert plan.needs_retrieval is False
        assert plan.needs_motion is False
        assert plan.needs_clarification is False

    def test_3_axis_populated(self):
        plan = PlanOutput(
            required_outputs=["exercise_protocol", "scope_disclaimer"],
            resolved_query="bai tap cho L4-L5",
            needs_retrieval=True,
            needs_motion=False,
            needs_clarification=False,
        )
        assert len(plan.required_outputs) == 2
        assert "exercise_protocol" in plan.required_outputs
        assert plan.needs_retrieval is True
        assert plan.needs_motion is False

    def test_json_mode_roundtrip(self):
        plan = PlanOutput(
            required_outputs=["red_flag_screen", "referral_advice"],
            resolved_query="dau nguc khi tap",
            needs_retrieval=False,
            needs_motion=False,
        )
        d = plan.model_dump()
        plan2 = PlanOutput(**d)
        assert plan2.required_outputs == plan.required_outputs
        assert plan2.resolved_query == plan.resolved_query

    def test_all_8_tags(self):
        """All 8 PT domain tags."""
        plan = PlanOutput(
            required_outputs=[
                "red_flag_screen", "referral_advice", "scope_disclaimer",
                "exercise_protocol", "exercise_steps", "contraindication",
                "evidence_citation", "motion_descriptor",
            ],
            resolved_query="test",
        )
        assert len(plan.required_outputs) == 8

    def test_unknown_tags_filtered_during_validation(self):
        """Planner post-validates: unknown tags are filtered out.
        But PlanOutput itself doesn't validate — the node does.
        This test just confirms PlanOutput accepts any strings.
        """
        plan = PlanOutput(
            required_outputs=["not_a_real_tag", "exercise_protocol"],
            resolved_query="test",
        )
        assert "not_a_real_tag" in plan.required_outputs

    def test_resolved_query_vi(self):
        plan = PlanOutput(resolved_query="bai tap cho dau lung duoi")
        assert "bai tap" in plan.resolved_query
        assert plan.required_outputs == []

    def test_needs_motion_flag(self):
        plan = PlanOutput(needs_motion=True, resolved_query="squat demo")
        assert plan.needs_motion is True
        assert plan.needs_retrieval is False

    def test_needs_clarification_flag(self):
        plan = PlanOutput(needs_clarification=True, resolved_query="bai tap")
        assert plan.needs_clarification is True


@pytest.mark.unit
class TestPlannerTags:
    """Verify PLANNER_TAGS vocabulary (D5: controlled vocab)."""

    def test_all_8_tags_present(self):
        expected = {
            "red_flag_screen",
            "referral_advice",
            "scope_disclaimer",
            "exercise_protocol",
            "exercise_steps",
            "contraindication",
            "evidence_citation",
            "motion_descriptor",
        }
        assert PLANNER_TAGS == expected

    def test_tags_are_frozenset(self):
        assert isinstance(PLANNER_TAGS, frozenset)

    def test_no_deprecated_tags(self):
        """No old tags leaked from 6-enum system."""
        deprecated = {"knowledge_query", "general_query", "greeting", "clarify", "visualize_motion"}
        assert PLANNER_TAGS.isdisjoint(deprecated)


@pytest.mark.unit
class TestPlannerNode:
    """Test planner_node logic (mocked LLM)."""

    @pytest.mark.asyncio
    async def test_planner_validates_tags_post_hoc(self):
        """Planner node filters unknown tags from LLM output."""
        from langgraph_agents.nodes.planner import planner_node
        from langchain_core.runnables import RunnableConfig

        fake_plan = PlanOutput(
            required_outputs=["exercise_protocol", "fake_tag_xyz"],
            resolved_query="test query",
            needs_retrieval=True,
        )

        with patch("langgraph_agents.nodes.planner.get_chat_model") as mock_llm:
            mock_llm.return_value.with_structured_output.return_value.ainvoke = AsyncMock(
                return_value=fake_plan
            )
            config = RunnableConfig(configurable={
                "user_id": "test", "session_id": "test-sess",
                "query": "test query", "persona_id": "eca_default",
                "request_id": "req-001",
            })
            state: AgentState = {"messages": [], "errors": [], "total_tokens": 0,
                                 "retry_count": 0}
            result = await planner_node(state, config)
            assert "fake_tag_xyz" not in result.get("required_outputs", [])
            assert "exercise_protocol" in result.get("required_outputs", [])

    @pytest.mark.asyncio
    async def test_planner_auto_adds_referral_with_red_flag(self):
        """D33: red_flag_screen → auto-add referral_advice."""
        from langgraph_agents.nodes.planner import planner_node
        from langchain_core.runnables import RunnableConfig

        fake_plan = PlanOutput(
            required_outputs=["red_flag_screen"],
            resolved_query="dau nguc",
            needs_retrieval=False,
        )

        with patch("langgraph_agents.nodes.planner.get_chat_model") as mock_llm:
            mock_llm.return_value.with_structured_output.return_value.ainvoke = AsyncMock(
                return_value=fake_plan
            )
            config = RunnableConfig(configurable={
                "user_id": "test", "session_id": "test-sess",
                "query": "dau nguc", "persona_id": "eca_default",
                "request_id": "req-001",
            })
            state: AgentState = {"messages": [], "errors": [], "total_tokens": 0,
                                 "retry_count": 0}
            result = await planner_node(state, config)
            assert "red_flag_screen" in result.get("required_outputs", [])
            assert "referral_advice" in result.get("required_outputs", [])

    @pytest.mark.asyncio
    async def test_planner_llm_error_fallback(self):
        """Planner returns safe fallback on LLM error (Gemini fallback unavailable)."""
        from langgraph_agents.nodes.planner import planner_node
        from langchain_core.runnables import RunnableConfig

        with patch("langgraph_agents.nodes.planner.get_chat_model") as mock_llm, \
             patch("langgraph_agents.nodes.planner.get_fallback_chat_model", return_value=None):
            mock_llm.return_value.with_structured_output.return_value.ainvoke = AsyncMock(
                side_effect=RuntimeError("LLM down")
            )
            config = RunnableConfig(configurable={
                "user_id": "test", "session_id": "test-sess",
                "query": "hello", "persona_id": "eca_default",
                "request_id": "req-001",
            })
            state: AgentState = {"messages": [], "errors": [], "total_tokens": 0,
                                 "retry_count": 0}
            result = await planner_node(state, config)
            assert result["needs_clarification"] is True
            assert len(result.get("errors", [])) >= 1

    @pytest.mark.asyncio
    async def test_planner_chain_broken_handles_none(self):
        """Planner returns fallback when structured output returns None."""
        from langgraph_agents.nodes.planner import planner_node
        from langchain_core.runnables import RunnableConfig

        with patch("langgraph_agents.nodes.planner.get_chat_model") as mock_llm:
            mock_llm.return_value.with_structured_output.return_value.ainvoke = AsyncMock(
                return_value=None
            )
            config = RunnableConfig(configurable={
                "user_id": "test", "session_id": "test-sess",
                "query": "hello", "persona_id": "eca_default",
                "request_id": "req-001",
            })
            state: AgentState = {"messages": [], "errors": [], "total_tokens": 0,
                                 "retry_count": 0}
            result = await planner_node(state, config)
            assert result.get("needs_clarification") is True

    @pytest.mark.asyncio
    async def test_planner_creates_resolved_query(self):
        """Planner resolves coreference (D9)."""
        from langgraph_agents.nodes.planner import planner_node
        from langchain_core.runnables import RunnableConfig

        fake_plan = PlanOutput(
            required_outputs=["exercise_protocol"],
            resolved_query="chi tiet bai tap bird-dog",  # resolved from "dao sau ve no"
            needs_retrieval=True,
        )

        with patch("langgraph_agents.nodes.planner.get_chat_model") as mock_llm:
            mock_llm.return_value.with_structured_output.return_value.ainvoke = AsyncMock(
                return_value=fake_plan
            )
            config = RunnableConfig(configurable={
                "user_id": "test", "session_id": "test-sess",
                "query": "dao sau ve no", "persona_id": "eca_default",
                "request_id": "req-001",
            })
            state: AgentState = {"messages": [], "errors": [], "total_tokens": 0,
                                 "retry_count": 0}
            result = await planner_node(state, config)
            assert result["resolved_query"] == "chi tiet bai tap bird-dog"


@pytest.mark.unit
class TestPlanOutputSerialization:
    """Verify JSON serialization for LangGraph checkpointing."""

    def test_model_dump_json(self):
        plan = PlanOutput(
            required_outputs=["exercise_protocol"],
            resolved_query="test",
            needs_retrieval=True,
            needs_motion=False,
            needs_clarification=False,
        )
        json_str = plan.model_dump_json()
        assert '"exercise_protocol"' in json_str
        assert '"needs_retrieval":true' in json_str

    def test_empty_plan_serializable(self):
        plan = PlanOutput()
        d = plan.model_dump()
        assert d["required_outputs"] == []
        assert d["needs_retrieval"] is False
