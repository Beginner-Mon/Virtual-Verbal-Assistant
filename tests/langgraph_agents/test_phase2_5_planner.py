"""Tests for planner node (Task 2.5.3)."""

import pytest

from langgraph_agents.nodes.planner import PlanOutput


@pytest.mark.unit
def test_plan_output_defaults():
    p = PlanOutput(intent="knowledge_query", confidence=0.9, expanded_query="test")
    assert p.intent == "knowledge_query"
    assert p.confidence == 0.9
    assert p.needs_clarification is False
    assert p.required_outputs == []
    assert p.search_strategy == []


@pytest.mark.unit
def test_plan_output_serialization():
    p = PlanOutput(
        intent="exercise_recommendation",
        confidence=0.85,
        expanded_query="lower back pain exercises",
        required_outputs=["exercise_name", "safety_warnings"],
        search_strategy=["pgvector_search"],
        constraints_detected=["avoid_hplx"],
    )
    d = p.model_dump()
    assert d["intent"] == "exercise_recommendation"
    assert "lower back" in d["expanded_query"]
    assert "pgvector_search" in d["search_strategy"]


@pytest.mark.unit
def test_plan_output_confidence_range():
    with pytest.raises(Exception):
        PlanOutput(intent="test", confidence=1.5, expanded_query="x")
    with pytest.raises(Exception):
        PlanOutput(intent="test", confidence=-0.1, expanded_query="x")


@pytest.mark.unit
@pytest.mark.asyncio
async def test_planner_forces_clarification_on_ambiguous_ltm(monkeypatch):
    """Regression: when memory.long_term.ambiguous=True, planner MUST set needs_clarification."""
    from langgraph_agents.nodes import planner as planner_mod

    async def _fake_invoke(_messages):
        # LLM erroneously says "no clarification needed" — planner must override
        return PlanOutput(
            intent="exercise_recommendation",
            confidence=0.85,
            expanded_query="bai tap lung",
            needs_clarification=False,
            clarification_question=None,
        )

    class _FakeStructured:
        async def ainvoke(self, messages):
            return await _fake_invoke(messages)

    class _FakeLLM:
        def with_structured_output(self, _model, **_kwargs):
            return _FakeStructured()

    monkeypatch.setattr(planner_mod, "get_chat_model", lambda role: _FakeLLM())

    state = {
        "messages": [],
        "errors": [],
        "memory_context": {
            "short_term": [],
            "long_term": {"ambiguous": True, "sessions": [{"session_id": "a"}, {"session_id": "b"}]},
            "user_profile": {},
        },
    }
    config = {"configurable": {"query": "ban con nho bai tap tuan truoc khong"}}

    out = await planner_mod.planner_node(state, config)
    assert out["needs_clarification"] is True
    assert out["plan"].get("clarification_question"), "Planner must fill clarification_question when LLM omitted it"
