"""Tests for conversation node — dual mode (Task 2.5.9)."""

import pytest

from langgraph_agents.nodes._persona_loader import get_persona, build_persona_prompt


@pytest.mark.unit
def test_persona_loader_get_default():
    persona = get_persona("eca_default")
    assert persona["persona_id"] == "eca_default"
    assert persona["identity"]  # non-empty
    assert persona["personality"]


@pytest.mark.unit
def test_persona_build_prompt():
    persona = get_persona("eca_default")
    prompt = build_persona_prompt(persona, "knowledge_query")
    assert persona["identity"] in prompt
    assert "Personality" in prompt
    assert "Rules" in prompt


@pytest.mark.unit
def test_persona_fallback():
    persona = get_persona("non_existent_persona")
    assert persona["persona_id"] == "non_existent_persona"
    assert persona["title"] == "ECA Default"
