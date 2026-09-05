"""Tests for conversation node — dual mode (Task 2.5.9)."""

import pytest

from langgraph_agents.nodes._persona_loader import get_persona, build_persona_prompt


@pytest.mark.unit
def test_persona_loader_get_default():
    # eca_default was deleted on 04-09; `anne` is the declared default now.
    persona = get_persona("anne")
    assert persona["persona_id"] == "anne"
    assert persona["identity"]  # non-empty
    assert persona["personality"]


@pytest.mark.unit
def test_persona_build_prompt():
    persona = get_persona("anne")
    prompt = build_persona_prompt(persona, "synthesize")
    assert persona["identity"] in prompt
    assert "Personality" in prompt
    assert "Rules" in prompt


@pytest.mark.unit
def test_a_missing_persona_raises_rather_than_substituting():
    """Renamed from test_persona_fallback — there is no fallback any more.

    A generic stand-in meant a broken character silently became a different
    one: the user kept looking at Bronya's avatar while reading words written
    for nobody. Owner's call — answering as the wrong character is worse than
    not answering at all.
    """
    from langgraph_agents.nodes._persona_loader import PersonaError

    with pytest.raises(PersonaError):
        get_persona("non_existent_persona")
