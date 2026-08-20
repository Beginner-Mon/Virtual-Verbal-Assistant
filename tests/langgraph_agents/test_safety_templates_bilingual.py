# -*- coding: utf-8 -*-
"""The grader injects safety text VERBATIM — it never calls an LLM.

So the language of that text is decided here, not by the model, and an English
answer used to receive a Vietnamese warning stapled to the front of it. That is
worst on red_flag_screen: the one line the reader must not skip.
"""

import pytest

from langgraph_agents.nodes.grader import (
    DEFAULT_SAFETY_TEMPLATES,
    DEFAULT_SAFETY_TEMPLATES_EN,
    get_safety_text,
    grader_node,
)
from langgraph_agents.state import AgentState

CHARACTERS = ["anne", "bronya", "miki", "hatsune-miku"]
SAFETY_TAGS = ["red_flag_screen", "referral_advice", "scope_disclaimer"]


@pytest.mark.unit
@pytest.mark.parametrize("slug", CHARACTERS + ["eca_default"])
@pytest.mark.parametrize("tag", SAFETY_TAGS)
def test_every_character_has_both_languages(slug, tag):
    vi = get_safety_text(tag, slug, "vi")
    en = get_safety_text(tag, slug, "en")
    assert vi and en
    assert vi != en, f"{slug}.{tag}: the English variant is just the Vietnamese one"


@pytest.mark.unit
@pytest.mark.parametrize("slug", CHARACTERS + ["eca_default"])
@pytest.mark.parametrize("tag", SAFETY_TAGS)
def test_language_variants_do_not_contain_the_other_alphabet(slug, tag):
    """A cheap smoke test that catches a copy-paste of the wrong line."""
    from langgraph_agents.shared.lang import _VN_SET

    en = get_safety_text(tag, slug, "en")
    assert not any(c in _VN_SET for c in en), f"{slug}.{tag}.en still reads Vietnamese"


@pytest.mark.unit
def test_default_falls_back_when_a_persona_defines_nothing():
    """A persona with no templates at all must still warn, in both languages."""
    for tag in SAFETY_TAGS:
        assert get_safety_text(tag, "does-not-exist", "vi") == DEFAULT_SAFETY_TEMPLATES[tag]
        assert get_safety_text(tag, "does-not-exist", "en") == DEFAULT_SAFETY_TEMPLATES_EN[tag]


@pytest.mark.unit
def test_persona_customisation_survives_a_missing_en_variant():
    """Resolution is `<tag>.<lang>` → `<tag>` → default.

    The persona's own line sits ABOVE the generic English default on purpose: a
    persona that customised its warning did so for a reason, and silently
    replacing it with boilerplate would drop that.
    """
    import langgraph_agents.nodes._persona_loader as loader

    loader._persona_cache["_probe"] = {
        "persona_id": "_probe",
        "safety_templates": {"red_flag_screen": "CUSTOM ONLY"},
    }
    try:
        assert get_safety_text("red_flag_screen", "_probe", "en") == "CUSTOM ONLY"
        assert get_safety_text("referral_advice", "_probe", "en") == \
            DEFAULT_SAFETY_TEMPLATES_EN["referral_advice"]
    finally:
        loader._persona_cache.pop("_probe", None)


@pytest.mark.unit
def test_default_lang_keeps_existing_callers_on_vietnamese():
    """`lang` defaults to "vi" so nothing that predates this change moves."""
    assert get_safety_text("red_flag_screen", "anne") == \
        get_safety_text("red_flag_screen", "anne", "vi")


@pytest.mark.asyncio
async def test_english_answer_gets_an_english_warning_injected():
    """End to end through the node — the behaviour Tri would actually observe."""
    state: AgentState = {
        "messages": [], "errors": [], "retry_count": 0, "total_tokens": 0,
        "required_outputs": ["red_flag_screen"],
        "final_answer": (
            "Try three sets of ten repetitions and stop if the pain goes above "
            "4 out of 10, then rest for a full day before the next session."
        ),
    }
    config = {"configurable": {"persona_id": "anne", "query": "What stretches should I do?"}}
    result = await grader_node(state, config)

    assert result["grader_result"] == "pass_with_warning"
    injected = result["final_answer"]
    assert get_safety_text("red_flag_screen", "anne", "en") in injected
    assert get_safety_text("red_flag_screen", "anne", "vi") not in injected


@pytest.mark.asyncio
async def test_vietnamese_answer_still_gets_the_vietnamese_warning():
    state: AgentState = {
        "messages": [], "errors": [], "retry_count": 0, "total_tokens": 0,
        "required_outputs": ["red_flag_screen"],
        "final_answer": "Bạn nên tập bài kéo giãn cơ lưng dưới, mỗi hiệp 10 lần, thở đều.",
    }
    config = {"configurable": {"persona_id": "anne", "query": "Tôi nên tập bài nào?"}}
    result = await grader_node(state, config)

    assert get_safety_text("red_flag_screen", "anne", "vi") in result["final_answer"]


@pytest.mark.asyncio
async def test_node_works_without_a_query_in_config():
    """/tts and older call sites do not put `query` in configurable."""
    state: AgentState = {
        "messages": [], "errors": [], "retry_count": 0, "total_tokens": 0,
        "required_outputs": ["red_flag_screen"],
        "final_answer": "Bạn nên ngừng tập và đi khám bác sĩ ngay.",
    }
    result = await grader_node(state, {"configurable": {"persona_id": "anne"}})
    assert result["grader_result"] in ("pass", "pass_with_warning")
