"""A character speaks one language per turn, and it is the one that was asked for.

This closes the gap `test_prompt_purity.py` names in its own docstring and
cannot cover. That file scans the prompt constants inside the nodes; the persona
reaches the model down a different path — `build_voice_card()` places the
character's sample lines LAST in the message list, deliberately, to buy recency
(see its docstring). Before the overlay split those lines were Vietnamese for
every character, so an English turn ended with Vietnamese as the most imitable
text in the whole prompt.

Also pins the two behaviours Owner asked for explicitly:
  - a persona that cannot be loaded RAISES; it is never silently replaced by a
    stand-in, because answering as the wrong character is worse than not
    answering
  - `identity` and the behavioural rules are shared across languages; only the
    way the character SOUNDS is per-language
"""

from __future__ import annotations

import pytest

from langgraph_agents.nodes._persona_loader import (
    build_persona_prompt,
    build_voice_card,
    get_persona,
)
from langgraph_agents.shared.lang import _VN_EXCLUSIVE

CHARACTERS = ["anne", "bronya", "hatsune-miku", "miki"]
MODES = ["chat", "clarify", "refuse", "synthesize"]

_VN_CHARS = frozenset(_VN_EXCLUSIVE + _VN_EXCLUSIVE.upper())


def _vietnamese_in(text: str) -> str:
    return "".join(sorted({c for c in text if c in _VN_CHARS}))


@pytest.mark.unit
@pytest.mark.parametrize("slug", CHARACTERS)
@pytest.mark.parametrize("mode", MODES)
def test_an_english_turn_carries_no_vietnamese(slug: str, mode: str) -> None:
    """Everything persona-derived that reaches the model, for an English turn.

    Both halves matter and the second is the one that used to fail: the voice
    card is the LAST message, closest to the generation point, and whatever sits
    there is what the model answers in the register of.
    """
    persona = get_persona(slug, "en")

    system = build_persona_prompt(persona, mode)
    voice_card = build_voice_card(persona, mode)

    for label, text in (("system prompt", system), ("voice card", voice_card)):
        leaked = _vietnamese_in(text)
        assert not leaked, (
            f"{slug} ({mode}) — the {label} of an ENGLISH turn contains "
            f"Vietnamese [{leaked}].\n\n{text}"
        )


@pytest.mark.unit
@pytest.mark.parametrize("slug", CHARACTERS)
def test_vietnamese_turn_still_sounds_vietnamese(slug: str) -> None:
    """The other direction — the split must not flatten the Vietnamese voice.

    A green `test_an_english_turn_carries_no_vietnamese` is trivially achievable
    by deleting the Vietnamese content. This is what stops that.
    """
    persona = get_persona(slug, "vi")
    card = build_voice_card(persona, "chat")
    assert _vietnamese_in(card), f"{slug}: the Vietnamese voice card reads as English"


@pytest.mark.unit
@pytest.mark.parametrize("slug", CHARACTERS)
def test_core_is_shared_and_only_the_voice_differs(slug: str) -> None:
    vi = get_persona(slug, "vi")
    en = get_persona(slug, "en")

    # Core: who the character is and what they may do. One source, both languages.
    for field in ("identity", "personality", "behavioral_rules", "response_formatting"):
        assert vi[field] == en[field], f"{slug}.{field} should not differ per language"

    # Overlay: how they sound, and text inserted verbatim.
    assert vi["safety_templates"] != en["safety_templates"]
    assert vi["ui_strings"] != en["ui_strings"]


@pytest.mark.unit
@pytest.mark.parametrize("slug", CHARACTERS)
def test_safety_templates_exist_in_both_languages(slug: str) -> None:
    for lang in ("vi", "en"):
        templates = get_persona(slug, lang)["safety_templates"]
        for tag in ("red_flag_screen", "referral_advice", "scope_disclaimer"):
            assert templates.get(tag), f"{slug}.{lang} is missing {tag}"


# ── Failure is loud, not silent ──────────────────────────────────────────────


@pytest.mark.unit
@pytest.mark.parametrize(
    "bad_slug",
    # eca_default is here deliberately: it was a real slug until 04-09, so a
    # stale client or a stale DB row can still ask for it, and it must fail
    # rather than resurrect a generic voice.
    ["does-not-exist", "eca_default", "../../../README", "..%2F..%2Fetc", "a/b"],
)
def test_an_unloadable_persona_raises(bad_slug: str) -> None:
    """No stand-in. Answering as the wrong character is worse than not answering.

    Covers the deleted personas too: `eca_default` is gone, and asking for it
    must fail rather than resurrect a generic voice.
    """
    with pytest.raises(Exception):
        get_persona(bad_slug, "vi")


@pytest.mark.unit
@pytest.mark.parametrize("bad_lang", ["fr", "", "../vi", "vi/../en"])
def test_an_unsupported_language_raises(bad_lang: str) -> None:
    """`lang` becomes part of a filesystem path, so it is validated like the slug."""
    with pytest.raises(Exception):
        get_persona("anne", bad_lang)


@pytest.mark.unit
def test_the_cache_key_is_still_a_plain_slug() -> None:
    """Guards the decision NOT to re-key the cache on (slug, lang).

    `test_a0_persona_security.py` reaches into `_persona_cache` with a string
    key to prove an invalid id never poisons it. Keying by tuple would have
    forced that security test to change shape to suit a feature; instead the
    cache holds every language under one slug and `get_persona` flattens on
    read. If someone re-keys it later, this fails and points at why.
    """
    from langgraph_agents.nodes._persona_loader import _persona_cache

    get_persona("anne", "vi")
    get_persona("anne", "en")

    assert "anne" in _persona_cache
    assert all(isinstance(k, str) for k in _persona_cache)
