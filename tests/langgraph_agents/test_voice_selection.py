# -*- coding: utf-8 -*-
"""Tests for services/vieneu_tts/voice.py and the language hop to SpeechLLm."""

import pytest

from langgraph_agents.services.vieneu_tts.voice import resolve_voice


VI = "Bạn nên tập bài kéo giãn cơ lưng dưới, mỗi hiệp 10 lần, thở đều."
EN = "Try three sets of ten repetitions, and stop if the pain goes above 4/10."


@pytest.mark.unit
@pytest.mark.parametrize("slug", ["anne", "bronya", "miki", "hatsune-miku"])
def test_path_follows_character_and_language(slug):
    assert resolve_voice(slug, VI) == (f"voices/{slug}_vi.wav", "vi")
    assert resolve_voice(slug, EN) == (f"voices/{slug}_en.wav", "en")


@pytest.mark.unit
def test_same_character_switches_file_with_the_answer_language():
    """The point of the whole change: one character, two voices."""
    vi_path, _ = resolve_voice("anne", VI)
    en_path, _ = resolve_voice("anne", EN)
    assert vi_path != en_path


@pytest.mark.unit
def test_query_breaks_the_tie_for_a_signal_free_answer():
    assert resolve_voice("anne", "OK", query="What stretches should I do?")[1] == "en"
    assert resolve_voice("anne", "OK", query="Tôi nên tập bài nào?")[1] == "vi"


@pytest.mark.unit
def test_persona_language_is_the_last_resort():
    """Neither answer nor query readable — the character's own language wins,
    rather than a hardcoded default that would mute half the catalog."""
    assert resolve_voice("anne", "OK", persona_lang="en")[1] == "en"
    assert resolve_voice("anne", "OK", persona_lang="vi")[1] == "vi"


@pytest.mark.unit
@pytest.mark.parametrize("bad", [
    "../../etc/passwd",
    "anne/../../secret",
    "a" * 65,
    "",
    "has space",
])
def test_unsafe_persona_id_yields_no_path(bad):
    """persona_id arrives in a request body and is about to become part of a
    path on another host. A rejected id must still return a usable language, so
    the caller degrades to the preset voice instead of failing the turn."""
    path, lang = resolve_voice(bad, VI)
    assert path is None
    assert lang in ("vi", "en")


@pytest.mark.unit
def test_language_reaches_the_speechllm_payload():
    """`language` was accepted by SpeechLLm from the start and never sent, so
    every generated file claimed to be English. Guards the wiring, not the API."""
    from langgraph_agents.services.vieneu_tts.client import VieNeuTTSClient

    payload = VieNeuTTSClient._payload("xin chào", "voices/anne_vi.wav", "vi")
    assert payload == {
        "text": "xin chào",
        "voice_path": "voices/anne_vi.wav",
        "language": "vi",
    }


@pytest.mark.unit
def test_payload_omits_what_it_does_not_have():
    from langgraph_agents.services.vieneu_tts.client import VieNeuTTSClient

    assert VieNeuTTSClient._payload("hello", None, None) == {"text": "hello"}


@pytest.mark.unit
def test_persona_voice_path_is_no_longer_consulted():
    """`voice_identity.voice_path` pinned one file per character and could not
    express "this character, in English". If it comes back, this test says so."""
    from langgraph_agents.nodes._persona_loader import _load_persona

    for slug in ("anne", "bronya", "miki", "hatsune-miku",
                 "eca_default", "eca_clinical", "eca_friendly"):
        assert not _load_persona(slug)["voice_identity"].get("voice_path"), (
            f"{slug}.md declares voice_path again; resolve_voice derives the "
            "name from the slug and will ignore it"
        )
