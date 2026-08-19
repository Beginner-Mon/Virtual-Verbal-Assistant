# -*- coding: utf-8 -*-
"""Tests for shared/lang.py — the vi/en detector behind TTS voice selection.

The cases are adversarial on purpose. A detector that only sees clean
Vietnamese and clean English passes with any implementation, including the
naive "has diacritics ⇒ Vietnamese" one that this module exists to avoid.
"""

import pytest

from langgraph_agents.shared.lang import detect_lang


# ── Clean prose — the boring majority of real traffic ────────────────────

@pytest.mark.unit
@pytest.mark.parametrize("text", [
    "Bạn nên tập bài kéo giãn cơ lưng dưới, 3 hiệp mỗi hiệp 10 lần, thở đều.",
    "⚠️ Đau ngực khi tập thể dục có thể là dấu hiệu nghiêm trọng. Bạn nên NGỪNG tập ngay.",
    "Vâng ạ.",
    "Chào bạn!",
])
def test_plain_vietnamese(text):
    assert detect_lang(text) == "vi"


@pytest.mark.unit
@pytest.mark.parametrize("text", [
    "Try three sets of ten repetitions, and stop if the pain goes above 4/10.",
    "Hold the plank for 30 seconds, rest 15 seconds, repeat three times.",
])
def test_plain_english(text):
    assert detect_lang(text) == "en"


# ── The reason this module is not a diacritic counter ────────────────────

@pytest.mark.unit
@pytest.mark.parametrize("text", [
    "The café near my fiancé's résumé exposé was naïve about the façade.",
    "Her role in the crêpe soufflé décor was a cliché déjà vu.",
    "A pâté of jalapeño and crème fraîche, naïvely served à la carte.",
    "São Paulo and Zürich are both over-rated, señor.",
])
def test_english_with_loanword_diacritics(text):
    """é è ê à â ô ï ç ü ñ ã carry no weight — only Vietnamese-exclusive marks do.

    The last case matters separately: 'São' is the one Portuguese form close
    enough to Vietnamese ã that ã had to be dropped from the exclusive set.
    """
    assert detect_lang(text) == "en"


# ── Vietnamese typed without tone marks — tier 1 scores zero ─────────────

@pytest.mark.unit
@pytest.mark.parametrize("text", [
    "toi bi dau lung va khong the ngu duoc",
    "ban co the cho toi biet bai tap nao phu hop khong",
    "toi bi dau lung, khong ngu duoc mấy hôm nay",
])
def test_vietnamese_without_diacritics(text):
    assert detect_lang(text) == "vi"


# ── Mixed text: which language is quoting which ──────────────────────────

@pytest.mark.unit
def test_vietnamese_quoting_english_exercise_name():
    assert detect_lang("Bài tập này gọi là 'Bird Dog' — hold for 10 seconds.") == "vi"


@pytest.mark.unit
def test_english_quoting_vietnamese_phrase():
    """The hardest case, and the reason `_decide` compares two densities.

    This scores 0.22 on the Vietnamese ratio; the Vietnamese sentence above
    scores 0.20. No threshold separates them — only the comparison against
    English marker density does.
    """
    assert detect_lang("The Vietnamese term is 'đau lưng' (lower back pain).") == "en"


@pytest.mark.unit
def test_english_with_vietnamese_proper_noun():
    text = ("This exercise, known as Đau Lưng Stretch, targets the erector spinae "
            "group and should be held for thirty seconds each side.")
    assert detect_lang(text) == "en"


@pytest.mark.unit
def test_vietnamese_dense_with_english_terms():
    assert detect_lang("Plank giữ 30 giây, sau đó nghỉ 15 giây rồi lặp lại 3 lần.") == "vi"


@pytest.mark.unit
def test_code_block_does_not_vote():
    """An English answer whose snippet happens to contain Vietnamese words."""
    text = "```python\nprint('xin chao ban')\n```\nHere is the script you asked for."
    assert detect_lang(text) == "en"


# ── No signal: the query decides, not a guess ────────────────────────────

@pytest.mark.unit
def test_short_answer_defers_to_query():
    assert detect_lang("Có.", query="Tôi có nên tập không?") == "vi"
    assert detect_lang("Hello", query="How are you today?") == "en"


@pytest.mark.unit
def test_short_answer_without_query_uses_fallback():
    assert detect_lang("OK") == "vi"
    assert detect_lang("OK", fallback="en") == "en"


@pytest.mark.unit
def test_query_only_consulted_when_answer_is_inconclusive():
    """A confident answer must win over a query in the other language.

    Otherwise a user asking in English and getting a Vietnamese answer (the
    model disobeying the LANGUAGE RULE) would be read aloud in the wrong voice
    — the failure this whole path exists to prevent.
    """
    vi_answer = "Bạn nên tập bài kéo giãn cơ lưng dưới, mỗi hiệp 10 lần."
    assert detect_lang(vi_answer, query="What stretches should I do?") == "vi"


@pytest.mark.unit
@pytest.mark.parametrize("text", ["", "   ", "123 456", "!!!", "😀🎉"])
def test_no_letters_never_raises(text):
    assert detect_lang(text) in ("vi", "en")


# ── The reason travels with the result, for the log line ────────────────

@pytest.mark.unit
def test_result_carries_a_reason():
    result = detect_lang("Bạn nên nghỉ ngơi và chườm ấm vùng thắt lưng.")
    assert result == "vi"
    assert result.reason
    assert "diacritics" in result.reason


@pytest.mark.unit
def test_fallback_reason_says_so():
    assert "fallback" in detect_lang("OK").reason


# ── Mutation checks: assert the design, not just the outcome ─────────────

@pytest.mark.unit
def test_marker_lists_do_not_collide():
    """A word in both lists cancels itself out and silently weakens the tier-2
    test. `the` was in the Vietnamese list in the first draft (from `the nao`)
    and flipped whole English sentences to Vietnamese."""
    from langgraph_agents.shared.lang import _EN_MARKERS, _VI_MARKERS
    assert not (_VI_MARKERS & _EN_MARKERS)


@pytest.mark.unit
@pytest.mark.parametrize("char", list("éèêàâôîûïçüñöãõ"))
def test_loanword_diacritics_are_not_in_the_exclusive_set(char):
    """Guards the single decision this module rests on. Adding any of these
    back would make every `café` read aloud in a Vietnamese voice."""
    from langgraph_agents.shared.lang import _VN_SET
    assert char not in _VN_SET


@pytest.mark.unit
@pytest.mark.parametrize("char", list("ăơưảẹđấếốỗựỷ"))
def test_vietnamese_exclusive_marks_are_present(char):
    from langgraph_agents.shared.lang import _VN_SET
    assert char in _VN_SET


@pytest.mark.unit
def test_detector_is_fast_enough_for_the_tts_path():
    """It runs immediately before a 30-45s CPU synthesis; a full clinical answer
    measures ~0.25 ms. The bound is loose on purpose — this asserts no accidental
    quadratic blow-up, not a benchmark."""
    import time
    text = ("**Đánh giá**\nCơn đau vùng thắt lưng sau khi ngồi lâu thường liên quan "
            "đến căng cơ dựng sống và co cứng cơ gấp hông.\n") * 20
    start = time.perf_counter()
    for _ in range(50):
        detect_lang(text)
    assert (time.perf_counter() - start) / 50 < 0.02
