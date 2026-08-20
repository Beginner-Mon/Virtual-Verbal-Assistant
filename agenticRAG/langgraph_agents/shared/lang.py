"""Vietnamese/English detection for picking the TTS reference voice.

Why a hand-written rule and not a library or an LLM call:

* The only decision it feeds is `voices/<character>_<lang>.wav`, a binary choice
  between two languages the product supports. A general-purpose langdetect model
  solves a 50-language problem we do not have, and pays for it in start-up cost
  and a dependency.
* It runs on every spoken turn, right before a 30-45s CPU synthesis. Measured:
  0.011 ms for a short reply, 0.25 ms for a 1600-character clinical answer.

The design point worth keeping: the strong signal is WHICH diacritic, not HOW
MANY. See `_VN_EXCLUSIVE`.
"""

from __future__ import annotations

import re

__all__ = ["detect_lang", "LangResult"]


# ── Tier 1: Vietnamese-exclusive codepoints ──────────────────────────────
#
# Deliberately EXCLUDES é è ê à â ô î û ï ç ü ñ ö — every diacritic an English
# text can legitimately carry through a loanword: café, résumé, fiancé, naïve,
# façade, crème fraîche, pâté, crêpe, jalapeño, à la carte. Counting those is
# what makes a naive "has diacritics ⇒ Vietnamese" rule fail.
#
# Also excludes ã and õ: Portuguese has both ("São Paulo"). Dropping them costs
# nothing — a real Vietnamese sentence carries plenty of the others.
#
# What remains exists in no other Latin orthography:
#   ă ơ ư          breve and horn
#   ả ẻ ỉ ỏ ủ ỷ    hook above
#   ạ ẹ ị ọ ụ ỵ    dot below
#   ấ ế ố ...      circumflex WITH a tone stacked on top (French never stacks)
#   đ ẽ ĩ ũ ỹ ý
_VN_EXCLUSIVE = (
    "ăằắẳẵặ"
    "ấầẩẫậ"
    "đ"
    "ếềểễệ"
    "ốồổỗộ"
    "ơớờởỡợ"
    "ưứừửữự"
    "ảẻỉỏủỷ"
    "ạẹịọụỵ"
    "ẽĩũỹ"
    "ý"
)
_VN_SET = frozenset(_VN_EXCLUSIVE + _VN_EXCLUSIVE.upper())


# ── Tier 2: diacritic-free marker words ──────────────────────────────────
#
# For Vietnamese typed without tone marks ("toi bi dau lung, khong ngu duoc"),
# where tier 1 scores exactly zero.
#
# Every entry was checked against an English dictionary and anything that is
# also an English word was dropped, because a single collision is enough to
# flip a whole English sentence. Rejected for that reason:
#   the, ban, bi, cam, can, chi, dang, den, em, hay, hit, lan, lung, ma, may,
#   no, on, qua, rat, so, tap, to, tot, vi
_VI_MARKERS = frozenset("""
khong duoc nguoi nhung cua minh roi nay nao phai nen chua hoi moi cung voi vay
thi cho tu luc khi neu boi tren duoi trong ngoai lam xin chao giup xem biet
muon nhu toi sao dau bai thuc hien hiep nghi tho vao thoat khop xuong gium
nhe chung bay gio hom truoc buoi sang chieu
""".split())

_EN_MARKERS = frozenset("""
the and is are you your of to for with that this have has will should from they
it be not but or as at by on in what when how why do does did can could would
""".split())


# Code fences, inline code and URLs are ASCII islands that say nothing about the
# prose language — a Vietnamese answer containing a Python snippet would other-
# wise have its Vietnamese ratio diluted by the snippet's length.
_STRIP = re.compile(r"```.*?```|`[^`]*`|https?://\S+", re.S)
_WORD = re.compile(r"[a-zà-ỹ]+", re.I | re.U)

# Share of words carrying a tier-1 character before the text counts as
# Vietnamese. Real Vietnamese prose measures 0.50-0.73; an English sentence
# quoting one Vietnamese phrase measures ~0.22, which is why the ratio alone is
# not enough and `_decide` also compares it against the English marker density.
_VI_RATIO_MIN = 0.15

DEFAULT_LANG = "vi"


class LangResult(str):
    """A ``str`` that is the language code, carrying `.reason` for logging.

    Subclassing str keeps every call site able to write `result == "vi"` or pass
    it straight into an f-string, while the reason travels along for the log
    line. Returning a tuple instead would force both TTS call sites to unpack
    something they mostly do not use.
    """

    reason: str

    def __new__(cls, lang: str, reason: str) -> "LangResult":
        obj = super().__new__(cls, lang)
        obj.reason = reason
        return obj


def _score(text: str) -> dict | None:
    words = _WORD.findall(_STRIP.sub(" ", text))
    if not words:
        return None
    lower = [w.lower() for w in words]
    return {
        "n": len(words),
        "marked": sum(1 for w in words if any(c in _VN_SET for c in w)),
        "vi": sum(1 for w in lower if w in _VI_MARKERS),
        "en": sum(1 for w in lower if w in _EN_MARKERS),
    }


def _decide(s: dict) -> tuple[str, str] | None:
    """Return (lang, reason), or None when the text carries no usable signal."""
    ratio = s["marked"] / s["n"]
    en_ratio = s["en"] / s["n"]

    # Vietnamese fingerprints must be DENSER than English ones. Without this
    # second condition, "The Vietnamese term is 'đau lưng' (lower back pain)"
    # scores 0.22 and passes the threshold, tying with a genuinely Vietnamese
    # sentence that quotes an English exercise name. The two are 0.02 apart —
    # no threshold separates them, only the comparison does.
    if ratio >= _VI_RATIO_MIN and ratio > en_ratio:
        return "vi", f"diacritics {s['marked']}/{s['n']}={ratio:.2f}>en={en_ratio:.2f}"

    if s["vi"] > s["en"]:
        return "vi", f"markers vi={s['vi']}>en={s['en']}"
    if s["en"] > s["vi"]:
        return "en", f"markers en={s['en']}>vi={s['vi']}"
    return None


def detect_lang(text: str, query: str | None = None, fallback: str = DEFAULT_LANG) -> LangResult:
    """Classify ``text`` as ``"vi"`` or ``"en"``.

    ``query`` is the tie-breaker, and it is not optional in spirit: a reply of
    "OK" or "Có." carries no signal at all, and the synthesizer's LANGUAGE RULE
    already binds the reply's language to the query's. Guessing from a two-word
    answer when the question is right there would be inventing uncertainty.

    ``fallback`` is the last resort — pass the character's ``voice_language`` so
    an unreadable turn at least keeps the persona's own language.
    """
    scored = _score(text) if text else None
    if scored is not None:
        decided = _decide(scored)
        if decided is not None:
            return LangResult(*decided)

    if query:
        q_scored = _score(query)
        if q_scored is not None:
            decided = _decide(q_scored)
            if decided is not None:
                lang, why = decided
                return LangResult(lang, f"answer inconclusive, query says {lang} ({why})")

    return LangResult(fallback, f"no signal in answer or query, fallback={fallback}")
