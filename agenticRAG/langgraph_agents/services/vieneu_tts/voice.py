"""Pick the TTS reference voice for a character and a reply.

    voices/<character-slug>_<lang>.wav        e.g. voices/anne_vi.wav

Two decisions worth stating, because both were the other way round before:

1. The name is DERIVED from `persona_id`, not read from the persona's
   `voice_identity.voice_path`. That field pinned one file per character, which
   cannot express "this character, in English" — and only three legacy personas
   ever set it, none of them selectable in the UI.

2. This module does NOT check that the file exists. The path is resolved by
   SpeechLLm against ITS OWN working directory; those are two processes and, in
   the deployed layout, two machines. Checking here would pass locally and lie
   in production. The existence check and the fall back to the preset voice both
   live in `SpeechLLm/src/services/vieneu_client.py`, where the filesystem is.
"""

from __future__ import annotations

import re

from langgraph_agents.shared.lang import detect_lang
from langgraph_agents.shared.logging import get_logger

logger = get_logger("langgraph.tts.voice")

__all__ = ["resolve_voice", "VOICES_DIR"]

VOICES_DIR = "voices"

# Same shape `_persona_loader` already enforces before touching the filesystem.
# `persona_id` reaches us from the request body, and it is about to become part
# of a path on another host.
_SAFE_SLUG = re.compile(r"[A-Za-z0-9_-]{1,64}")


def resolve_voice(
    persona_id: str,
    text: str,
    query: str | None = None,
    persona_lang: str = "vi",
) -> tuple[str | None, str]:
    """Return ``(voice_path, lang)`` for a reply about to be spoken.

    ``voice_path`` is None only when ``persona_id`` is unusable as a filename —
    SpeechLLm then falls back to its preset voice, same as for a missing file.
    ``lang`` is always a real code, so the caller can pass it downstream.
    """
    lang = detect_lang(text, query=query, fallback=persona_lang)

    if not persona_id or not _SAFE_SLUG.fullmatch(persona_id):
        logger.warning(
            "voice_unsafe_persona_id",
            extra={"persona_id": (persona_id or "")[:80], "lang": str(lang)},
        )
        return None, str(lang)

    voice_path = f"{VOICES_DIR}/{persona_id}_{lang}.wav"
    logger.info(
        "voice_selected",
        extra={
            "persona_id": persona_id,
            "lang": str(lang),
            # Why this language, in the log rather than in a debugger: a wrong
            # voice is only audible, never visible, and "wrong voice" sounds
            # exactly like "right voice, bad model" to whoever reports it.
            "lang_reason": lang.reason,
            "voice_path": voice_path,
        },
    )
    return voice_path, str(lang)
