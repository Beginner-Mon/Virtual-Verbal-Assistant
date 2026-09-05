"""Persona loader — parse persona MD files, cache, build prompts.

Extracted from conversation.py for reuse.
"""

import json
import logging
import re
from pathlib import Path
from typing import Optional

import yaml

logger = logging.getLogger("langgraph.nodes.persona")

_persona_cache: dict[str, dict] = {}


def _load_config() -> dict:
    config_path = Path(__file__).resolve().parents[3] / "config" / "langgraph.yaml"
    if config_path.exists():
        with open(config_path, "r", encoding="utf-8") as f:
            return yaml.safe_load(f).get("langgraph", {})
    return {}


_CFG = _load_config()
_DEFAULT_PERSONA = _CFG.get("persona", {}).get("default", "anne")

# The languages a character can be authored in. A `lang` becomes part of a file
# path, so this is an allowlist and not merely a hint.
LANGS = ("vi", "en")
DEFAULT_LANG = "en"

# Sections that describe WHO the character is and what they may do. Written once,
# in English, in `_core.md`. The model reads them in English and still answers in
# the user's language — the same way every other instruction in the prompt works.
_CORE_SECTIONS = (
    "identity", "personality", "behavioral_rules", "response_formatting",
)

# Sections the model IMITATES, or that are inserted verbatim into something a
# person reads. These cannot be shared: "Xưng 'mình', gọi người dùng là 'bạn'"
# has no English equivalent, which is why `vi.md` and `en.md` are two authored
# voices rather than one voice and a translation.
_OVERLAY_SECTIONS = ("voice", "examples", "safety_templates", "ui_strings")


class PersonaError(RuntimeError):
    """A character could not be loaded, or was asked for in an unknown language.

    Raised rather than substituted. There used to be a `_fallback_persona()` that
    returned a generic "ECA Default" stand-in for any bad or missing id, which
    meant a broken persona silently became a DIFFERENT character: the user still
    saw Bronya's avatar while reading words written for nobody. Owner's call —
    answering as the wrong character is worse than not answering.
    """


def _resolve_personas_dir() -> Path:
    cfg_dir = _CFG.get("persona", {}).get("personas_dir", "langgraph_agents/personas")
    if not Path(cfg_dir).is_absolute():
        return Path(__file__).resolve().parents[2] / cfg_dir
    return Path(cfg_dir)


def _parse_voice_identity(text: str) -> dict:
    result: dict[str, Optional[str]] = {"voice_path": None, "language": "vi"}
    if not text:
        return result
    for line in text.split("\n"):
        if ":" in line:
            key, _, val = line.partition(":")
            key = key.strip().lower()
            val = val.strip().strip('"').strip("'")
            if key == "voice_path":
                result["voice_path"] = val
            elif key == "language":
                result["language"] = val
    return result


def _parse_key_values(text: str) -> dict[str, str]:
    """Parse a `key: "value"` section into a dict. One entry per line.

    Used by `## Safety Templates` and `## UI Strings` alike — the two sections
    have the same shape, so they share the parser rather than growing a second
    copy of the quote-stripping.
    """
    if not text:
        return {}
    result: dict[str, str] = {}
    for line in text.split("\n"):
        line = line.strip()
        if not line or ":" not in line:
            continue
        key, _, val = line.partition(":")
        key = key.strip()
        val = val.strip().strip('"').strip("'")
        if key and val:
            result[key] = val
    return result


def _safe_persona_file(persona_id: str, filename: str) -> Path:
    """`personas/<persona_id>/<filename>`, proven to be inside `personas/`.

    Two fences, both kept from the previous flat-file loader and now applied to
    the language segment as well, since `lang` also becomes part of a path:

      1. the id must match a strict character class before it touches the
         filesystem at all
      2. the RESOLVED path must still sit inside `personas/` — which catches
         anything symlinks or `..` could do that the regex did not anticipate

    Raises rather than returning a stand-in. See `PersonaError`.
    """
    if not re.fullmatch(r"[A-Za-z0-9_-]{1,64}", persona_id):
        logger.warning("invalid_persona_id", extra={"persona_id": persona_id[:80]})
        raise PersonaError(f"invalid persona id: {persona_id[:80]!r}")

    personas_dir = _resolve_personas_dir()
    resolved = (personas_dir / persona_id / filename).resolve()

    try:
        resolved.relative_to(personas_dir.resolve())
    except ValueError:
        logger.warning(
            "persona_path_traversal_attempt", extra={"persona_id": persona_id[:80]}
        )
        raise PersonaError(f"persona path escapes the personas directory: {persona_id[:80]!r}")

    if not resolved.is_file():
        logger.warning("persona_file_missing", extra={"path": str(resolved)})
        raise PersonaError(f"persona file not found: {resolved}")

    return resolved


def _parse_sections(content: str) -> dict:
    """Parse one markdown file into the section dict the rest of this module uses.

    Shared by `_core.md` and the language overlays — they have the same shape and
    differ only in which sections they are expected to carry.
    """
    sections: dict[str, str] = {}
    current_header = "identity"
    current_body: list[str] = []

    for line in content.split("\n"):
        header_match = re.match(r"^##\s+(.+)$", line)
        if header_match:
            if current_body:
                sections[current_header] = "\n".join(current_body).strip()
            current_header = header_match.group(1).lower().replace(" ", "_")
            current_body = []
        else:
            current_body.append(line)

    if current_body and current_header:
        sections[current_header] = "\n".join(current_body).strip()

    safety_raw = sections.pop("safety_templates", "")
    safety_templates = _parse_key_values(safety_raw)

    # Display copy for the chat surface — greeting, stage labels, error lines,
    # input placeholder. Popped like the two above so it never leaks into the
    # system prompt: these strings are for the user's screen, not the model.
    ui_raw = sections.pop("ui_strings", "")
    ui_strings = _parse_key_values(ui_raw)
    # ── Normalize greeting to nested object ──────────────────────────────
    # Authoring is flat dot-keys in markdown (greeting.morning: "...") for
    # readability — one line per slot — but the DB and frontend use a nested
    # object `greeting: {morning, afternoon, evening, night}`. This block
    # assembles the object so sync_personas_to_db.py writes the nested shape
    # and so that a plain string `greeting: "hi"` (old data) would not shadow it.
    # Also handles the case where greeting was authored as a JSON string
    # `greeting: {"morning": "..."}` in one line.
    _greeting_slots: dict[str, str] = {}
    for _k in list(ui_strings.keys()):
        if _k.startswith("greeting."):
            _slot = _k.split(".", 1)[1].strip()
            if _slot in ("morning", "afternoon", "evening", "night"):
                _greeting_slots[_slot] = ui_strings.pop(_k)
    # JSON single-line form: greeting: {"morning": "...", ...}
    if "greeting" in ui_strings and isinstance(ui_strings["greeting"], str):
        _raw_g = ui_strings["greeting"].strip()
        if _raw_g.startswith("{"):
            try:
                _parsed = json.loads(_raw_g)
                if isinstance(_parsed, dict):
                    for _s in ("morning", "afternoon", "evening", "night"):
                        if _s in _parsed and isinstance(_parsed[_s], str) and _parsed[_s].strip():
                            _greeting_slots[_s] = _parsed[_s].strip()
                    ui_strings.pop("greeting", None)
            except (ValueError, TypeError):
                pass
    if _greeting_slots:
        # Old flat string `greeting: "..."` must not survive alongside the object —
        # frontend now expects an object and "bỏ fallback" per spec.
        ui_strings.pop("greeting", None)
        ui_strings["greeting"] = _greeting_slots

    voice_lines = sections.pop("voice_identity", "")
    voice_identity = _parse_voice_identity(voice_lines)

    title_match = re.match(r"^#\s+(.+)$", content, re.MULTILINE)
    title = title_match.group(1).strip() if title_match else ""

    return {
        "title": title,
        "identity": sections.get("identity", ""),
        "voice_identity": voice_identity,
        "personality": sections.get("personality", ""),
        # `voice` is speech habits, `examples` verbatim sample lines — the two
        # sections a model actually imitates, which is why they live in the
        # language overlay rather than the shared core.
        "voice": sections.get("voice", ""),
        "examples": sections.get("examples", ""),
        "behavioral_rules": sections.get("behavioral_rules", ""),
        "response_formatting": sections.get("response_formatting", ""),
        "safety_templates": safety_templates,
        "ui_strings": ui_strings,
    }


def _load_persona(persona_id: str) -> dict:
    """Read `_core.md` plus every language overlay into ONE cache entry.

    Deliberately NOT one entry per (slug, language). `test_a0_persona_security`
    reaches into `_persona_cache` with a plain string key to prove an invalid id
    cannot poison it; re-keying on a tuple would have forced a security test to
    change shape in order to suit a feature. So the cache stores everything under
    the slug and `get_persona` flattens on read — a few dict lookups, no cost.
    """
    core = _parse_sections(_safe_persona_file(persona_id, "_core.md").read_text("utf-8"))

    locales: dict[str, dict] = {}
    for lang in LANGS:
        try:
            overlay = _parse_sections(
                _safe_persona_file(persona_id, f"{lang}.md").read_text("utf-8")
            )
        except PersonaError:
            # A character may ship only some languages. Asking for a missing one
            # raises in `get_persona`; it must not stop the ones that exist from
            # loading.
            continue
        locales[lang] = {k: overlay[k] for k in _OVERLAY_SECTIONS}

    if not locales:
        raise PersonaError(f"{persona_id!r} has no language overlay")

    return {
        "persona_id": persona_id,
        "title": core["title"],
        "voice_identity": core["voice_identity"],
        **{k: core[k] for k in _CORE_SECTIONS},
        "locales": locales,
    }


def get_persona(persona_id: str, lang: str = DEFAULT_LANG) -> dict:
    """One character in one language, flattened to the shape callers expect.

    The return value looks exactly like it did before overlays existed — core
    fields and overlay fields side by side — so `build_persona_prompt`,
    `build_voice_card` and `get_safety_text` read `persona["voice"]` and
    `persona["safety_templates"]` unchanged. They simply receive the right
    language's copy.

    Raises `PersonaError` for an unknown character or an unknown language. There
    is no stand-in: see `PersonaError`.
    """
    if lang not in LANGS:
        raise PersonaError(f"unsupported persona language: {lang!r}")

    parsed = _persona_cache.get(persona_id)
    if parsed is None:
        parsed = _load_persona(persona_id)
        _persona_cache[persona_id] = parsed

    overlay = parsed["locales"].get(lang)
    if overlay is None:
        raise PersonaError(f"{persona_id!r} has no {lang!r} overlay")

    flat = {k: v for k, v in parsed.items() if k != "locales"}
    flat.update(overlay)
    return flat


# Last-resort chat-surface copy, per language, for a persona that ships no
# `## UI Strings`. Deliberately character-neutral: a wrong name is worse than no
# name. This is the ONLY user-visible text left in this module, and it exists so
# that a missing string renders as something readable rather than as "".
_DEFAULT_UI_STRINGS: dict[str, dict[str, str]] = {
    "en": {
        "error_system": "Something went wrong on our side. Please try again shortly.",
        "error_partial": "There was a small problem, but here is what I can tell you.",
        "error_unavailable": "I could not handle that request.",
    },
    "vi": {
        "error_system": "Xin lỗi, hệ thống đang gặp sự cố. Vui lòng thử lại sau.",
        "error_partial": "Đã có lỗi nhỏ, nhưng tôi vẫn cố gắng trả lời.",
        "error_unavailable": "Xin lỗi, tôi không thể xử lý yêu cầu này.",
    },
}


def get_ui_string(
    persona_id: str, key: str, lang: str = DEFAULT_LANG, default: str = ""
) -> str:
    """One chat-surface string for a character, in one language.

    Display copy only — this must never be fed to the model. Keeping the read
    path separate from build_persona_prompt is what stops UI text drifting into
    the model's instructions, and it is why `ui_strings` is popped out of
    `sections` during parsing rather than left to render with the rest.

    Never raises. Unlike `get_persona`, every caller here is already ON an error
    path — `api/main.py` reaching for `error_unavailable` because the graph
    produced nothing — and a loader failure there would replace a readable
    message with a stack trace.
    """
    try:
        value = (get_persona(persona_id, lang).get("ui_strings") or {}).get(key)
    except PersonaError:
        value = None
    if value:
        return value
    table = _DEFAULT_UI_STRINGS.get(lang) or _DEFAULT_UI_STRINGS[DEFAULT_LANG]
    return default or table.get(key, "")


def _normalise_db_persona(slug: str, persona: dict) -> dict:
    """Accept both the overlay shape and the flat shape rows were written in before.

    A row written by an older `sync_personas_to_db.py` has no `locales` key: its
    fields sit at the top level and are Vietnamese. Reading it as the `vi`
    overlay means the deploy order does not matter — a new backend serves an old
    row, and a new row does not break an old backend.
    """
    if isinstance(persona.get("locales"), dict) and persona["locales"]:
        out = dict(persona)
        out.setdefault("persona_id", slug)
        return out

    core = {k: persona.get(k, "") for k in _CORE_SECTIONS}
    overlay = {
        "voice": persona.get("voice", ""),
        "examples": persona.get("examples", ""),
        "safety_templates": persona.get("safety_templates") or {},
        "ui_strings": persona.get("ui_strings") or {},
    }
    logger.info("persona_preload_legacy_shape", extra={"slug": slug})
    return {
        "persona_id": slug,
        "title": persona.get("title", ""),
        "voice_identity": persona.get("voice_identity") or {},
        **core,
        "locales": {"vi": overlay},
    }


async def preload_personas_from_db() -> int:
    """Load every active character's persona into the cache. Returns the count.

    Called once from the FastAPI lifespan. `get_persona` stays synchronous:
    the alternative — querying inside it — makes it async and forces all four
    call sites (synthesizer, grader, and both TTS paths) to change, to buy a
    round trip to Neon in Singapore on every cache miss. The catalog is four
    rows that change approximately never, so reading it at startup costs one
    query and nothing at request time.

    Never raises. A DB that is down at startup leaves the cache empty and every
    lookup falls through to personas/*.md exactly as before — the markdown files
    stay in the repo precisely so this degradation is a non-event.

    Consequence to know: a persona edited in the DB needs a process restart to
    take effect. That is already true of the markdown files (no TTL, no
    invalidation), so this changes nothing operationally.
    """
    try:
        from langgraph_agents.shared import get_pg_client

        pg = get_pg_client()
        await pg.connect()
        rows = await pg.fetch(
            "SELECT slug, persona FROM characters WHERE is_active ORDER BY sort_order"
        )
    except Exception as exc:
        logger.warning(
            "persona_preload_failed", extra={"error": str(exc)},
        )
        return 0

    loaded = 0
    for row in rows:
        slug = row["slug"]
        persona = row["persona"]

        # asyncpg hands back JSONB as a string unless a codec is registered.
        if isinstance(persona, (str, bytes)):
            try:
                persona = json.loads(persona)
            except (ValueError, TypeError):
                logger.warning("persona_preload_bad_json", extra={"slug": slug})
                continue

        if not isinstance(persona, dict) or not persona.get("identity"):
            # An empty/partial JSONB would otherwise shadow a perfectly good
            # markdown file with a persona that renders a blank system prompt.
            logger.warning("persona_preload_incomplete", extra={"slug": slug})
            continue

        _persona_cache[slug] = _normalise_db_persona(slug, persona)
        loaded += 1

    logger.info("persona_preload", extra={"count": loaded, "rows": len(rows)})
    return loaded


_MODE_HINTS = {
    "chat":        "This turn is a casual greeting or general chat — be brief and warm.",
    "clarify":     "This turn is asking the user a clarification question — be concise and inviting.",
    "refuse":      "This turn is refusing to answer (out of scope / no reliable sources) — be polite, explain why, and refer to a professional.",
    "synthesize":  "This turn is delivering factual content from retrieved sources — keep claims accurate, cite sources when available, and preserve all safety warnings verbatim.",
}


def persona_name(persona: dict) -> str:
    """Display name for the character.

    `identity` is authored as prose but older files start it with a
    "Name: X | Role: ..." metadata line, so try that first and fall back to the
    file's `# Heading`. The two disagree on purpose for eca_default, whose
    heading is "ECA Default" and whose character is Seele.
    """
    identity = persona.get("identity") or ""
    match = re.search(r"\bName:\s*([^|\n]+)", identity)
    if match:
        return match.group(1).strip()
    title = (persona.get("title") or "").strip()
    return title or persona.get("persona_id", "the assistant")


def _example_lines(persona: dict, mode: str, limit: int = 2) -> list[str]:
    """Sample lines from `## Examples`, preferring ones tagged for this mode.

    Authored as `- (mode) text`, e.g. `- (chat) Chào bạn!`. An untagged line is
    usable in any mode. Falls back to untagged/other lines so a persona that
    only wrote two examples still gets a card.
    """
    raw = persona.get("examples") or ""
    if not raw:
        return []

    tagged: list[str] = []
    untagged: list[str] = []
    for line in raw.split("\n"):
        line = line.strip().lstrip("-").strip()
        if not line:
            continue
        match = re.match(r"^\((\w+)\)\s*(.+)$", line)
        if match:
            if match.group(1).lower() == mode:
                tagged.append(match.group(2).strip())
        else:
            untagged.append(line)

    return (tagged + untagged)[:limit]


def build_voice_card(persona: dict, mode: str) -> str:
    """A short reminder of who is speaking, placed LAST in the message list.

    The full persona block sits at the top of the system prompt, which is right
    for prompt caching and wrong for everything else: by the time the model
    starts generating it has read several thousand tokens of retrieved evidence,
    tag contracts and formatting instructions — all of them more recent, more
    concrete and more imitable than "Tone: Warm, professional, encouraging".

    Restating the voice in ~100 tokens at the very end buys the recency that
    would otherwise need a second LLM call to restyle the answer. That second
    call would cost a full round trip *and* break streaming, because a restyling
    pass cannot emit its first token until the content pass has finished.

    Deliberately small. This is a reminder, not a second copy of the persona:
    duplicating the whole block would pay for those tokens twice and give the
    model two places to disagree with itself.
    """
    name = persona_name(persona)
    parts = [f"## Who is speaking\nYou are {name}."]

    voice = (persona.get("voice") or persona.get("personality") or "").strip()
    if voice:
        parts.append(voice)

    examples = _example_lines(persona, mode)
    if examples:
        quoted = "\n".join(f'  "{e}"' for e in examples)
        parts.append(f"Lines that sound like you:\n{quoted}")

    parts.append(
        f"Write the next message as {name}. Keep the facts and every safety "
        f"warning exactly as required above — only the voice is yours."
    )
    return "\n\n".join(parts)


def build_persona_prompt(persona: dict, mode: str) -> str:
    """Build system prompt from persona sections for LLM styling/generation.

    `mode` is the synthesizer mode: 'chat' | 'clarify' | 'refuse' | 'synthesize'.
    Adds a one-line nudge so persona instructions adapt to the current turn type
    without bloating the system message.
    """
    hint = _MODE_HINTS.get(mode, "")
    hint_block = f"\n\n## Turn hint\n{hint}" if hint else ""

    # Optional — personas written before the schema grew still render fine.
    voice = (persona.get("voice") or "").strip()
    voice_block = f"\n\n## How you speak\n{voice}" if voice else ""

    return f"""You are {persona['identity']}

## Your Personality
{persona['personality']}{voice_block}

## Rules
{persona['behavioral_rules']}

## Formatting
{persona['response_formatting']}{hint_block}"""
