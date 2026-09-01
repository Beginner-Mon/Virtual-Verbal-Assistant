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
_DEFAULT_PERSONA = _CFG.get("persona", {}).get("default", "eca_default")


def _resolve_personas_dir() -> Path:
    cfg_dir = _CFG.get("persona", {}).get("personas_dir", "langgraph_agents/personas")
    if not Path(cfg_dir).is_absolute():
        return Path(__file__).resolve().parents[2] / cfg_dir
    return Path(cfg_dir)


def _fallback_persona(persona_id: str) -> dict:
    return {
        "persona_id": persona_id,
        "_fallback": True,   # not cached by get_persona (avoid unbounded cache from bad ids)
        "title": "ECA Default",
        "identity": "Name: ECA | Role: Physical therapy AI assistant",
        "voice_identity": {"voice_path": None, "language": "vi"},
        "personality": "Tone: Warm, professional, encouraging",
        "behavioral_rules": "Use Vietnamese by default. Keep responses helpful.",
        "response_formatting": "Keep under 300 words.",
        "safety_templates": {},
        "ui_strings": {},
    }


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


def _load_persona(persona_id: str) -> dict:
    # ── Defense in depth: validate persona_id before using as path ──
    if not re.fullmatch(r"[A-Za-z0-9_-]{1,64}", persona_id):
        logger.warning("invalid_persona_id", extra={"persona_id": persona_id[:80]})
        return _fallback_persona(persona_id)

    personas_dir = _resolve_personas_dir()
    resolved = (personas_dir / f"{persona_id}.md").resolve()

    # Containment check: resolved path must be inside personas_dir
    try:
        resolved.relative_to(personas_dir.resolve())
    except ValueError:
        logger.warning("persona_path_traversal_attempt", extra={"persona_id": persona_id[:80]})
        return _fallback_persona(persona_id)

    if not resolved.exists():
        logger.warning("Persona file not found: %s, using fallback", resolved)
        return _fallback_persona(persona_id)

    content = resolved.read_text(encoding="utf-8")

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
        "persona_id": persona_id,
        "title": title,
        "identity": sections.get("identity", ""),
        "voice_identity": voice_identity,
        "personality": sections.get("personality", ""),
        # Optional. `voice` is speech habits, `examples` verbatim sample lines —
        # the two sections a model actually imitates. Personas written before
        # they existed return "" and render exactly as they used to.
        "voice": sections.get("voice", ""),
        "examples": sections.get("examples", ""),
        "behavioral_rules": sections.get("behavioral_rules", ""),
        "response_formatting": sections.get("response_formatting", ""),
        "safety_templates": safety_templates,
        "ui_strings": ui_strings,
    }


def get_persona(persona_id: str) -> dict:
    cached = _persona_cache.get(persona_id)
    if cached is not None:
        return cached
    persona = _load_persona(persona_id)
    # Only cache real personas — fallbacks (bad/missing id) are not cached so a
    # flood of distinct invalid ids can't grow the cache unbounded.
    if not persona.get("_fallback"):
        _persona_cache[persona_id] = persona
    return persona


# Chat-surface copy used when a persona ships no `## UI Strings` section. These
# are verbatim the strings that used to be hard-coded at the call sites, so a
# persona without its own copy behaves exactly as it did before.
_DEFAULT_UI_STRINGS = {
    "error_system": "Xin lỗi, hệ thống đang gặp sự cố. Vui lòng thử lại sau.",
    "error_partial": "Đã có lỗi nhỏ, nhưng tôi vẫn cố gắng trả lời.",
    "error_unavailable": "Xin lỗi, tôi không thể xử lý yêu cầu này.",
}


def get_ui_string(persona_id: str, key: str, default: str = "") -> str:
    """One chat-surface string for a character, with a safe fallback.

    Display copy only — this must never be fed to the model. Keeping the read
    path separate from build_persona_prompt is what stops UI text drifting into
    the model's instructions, and it is why `ui_strings` is popped out of
    `sections` during parsing rather than left to render with the rest.
    """
    value = (get_persona(persona_id).get("ui_strings") or {}).get(key)
    if value:
        return value
    return default or _DEFAULT_UI_STRINGS.get(key, "")


_PERSONA_SECTIONS = (
    "title", "identity", "voice_identity", "personality",
    "behavioral_rules", "response_formatting", "safety_templates",
    # `voice` (speech habits) and `examples` (verbatim sample lines) are the two
    # sections that actually move a model. Both are optional: a persona without
    # them renders exactly as before.
    "voice", "examples",
    # Chat-surface copy. Never reaches the model — see get_ui_string.
    "ui_strings",
)

# Sections whose default is an empty dict rather than an empty string. Keyed by
# suffix because the alternative is a second list to keep in sync with the one
# above.
_DICT_SECTION_SUFFIXES = ("_templates", "_identity", "_strings")


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

        persona.setdefault("persona_id", slug)
        for section in _PERSONA_SECTIONS:
            persona.setdefault(
                section, {} if section.endswith(_DICT_SECTION_SUFFIXES) else ""
            )
        _persona_cache[slug] = persona
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
