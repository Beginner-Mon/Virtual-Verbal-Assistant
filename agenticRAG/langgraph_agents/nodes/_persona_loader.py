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


def _parse_safety_templates(text: str) -> dict[str, str]:
    """Parse ## Safety Templates section into {tag: template_text} dict."""
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
    safety_templates = _parse_safety_templates(safety_raw)

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
        "behavioral_rules": sections.get("behavioral_rules", ""),
        "response_formatting": sections.get("response_formatting", ""),
        "safety_templates": safety_templates,
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


_PERSONA_SECTIONS = (
    "title", "identity", "voice_identity", "personality",
    "behavioral_rules", "response_formatting", "safety_templates",
)


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
            persona.setdefault(section, {} if section.endswith(("_templates", "_identity")) else "")
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


def build_persona_prompt(persona: dict, mode: str) -> str:
    """Build system prompt from persona sections for LLM styling/generation.

    `mode` is the synthesizer mode: 'chat' | 'clarify' | 'refuse' | 'synthesize'.
    Adds a one-line nudge so persona instructions adapt to the current turn type
    without bloating the system message.
    """
    hint = _MODE_HINTS.get(mode, "")
    hint_block = f"\n\n## Turn hint\n{hint}" if hint else ""
    return f"""You are {persona['identity']}.

## Your Personality
{persona['personality']}

## Rules
{persona['behavioral_rules']}

## Formatting
{persona['response_formatting']}{hint_block}"""
