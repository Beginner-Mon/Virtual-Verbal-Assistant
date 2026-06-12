"""Persona loader — parse persona MD files, cache, build prompts.

Extracted from conversation.py for reuse.
"""

import logging
import re
from pathlib import Path
from typing import Optional

import yaml

logger = logging.getLogger("langgraph.nodes.persona")

_persona_cache: dict[str, dict] = {}


def _load_config() -> dict:
    config_path = Path(__file__).resolve().parents[4] / "config" / "langgraph.yaml"
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
        "title": "ECA Default",
        "identity": "Name: ECA | Role: Physical therapy AI assistant",
        "voice_identity": {"voice_path": None, "language": "vi"},
        "personality": "Tone: Warm, professional, encouraging",
        "behavioral_rules": "Use Vietnamese by default. Keep responses helpful.",
        "response_formatting": "Keep under 300 words.",
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


def _load_persona(persona_id: str) -> dict:
    personas_dir = _resolve_personas_dir()
    filepath = personas_dir / f"{persona_id}.md"

    if not filepath.exists():
        logger.warning("Persona file not found: %s, using fallback", filepath)
        return _fallback_persona(persona_id)

    content = filepath.read_text(encoding="utf-8")

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
    }


def get_persona(persona_id: str) -> dict:
    if persona_id not in _persona_cache:
        _persona_cache[persona_id] = _load_persona(persona_id)
    return _persona_cache[persona_id]


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
