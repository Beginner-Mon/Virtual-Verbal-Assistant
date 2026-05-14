"""Lightweight entity tag extraction helpers for retrieval metadata.

Tags are intentionally compact and ASCII-only so they are safe to persist in
vector metadata fields across backends.
"""

from __future__ import annotations

import re
from typing import Iterable, List


_TAG_PATTERNS: dict[str, tuple[str, ...]] = {
    "neck": ("neck", "cervical"),
    "shoulder": ("shoulder", "rotator cuff", "scapula", "scapular"),
    "back": ("back", "lumbar", "spine", "thoracic"),
    "knee": ("knee", "patella", "acl", "meniscus"),
    "hip": ("hip", "glute", "gluteal", "pelvic"),
    "ankle": ("ankle", "calf", "achilles"),
    "posture": ("posture", "alignment", "rounded shoulder", "slouch"),
    "mobility": ("mobility", "range of motion", "rom", "flexibility"),
    "strength": ("strength", "resistance", "progressive overload"),
    "pain_relief": ("pain", "ache", "sore", "relief"),
    "rehab": ("rehab", "rehabilitation", "physical therapy", "physio"),
    "warmup": ("warm up", "warm-up", "activation"),
    "exercise_recommendation": (
        "recommend",
        "suggest",
        "routine",
        "program",
        "plan",
        "list",
    ),
    "visualize_motion": (
        "show me",
        "visualize",
        "visualise",
        "demonstrate",
        "animate",
        "motion",
    ),
}


def _normalize_text(text: str) -> str:
    lowered = (text or "").strip().lower()
    lowered = re.sub(r"\s+", " ", lowered)
    return lowered


def extract_entity_tags(text: str, max_tags: int = 8) -> List[str]:
    """Extract deterministic entity tags from free-form text."""
    normalized = _normalize_text(text)
    if not normalized:
        return []

    tags: List[str] = []
    for tag, keywords in _TAG_PATTERNS.items():
        if any(keyword in normalized for keyword in keywords):
            tags.append(tag)
        if len(tags) >= max_tags:
            break

    return tags


def encode_entity_tags(tags: Iterable[str]) -> str:
    """Encode tags into a compact metadata-safe string."""
    cleaned: List[str] = []
    seen: set[str] = set()
    for raw_tag in tags:
        tag = str(raw_tag or "").strip().lower()
        if not tag or tag in seen:
            continue
        seen.add(tag)
        cleaned.append(tag)
    return ",".join(cleaned)


def decode_entity_tags(raw: object) -> List[str]:
    """Decode tags from metadata fields that may be list/tuple/string."""
    if raw is None:
        return []

    if isinstance(raw, (list, tuple, set)):
        values = list(raw)
    else:
        values = str(raw).split(",")

    tags: List[str] = []
    seen: set[str] = set()
    for value in values:
        tag = str(value or "").strip().lower()
        if not tag or tag in seen:
            continue
        seen.add(tag)
        tags.append(tag)
    return tags
