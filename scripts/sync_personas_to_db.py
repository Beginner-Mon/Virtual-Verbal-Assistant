#!/usr/bin/env python3
"""Push personas/*.md into characters.persona — the step that makes an edit real.

`preload_personas_from_db()` fills the same cache `get_persona()` reads, and it
runs at FastAPI startup. A row whose `persona` JSONB is stale therefore SHADOWS
the markdown file: editing personas/anne.md and restarting the service changes
nothing at all, silently, because the DB copy wins. This script is how the two
are reconciled.

Separate from upload_characters_to_s3.py on purpose. That script seeds a
character whole — GLB parsing, content hash, S3 upload, avatar profile, persona
— and it cannot run any more: the .vrm files were removed from the repo once
they lived on the CDN, and scripts/characters.seed.json (its fallback) was
deleted too, so it exits with the recovery instructions in build_records().
Persona text meanwhile changes on every iteration of writing a character and
needs none of that machinery. This touches one column.

Only UPDATEs. Creating a character row is still the seed script's job, so a
typo'd slug reports "not in the database" rather than inserting a half row with
no model attached.

    python scripts/sync_personas_to_db.py --dry-run     # show what would change
    python scripts/sync_personas_to_db.py anne          # one character
    python scripts/sync_personas_to_db.py               # every active character

Requires VVA_PG_DSN — the same variable the backend and Alembic read, so there
is no way to write personas into one database while the app reads another.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "agenticRAG"))

# Sections compared and reported. Order matters only for readable output.
_REPORTED = (
    "identity", "personality", "voice", "behavioral_rules",
    "response_formatting", "examples", "safety_templates", "ui_strings",
)


def load_from_markdown(slug: str) -> dict:
    """Parse personas/{slug}.md with the backend's own loader.

    Importing `_load_persona` rather than reimplementing the parse is what stops
    the JSONB in the DB drifting from what get_persona() produces at runtime —
    the same reasoning as upload_characters_to_s3.load_persona.
    """
    from langgraph_agents.nodes._persona_loader import _load_persona

    persona = _load_persona(slug)
    if persona.get("_fallback"):
        raise SystemExit(
            f"personas/{slug}.md is missing or unparseable — refusing to write a "
            f"fallback persona over a real one."
        )
    return persona


def _summarise(value) -> str:
    if isinstance(value, dict):
        return json.dumps(value, ensure_ascii=False, sort_keys=True)
    return " ".join(str(value or "").split())


def diff_sections(old: dict, new: dict) -> list[str]:
    """Section names that differ, for a report worth reading before a write."""
    changed = []
    for section in _REPORTED:
        if _summarise(old.get(section)) != _summarise(new.get(section)):
            changed.append(section)
    return changed


async def sync(slugs: list[str], dry_run: bool) -> int:
    from langgraph_agents.shared import get_pg_client

    pg = get_pg_client()
    await pg.connect()

    rows = await pg.fetch(
        "SELECT slug, persona FROM characters WHERE is_active ORDER BY sort_order"
    )
    existing = {}
    for row in rows:
        persona = row["persona"]
        # asyncpg hands back JSONB as a string unless a codec is registered.
        if isinstance(persona, (str, bytes)):
            try:
                persona = json.loads(persona)
            except (ValueError, TypeError):
                persona = {}
        existing[row["slug"]] = persona if isinstance(persona, dict) else {}

    targets = slugs or sorted(existing)
    unknown = [s for s in targets if s not in existing]
    if unknown:
        raise SystemExit(
            f"not in the database (or not active): {', '.join(unknown)}\n"
            f"known slugs: {', '.join(sorted(existing))}\n"
            f"This script only UPDATEs — seed a new character with "
            f"upload_characters_to_s3.py."
        )

    written = 0
    for slug in targets:
        new = load_from_markdown(slug)
        changed = diff_sections(existing[slug], new)

        if not changed:
            print(f"  {slug:14} unchanged")
            continue

        print(f"  {slug:14} {'would update' if dry_run else 'updating'}: {', '.join(changed)}")
        for section in changed:
            before = _summarise(existing[slug].get(section))
            after = _summarise(new.get(section))
            print(f"      {section}")
            print(f"        - {before[:110] or '(empty)'}")
            print(f"        + {after[:110] or '(empty)'}")

        if not dry_run:
            # `ui_strings` is written twice on purpose: once inside `persona`,
            # which is what get_persona() serves the backend, and once into its
            # own column, which is the only one the catalog Lambda is allowed to
            # expose (persona is the system prompt and stays server-side). Both
            # come from the same parsed dict in the same statement, so they
            # cannot drift.
            await pg.execute(
                "UPDATE characters SET persona = $1, ui_strings = $2, "
                "updated_at = now() WHERE slug = $3",
                json.dumps(new, ensure_ascii=False),
                json.dumps(new.get("ui_strings") or {}, ensure_ascii=False),
                slug,
            )
        written += 1

    return written


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("slugs", nargs="*", help="characters to sync (default: all active)")
    ap.add_argument("--dry-run", action="store_true",
                    help="show the diff without writing")
    args = ap.parse_args()

    count = asyncio.run(sync(args.slugs, args.dry_run))

    if args.dry_run:
        print(f"\n{count} character(s) would change. Nothing written.")
    else:
        print(f"\n{count} character(s) updated. "
              f"Restart the LangGraph service — the persona cache is filled once, "
              f"at startup.")


if __name__ == "__main__":
    main()
