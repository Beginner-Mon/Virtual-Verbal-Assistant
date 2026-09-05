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

Requires VVA_PG_DSN_OWNER (falls back to VVA_PG_DSN for the read-only
dry-run). The application role cannot write `characters` — 007_rls lists it as
SYSTEM_READONLY because the catalog belongs to nobody. Same variables the
backend and Alembic read, so there is no way to write personas into one
database while the app reads another.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import sys
from pathlib import Path

# This script's whole job is printing persona text, which is Vietnamese, at a
# Windows console defaulting to cp1252. Without this the dry-run — the step that
# exists so nobody writes blind — dies on the first accented character.
# Same fix as start_services.py; third occurrence of this in the repo.
for _stream in (sys.stdout, sys.stderr):
    try:
        _stream.reconfigure(encoding="utf-8", errors="replace")
    except (AttributeError, OSError):
        pass

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "agenticRAG"))

# Sections compared and reported. Order matters only for readable output.
# Core sections are shared across languages; the rest live per language under
# `locales`. Reported separately so a diff says WHICH language changed rather
# than just "locales".
_REPORTED_CORE = (
    "title", "identity", "personality", "behavioral_rules", "response_formatting",
)
_REPORTED_OVERLAY = ("voice", "examples", "safety_templates", "ui_strings")


def load_from_markdown(slug: str) -> dict:
    """Parse personas/{slug}.md with the backend's own loader.

    Importing `_load_persona` rather than reimplementing the parse is what stops
    the JSONB in the DB drifting from what get_persona() produces at runtime —
    the same reasoning as upload_characters_to_s3.load_persona.
    """
    from langgraph_agents.nodes._persona_loader import PersonaError, _load_persona

    try:
        return _load_persona(slug)
    except PersonaError as exc:
        # There is no fallback persona to guard against any more — the loader
        # raises. Turning that into a clean exit keeps the failure readable
        # instead of a traceback, and stops a half-written character reaching
        # the row that a working one is being served from.
        raise SystemExit(f"personas/{slug}/ cannot be loaded: {exc}")


def _public_ui_strings(persona: dict) -> dict:
    """`{lang: {...}}` for the `ui_strings` column the catalog Lambda serves.

    A projection of `persona`, not a second source: both come from the same
    parsed dict in the same UPDATE, so the column and the JSONB cannot drift.
    The column exists because `persona` is the system prompt and must never
    reach a browser, while this is display copy that only exists to.
    """
    return {
        lang: overlay.get("ui_strings") or {}
        for lang, overlay in (persona.get("locales") or {}).items()
    }


def _summarise(value) -> str:
    if isinstance(value, dict):
        return json.dumps(value, ensure_ascii=False, sort_keys=True)
    if isinstance(value, (list, tuple)):
        return " | ".join(" ".join(str(v).split()) for v in value)
    return " ".join(str(value or "").split())


def _section_value(persona: dict, section: str):
    """Read a section named the way `diff_sections` names it.

    Overlay sections are reported as `"en.examples"`, which is a LABEL, not a
    key — `persona["en.examples"]` is always None. Resolving it back through
    `locales` is what makes the dry-run show the overlay text; without this the
    report printed `(empty) -> (empty)` for voice, examples, safety templates
    and UI strings, which is every section a persona edit actually touches. The
    dry-run exists so nobody writes blind, so this was the whole point of it
    failing silently.
    """
    lang, sep, name = section.partition(".")
    if not sep:
        return persona.get(section)

    locales = persona.get("locales") or {}
    if not locales:
        # A row written before the overlay split has no `locales` and its fields
        # sit at the top level, in Vietnamese. `_normalise_db_persona` reads such
        # a row as the `vi` overlay, so the report has to agree — otherwise this
        # very migration prints "vi.examples: (empty) -> ..." over a row that
        # does have examples, and understates what is being replaced.
        return persona.get(name) if lang == "vi" else None

    return (locales.get(lang) or {}).get(name)


def diff_sections(old: dict, new: dict) -> list[str]:
    """Section names that differ, for a report worth reading before a write."""
    changed = []
    for section in _REPORTED_CORE:
        if _summarise(old.get(section)) != _summarise(new.get(section)):
            changed.append(section)

    old_locales = old.get("locales") or {}
    new_locales = new.get("locales") or {}
    for lang in sorted(set(old_locales) | set(new_locales)):
        o, n = old_locales.get(lang) or {}, new_locales.get(lang) or {}
        for section in _REPORTED_OVERLAY:
            if _summarise(o.get(section)) != _summarise(n.get(section)):
                changed.append(f"{lang}.{section}")
    return changed


def _writer_client():
    """A client that can actually UPDATE `characters`.

    `get_pg_client()` connects as the application role, and since 007_rls that
    role has `characters` in SYSTEM_READONLY — the catalog belongs to nobody and
    the app must not write it. So this script's own docstring was wrong: with
    only VVA_PG_DSN it fails with `permission denied for table characters`, and
    has done since that migration.

    Same precedence alembic/env.py uses for the same reason: owner first, app
    role second, so a machine that has only the app DSN still gets the readable
    dry-run rather than an import-time failure.
    """
    import os

    from langgraph_agents.db.postgres import PostgresClient

    dsn = os.environ.get("VVA_PG_DSN_OWNER") or os.environ.get("VVA_PG_DSN")
    return PostgresClient(dsn) if dsn else None


async def sync(slugs: list[str], dry_run: bool) -> int:
    from langgraph_agents.shared import get_pg_client

    pg = _writer_client() or get_pg_client()
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
            before = _summarise(_section_value(existing[slug], section))
            after = _summarise(_section_value(new, section))
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
                json.dumps(_public_ui_strings(new), ensure_ascii=False),
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
