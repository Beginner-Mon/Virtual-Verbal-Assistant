#!/usr/bin/env python3
"""Push `personas/<slug>.md` into `characters.persona` for rows that exist.

Why this exists next to upload_characters_to_s3.py, which already seeds the
same column: that script also rewrites `vrm_url` from a `--cdn` argument. When
the only thing that changed is a markdown file, asking an operator to re-supply
the CloudFront base URL invites a typo that silently repoints every avatar.

This touches exactly two columns, both derived from the markdown:
    persona          (JSONB, parsed by the backend's own loader)
    voice_language   (voice_identity.language)

Why it is needed at all: `preload_personas_from_db()` fills the persona cache
from the DB at startup, and `get_persona()` reads that cache before it reads
any file. A persona edited only in markdown therefore has no effect anywhere a
database is reachable — the file looks authoritative and is not.

    python scripts/sync_personas_to_db.py --dry-run
    python scripts/sync_personas_to_db.py

Reads VVA_PG_DSN through the service's own loader, so it cannot write to a
different database from the one the backend reads.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "agenticRAG"))

from langgraph_agents.shared.env import load_env  # noqa: E402


async def main(dry_run: bool) -> int:
    load_env()
    from langgraph_agents.db.postgres import get_default_dsn
    from langgraph_agents.nodes._persona_loader import _load_persona

    import asyncpg

    conn = await asyncpg.connect(get_default_dsn(), timeout=20)
    try:
        rows = await conn.fetch("SELECT slug, persona FROM characters ORDER BY sort_order")
        if not rows:
            print("characters table is empty — run upload_characters_to_s3.py first")
            return 1

        changed = 0
        for row in rows:
            slug = row["slug"]
            parsed = _load_persona(slug)
            if parsed.get("_fallback"):
                # Refuse rather than overwrite a real persona with the generic
                # fallback, which is what a renamed or deleted .md would do.
                print(f"  {slug:16} SKIP — personas/{slug}.md missing or unparseable")
                continue

            current = row["persona"]
            if isinstance(current, (str, bytes)):
                try:
                    current = json.loads(current)
                except (ValueError, TypeError):
                    current = None

            if current == parsed:
                print(f"  {slug:16} unchanged")
                continue

            lang = parsed.get("voice_identity", {}).get("language", "vi")
            before = set((current or {}).get("safety_templates", {}))
            after = set(parsed.get("safety_templates", {}))
            added = sorted(after - before)
            print(f"  {slug:16} UPDATE  lang={lang}"
                  + (f"  +templates: {', '.join(added)}" if added else ""))

            if not dry_run:
                await conn.execute(
                    "UPDATE characters SET persona = $2::jsonb, voice_language = $3, "
                    "updated_at = now() WHERE slug = $1",
                    slug, json.dumps(parsed, ensure_ascii=False), lang,
                )
            changed += 1

        print(f"\n{changed} row(s) {'would be' if dry_run else ''} updated"
              f"{' (dry run, nothing written)' if dry_run else ''}")
        return 0
    finally:
        await conn.close()


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--dry-run", action="store_true", help="show the diff, write nothing")
    raise SystemExit(asyncio.run(main(ap.parse_args().dry_run)))
