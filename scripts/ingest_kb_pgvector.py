"""Ingest the exercise knowledge base into PostgreSQL/pgvector.

Replaces the legacy ChromaDB ingest scripts (`ingest_medquad.py`,
`ingest_humanml3d.py`) which target the pre-migration architecture. Those wrote
to ChromaDB; the current stack reads `documents` + `kb_embeddings` via
`kb_search` (see agenticRAG/langgraph_agents/tools/pgvector_tool.py), and that
table was EMPTY — every clinical/exercise question fell through to refuse or
web fallback. This script closes that gap.

Source: agenticRAG/agentic_rag_gemini/data/knowledge_base/documents.txt
  ~2918 records delimited by a line containing only `---`, each shaped:

      Exercise: <name>
      Type: <type>
      Target Body Part: <part>
      Equipment: <equipment>
      Difficulty Level: <level>
      Description:
      <free text>
      Keywords:
      <comma list>

Embeddings: `intfloat/multilingual-e5-small` (384-dim) via the shared service,
which auto-prepends the REQUIRED `passage: ` prefix — matching the `query: `
prefix `kb_search` uses at read time. Mismatched prefixes silently degrade
recall, so both sides must go through the shared service.

Usage:
    # full ingest (idempotent — clears this source_type first)
    python scripts/ingest_kb_pgvector.py --reset

    # smoke test on a small slice
    python scripts/ingest_kb_pgvector.py --reset --limit 50
"""

from __future__ import annotations

# Offline embedding load MUST be set before the embedding stack is imported —
# huggingface_hub caches these flags at import time (see fixes/retrieval-perf P1).
import os

os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")

import argparse
import asyncio
import re
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "agenticRAG"))

DEFAULT_SOURCE = (
    REPO_ROOT
    / "agenticRAG"
    / "agentic_rag_gemini"
    / "data"
    / "knowledge_base"
    / "documents.txt"
)

SOURCE_TYPE = "exercise_db"
EMBED_BATCH = 64

# The export was produced by pandas: missing descriptions came through as the
# literal string "nan". Embedding that word adds noise, so it is dropped and the
# record is indexed on its structured fields alone (still useful for
# "what targets lower back" style queries).
_NULLISH = {"nan", "none", "null", ""}


# ── Parsing ───────────────────────────────────────────────────────────────

def _field(record: str, name: str) -> str:
    match = re.search(rf"^{re.escape(name)}:[ \t]*(.*?)[ \t]*$", record, re.M)
    return match.group(1).strip() if match else ""


def _description(record: str) -> str:
    match = re.search(r"^Description:[ \t]*\n(.*?)(?:\n\nKeywords:|\Z)", record, re.S | re.M)
    text = match.group(1).strip() if match else ""
    return "" if text.lower() in _NULLISH else text


def parse_records(path: Path) -> list[dict]:
    """Split the flat text export into structured exercise records."""
    raw = path.read_text(encoding="utf-8")
    chunks = [c.strip() for c in re.split(r"\n---\n", raw) if c.strip()]

    records: list[dict] = []
    for chunk in chunks:
        name = _field(chunk, "Exercise")
        if not name:
            continue  # not a record (stray text) — skip rather than index garbage
        records.append({
            "name": name,
            "type": _field(chunk, "Type"),
            "body_part": _field(chunk, "Target Body Part"),
            "equipment": _field(chunk, "Equipment"),
            "difficulty": _field(chunk, "Difficulty Level"),
            "keywords": _field(chunk, "Keywords"),
            "description": _description(chunk),
        })
    return records


def compose_content(rec: dict) -> str:
    """Build the text that actually gets embedded + returned to the synthesizer.

    Structured fields are inlined (not just kept as metadata) so body-part and
    equipment queries have lexical/semantic surface to match against — metadata
    columns are invisible to vector search.
    """
    lines = [f"Exercise: {rec['name']}"]

    facts = [
        ("Type", rec["type"]),
        ("Target body part", rec["body_part"]),
        ("Equipment", rec["equipment"]),
        ("Difficulty", rec["difficulty"]),
    ]
    inline = " | ".join(f"{label}: {value}" for label, value in facts if value)
    if inline:
        lines.append(inline)

    if rec["description"]:
        lines.append(rec["description"])
    if rec["keywords"]:
        lines.append(f"Keywords: {rec['keywords']}")

    return "\n".join(lines)


# ── Ingest ────────────────────────────────────────────────────────────────

async def ingest(records: list[dict], *, reset: bool) -> None:
    """Embed everything first, write only once every vector exists.

    The old order was: delete, then embed for ~9 minutes, then insert. On
    05/08 `embed_passages` segfaulted partway through — exit 139, no traceback —
    and because the delete had already committed, the knowledge base was left
    EMPTY. Every question was refused until a full re-ingest finished.

    Embedding now happens against an untouched database, and the destructive
    part is a single transaction at the end. A crash during the slow, fragile,
    native-code phase now costs nothing but time.

    Buffering is cheap at this scale: 2918 records x 384 dims x 4 bytes is about
    4.5 MB.
    """
    from langgraph_agents.shared import get_embedding_service, get_pg_client

    pg = get_pg_client()
    await pg.connect()

    existing = await pg.fetchval(
        "SELECT COUNT(*) FROM documents WHERE source_type = $1", SOURCE_TYPE
    )
    if existing and not reset:
        print(
            f"ABORT: {existing} '{SOURCE_TYPE}' documents already present.\n"
            "       Re-run with --reset to replace them (there is no unique\n"
            "       constraint on documents, so a plain re-run would duplicate)."
        )
        return

    print("Loading embedding model (offline)...")
    embed = get_embedding_service()

    total = len(records)
    # (record, content, vector) triples, built before anything is deleted.
    staged: list[tuple[dict, str, object]] = []

    for start in range(0, total, EMBED_BATCH):
        batch = records[start:start + EMBED_BATCH]
        contents = [compose_content(r) for r in batch]

        # Batch encode — one model call per batch instead of per record.
        # This is the line that segfaulted. Nothing has been deleted yet.
        vectors = embed.embed_passages(contents)

        staged.extend(zip(batch, contents, vectors))
        print(f"  {len(staged)}/{total} embedded", end="\r", flush=True)

    print(f"\nEmbedded {len(staged)} records. Writing to the database...")

    # Reaching for the pool directly: PostgresClient has no transaction helper,
    # and the delete and the inserts must land together or not at all. Without
    # this, a failure between them leaves a partially populated KB, which is
    # harder to notice than an empty one.
    async with pg._pool.acquire() as conn:  # noqa: SLF001 — maintenance script
        async with conn.transaction():
            if reset and existing:
                # CASCADE on kb_embeddings.document_id removes the embeddings.
                await conn.execute(
                    "DELETE FROM documents WHERE source_type = $1", SOURCE_TYPE
                )
                print(f"  deleted {existing} existing '{SOURCE_TYPE}' documents")

            inserted = 0
            for rec, content, vector in staged:
                doc_id = await conn.fetchval(
                    """
                    INSERT INTO documents (source_type, external_id, title, metadata)
                    VALUES ($1, $2, $3, $4::jsonb)
                    RETURNING id
                    """,
                    SOURCE_TYPE,
                    rec["name"],
                    rec["name"],
                    _metadata_json(rec),
                )
                await conn.execute(
                    """
                    INSERT INTO kb_embeddings (document_id, chunk_index, content, embedding)
                    VALUES ($1, 0, $2, $3)
                    """,
                    doc_id,
                    content,
                    vector,
                )
                inserted += 1

    print(f"Done: {inserted} documents + {inserted} embeddings.")


def _metadata_json(rec: dict) -> str:
    import json

    return json.dumps(
        {
            "type": rec["type"],
            "body_part": rec["body_part"],
            "equipment": rec["equipment"],
            "difficulty": rec["difficulty"],
            "keywords": rec["keywords"],
            "has_description": bool(rec["description"]),
        },
        ensure_ascii=False,
    )


# ── CLI ───────────────────────────────────────────────────────────────────

BACKEND_HEALTH_URL = "http://127.0.0.1:8000/health"


def _backend_is_running(timeout: float = 2.0) -> bool:
    """Is the API server up on this machine?

    Two processes loading torch at once is what killed the ingest on 05/08: the
    embedding call died with SIGSEGV and no traceback, which reads like a bug in
    this script rather than contention. Checking costs 2 seconds and removes an
    entire class of confusing failure.
    """
    import urllib.error
    import urllib.request

    try:
        with urllib.request.urlopen(BACKEND_HEALTH_URL, timeout=timeout) as resp:
            return resp.status == 200
    except (urllib.error.URLError, OSError, ValueError):
        return False


def main() -> int:
    parser = argparse.ArgumentParser(description="Ingest exercise KB into pgvector.")
    parser.add_argument("--source", type=Path, default=DEFAULT_SOURCE, help="documents.txt path")
    parser.add_argument("--reset", action="store_true", help="delete existing rows of this source_type first")
    parser.add_argument("--limit", type=int, default=0, help="only ingest the first N records (smoke test)")
    parser.add_argument(
        "--force",
        action="store_true",
        help="run even while the backend is up (it will probably segfault)",
    )
    args = parser.parse_args()

    if _backend_is_running() and not args.force:
        print(
            "ABORT: the backend is running on :8000.\n"
            "\n"
            "  Embedding here loads torch in a second process and dies with\n"
            "  SIGSEGV (exit 139, no traceback). This has happened before.\n"
            "\n"
            "  1. Stop the backend (Ctrl+C in its terminal)\n"
            "  2. Re-run this command\n"
            "  3. Start the backend again\n"
            "\n"
            "  --force skips this check if you know better."
        )
        return 2

    if not args.source.exists():
        print(f"ERROR: source not found: {args.source}")
        return 1

    records = parse_records(args.source)
    if not records:
        print("ERROR: no records parsed — check the delimiter/format.")
        return 1

    with_desc = sum(1 for r in records if r["description"])
    print(f"Parsed {len(records)} records ({with_desc} with a real description, "
          f"{len(records) - with_desc} metadata-only).")

    if args.limit:
        records = records[: args.limit]
        print(f"Limiting to {len(records)} records (smoke test).")

    asyncio.run(ingest(records, reset=args.reset))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
