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
    if reset and existing:
        # CASCADE on kb_embeddings.document_id removes the embeddings too.
        await pg.execute("DELETE FROM documents WHERE source_type = $1", SOURCE_TYPE)
        print(f"Reset: deleted {existing} existing '{SOURCE_TYPE}' documents (+ embeddings).")

    print("Loading embedding model (offline)...")
    embed = get_embedding_service()

    total = len(records)
    inserted = 0

    for start in range(0, total, EMBED_BATCH):
        batch = records[start:start + EMBED_BATCH]
        contents = [compose_content(r) for r in batch]

        # Batch encode — one model call per batch instead of per record.
        vectors = embed.embed_passages(contents)

        for rec, content, vector in zip(batch, contents, vectors):
            doc_id = await pg.fetchval(
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
            await pg.execute(
                """
                INSERT INTO kb_embeddings (document_id, chunk_index, content, embedding)
                VALUES ($1, 0, $2, $3)
                """,
                doc_id,
                content,
                vector,
            )
            inserted += 1

        print(f"  {inserted}/{total} indexed", end="\r", flush=True)

    print(f"\nDone: {inserted} documents + {inserted} embeddings.")


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

def main() -> int:
    parser = argparse.ArgumentParser(description="Ingest exercise KB into pgvector.")
    parser.add_argument("--source", type=Path, default=DEFAULT_SOURCE, help="documents.txt path")
    parser.add_argument("--reset", action="store_true", help="delete existing rows of this source_type first")
    parser.add_argument("--limit", type=int, default=0, help="only ingest the first N records (smoke test)")
    args = parser.parse_args()

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
