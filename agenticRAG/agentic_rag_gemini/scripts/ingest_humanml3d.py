#!/usr/bin/env python3
"""Ingest HumanML3D JSONL into ChromaDB humanml3d_library collection.

Usage:
    python scripts/ingest_humanml3d.py

Reads:  data/knowledge_base/humanml3d_descriptions.jsonl
Writes: ChromaDB collection 'humanml3d_library' (batch upsert, 200/batch)
"""

import json
import re
import sys
import time
from pathlib import Path
from typing import List, Dict

SCRIPT_DIR = Path(__file__).resolve().parent
SERVICE_ROOT = SCRIPT_DIR.parent
JSONL_PATH = SERVICE_ROOT / "data" / "knowledge_base" / "humanml3d_descriptions.jsonl"

COLLECTION_NAME = "humanml3d_library"
EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
BATCH_SIZE = 200
CHROMA_HOST = "localhost"
CHROMA_PORT = 8100


def _clean_pos_tags(text: str) -> str:
    """Strip POS-tag annotations that follow '#' in HumanML3D raw captions."""
    # "a man walks.#a/DET man/NOUN walk/VERB#0.0#0.0" → "a man walks."
    cleaned = text.split("#")[0].strip()
    return re.sub(r"\s+", " ", cleaned) if cleaned else text.strip()


def main() -> int:
    if not JSONL_PATH.exists():
        print(f"ERROR: JSONL not found at {JSONL_PATH}")
        print("Run  python scripts/download_humanml3d.py  first.")
        return 1

    # ── Load JSONL ──────────────────────────────────────────────
    print(f"Loading {JSONL_PATH.name}...")
    entries: List[Dict] = []
    seen = set()

    with open(JSONL_PATH, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            obj = json.loads(line)
            text = _clean_pos_tags(obj.get("text_description", ""))
            if not text or len(text) < 5:
                continue
            # Deduplicate by lowercase text
            key = text.lower()
            if key in seen:
                continue
            seen.add(key)

            entries.append({
                "id": str(obj.get("id", len(entries))),
                "text": text,
                "motion_id": str(obj.get("id", "")),
                "duration": obj.get("duration"),
                "num_frames": obj.get("num_frames"),
            })

    print(f"Loaded {len(entries)} unique clean entries")

    # ── Generate embeddings ─────────────────────────────────────
    print(f"Loading embedding model: {EMBEDDING_MODEL}")
    from sentence_transformers import SentenceTransformer

    model = SentenceTransformer(EMBEDDING_MODEL)
    texts = [e["text"] for e in entries]

    print(f"Encoding {len(texts)} texts (this may take a minute)...")
    t0 = time.time()
    embeddings = model.encode(texts, batch_size=256, show_progress_bar=True, normalize_embeddings=True)
    print(f"Encoding done in {time.time() - t0:.1f}s")

    # ── Connect to ChromaDB ─────────────────────────────────────
    import chromadb
    from chromadb.config import Settings

    print(f"Connecting to ChromaDB at {CHROMA_HOST}:{CHROMA_PORT}...")
    client = chromadb.HttpClient(
        host=CHROMA_HOST,
        port=CHROMA_PORT,
        settings=Settings(anonymized_telemetry=False),
    )
    client.heartbeat()  # Verify connection
    print("Connected to ChromaDB")

    # Delete and recreate collection for fresh ingestion
    try:
        client.delete_collection(COLLECTION_NAME)
        print(f"Deleted existing collection '{COLLECTION_NAME}'")
    except Exception:
        pass  # Collection didn't exist

    collection = client.get_or_create_collection(
        name=COLLECTION_NAME,
        metadata={"description": "HumanML3D motion descriptions — Semantic Thesaurus"},
        embedding_function=None,  # We provide pre-computed embeddings
    )

    # ── Batch upsert ────────────────────────────────────────────
    total = len(entries)
    print(f"Ingesting {total} entries in batches of {BATCH_SIZE}...")
    t0 = time.time()

    for start in range(0, total, BATCH_SIZE):
        end = min(start + BATCH_SIZE, total)
        batch_entries = entries[start:end]
        batch_embeddings = embeddings[start:end].tolist()

        ids = [f"hml3d_{e['id']}" for e in batch_entries]
        documents = [e["text"] for e in batch_entries]
        metadatas = []
        for e in batch_entries:
            meta = {"motion_id": e["motion_id"]}
            if e["duration"] is not None:
                meta["duration"] = float(e["duration"])
            if e["num_frames"] is not None:
                meta["num_frames"] = int(e["num_frames"])
            metadatas.append(meta)

        collection.upsert(
            ids=ids,
            documents=documents,
            embeddings=batch_embeddings,
            metadatas=metadatas,
        )

        pct = end / total * 100
        print(f"  [{end}/{total}] {pct:.0f}%")

    elapsed = time.time() - t0
    final_count = collection.count()
    print(f"\nDone! Ingested {final_count} entries into '{COLLECTION_NAME}' in {elapsed:.1f}s")

    # ── Smoke test ──────────────────────────────────────────────
    print("\n--- Smoke test: querying 'a person walks forward' ---")
    test_emb = model.encode(["a person walks forward"], normalize_embeddings=True).tolist()
    results = collection.query(query_embeddings=test_emb, n_results=3)
    for i, doc in enumerate(results["documents"][0]):
        dist = results["distances"][0][i]
        print(f"  {i+1}. (dist={dist:.4f}) {doc}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
