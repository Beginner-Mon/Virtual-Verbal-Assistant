#!/usr/bin/env python3
"""Ingest MedQuAD (NIH) dataset into ChromaDB medquad_library collection.

SCAFFOLDING — This script defines the schema and ingestion pipeline
for MedQuAD data.  Actual data must be downloaded from HuggingFace
before running.

Schema (4 fields per entry):
    - Question:  The medical question text (stored as document)
    - Answer:    The full answer text (stored in metadata)
    - Topic:     Medical topic category (e.g., "Diabetes", "Heart Disease")
    - Split:     Dataset split (train/val/test)

Usage:
    # 1) Download dataset first (future step):
    #    pip install datasets
    #    python scripts/ingest_medquad.py --source huggingface
    #
    # 2) Or from local JSONL:
    #    python scripts/ingest_medquad.py --source jsonl --path data/medquad.jsonl

    python scripts/ingest_medquad.py
"""

import argparse
import json
import sys
import time
from pathlib import Path
from typing import List, Dict

SCRIPT_DIR = Path(__file__).resolve().parent
SERVICE_ROOT = SCRIPT_DIR.parent

COLLECTION_NAME = "medquad_library"
EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
BATCH_SIZE = 200
CHROMA_HOST = "localhost"
CHROMA_PORT = 8100

# Default local JSONL path (will be created by download step in the future)
DEFAULT_JSONL_PATH = SERVICE_ROOT / "data" / "knowledge_base" / "medquad.jsonl"


def load_from_jsonl(path: Path) -> List[Dict]:
    """Load MedQuAD entries from a local JSONL file.

    Expected JSONL schema per line:
        {"question": "...", "answer": "...", "topic": "...", "split": "train"}
    """
    if not path.exists():
        print(f"ERROR: JSONL file not found at {path}")
        print("Please download MedQuAD data first.")
        return []

    entries = []
    with open(path, "r", encoding="utf-8") as f:
        for line_num, line in enumerate(f, 1):
            line = line.strip()
            if not line:
                continue
            try:
                obj = json.loads(line)
            except json.JSONDecodeError:
                print(f"WARNING: Skipping invalid JSON on line {line_num}")
                continue

            question = (obj.get("question") or "").strip()
            answer = (obj.get("answer") or "").strip()
            topic = (obj.get("topic") or "general").strip()
            split = (obj.get("split") or "train").strip()

            if not question or not answer:
                continue

            entries.append({
                "id": f"medquad_{line_num}",
                "question": question,
                "answer": answer,
                "topic": topic,
                "split": split,
            })

    return entries


def load_from_huggingface() -> List[Dict]:
    """Load MedQuAD from HuggingFace datasets.

    Requires: pip install datasets

    This is a placeholder — the actual HuggingFace dataset name
    may vary.  Common options:
        - "keivalya/MedQuAD-MedicalQnADataset"
        - "lavita/MedQuAD"
    """
    try:
        from datasets import load_dataset
    except ImportError:
        print("ERROR: 'datasets' package not installed.")
        print("Run: pip install datasets")
        return []

    dataset_name = "keivalya/MedQuAD-MedicalQnADataset"
    print(f"Loading dataset '{dataset_name}' from HuggingFace...")

    try:
        ds = load_dataset(dataset_name)
    except Exception as exc:
        print(f"ERROR: Failed to load dataset: {exc}")
        return []

    entries = []
    for split_name, split_data in ds.items():
        for i, row in enumerate(split_data):
            question = (row.get("Question") or row.get("question") or "").strip()
            answer = (row.get("Answer") or row.get("answer") or "").strip()

            if not question or not answer:
                continue

            entries.append({
                "id": f"medquad_{split_name}_{i}",
                "question": question,
                "answer": answer,
                "topic": "general",
                "split": split_name,
            })

    return entries


def main() -> int:
    parser = argparse.ArgumentParser(description="Ingest MedQuAD into ChromaDB")
    parser.add_argument(
        "--source", choices=["jsonl", "huggingface"], default="jsonl",
        help="Data source (default: jsonl)",
    )
    parser.add_argument(
        "--path", type=str, default=str(DEFAULT_JSONL_PATH),
        help="Path to JSONL file (only used with --source jsonl)",
    )
    args = parser.parse_args()

    # ── Load data ──────────────────────────────────────────────
    if args.source == "huggingface":
        entries = load_from_huggingface()
    else:
        entries = load_from_jsonl(Path(args.path))

    if not entries:
        print("No entries to ingest. Exiting.")
        return 1

    print(f"Loaded {len(entries)} MedQuAD entries")

    # ── Generate embeddings ────────────────────────────────────
    print(f"Loading embedding model: {EMBEDDING_MODEL}")
    from sentence_transformers import SentenceTransformer

    model = SentenceTransformer(EMBEDDING_MODEL)
    questions = [e["question"] for e in entries]

    print(f"Encoding {len(questions)} questions...")
    t0 = time.time()
    embeddings = model.encode(
        questions,
        batch_size=256,
        show_progress_bar=True,
        normalize_embeddings=True,
    )
    print(f"Encoding done in {time.time() - t0:.1f}s")

    # ── Connect to ChromaDB ────────────────────────────────────
    import chromadb
    from chromadb.config import Settings

    print(f"Connecting to ChromaDB at {CHROMA_HOST}:{CHROMA_PORT}...")
    client = chromadb.HttpClient(
        host=CHROMA_HOST,
        port=CHROMA_PORT,
        settings=Settings(anonymized_telemetry=False),
    )
    client.heartbeat()
    print("Connected to ChromaDB")

    # Delete and recreate for clean ingestion
    try:
        client.delete_collection(COLLECTION_NAME)
        print(f"Deleted existing collection '{COLLECTION_NAME}'")
    except Exception:
        pass

    collection = client.get_or_create_collection(
        name=COLLECTION_NAME,
        metadata={
            "description": "MedQuAD (NIH) Medical QA — Question-Answer pairs",
            "schema": "question, answer, topic, split",
        },
        embedding_function=None,
    )

    # ── Batch upsert ───────────────────────────────────────────
    total = len(entries)
    print(f"Ingesting {total} entries in batches of {BATCH_SIZE}...")
    t0 = time.time()

    for start in range(0, total, BATCH_SIZE):
        end = min(start + BATCH_SIZE, total)
        batch = entries[start:end]
        batch_embeddings = embeddings[start:end].tolist()

        ids = [e["id"] for e in batch]
        documents = [e["question"] for e in batch]
        metadatas = []
        for e in batch:
            metadatas.append({
                "answer": e["answer"][:2000],  # Truncate very long answers
                "topic": e["topic"],
                "split": e["split"],
            })

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

    # ── Smoke test ─────────────────────────────────────────────
    print("\n--- Smoke test: querying 'What causes diabetes?' ---")
    test_emb = model.encode(
        ["What causes diabetes?"], normalize_embeddings=True,
    ).tolist()
    results = collection.query(query_embeddings=test_emb, n_results=3)
    for i, doc in enumerate(results["documents"][0]):
        dist = results["distances"][0][i]
        topic = results["metadatas"][0][i].get("topic", "?")
        print(f"  {i+1}. (dist={dist:.4f}, topic={topic}) {doc[:80]}...")

    return 0


if __name__ == "__main__":
    sys.exit(main())
