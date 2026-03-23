"""Top-K motion candidate retrieval via ChromaDB (with JSONL fallback).

Primary path:  query the `humanml3d_library` ChromaDB collection.
Fallback path: if ChromaDB is unreachable or empty, load the local JSONL
               and do in-memory cosine search (original approach).
"""

from dataclasses import dataclass
import json
import os
import re
from typing import Any, Dict, List

import chromadb
from chromadb.config import Settings

from utils.logger import get_logger

logger = get_logger(__name__)

# ── ChromaDB settings ─────────────────────────────────────────────
CHROMA_HOST = os.getenv("CHROMA_HOST", "localhost")
CHROMA_PORT = int(os.getenv("CHROMA_PORT", "8100"))
COLLECTION_NAME = "humanml3d_library"

# ── Fallback JSONL path ───────────────────────────────────────────
JSONL_PATH = os.getenv(
    "HUMANML3D_DESCRIPTIONS_PATH",
    "./data/knowledge_base/humanml3d_descriptions.jsonl",
)
EMBEDDING_MODEL = os.getenv(
    "MOTION_EMBEDDING_MODEL",
    "sentence-transformers/all-MiniLM-L6-v2",
)


def _clean_pos_tags(text: str) -> str:
    """Strip POS-tag annotations that follow '#'."""
    cleaned = text.split("#")[0].strip()
    return re.sub(r"\s+", " ", cleaned) if cleaned else text.strip()


@dataclass
class MotionCandidate:
    candidate_id: str
    text_description: str
    motion_prompt: str
    score: float
    motion_id: str = ""       # Original HumanML3D motion ID
    duration: float = 0.0     # Motion duration in seconds


class MotionCandidateRetriever:
    """Retrieve Top-K HumanML3D candidates — ChromaDB first, JSONL fallback."""

    def __init__(self) -> None:
        self._use_chromadb = False
        self._collection = None
        self._embedder = None

        # ── Try ChromaDB first ──────────────────────────────────
        try:
            client = chromadb.HttpClient(
                host=CHROMA_HOST,
                port=CHROMA_PORT,
                settings=Settings(anonymized_telemetry=False),
            )
            client.heartbeat()
            col = client.get_collection(name=COLLECTION_NAME, embedding_function=None)
            count = col.count()
            if count > 0:
                self._collection = col
                self._use_chromadb = True
                logger.info(
                    "MotionCandidateRetriever: using ChromaDB collection '%s' (%d entries)",
                    COLLECTION_NAME, count,
                )
            else:
                logger.warning(
                    "ChromaDB collection '%s' exists but is empty — falling back to JSONL",
                    COLLECTION_NAME,
                )
        except Exception as exc:
            logger.warning("ChromaDB unavailable (%s) — falling back to JSONL", exc)

        # ── Load embedding model (needed for both paths) ────────
        try:
            from sentence_transformers import SentenceTransformer
            self._embedder = SentenceTransformer(EMBEDDING_MODEL)
        except Exception as exc:
            logger.warning("Cannot load embedding model %s: %s", EMBEDDING_MODEL, exc)

        # ── Fallback: load JSONL into memory ────────────────────
        if not self._use_chromadb:
            self._rows = self._load_jsonl()
            self._embeddings = self._embed_rows()
        else:
            self._rows = []
            self._embeddings = []

    # ── JSONL fallback helpers ──────────────────────────────────

    def _load_jsonl(self) -> List[Dict[str, str]]:
        if not os.path.exists(JSONL_PATH):
            logger.warning("JSONL not found at %s", JSONL_PATH)
            return []

        rows: List[Dict[str, str]] = []
        with open(JSONL_PATH, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    obj = json.loads(line)
                except json.JSONDecodeError:
                    continue

                text = _clean_pos_tags(
                    obj.get("text_description")
                    or obj.get("description")
                    or obj.get("caption")
                    or ""
                )
                if not text:
                    continue

                rows.append({
                    "candidate_id": str(obj.get("id", len(rows))),
                    "text_description": text,
                    "motion_prompt": text,
                })

        logger.info("Loaded %d JSONL fallback descriptions", len(rows))
        return rows

    def _embed_rows(self):
        if not self._rows or self._embedder is None:
            return []
        texts = [r["text_description"] for r in self._rows]
        return self._embedder.encode(texts, normalize_embeddings=True)

    # ── Public API ──────────────────────────────────────────────

    def retrieve_top_k(self, query: str, k: int = 5) -> List[MotionCandidate]:
        """Retrieve top-K motion candidates for the given query."""
        if self._use_chromadb:
            return self._retrieve_chromadb(query, k)
        return self._retrieve_fallback(query, k)

    # ── ChromaDB path ───────────────────────────────────────────

    def _retrieve_chromadb(self, query: str, k: int) -> List[MotionCandidate]:
        if self._embedder is None or self._collection is None:
            return [MotionCandidate("fallback", query, query, 0.0)]

        query_emb = self._embedder.encode([query], normalize_embeddings=True).tolist()
        results = self._collection.query(query_embeddings=query_emb, n_results=k)

        candidates: List[MotionCandidate] = []
        for i, doc in enumerate(results["documents"][0]):
            dist = results["distances"][0][i] if "distances" in results else 0.0
            similarity = max(0.0, 1.0 - dist / 2.0)
            meta = results["metadatas"][0][i] if "metadatas" in results else {}
            cid = meta.get("motion_id", results["ids"][0][i])

            candidates.append(MotionCandidate(
                candidate_id=str(cid),
                text_description=doc,
                motion_prompt=doc,
                score=similarity,
                motion_id=str(cid),
                duration=float(meta.get("duration", 0.0)),
            ))

        if not candidates:
            return [MotionCandidate("fallback", query, query, 0.0)]

        logger.info(
            "ChromaDB retrieval: top-1='%s' (score=%.3f) for query='%s'",
            candidates[0].text_description[:60], candidates[0].score, query[:60],
        )
        return candidates

    # ── JSONL fallback path ─────────────────────────────────────

    def _retrieve_fallback(self, query: str, k: int) -> List[MotionCandidate]:
        if not self._rows or len(self._embeddings) == 0 or self._embedder is None:
            return [MotionCandidate("fallback", query, query, 0.0)]

        query_emb = self._embedder.encode([query], normalize_embeddings=True)[0]
        scores = self._embeddings @ query_emb

        top_k = min(k, len(self._rows))
        top_indexes = sorted(range(len(scores)), key=lambda i: float(scores[i]), reverse=True)[:top_k]

        candidates: List[MotionCandidate] = []
        for idx in top_indexes:
            row = self._rows[idx]
            candidates.append(MotionCandidate(
                candidate_id=row["candidate_id"],
                text_description=row["text_description"],
                motion_prompt=row["motion_prompt"],
                score=float(scores[idx]),
            ))

        logger.info(
            "JSONL fallback retrieval: top-1='%s' (score=%.3f) for query='%s'",
            candidates[0].text_description[:60], candidates[0].score, query[:60],
        )
        return candidates
