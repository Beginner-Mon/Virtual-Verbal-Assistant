"""Parallel Semantic Bridge — Motion Strategy Agent for HumanML3D vocabulary.

This module implements the ECA Motion Logic Controller that converts vague user
exercise queries into HumanML3D-compatible motion descriptions.  It acts as a
"universal key" that either retrieves a near-perfect match from the library or
generates a new kinematic description via Gemini.

Pipeline position:
    User Query → [Task B: SemanticBridge] → semantic_bridge_prompt → DART /generate
                 [Task A: RAG Synthesis]  → text_answer            → UI

Decision Flow:
    1. Semantic Analysis  — deconstruct into core skeletal movements
    2. Library-First Check — probe HumanML3D (23.4k rows) via vector search
    3. Confidence Gate:
       • HIGH  (≥ 0.85) → Direct Retrieval  (zero-latency, high fidelity)
       • LOW/MED         → Generation via Gemini with few-shot grounding
    4. Output: Full Kinematic Description (never truncated, never just "a person")

Key design decisions:
    - Uses gemini-2.5-flash with max_tokens=200 for complete descriptions.
    - Few-shot examples from humanml3d_library vector DB ground the vocabulary.
    - Anti-truncation rules baked into the system prompt.
    - Falls back gracefully to the best library match on any failure.

Supports both ChromaDB (legacy) and Qdrant Cloud backends, selected by
the ``VECTOR_DB_TYPE`` environment variable.
"""

import json
import os
import time
from typing import Dict, List, Optional, Tuple

from utils.logger import get_logger
from utils.gemini_client import GeminiClientWrapper

logger = get_logger(__name__)

# ── Vector DB settings ────────────────────────────────────────────
VECTOR_DB_TYPE = os.getenv("VECTOR_DB_TYPE", "chromadb").strip().lower()
# ChromaDB (legacy)
CHROMA_HOST = os.getenv("CHROMA_HOST", "localhost")
CHROMA_PORT = int(os.getenv("CHROMA_PORT", "8100"))
# Qdrant
QDRANT_URL = os.getenv("QDRANT_URL", "http://localhost:6333")
QDRANT_API_KEY = os.getenv("QDRANT_API_KEY") or None

COLLECTION_NAME = "humanml3d_library"
FEW_SHOT_K = int(os.getenv("SEMANTIC_BRIDGE_FEW_SHOT_K", "5"))

# ── Gemini settings ──────────────────────────────────────────────
BRIDGE_MODEL = os.getenv("SEMANTIC_BRIDGE_MODEL", "gemini-2.5-flash")
BRIDGE_MAX_TOKENS = int(os.getenv("SEMANTIC_BRIDGE_MAX_TOKENS", "200"))
BRIDGE_TEMPERATURE = float(os.getenv("SEMANTIC_BRIDGE_TEMPERATURE", "0.3"))

# ── Strategy thresholds ──────────────────────────────────────────
DIRECT_HIT_THRESHOLD = float(os.getenv("SEMANTIC_BRIDGE_DIRECT_HIT", "0.85"))

# ── System Prompt: Motion Strategy Agent ─────────────────────────
SYSTEM_PROMPT = """\
You are the ECA Motion Logic Controller — a kinematic translator that converts \
exercise requests into HumanML3D-compatible motion descriptions.

CRITICAL RULES:
1. Output ONLY the physical motion description. No conversation, no markdown, no labels.
2. ALWAYS start with "a person" followed by detailed action verbs.
3. Describe BODY MOVEMENTS: joints, limbs, spatial orientation, speed, repetition.
4. Use simple physical verbs: walks, bends, raises, lowers, steps, extends, rotates, \
squats, reaches, pushes, pulls, lifts, swings, kicks, jumps, lunges.
5. NEVER output just "a person" alone — that is a FAILURE. You must describe the FULL motion.
6. NEVER use exercise brand names (burpee, deadlift). Decompose into skeletal movements.
7. Minimum 8 words, maximum 25 words.
8. Match the linguistic style of the reference captions from the motion library.

EXAMPLES of correct output:
- "a person stands and slowly raises both arms overhead then lowers them back down"
- "a person bends their knees into a squat position and then stands back up repeatedly"
- "a person tucks their chin toward their chest and holds briefly then releases"
- "a person steps forward with right foot lunging down then returns to standing"
- "a person walks forward in a straight line at a moderate pace"
"""


class SemanticBridgeService:
    """Motion Strategy Agent: user query → HumanML3D kinematic description.

    Implements a two-path decision gate:
    - HIGH confidence (≥ 0.85): Direct retrieval from HumanML3D library
    - LOW/MED confidence:       Gemini generation with few-shot grounding

    Designed to complete in < 2 seconds total.
    """

    def __init__(self) -> None:
        self._collection = None  # ChromaDB collection (legacy)
        self._qdrant_client = None  # Qdrant client
        self._embedder = None
        self._client = GeminiClientWrapper()
        self._vector_db_type = VECTOR_DB_TYPE

        if self._vector_db_type == "qdrant":
            self._init_qdrant()
        else:
            self._init_chromadb()

        self._init_embedder()

    # ── Initialization ───────────────────────────────────────────

    def _init_chromadb(self) -> None:
        """Connect to the humanml3d_library ChromaDB collection."""
        try:
            import chromadb
            from chromadb.config import Settings

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
                logger.info(
                    "SemanticBridge: ChromaDB '%s' ready (%d entries)",
                    COLLECTION_NAME,
                    count,
                )
            else:
                logger.warning(
                    "SemanticBridge: ChromaDB '%s' empty — few-shot disabled",
                    COLLECTION_NAME,
                )
        except Exception as exc:
            logger.warning("SemanticBridge: ChromaDB unavailable (%s) — few-shot disabled", exc)

    def _init_qdrant(self) -> None:
        """Connect to the humanml3d_library Qdrant collection."""
        try:
            from qdrant_client import QdrantClient

            self._qdrant_client = QdrantClient(url=QDRANT_URL, api_key=QDRANT_API_KEY, timeout=10)
            info = self._qdrant_client.get_collection(COLLECTION_NAME)
            count = info.points_count
            if count > 0:
                logger.info(
                    "SemanticBridge: Qdrant '%s' ready (%d entries)",
                    COLLECTION_NAME,
                    count,
                )
            else:
                logger.warning(
                    "SemanticBridge: Qdrant '%s' empty — few-shot disabled",
                    COLLECTION_NAME,
                )
                self._qdrant_client = None
        except Exception as exc:
            logger.warning("SemanticBridge: Qdrant unavailable (%s) — few-shot disabled", exc)
            self._qdrant_client = None

    def _init_embedder(self) -> None:
        """Load the sentence-transformer model for query embedding."""
        try:
            from sentence_transformers import SentenceTransformer

            model_name = os.getenv(
                "MOTION_EMBEDDING_MODEL",
                "sentence-transformers/all-MiniLM-L6-v2",
            )
            self._embedder = SentenceTransformer(model_name)
        except Exception as exc:
            logger.warning("SemanticBridge: Cannot load embedder (%s) — few-shot disabled", exc)

    # ── Library Retrieval ────────────────────────────────────────

    def _retrieve_candidates(
        self, query: str, k: int = FEW_SHOT_K,
    ) -> List[Dict[str, any]]:
        """Retrieve top-K HumanML3D candidates with scores."""
        if self._embedder is None:
            return []

        if self._vector_db_type == "qdrant":
            return self._retrieve_candidates_qdrant(query, k)
        return self._retrieve_candidates_chromadb(query, k)

    def _retrieve_candidates_chromadb(self, query: str, k: int) -> List[Dict[str, any]]:
        if self._collection is None:
            return []
        try:
            query_emb = self._embedder.encode([query], normalize_embeddings=True).tolist()
            results = self._collection.query(query_embeddings=query_emb, n_results=k)

            candidates = []
            docs = results.get("documents", [[]])[0]
            distances = results.get("distances", [[]])[0]
            metadatas = results.get("metadatas", [[]])[0]
            ids = results.get("ids", [[]])[0]

            for i, doc in enumerate(docs):
                if not doc:
                    continue
                dist = distances[i] if i < len(distances) else 0.0
                similarity = max(0.0, 1.0 - dist / 2.0)
                meta = metadatas[i] if i < len(metadatas) else {}
                cid = meta.get("motion_id", ids[i] if i < len(ids) else f"unk_{i}")

                candidates.append({
                    "text": doc,
                    "score": similarity,
                    "motion_id": str(cid),
                    "duration": float(meta.get("duration", 0.0)),
                })

            return candidates
        except Exception as exc:
            logger.warning("SemanticBridge: Library retrieval failed: %s", exc)
            return []

    def _retrieve_candidates_qdrant(self, query: str, k: int) -> List[Dict[str, any]]:
        if self._qdrant_client is None:
            return []
        try:
            query_emb = self._embedder.encode([query], normalize_embeddings=True).tolist()[0]
            results = self._qdrant_client.search(
                collection_name=COLLECTION_NAME,
                query_vector=query_emb,
                limit=k,
            )

            candidates = []
            for point in results:
                payload = point.payload or {}
                doc = payload.get("text", "")
                if not doc:
                    continue
                candidates.append({
                    "text": doc,
                    "score": point.score,
                    "motion_id": str(payload.get("motion_id", point.id)),
                    "duration": float(payload.get("duration", 0.0)),
                })

            return candidates
        except Exception as exc:
            logger.warning("SemanticBridge: Qdrant retrieval failed: %s", exc)
            return []

    # ── Core Translation ─────────────────────────────────────────

    def translate(self, user_query: str) -> str:
        """Translate a user query into a HumanML3D-compatible description.

        Decision flow:
        1. Probe HumanML3D library for semantic match
        2. If HIGH confidence (≥ 0.85) → return library description directly
        3. If LOW/MED → call Gemini with library examples as few-shot grounding

        Args:
            user_query: The raw user exercise query.

        Returns:
            A full kinematic description (8-25 words, starting with "a person").
        """
        t0 = time.perf_counter()

        # ── Step 1: Library-First Check ──────────────────────────
        candidates = self._retrieve_candidates(user_query)
        retrieval_ms = (time.perf_counter() - t0) * 1000

        top1_score = candidates[0]["score"] if candidates else 0.0
        top1_text = candidates[0]["text"] if candidates else ""

        # ── Step 2: Confidence Gate ──────────────────────────────
        if top1_score >= DIRECT_HIT_THRESHOLD and top1_text:
            total_ms = (time.perf_counter() - t0) * 1000
            result = self._ensure_quality(top1_text)
            logger.info(
                "SemanticBridge DIRECT HIT in %.0fms (score=%.3f): "
                "'%s' → '%s'",
                total_ms, top1_score,
                user_query[:50], result[:80],
            )
            return result

        # ── Step 3: Generation Path — Gemini with few-shot ───────
        reference_captions = [c["text"] for c in candidates[:5] if c.get("text")]

        example_block = ""
        if reference_captions:
            numbered = "\n".join(
                f'  {i+1}. "{cap}"' for i, cap in enumerate(reference_captions)
            )
            example_block = (
                f"\nReference captions from the HumanML3D motion library "
                f"(match your style to these):\n{numbered}\n"
            )

        user_prompt = (
            f"{example_block}\n"
            f'User exercise request: "{user_query}"\n\n'
            f"Write a single complete kinematic description of this motion. "
            f"Start with 'a person' and describe the full body movement "
            f"(joints, limbs, direction, speed). 8-25 words."
        )

        try:
            response = self._client.chat.completions.create(
                model=BRIDGE_MODEL,
                messages=[
                    {"role": "system", "content": SYSTEM_PROMPT},
                    {"role": "user", "content": user_prompt},
                ],
                temperature=BRIDGE_TEMPERATURE,
                max_tokens=BRIDGE_MAX_TOKENS,
            )
            result = response.choices[0].message.content.strip()
            result = self._clean_output(result)
            result = self._ensure_quality(result)

            total_ms = (time.perf_counter() - t0) * 1000
            strategy = "generation"
            logger.info(
                "SemanticBridge %s in %.0fms (retrieval=%.0fms, llm=%.0fms, "
                "top1_score=%.3f): '%s' → '%s'",
                strategy, total_ms, retrieval_ms, total_ms - retrieval_ms,
                top1_score, user_query[:50], result[:80],
            )
            return result

        except Exception as exc:
            total_ms = (time.perf_counter() - t0) * 1000
            logger.error(
                "SemanticBridge: Gemini failed in %.0fms: %s — "
                "falling back to best library match",
                total_ms, exc,
            )
            if top1_text:
                return self._ensure_quality(top1_text)
            return self._ensure_quality(user_query)

    # ── Output Quality Guards ────────────────────────────────────

    def _clean_output(self, text: str) -> str:
        """Strip accidental formatting from LLM output."""
        text = text.strip()

        if text.startswith("```"):
            text = text.strip("`").strip()

        if (text.startswith('"') and text.endswith('"')) or \
           (text.startswith("'") and text.endswith("'")):
            text = text[1:-1].strip()

        for prefix in ("kinematic description:", "output:", "rewritten caption:",
                        "description:", "motion:", "caption:"):
            if text.lower().startswith(prefix):
                text = text[len(prefix):].strip()

        lines = text.strip().splitlines()
        cleaned_lines = []
        for line in lines:
            lower = line.strip().lower()
            if lower.startswith("chosen path") or lower.startswith("optimized prompt"):
                continue
            if lower.startswith("path:") or lower.startswith("strategy:"):
                continue
            cleaned_lines.append(line.strip())
        text = " ".join(cleaned_lines).strip()

        return text

    def _ensure_quality(self, text: str) -> str:
        """Validate output meets minimum quality bar."""
        text = text.strip()
        if not text:
            return "a person performs a standing exercise movement"

        lower = text.lower()
        if not lower.startswith("a person"):
            text = f"a person {text}"

        words = text.split()
        if len(words) < 5:
            logger.warning(
                "SemanticBridge: output too short (%d words): '%s' — "
                "appending fallback action",
                len(words), text,
            )
            text = f"{text} performs a controlled exercise movement with their body"

        if len(words) > 30:
            text = " ".join(words[:30])

        return text

    # ── Async wrapper ────────────────────────────────────────────

    async def translate_async(self, user_query: str) -> str:
        """Async wrapper that runs translate() in a thread pool executor."""
        import asyncio

        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(None, self.translate, user_query)
