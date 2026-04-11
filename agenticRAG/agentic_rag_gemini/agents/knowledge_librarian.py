"""ECA Knowledge Librarian — Unified multi-collection retrieval agent.

Lightweight Hub & Spoke Architecture:
    HUB  = KnowledgeLibrarian.retrieve()  — routing, confidence gating, fact packaging
    SPOKE = SLM Classification (qwen:0.5b) — only when keyword routing is ambiguous

Data Layers:
    1. HumanML3D    → humanml3d_library (kinematic motion descriptions)
    2. Documents    → user_{id}_documents (uploaded PDFs, DOCX, exercise KB)
    3. User Context → user_{id}_collection (conversation memory)
    4. MedQuAD      → medquad_library (NIH medical QA — scaffolding only)

Design Principles:
    - Keyword fast-path handles 70%+ of queries (zero latency, zero LLM)
    - SLM classification only fires when keywords are ambiguous (~150ms)
    - Fact summarization is inline (regex, no LLM) — 80% token savings
    - KeywordExtractor provides motion verb fallback when DIRECT_MATCH fails
    - Shared qwen:0.5b model with SafetyFilter + Orchestrator (0 extra RAM)

Output Contract:
    {
        "source_collection": str,
        "retrieved_facts": List[str],         # Summarized, not raw chunks
        "motion_metadata": dict | None,
        "confidence_score": float,
        "query_used": str,
        "strategy": str,
        "direct_match": bool,
        "keyword_match": bool,                # NEW: verb extracted by KeywordExtractor
        "no_context_signal": str | None,
        "slm_classification_used": bool,      # NEW: whether SLM was called
        "elapsed_ms": float,
    }
"""

import json
import os
import re
import time
from typing import Any, Dict, List, Optional

from utils.logger import get_logger

logger = get_logger(__name__)

# ── Configurable thresholds ──────────────────────────────────────
RAG_CONFIDENCE_THRESHOLD = float(
    os.getenv("RAG_CONFIDENCE_THRESHOLD", "0.80")
)
DIRECT_MATCH_THRESHOLD = float(
    os.getenv("RAG_DIRECT_MATCH_THRESHOLD", "0.95")
)
# Gemini Gate: below this confidence, signal that Gemini should handle it
GEMINI_GATE_THRESHOLD = float(
    os.getenv("GEMINI_GATE_THRESHOLD", "0.60")
)
# Fact summarization limits
MAX_FACTS = int(os.getenv("LIBRARIAN_MAX_FACTS", "5"))
MAX_FACT_LENGTH = int(os.getenv("LIBRARIAN_MAX_FACT_LENGTH", "200"))

# ── MedQuAD collection (scaffolding — not yet populated) ─────────
MEDQUAD_COLLECTION_NAME = "medquad_library"

# ── Source identifiers ───────────────────────────────────────────
SOURCE_HUMANML3D = "humanml3d_library"
SOURCE_DOCUMENTS = "documents"
SOURCE_USER_CONTEXT = "user_context"
SOURCE_MEDQUAD = "medquad_library"
SOURCE_NONE = "none"

# ── Keyword lists for source routing (fast-path) ────────────────
_MEDICAL_KEYWORDS = (
    "symptom", "symptoms", "diagnosis", "disease", "medication",
    "treatment", "medicine", "doctor", "clinical", "chronic",
    "infection", "blood pressure", "diabetes", "fever",
    "headache", "nausea", "dizziness", "fatigue",
)

_MOTION_KEYWORDS = (
    "show me", "demonstrate", "visualize", "visualise", "animate",
    "animation", "3d", "glb", "motion", "how to do",
)

_PERSONAL_KEYWORDS = (
    "remember", "my history", "my record", "my document",
    "my file", "you told me", "last time", "previously",
    "my injury", "my condition", "my data",
)

# ── SLM Classification Prompt ────────────────────────────────────
_SLM_CLASSIFY_PROMPT = """Classify this query into one category.

Categories:
- "motion": user wants to SEE, VISUALIZE, or DO a physical exercise/movement
- "medical": query about symptoms, diseases, medications, diagnosis, treatment
- "personal": user references their own history, files, past conversations
- "exercise": query about exercise recommendations, techniques, or physical therapy

Query: "{query}"

JSON output (only):
{{"category": "", "entity": ""}}"""


class KnowledgeLibrarian:
    """Lightweight knowledge retrieval agent — Hub & Spoke architecture.

    Implements the 5-step Retrieval Protocol:
        1. Source Identification  — keyword fast-path + SLM fallback
        2. Query Refinement       — extract key entities, strip fluff
        3. Fact Extraction        — delegate to ChromaDB tools
        4. Fact Summarization     — inline extraction (no LLM)
        5. Confidence Scoring     — threshold gating + DIRECT_MATCH + Gemini Gate
    """

    def __init__(
        self,
        memory_tool=None,
        document_tool=None,
        motion_retriever=None,
        exercise_detector=None,
        keyword_extractor=None,
    ) -> None:
        """Inject existing tool instances (Wrapper pattern).

        Args:
            memory_tool:       MemoryTool instance (user conversation memory).
            document_tool:     DocumentRetrievalTool instance (exercise KB + uploads).
            motion_retriever:  MotionCandidateRetriever instance (HumanML3D).
            exercise_detector: ExerciseDetector instance (entity extraction).
            keyword_extractor: KeywordExtractor instance (motion verb extraction).
        """
        self._memory_tool = memory_tool
        self._document_tool = document_tool
        self._motion_retriever = motion_retriever
        self._exercise_detector = exercise_detector
        self._keyword_extractor = keyword_extractor

        # SLM client for ambiguous query classification
        # Uses qwen2.5:1.5b for better accuracy (~90%+ classification)
        # Separate from SafetyFilter/Orchestrator's qwen:0.5b — worth the ~1GB for accuracy
        self._slm_client = None
        self._slm_available = False
        self._slm_model = "qwen2.5:1.5b"
        try:
            from utils.ollama_client import OllamaClient
            self._slm_client = OllamaClient(model_name=self._slm_model)
            self._slm_client.timeout = 8  # Slightly higher timeout for 1.5b model
            if self._slm_client.check_connection():
                available = self._slm_client.list_models()
                if any(self._slm_model in m for m in available):
                    self._slm_available = True
                    logger.info(
                        "KnowledgeLibrarian: SLM classification ready (%s)",
                        self._slm_model,
                    )
                else:
                    logger.warning(
                        "KnowledgeLibrarian: %s not found in Ollama (available: %s) — "
                        "keyword-only routing. Pull with: ollama pull %s",
                        self._slm_model, available[:5], self._slm_model,
                    )
            else:
                logger.info("KnowledgeLibrarian: Ollama not reachable — keyword-only routing")
        except Exception:
            logger.info("KnowledgeLibrarian: SLM client init failed — keyword-only routing")

        # MedQuAD: lazy-loaded ChromaDB collection (scaffolding)
        self._medquad_collection = None
        self._medquad_embedder = None
        self._init_medquad()

        logger.info(
            "KnowledgeLibrarian initialized | memory=%s doc=%s motion=%s "
            "keyword_extractor=%s slm=%s medquad=%s",
            self._memory_tool is not None,
            self._document_tool is not None,
            self._motion_retriever is not None,
            self._keyword_extractor is not None,
            self._slm_available,
            self._medquad_collection is not None,
        )

    # ── MedQuAD scaffolding ──────────────────────────────────────

    def _init_medquad(self) -> None:
        """Try to connect to the medquad_library ChromaDB collection.

        This is scaffolding — the collection won't exist until data is ingested.
        Failure is non-fatal; MedQuAD queries will gracefully return no results.
        """
        try:
            import chromadb
            from chromadb.config import Settings

            host = os.getenv("CHROMA_HOST", "localhost")
            port = int(os.getenv("CHROMA_PORT", "8100"))

            client = chromadb.HttpClient(
                host=host,
                port=port,
                settings=Settings(anonymized_telemetry=False),
            )
            client.heartbeat()
            col = client.get_collection(
                name=MEDQUAD_COLLECTION_NAME,
                embedding_function=None,
            )
            count = col.count()
            if count > 0:
                self._medquad_collection = col
                logger.info(
                    "KnowledgeLibrarian: MedQuAD collection ready (%d entries)",
                    count,
                )
            else:
                logger.info(
                    "KnowledgeLibrarian: MedQuAD collection exists but is empty — "
                    "medical QA layer disabled until data is ingested"
                )
        except Exception:
            # Expected: collection doesn't exist yet
            logger.info(
                "KnowledgeLibrarian: MedQuAD collection not found — "
                "medical QA layer disabled (run ingest_medquad.py to enable)"
            )

    # ──────────────────────────────────────────────────────────────
    # PUBLIC API
    # ──────────────────────────────────────────────────────────────

    def retrieve(
        self,
        user_query: str,
        user_id: str = "guest",
        conversation_history: Optional[List[Dict[str, str]]] = None,
        intent_hint: Optional[str] = None,
    ) -> Dict[str, Any]:
        """Unified retrieval — single entry point for all knowledge queries.

        Args:
            user_query:           Raw user input.
            user_id:              User identifier (multi-tenant isolation).
            conversation_history: Recent conversation turns.
            intent_hint:          Optional intent from the orchestrator
                                  (e.g. "visualize_motion", "knowledge_query").

        Returns:
            Structured JSON-serializable dict with source_collection,
            retrieved_facts (summarized), motion_metadata, confidence_score, etc.
        """
        t0 = time.perf_counter()

        # ── Step 1: Source Identification ─────────────────────────
        source, slm_used = self._identify_source(user_query, intent_hint)

        # ── Step 2: Query Refinement ─────────────────────────────
        refined_query = self._refine_query(user_query)

        # ── Step 3: Fact Extraction (delegate to appropriate tool) ──
        raw_results, motion_metadata = self._extract_facts(
            source=source,
            query=refined_query,
            user_id=user_id,
            conversation_history=conversation_history,
        )

        # ── Step 4: Confidence Scoring ───────────────────────────
        best_score = 0.0
        for result in raw_results:
            similarity = result.get("similarity") or result.get("score", 0.0)
            if similarity > best_score:
                best_score = similarity

        # ── Step 4b: Fact Summarization ──────────────────────────
        # Instead of passing raw chunks, extract core facts to save tokens
        retrieved_facts = self._summarize_facts(raw_results)

        strategy = raw_results[0].get("strategy", "vector") if raw_results else "vector"
        direct_match = best_score >= DIRECT_MATCH_THRESHOLD

        # ── Step 4c: Motion verb fallback via KeywordExtractor ───
        keyword_match = False
        if (
            source == SOURCE_HUMANML3D
            and not direct_match
            and self._keyword_extractor is not None
            and motion_metadata is None
        ):
            verb = self._keyword_extractor.extract(user_query)
            if verb:
                motion_metadata = {
                    "motion_id": None,
                    "text_description": f"a person performs {verb}",
                    "direct_match": False,
                    "keyword_match": True,
                    "verb": verb,
                }
                keyword_match = True
                logger.info(
                    "KnowledgeLibrarian: keyword fallback → verb='%s' (0 LLM calls)",
                    verb,
                )

        # Build output
        elapsed_ms = (time.perf_counter() - t0) * 1000
        output: Dict[str, Any] = {
            "source_collection": source if retrieved_facts else SOURCE_NONE,
            "retrieved_facts": retrieved_facts,
            "motion_metadata": motion_metadata,
            "confidence_score": round(best_score, 4),
            "query_used": refined_query,
            "strategy": strategy,
            "direct_match": direct_match,
            "keyword_match": keyword_match,
            "no_context_signal": None,
            "slm_classification_used": slm_used,
            "elapsed_ms": round(elapsed_ms, 1),
        }

        if not retrieved_facts:
            output["source_collection"] = SOURCE_NONE
            output["no_context_signal"] = "NO_LOCAL_CONTEXT_FOUND"
            logger.info(
                "KnowledgeLibrarian: NO_LOCAL_CONTEXT_FOUND for '%s' "
                "(best_score=%.3f < threshold=%.2f) [%.0fms]",
                user_query[:50], best_score, RAG_CONFIDENCE_THRESHOLD, elapsed_ms,
            )
        else:
            logger.info(
                "KnowledgeLibrarian: %s → %d facts (score=%.3f, direct=%s, keyword=%s, slm=%s) [%.0fms]",
                source, len(retrieved_facts), best_score,
                direct_match, keyword_match, slm_used, elapsed_ms,
            )

        return output

    # ──────────────────────────────────────────────────────────────
    # Step 1: Source Identification (keyword fast-path + SLM fallback)
    # ──────────────────────────────────────────────────────────────

    def _identify_source(
        self, query: str, intent_hint: Optional[str] = None,
    ) -> tuple:
        """Determine which data layer to query.

        Returns:
            (source: str, slm_used: bool)

        Priority:
          1. Respect explicit intent hints from the orchestrator.
          2. Keyword fast-path (handles 70%+ of queries, ~1ms)
          3. SLM classification fallback (handles ambiguous queries, ~150ms)
          4. Default → documents
        """
        q = (query or "").strip().lower()
        slm_used = False

        # ── Orchestrator hint overrides ──────────────────────────
        if intent_hint == "visualize_motion":
            return SOURCE_HUMANML3D, False
        if intent_hint == "conversation":
            return SOURCE_USER_CONTEXT, False

        # ── Keyword fast-path ────────────────────────────────────
        if any(kw in q for kw in _PERSONAL_KEYWORDS):
            return SOURCE_USER_CONTEXT, False

        if self._medquad_collection is not None:
            if any(kw in q for kw in _MEDICAL_KEYWORDS):
                return SOURCE_MEDQUAD, False

        if any(kw in q for kw in _MOTION_KEYWORDS):
            return SOURCE_HUMANML3D, False

        # ── SLM classification fallback (ambiguous queries) ──────
        if self._slm_available:
            slm_source = self._slm_classify(query)
            if slm_source:
                return slm_source, True

        # Default: exercise knowledge base
        return SOURCE_DOCUMENTS, False

    def _slm_classify(self, query: str) -> Optional[str]:
        """Classify query using local SLM (qwen:0.5b, shared instance).

        Only called when keyword fast-path can't determine the source.
        Returns None on failure (caller falls through to default).
        """
        try:
            prompt = _SLM_CLASSIFY_PROMPT.format(query=query[:200])
            raw = self._slm_client.generate(
                prompt=prompt,
                format="json",
                temperature=0.1,
                max_tokens=32,
            )

            parsed = json.loads(raw)
            category = parsed.get("category", "").lower().strip()

            category_map = {
                "motion": SOURCE_HUMANML3D,
                "medical": SOURCE_MEDQUAD if self._medquad_collection else SOURCE_DOCUMENTS,
                "personal": SOURCE_USER_CONTEXT,
                "exercise": SOURCE_DOCUMENTS,
            }

            source = category_map.get(category)
            if source:
                logger.debug(
                    "KnowledgeLibrarian SLM classified '%s' → %s (category=%s)",
                    query[:40], source, category,
                )
                return source

            logger.debug(
                "KnowledgeLibrarian SLM returned unknown category '%s'", category
            )
            return None

        except Exception as exc:
            logger.debug("KnowledgeLibrarian SLM classification failed: %s", exc)
            return None

    # ──────────────────────────────────────────────────────────────
    # Step 2: Query Refinement
    # ──────────────────────────────────────────────────────────────

    def _refine_query(self, query: str) -> str:
        """Strip conversational fluff and extract key entities.

        Examples:
            "Hey can you please show me how to do a cartwheel?" → "cartwheel"
            "What exercises help with lower back pain?"         → "exercises lower back pain"
        """
        q = (query or "").strip()
        if not q:
            return q

        # 1) Try ExerciseDetector for precise entity extraction
        if self._exercise_detector is not None:
            try:
                detected = self._exercise_detector.detect_exercise(q)
                if detected:
                    logger.debug(
                        "KnowledgeLibrarian: ExerciseDetector extracted '%s' from '%s'",
                        detected, q[:40],
                    )
                    return detected
            except Exception:
                pass

        # 2) Strip common conversational patterns
        stripped = q.lower()
        for prefix in (
            "hey ", "hi ", "hello ", "can you ", "could you ", "please ",
            "show me how to ", "show me ", "how to do a ", "how to do ",
            "how to ", "demonstrate ", "visualize ", "what is ",
        ):
            if stripped.startswith(prefix):
                stripped = stripped[len(prefix):]

        # Remove trailing question mark and whitespace
        stripped = stripped.rstrip("?").strip()

        return stripped if len(stripped) >= 2 else q

    # ──────────────────────────────────────────────────────────────
    # Step 3: Fact Extraction
    # ──────────────────────────────────────────────────────────────

    def _extract_facts(
        self,
        source: str,
        query: str,
        user_id: str,
        conversation_history: Optional[List[Dict[str, str]]] = None,
    ) -> tuple:
        """Delegate to the appropriate tool and collect results.

        Returns:
            (raw_results: List[dict], motion_metadata: dict | None)
        """
        motion_metadata = None

        if source == SOURCE_HUMANML3D:
            results, motion_metadata = self._retrieve_humanml3d(query)

        elif source == SOURCE_USER_CONTEXT:
            results = self._retrieve_user_context(query, user_id)

        elif source == SOURCE_MEDQUAD:
            results = self._retrieve_medquad(query)

        elif source == SOURCE_DOCUMENTS:
            results = self._retrieve_documents(query, user_id)

        else:
            results = []

        return results, motion_metadata

    # ── HumanML3D retrieval ──────────────────────────────────────

    def _retrieve_humanml3d(self, query: str) -> tuple:
        """Retrieve motion candidates from HumanML3D library.

        If a DIRECT_MATCH (>0.95) is found, populates motion_metadata
        with the NPZ/GLB path so downstream can skip Text-to-Motion.

        Returns:
            (results: List[dict], motion_metadata: dict | None)
        """
        motion_metadata = None

        if self._motion_retriever is None:
            return [], None

        try:
            candidates = self._motion_retriever.retrieve_top_k(query, k=5)
        except Exception as exc:
            logger.warning("KnowledgeLibrarian: HumanML3D retrieval failed: %s", exc)
            return [], None

        results = []
        for candidate in candidates:
            entry = {
                "document": candidate.text_description,
                "text_description": candidate.text_description,
                "similarity": candidate.score,
                "score": candidate.score,
                "motion_id": candidate.motion_id,
                "duration": candidate.duration,
                "strategy": "vector",
            }
            results.append(entry)

        # Check for DIRECT_MATCH on top-1
        if candidates and candidates[0].score >= DIRECT_MATCH_THRESHOLD:
            top = candidates[0]
            motion_metadata = {
                "motion_id": top.motion_id,
                "text_description": top.text_description,
                "duration": top.duration,
                "direct_match": True,
                "keyword_match": False,
                "similarity_score": round(top.score, 4),
            }
            logger.info(
                "KnowledgeLibrarian: DIRECT_MATCH on HumanML3D "
                "(score=%.3f): '%s' → motion_id=%s",
                top.score, top.text_description[:60], top.motion_id,
            )

        return results, motion_metadata

    # ── User Context retrieval ───────────────────────────────────

    def _retrieve_user_context(
        self, query: str, user_id: str,
    ) -> List[Dict[str, Any]]:
        """Retrieve from user-scoped conversation memory.

        Multi-tenant isolation: only accesses user_{id}_collection.
        """
        if self._memory_tool is None:
            return []

        try:
            results = self._memory_tool.retrieve_memory(
                user_id=user_id,
                query=query,
                top_k=5,
            )
            # Normalize output format
            normalized = []
            for r in results:
                normalized.append({
                    "document": r.get("document", ""),
                    "similarity": r.get("similarity", 0.0),
                    "metadata": r.get("metadata", {}),
                    "strategy": "vector",
                })
            return normalized
        except Exception as exc:
            logger.warning(
                "KnowledgeLibrarian: User context retrieval failed for user=%s: %s",
                user_id, exc,
            )
            return []

    # ── Document KB retrieval ────────────────────────────────────

    def _retrieve_documents(
        self, query: str, user_id: str,
    ) -> List[Dict[str, Any]]:
        """Retrieve from exercise KB + user-uploaded documents.

        Multi-tenant isolation: DocumentRetrievalTool internally
        scopes queries to user_{id}_documents collection.
        """
        if self._document_tool is None:
            return []

        try:
            # Use hybrid search for best coverage
            results = self._document_tool.search_documents(
                query=query,
                user_id=user_id,
                top_k=5,
                search_method="hybrid",
            )
            # Normalize output format
            normalized = []
            for r in results:
                normalized.append({
                    "document": r.get("document", ""),
                    "similarity": r.get("similarity", 0.0),
                    "metadata": r.get("metadata", {}),
                    "strategy": "hybrid",
                })
            return normalized
        except Exception as exc:
            logger.warning(
                "KnowledgeLibrarian: Document retrieval failed for user=%s: %s",
                user_id, exc,
            )
            return []

    # ── MedQuAD retrieval (scaffolding) ──────────────────────────

    def _retrieve_medquad(self, query: str) -> List[Dict[str, Any]]:
        """Retrieve from MedQuAD medical QA collection.

        Scaffolding: returns empty if the collection doesn't exist or
        hasn't been populated yet.
        """
        if self._medquad_collection is None:
            logger.debug("KnowledgeLibrarian: MedQuAD not available, skipping")
            return []

        try:
            # Lazy-load embedder (same model used everywhere)
            if self._medquad_embedder is None:
                from sentence_transformers import SentenceTransformer
                model_name = os.getenv(
                    "MOTION_EMBEDDING_MODEL",
                    "sentence-transformers/all-MiniLM-L6-v2",
                )
                self._medquad_embedder = SentenceTransformer(model_name)

            query_emb = self._medquad_embedder.encode(
                [query], normalize_embeddings=True,
            ).tolist()
            results = self._medquad_collection.query(
                query_embeddings=query_emb,
                n_results=5,
            )

            normalized = []
            docs = results.get("documents", [[]])[0]
            distances = results.get("distances", [[]])[0]
            metadatas = results.get("metadatas", [[]])[0]

            for i, doc in enumerate(docs):
                if not doc:
                    continue
                dist = distances[i] if i < len(distances) else 0.0
                similarity = max(0.0, 1.0 - dist / 2.0)
                meta = metadatas[i] if i < len(metadatas) else {}

                normalized.append({
                    "document": doc,
                    "similarity": similarity,
                    "metadata": {
                        "topic": meta.get("topic", ""),
                        "split": meta.get("split", ""),
                        "source": "medquad_nih",
                    },
                    "strategy": "vector",
                })

            return normalized
        except Exception as exc:
            logger.warning("KnowledgeLibrarian: MedQuAD retrieval failed: %s", exc)
            return []

    # ──────────────────────────────────────────────────────────────
    # Step 4: Fact Summarization (inline, no LLM)
    # ──────────────────────────────────────────────────────────────

    def _summarize_facts(self, raw_results: List[Dict[str, Any]]) -> List[str]:
        """Extract core facts from raw chunks — saves tokens for downstream LLM.

        Rules:
        1. Filter by confidence threshold (RAG_CONFIDENCE_THRESHOLD)
        2. Split into sentences, keep only substantive ones (20-200 chars)
        3. Strip boilerplate (section headers, citations, formatting)
        4. Deduplicate by first 50 chars
        5. Cap at MAX_FACTS facts

        Returns:
            List of summarized fact strings (max MAX_FACTS items).
        """
        facts: List[str] = []
        seen: set = set()

        for result in raw_results:
            similarity = result.get("similarity") or result.get("score", 0.0)
            if similarity < RAG_CONFIDENCE_THRESHOLD:
                continue

            doc = (
                result.get("document")
                or result.get("text_description")
                or result.get("text")
                or ""
            )
            if not doc:
                continue

            # Split into sentences
            sentences = re.split(r'[.!?]\s+', doc)
            for sentence in sentences:
                sentence = sentence.strip()

                # Length filter: skip too short or too long
                if len(sentence) < 20 or len(sentence) > MAX_FACT_LENGTH:
                    continue

                # Strip common boilerplate patterns
                if self._is_boilerplate(sentence):
                    continue

                # Dedup by first 50 chars (case-insensitive)
                key = sentence[:50].lower()
                if key in seen:
                    continue
                seen.add(key)

                facts.append(sentence)
                if len(facts) >= MAX_FACTS:
                    break

            if len(facts) >= MAX_FACTS:
                break

        return facts

    @staticmethod
    def _is_boilerplate(sentence: str) -> bool:
        """Check if a sentence is boilerplate (headers, citations, formatting).

        Returns True if the sentence should be filtered out.
        """
        s = sentence.lower().strip()

        # Common boilerplate patterns
        boilerplate_prefixes = (
            "references:", "bibliography:", "source:", "citation:",
            "table of contents", "figure ", "fig.", "table ",
            "copyright", "©", "all rights reserved",
            "page ", "chapter ", "section ",
        )
        if any(s.startswith(bp) for bp in boilerplate_prefixes):
            return True

        # Too many numbers (likely a table row or reference)
        digit_ratio = sum(c.isdigit() for c in s) / max(len(s), 1)
        if digit_ratio > 0.4:
            return True

        # Too many special characters (likely formatting)
        special_chars = sum(c in "[]{}()<>|\\/*#~_^" for c in s)
        if special_chars > 3:
            return True

        return False

    # ──────────────────────────────────────────────────────────────
    # Utilities
    # ──────────────────────────────────────────────────────────────

    def health_check(self) -> Dict[str, Any]:
        """Report which data layers are operational."""
        return {
            "memory_tool": self._memory_tool is not None,
            "document_tool": self._document_tool is not None,
            "motion_retriever": self._motion_retriever is not None,
            "exercise_detector": self._exercise_detector is not None,
            "keyword_extractor": self._keyword_extractor is not None,
            "slm_available": self._slm_available,
            "medquad_collection": self._medquad_collection is not None,
            "confidence_threshold": RAG_CONFIDENCE_THRESHOLD,
            "direct_match_threshold": DIRECT_MATCH_THRESHOLD,
            "gemini_gate_threshold": GEMINI_GATE_THRESHOLD,
        }
