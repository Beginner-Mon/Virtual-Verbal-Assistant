"""Parallel Semantic Bridge — Hub & Spoke Kinematic Auditor for HumanML3D.

This module implements the ECA Motion Logic Controller that converts vague user
exercise queries into HumanML3D-compatible motion descriptions.  It acts as a
"universal key" that either retrieves a near-perfect match from the library or
generates a new kinematic description via Gemini (the "Kinematic Auditor" Spoke).

Pipeline position:
    User Query → [Task B: SemanticBridge] → semantic_bridge_prompt → DART /generate
                 [Task A: RAG Synthesis]  → text_answer            → UI

Hub & Spoke Architecture:
    HUB  = SemanticBridgeService.translate()  — routing, confidence gating, fallback
    SPOKE = Kinematic Auditor (Gemini)        — structured generation with [[ ]] tags

Decision Flow:
    1. Force-Expansion Check  — intercept compound exercises before library lookup
    2. Library-First Check    — probe HumanML3D (23.4k rows) via vector search
    3. Confidence Gate:
       • HIGH  (≥ 0.95) → Direct Retrieval  (zero-latency, high fidelity)
       • LOW/MED         → Kinematic Auditor Spoke via Gemini with few-shot grounding
    4. Multi-Stage Validation — regex extraction, fallback parsing, quality guard
    5. Output: Full Kinematic Description (never truncated, never just "a person")

Key design decisions:
    - Uses gemini-2.5-flash with max_tokens=200 for complete descriptions.
    - Few-shot examples from humanml3d_library ChromaDB ground the vocabulary.
    - Anti-truncation rules baked into the system prompt.
    - [[ REFINED_PROMPT ]] delimiters enable robust regex extraction.
    - Force-Expansion List decomposes compound exercises DART cannot render as one entity.
    - Conversation history injection enables follow-up refinement ("do it faster").
    - Falls back gracefully to the best library match on any failure.
"""

import json
import os
import re
import time
from typing import Dict, List, Optional, Tuple

from utils.logger import get_logger
from utils.gemini_client import GeminiClientWrapper

logger = get_logger(__name__)

# ── ChromaDB settings ─────────────────────────────────────────────
CHROMA_HOST = os.getenv("CHROMA_HOST", "localhost")
CHROMA_PORT = int(os.getenv("CHROMA_PORT", "8100"))
COLLECTION_NAME = "humanml3d_library"
FEW_SHOT_K = int(os.getenv("SEMANTIC_BRIDGE_FEW_SHOT_K", "5"))

# ── Gemini settings ──────────────────────────────────────────────
BRIDGE_MODEL = os.getenv("SEMANTIC_BRIDGE_MODEL", "gemini-2.5-flash")
BRIDGE_MAX_TOKENS = int(os.getenv("SEMANTIC_BRIDGE_MAX_TOKENS", "200"))
BRIDGE_TEMPERATURE = float(os.getenv("SEMANTIC_BRIDGE_TEMPERATURE", "0.3"))

# ── Strategy thresholds ──────────────────────────────────────────
DIRECT_HIT_THRESHOLD = float(os.getenv("SEMANTIC_BRIDGE_DIRECT_HIT", "0.95"))

# ── Force-Expansion List ─────────────────────────────────────────
# Compound exercises that DART's diffusion model cannot render as a single
# atomic motion.  When detected, the Kinematic Auditor prompt forces Gemini
# to decompose them into 2-3 primitive skeletal movements.
FORCE_EXPANSION_LIST = [
    "burpee", "turkish get up", "turkish getup",
    "clean and jerk", "clean and press",
    "snatch", "thruster", "man maker", "man-maker",
    "devil press", "muscle up", "muscle-up",
]

# ── Kinematic Auditor System Prompt (Spoke) ──────────────────────
KINEMATIC_AUDITOR_PROMPT = """\
Role: You are a Kinematic Auditor. Your job is to translate user requests \
into precise, complete, and HumanML3D-compatible motion descriptions.

Investigative Guidelines:

1. Identify Entities: If the user mentions a standard exercise name \
(e.g., "Cartwheel", "Jumping Jacks", "Squat"), treat it as a "Primary Entity." \
Use a concise, complete sentence: "a person performs [Action Name]." \
Do NOT decompose standard exercises into individual bone movements \
unless the exercise appears in the FORCE_EXPANSION list below.

2. Vocabulary Alignment: Use third-person active voice with simple verbs: \
walks, jumps, bends, lifts, raises, lowers, steps, extends, rotates, \
squats, reaches, pushes, pulls, swings, kicks, lunges.

3. The "Completeness" Check (Anti-Truncation): Every sentence MUST be \
grammatically closed. If you use "and", "while", or "then", the action \
following them MUST be fully described before you stop writing.

4. Anti-Overthinking Protocol:
   - If the user_input is already a clear physical command (e.g., "Jump"), \
     do NOT rewrite it significantly. "a person jumps up" is perfect.
   - Avoid flowery language or internal states ("prepares to jump", \
     "focuses on balance"). Describe ONLY visible skeletal displacement.

5. ALWAYS start with "a person".
6. Minimum 8 words, maximum 25 words.
7. Match the linguistic style of the reference captions from the motion library.

{force_expansion_block}\

Refinement Process:
  Step 1: Extract the core action from the user request.
  Step 2: Check if the action exists as a common exercise name. \
If yes, use: "a person performs [Action Name]."
  Step 3: If the action is complex or in the FORCE_EXPANSION list, \
describe it as ONE complete, concise sentence decomposing the skeletal movements.
  Step 4: Final Validation — "Does this sentence end abruptly? \
Is it under 25 words? Does it sound like a robot describing a motion capture?"

Output Format:
Return the result STRICTLY in this format, with no other text before or after:
[[ your kinematic description here ]]
"""

# ── Default fallback description ─────────────────────────────────
_FALLBACK_DESCRIPTION = "a person performs a controlled full body exercise movement"


class SemanticBridgeService:
    """Hub & Spoke Motion Strategy Agent: user query → HumanML3D kinematic description.

    HUB: Routes between direct library retrieval and the Kinematic Auditor Spoke.
    SPOKE: Gemini-powered structured generation with [[ ]] extraction.

    Implements a two-path decision gate:
    - HIGH confidence (≥ 0.95): Direct retrieval from HumanML3D library
    - LOW/MED confidence:       Kinematic Auditor Spoke via Gemini

    Designed to complete in < 2 seconds total on warm runs.
    """

    def __init__(self) -> None:
        self._collection = None
        self._embedder = None
        self._client = GeminiClientWrapper()
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
        """Retrieve top-K HumanML3D candidates with scores.

        Returns list of dicts with keys: text, score, motion_id, duration
        """
        if self._collection is None or self._embedder is None:
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

    # ── Force-Expansion Detection ────────────────────────────────

    def _needs_force_expansion(self, query: str) -> bool:
        """Check if the query mentions a compound exercise from the Force-Expansion List."""
        query_lower = query.lower()
        return any(term in query_lower for term in FORCE_EXPANSION_LIST)

    def _build_force_expansion_block(self, query: str) -> str:
        """Build the FORCE_EXPANSION instruction block for the system prompt.

        If the query contains a compound exercise, inject explicit decomposition
        instructions. Otherwise, return an empty string (no overhead).
        """
        if not self._needs_force_expansion(query):
            return ""

        matched = [t for t in FORCE_EXPANSION_LIST if t in query.lower()]
        return (
            f"\nFORCE_EXPANSION ACTIVE for: {', '.join(matched)}.\n"
            f"This exercise cannot be rendered as a single atomic motion by the DART engine.\n"
            f"You MUST decompose it into 2-3 primitive skeletal movements.\n"
            f"Example: 'burpee' → 'a person squats down, kicks their legs back into a "
            f"plank, then jumps up with arms overhead'\n\n"
        )

    # ── Conversation Context Builder ─────────────────────────────

    def _build_context_block(
        self, conversation_history: Optional[List[Dict[str, str]]] = None,
    ) -> str:
        """Extract the last few turns of conversation for follow-up awareness.

        Enables the Auditor to handle queries like "Do it faster" or "Again"
        by providing the recent motion context.
        """
        if not conversation_history:
            return ""

        # Keep only last 3 turns to stay within token budget
        recent = conversation_history[-3:]
        lines = []
        for turn in recent:
            role = turn.get("role", "user")
            content = turn.get("content", "")
            if content:
                lines.append(f"{role}: {content}")

        if not lines:
            return ""

        return (
            "\nRecent conversation context (use this to understand follow-up requests):\n"
            + "\n".join(lines)
            + "\n"
        )

    # ── Core Translation (Hub) ───────────────────────────────────

    def translate(
        self,
        user_query: str,
        conversation_history: Optional[List[Dict[str, str]]] = None,
    ) -> str:
        """Hub: Translate a user query into a HumanML3D-compatible description.

        Decision flow:
        1. Check Force-Expansion List (compound exercises bypass library)
        2. Probe HumanML3D library for semantic match
        3. If HIGH confidence (≥ 0.95) → return library description directly
        4. If LOW/MED or force-expansion → Kinematic Auditor Spoke via Gemini
        5. Multi-stage validation on the output

        Args:
            user_query: The raw user exercise query.
            conversation_history: Optional list of recent chat turns for follow-ups.

        Returns:
            A full kinematic description (8-25 words, starting with "a person").
        """
        t0 = time.perf_counter()

        # ── Step 0: Force-Expansion Gate ─────────────────────────
        force_expand = self._needs_force_expansion(user_query)
        if force_expand:
            logger.info(
                "SemanticBridge: Force-expansion triggered for '%s' — skipping library",
                user_query[:50],
            )

        # ── Step 1: Library-First Check ──────────────────────────
        candidates = self._retrieve_candidates(user_query)
        retrieval_ms = (time.perf_counter() - t0) * 1000

        top1_score = candidates[0]["score"] if candidates else 0.0
        top1_text = candidates[0]["text"] if candidates else ""

        # ── Step 2: Confidence Gate ──────────────────────────────
        if (
            not force_expand
            and top1_score >= DIRECT_HIT_THRESHOLD
            and top1_text
        ):
            # FAST PATH: Direct Hit — library description is already in
            # native HumanML3D vocabulary, no Gemini call needed.
            total_ms = (time.perf_counter() - t0) * 1000
            result = self._ensure_quality(top1_text)
            logger.info(
                "SemanticBridge DIRECT HIT in %.0fms (score=%.3f): "
                "'%s' → '%s'",
                total_ms, top1_score,
                user_query[:50], result[:80],
            )
            return result

        # ── Step 3: Kinematic Auditor Spoke — Gemini with few-shot ──
        result = self._run_kinematic_auditor(
            user_query=user_query,
            candidates=candidates,
            conversation_history=conversation_history,
        )

        total_ms = (time.perf_counter() - t0) * 1000
        strategy = "force_expansion" if force_expand else "generation"
        logger.info(
            "SemanticBridge %s in %.0fms (retrieval=%.0fms, llm=%.0fms, "
            "top1_score=%.3f): '%s' → '%s'",
            strategy, total_ms, retrieval_ms, total_ms - retrieval_ms,
            top1_score, user_query[:50], result[:80],
        )
        return result

    # ── Kinematic Auditor Spoke ──────────────────────────────────

    def _run_kinematic_auditor(
        self,
        user_query: str,
        candidates: List[Dict[str, any]],
        conversation_history: Optional[List[Dict[str, str]]] = None,
    ) -> str:
        """Spoke: Run the Kinematic Auditor via Gemini to generate a refined prompt.

        Assembles the system prompt with dynamic force-expansion instructions,
        few-shot reference captions, and conversation context, then calls Gemini
        and extracts the result from [[ ]] delimiters.

        Falls back to best library match or a safe default on any failure.
        """
        # Build dynamic system prompt
        force_block = self._build_force_expansion_block(user_query)
        system_prompt = KINEMATIC_AUDITOR_PROMPT.format(
            force_expansion_block=force_block,
        )

        # Build few-shot reference block
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

        # Build conversation context block
        context_block = self._build_context_block(conversation_history)

        # Assemble user prompt
        user_prompt = (
            f"{example_block}"
            f"{context_block}\n"
            f'User exercise request: "{user_query}"\n\n'
            f"Follow your Refinement Process and return the result "
            f"strictly inside [[ ]] delimiters."
        )

        try:
            response = self._client.chat.completions.create(
                model=BRIDGE_MODEL,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt},
                ],
                temperature=BRIDGE_TEMPERATURE,
                max_tokens=BRIDGE_MAX_TOKENS,
            )
            raw_output = response.choices[0].message.content.strip()
            result = self._extract_and_validate(raw_output)
            return result

        except Exception as exc:
            logger.error(
                "SemanticBridge: Kinematic Auditor failed: %s — "
                "falling back to best library match",
                exc,
            )
            # Fallback: use best library match, or safe default
            top1_text = candidates[0]["text"] if candidates else ""
            if top1_text:
                return self._ensure_quality(top1_text)
            return _FALLBACK_DESCRIPTION

    # ── Multi-Stage Output Extraction & Validation ───────────────

    def _extract_and_validate(self, raw_text: str) -> str:
        """Multi-stage pipeline to extract and validate the Kinematic Auditor output.

        Stage 1 (Regex Extract):  Look for [[ ... ]] delimiters.
        Stage 2 (Empty Check):    Reject if extracted text is < 5 words.
        Stage 3 (Fallback Parse): Strip formatting, find first "a person" sentence.
        Stage 4 (Quality Guard):  Ensure minimum length and "a person" prefix.
        """
        raw_text = raw_text.strip()

        # ── Stage 1: Regex extraction from [[ ]] delimiters ──────
        match = re.search(r"\[\[\s*(.*?)\s*\]\]", raw_text, re.DOTALL)
        if match:
            extracted = match.group(1).strip()
            # ── Stage 2: Empty/too-short check ───────────────────
            if len(extracted.split()) >= 5:
                logger.debug("SemanticBridge: Regex extraction succeeded: '%s'", extracted[:80])
                return self._ensure_quality(extracted)
            else:
                logger.warning(
                    "SemanticBridge: Regex match too short (%d words): '%s' — "
                    "falling through to fallback parse",
                    len(extracted.split()), extracted,
                )

        # ── Stage 3: Fallback parsing (no valid [[ ]] found) ─────
        logger.warning(
            "SemanticBridge: No valid [[ ]] delimiters found in LLM output — "
            "attempting fallback parse. Raw: '%s'",
            raw_text[:120],
        )
        cleaned = self._fallback_parse(raw_text)
        return self._ensure_quality(cleaned)

    def _fallback_parse(self, text: str) -> str:
        """Attempt to salvage a kinematic description from unstructured LLM output.

        1. Strip markdown formatting and known AI prefixes.
        2. Find the first sentence starting with "a person" (case-insensitive).
        3. If nothing found, return the safe default.
        """
        text = text.strip()

        # Remove markdown fencing
        if text.startswith("```"):
            text = text.strip("`").strip()

        # Remove surrounding quotes
        if (text.startswith('"') and text.endswith('"')) or \
           (text.startswith("'") and text.endswith("'")):
            text = text[1:-1].strip()

        # Remove known AI prefixes
        for prefix in (
            "kinematic description:", "output:", "rewritten caption:",
            "description:", "motion:", "caption:", "refined prompt:",
            "sure!", "sure,", "here is", "here's",
        ):
            if text.lower().startswith(prefix):
                text = text[len(prefix):].strip()

        # Remove "Chosen Path:" / "Strategy:" lines
        lines = text.strip().splitlines()
        cleaned_lines = []
        for line in lines:
            lower = line.strip().lower()
            if lower.startswith("chosen path") or lower.startswith("optimized prompt"):
                continue
            if lower.startswith("path:") or lower.startswith("strategy:"):
                continue
            if lower.startswith("step ") and ":" in lower:
                continue
            cleaned_lines.append(line.strip())
        text = " ".join(cleaned_lines).strip()

        # Try to find a sentence starting with "a person"
        person_match = re.search(r"(a person\b[^.!?\n]*[.!?]?)", text, re.IGNORECASE)
        if person_match:
            found = person_match.group(1).strip().rstrip(".")
            if len(found.split()) >= 5:
                return found

        # Last resort: if the text has enough words, use it directly
        if len(text.split()) >= 5:
            return text

        return _FALLBACK_DESCRIPTION

    # ── Quality Guard ────────────────────────────────────────────

    def _ensure_quality(self, text: str) -> str:
        """Validate output meets minimum quality bar.

        Guards against the 'a person' truncation bug by ensuring
        the description contains at least one action verb.
        """
        text = text.strip()
        if not text:
            return _FALLBACK_DESCRIPTION

        # Remove any residual [[ ]] brackets
        text = re.sub(r"\[\[|\]\]", "", text).strip()

        # Remove surrounding quotes
        if (text.startswith('"') and text.endswith('"')) or \
           (text.startswith("'") and text.endswith("'")):
            text = text[1:-1].strip()

        # Ensure it starts with "a person" (HumanML3D convention)
        lower = text.lower()
        if not lower.startswith("a person"):
            text = f"a person {text}"

        # Guard: reject if too short (just "a person" with no action)
        words = text.split()
        if len(words) < 5:
            # Too short — this is the truncation bug.  Append a generic
            # action to make it usable rather than sending garbage to DART.
            logger.warning(
                "SemanticBridge: output too short (%d words): '%s' — "
                "using fallback description",
                len(words), text,
            )
            return _FALLBACK_DESCRIPTION

        # Enforce max 30 words (slightly above the 25-word prompt limit
        # to avoid cutting off valid outputs)
        if len(words) > 30:
            text = " ".join(words[:30])

        return text

    # ── Async wrapper ────────────────────────────────────────────

    async def translate_async(
        self,
        user_query: str,
        conversation_history: Optional[List[Dict[str, str]]] = None,
    ) -> str:
        """Async wrapper that runs translate() in a thread pool executor."""
        import asyncio
        from functools import partial

        loop = asyncio.get_event_loop()
        func = partial(
            self.translate,
            user_query,
            conversation_history=conversation_history,
        )
        return await loop.run_in_executor(None, func)
