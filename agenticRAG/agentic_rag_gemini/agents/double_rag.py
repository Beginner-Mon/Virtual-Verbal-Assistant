"""Double-RAG agent — Hub & Spoke kinematic dispatcher.

Pipeline (extracted from the legacy ``OrchestratorAgent.process_query`` so the
router only routes):

    1. Clinical Dispatch
       Pull top-K clinical/exercise documents for the expanded query.
    2. Constraint Extraction
       Tiny LLM call that condenses safety / biomechanical limits into ≤ 2
       sentences.  Cached in-process for the lifetime of one request.
    3. Conditioned Motion Search
       Re-query the HumanML3D motion library using the constraint-augmented
       HyDE document so DART receives a vocabulary-aligned caption.

The output ``DoubleRAGResult`` is exactly the bundle the orchestrator and the
RAG pipeline need; nothing more leaks across the boundary.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

from agents.tools.document_retrieval_tool import DocumentRetrievalTool
from agents.tools.motion_candidate_retriever import MotionCandidate, MotionCandidateRetriever
from core.tracing import AgentTrace, TraceStage
from utils.gemini_client import GeminiClientWrapper
from utils.logger import get_logger

logger = get_logger(__name__)


_CONSTRAINT_PROMPT_TEMPLATE = (
    "You are a clinical expert. Extract the key physical constraints, safety "
    "warnings, and targeted muscles from the provided clinical text.\n"
    "Ensure the output is extremely brief, no more than 2 sentences. "
    "Focus ONLY on biomechanical rules and limitations.\n\n"
    "Clinical Text:\n{clinical_text}"
)


@dataclass
class DoubleRAGResult:
    """Bundle of artefacts produced by the Double-RAG agent."""

    clinical_docs: List[Dict[str, Any]] = field(default_factory=list)
    constraints: str = "No specific constraints."
    motion_candidates: List[MotionCandidate] = field(default_factory=list)
    refined_hyde_document: Optional[str] = None

    @property
    def has_motion_match(self) -> bool:
        return bool(self.motion_candidates)


class DoubleRAGAgent:
    """Hub: routes between clinical retrieval, constraint LLM, and motion search."""

    def __init__(
        self,
        document_tool: DocumentRetrievalTool,
        motion_retriever: Optional[MotionCandidateRetriever] = None,
        client: Optional[GeminiClientWrapper] = None,
        constraint_model: str = "gemini-2.5-flash",
        constraint_max_tokens: int = 150,
        clinical_top_k: int = 5,
        motion_top_k: int = 1,
    ) -> None:
        self._document_tool = document_tool
        self._motion_retriever = motion_retriever or MotionCandidateRetriever()
        self._client = client or GeminiClientWrapper()
        self._constraint_model = constraint_model
        self._constraint_max_tokens = int(constraint_max_tokens)
        self._clinical_top_k = int(clinical_top_k)
        self._motion_top_k = int(motion_top_k)

    # ── Public entrypoint ──────────────────────────────────────────────────
    def run(
        self,
        expanded_query: str,
        hyde_document: str,
        user_id: Optional[str] = None,
        trace: Optional[AgentTrace] = None,
    ) -> DoubleRAGResult:
        """Run the full Double-RAG pipeline.

        Args:
            expanded_query:  Query to use for clinical dispatch (already enriched).
            hyde_document:   Hypothetical-Document for the motion-search step.
            user_id:         Optional user ID for per-user document scoping.
            trace:           Optional ``AgentTrace`` for telemetry.

        Returns:
            ``DoubleRAGResult`` with the three artefacts.
        """
        result = DoubleRAGResult(refined_hyde_document=hyde_document)

        # ── Stage 1: Clinical Dispatch ────────────────────────────────────
        with TraceStage(trace, "double_rag.clinical_dispatch"):
            clinical_docs = self._safe_clinical_dispatch(expanded_query, user_id)
        result.clinical_docs = clinical_docs
        if trace is not None:
            trace.add_decision(
                "double_rag.clinical_dispatch",
                doc_count=len(clinical_docs),
                expanded_query=expanded_query[:120],
            )

        # ── Stage 2: Constraint Extraction ────────────────────────────────
        with TraceStage(trace, "double_rag.constraint_extraction"):
            constraints = self._extract_constraints(clinical_docs, trace=trace)
        result.constraints = constraints
        if trace is not None:
            trace.add_decision(
                "double_rag.constraint_extraction",
                constraints_preview=constraints[:160],
            )

        # ── Stage 3: Conditioned Motion Search ────────────────────────────
        conditioned_query = (
            f"{hyde_document} Constraints: {constraints}".strip()
        )
        with TraceStage(trace, "double_rag.motion_search"):
            motion_candidates = self._safe_motion_search(conditioned_query)
        result.motion_candidates = motion_candidates
        if motion_candidates:
            result.refined_hyde_document = motion_candidates[0].text_description

        if trace is not None:
            trace.add_decision(
                "double_rag.motion_search",
                candidate_count=len(motion_candidates),
                top_score=(motion_candidates[0].score if motion_candidates else None),
                refined_hyde_preview=(result.refined_hyde_document or "")[:120],
            )

        return result

    # ── Stage helpers ──────────────────────────────────────────────────────
    def _safe_clinical_dispatch(self, query: str, user_id: Optional[str]) -> List[Dict[str, Any]]:
        try:
            return self._document_tool.search_documents(
                query=query,
                user_id=user_id,
                top_k=self._clinical_top_k,
            )
        except Exception as exc:  # noqa: BLE001
            logger.error("DoubleRAG.clinical_dispatch failed: %s", exc)
            return []

    def _safe_motion_search(self, conditioned_query: str) -> List[MotionCandidate]:
        try:
            return self._motion_retriever.retrieve_top_k(conditioned_query, k=self._motion_top_k)
        except Exception as exc:  # noqa: BLE001
            logger.error("DoubleRAG.motion_search failed: %s", exc)
            return []

    def _extract_constraints(
        self,
        clinical_docs: List[Dict[str, Any]],
        trace: Optional[AgentTrace] = None,
    ) -> str:
        if not clinical_docs:
            return "No specific constraints."

        clinical_text = "\n".join(
            (doc.get("document") or doc.get("content") or "") for doc in clinical_docs[:3]
        ).strip()
        if not clinical_text:
            return "No specific constraints."

        prompt = _CONSTRAINT_PROMPT_TEMPLATE.format(clinical_text=clinical_text)
        try:
            response = self._client.chat.completions.create(
                model=self._constraint_model,
                messages=[{"role": "user", "content": prompt}],
                temperature=0.1,
                max_tokens=self._constraint_max_tokens,
            )
            if trace is not None:
                trace.increment_llm_calls(1)
            content = response.choices[0].message.content if response and response.choices else ""
            return (content or "").strip() or "General safe range of motion."
        except Exception as exc:  # noqa: BLE001
            logger.error("DoubleRAG.extract_constraints failed: %s", exc)
            return "General safe range of motion."


__all__ = ["DoubleRAGAgent", "DoubleRAGResult"]
