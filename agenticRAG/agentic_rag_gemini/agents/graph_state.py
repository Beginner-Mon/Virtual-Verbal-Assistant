"""LangGraph state definition for the Agentic RAG orchestrator.

Defines the TypedDict that flows through the StateGraph nodes.
Each node reads from and writes to these fields.
"""

from typing import Any, Dict, List, Optional
from typing_extensions import TypedDict


class AgentState(TypedDict, total=False):
    """State that flows through the LangGraph orchestration pipeline.

    Fields
    ------
    query : str
        Original user query.
    user_id : str
        Current user identifier.
    conversation_history : list
        Recent conversation turns for follow-up awareness.
    user_context : dict
        Additional user context (physical conditions, preferences, etc.)

    # Classification node outputs
    intent : str
        One of: conversation, knowledge_query, exercise_recommendation, visualize_motion.
    action : str
        One of: call_llm, generate_motion, hybrid, clarify.
    confidence : float
        Model's confidence in the classification (0.0–1.0).
    reasoning : str
        One-sentence explanation of the routing decision.
    exercise_name : str | None
        Detected exercise name (for visualize_motion intent).
    needs_rag : bool
        Whether the query needs retrieval-augmented generation.
    generate_motion : bool
        Whether motion animation should be generated.
    motion_type : str | None
        Type of motion to generate.

    # Query transformation node outputs
    expanded_query : str
        Query enriched with extra keywords for better retrieval.
    hyde_document : str
        Hypothetical document for HyDE-style retrieval.

    # Tool execution node outputs
    tool_results : dict
        Results from concurrent tool execution (memory, documents, web_search).
    selected_tools : list
        List of tool names that were run.

    # Double-RAG node outputs
    clinical_docs : list
        Retrieved clinical documents.
    constraints : str
        Extracted safety constraints from clinical documents.
    motion_candidates : list
        Retrieved motion candidates conditioned on constraints.
    double_rag_meta : dict
        Metadata from the Double-RAG process.

    # Final assembled output
    action_plan : dict
        The complete action plan ready for the RAG pipeline / API response.
    """

    # ── Input ──
    query: str
    user_id: str
    conversation_history: Optional[List[Dict[str, str]]]
    user_context: Optional[Dict[str, Any]]

    # ── Classification ──
    intent: str
    action: str
    confidence: float
    reasoning: str
    exercise_name: Optional[str]
    needs_rag: bool
    generate_motion: bool
    motion_type: Optional[str]

    # ── Query Transformation ──
    expanded_query: str
    hyde_document: str

    # ── Tool Results ──
    tool_results: Dict[str, Any]
    selected_tools: List[str]

    # ── Double-RAG ──
    clinical_docs: List[Dict[str, Any]]
    constraints: str
    motion_candidates: list
    double_rag_meta: Dict[str, Any]

    # ── Output ──
    action_plan: Dict[str, Any]
