"""LangGraph orchestrator — replaces the procedural OrchestratorAgent.

Builds a ``StateGraph`` with 5 nodes::

    classify ──→ transform_query ──→ execute_tools ──→ double_rag ──→ assemble

Conditional edges:
  - After ``classify``:  if intent == "conversation" → skip to ``assemble``.
  - After ``execute_tools``: if no RAG needed → skip ``double_rag``.

Usage:
    from agents.orchestrator_graph import build_orchestrator_graph

    graph = build_orchestrator_graph(
        memory_tool=memory_tool,
        document_tool=document_tool,
        web_search_tool=web_search_tool,
    )
    result = graph.invoke({
        "query": "show me how to do a squat",
        "user_id": "user_123",
    })
    action_plan = result["action_plan"]
"""

from typing import Optional

from langgraph.graph import StateGraph, END

from agents.graph_state import AgentState
from agents.graph_nodes import (
    make_classify_node,
    make_transform_query_node,
    make_execute_tools_node,
    make_double_rag_node,
    make_assemble_node,
)
from agents.tools.memory_tool import MemoryTool
from agents.tools.document_retrieval_tool import DocumentRetrievalTool
from agents.tools.web_search_tool import WebSearchTool
from utils.llm_provider import get_llm
from utils.logger import get_logger

logger = get_logger(__name__)


def build_orchestrator_graph(
    memory_tool: Optional[MemoryTool] = None,
    document_tool: Optional[DocumentRetrievalTool] = None,
    web_search_tool: Optional[WebSearchTool] = None,
    query_transformer=None,
    motion_retriever=None,
    llm=None,
    config=None,
) -> StateGraph:
    """Build and compile the LangGraph orchestration pipeline.

    Args:
        memory_tool:       MemoryTool instance (injected from app).
        document_tool:     DocumentRetrievalTool instance.
        web_search_tool:   WebSearchTool instance.
        query_transformer: QueryTransformer instance (or None to skip).
        motion_retriever:  MotionCandidateRetriever instance (or None).
        llm:               LangChain ChatModel. If None, creates one via get_llm().
        config:            Orchestrator config section.

    Returns:
        Compiled StateGraph that accepts AgentState input and returns
        AgentState with ``action_plan`` populated.
    """
    from config import get_config

    if config is None:
        config = get_config().orchestrator
    if llm is None:
        llm = get_llm(
            model=config.model,
            temperature=0.0,
            max_tokens=1024,
        )

    # ── Create node functions ──────────────────────────────────────────────
    classify_fn = make_classify_node(llm, config)
    transform_fn = make_transform_query_node(query_transformer)
    tools_fn = make_execute_tools_node(memory_tool, document_tool, web_search_tool)
    double_rag_fn = make_double_rag_node(llm, document_tool, motion_retriever, config)
    assemble_fn = make_assemble_node()

    # ── Build graph ────────────────────────────────────────────────────────
    graph = StateGraph(AgentState)

    graph.add_node("classify", classify_fn)
    graph.add_node("transform_query", transform_fn)
    graph.add_node("execute_tools", tools_fn)
    graph.add_node("double_rag", double_rag_fn)
    graph.add_node("assemble", assemble_fn)

    # ── Entry point ────────────────────────────────────────────────────────
    graph.set_entry_point("classify")

    # ── Conditional edge after classify ────────────────────────────────────
    def _after_classify(state: AgentState) -> str:
        """Skip heavy processing for simple conversation queries."""
        intent = state.get("intent", "knowledge_query")
        if intent == "conversation":
            logger.info("Graph: conversation intent → skipping to assemble")
            return "assemble"
        return "transform_query"

    graph.add_conditional_edges(
        "classify",
        _after_classify,
        {
            "assemble": "assemble",
            "transform_query": "transform_query",
        },
    )

    # ── Linear edges ──────────────────────────────────────────────────────
    graph.add_edge("transform_query", "execute_tools")

    # ── Conditional edge after execute_tools ──────────────────────────────
    def _after_tools(state: AgentState) -> str:
        """Skip double-RAG when it's not needed."""
        needs_rag = state.get("needs_rag", False)
        action = state.get("action", "call_llm")
        if needs_rag or action == "generate_motion":
            return "double_rag"
        return "assemble"

    graph.add_conditional_edges(
        "execute_tools",
        _after_tools,
        {
            "double_rag": "double_rag",
            "assemble": "assemble",
        },
    )

    graph.add_edge("double_rag", "assemble")
    graph.add_edge("assemble", END)

    # ── Compile ────────────────────────────────────────────────────────────
    compiled = graph.compile()

    logger.info(
        "LangGraph orchestrator compiled: "
        "classify → transform_query → execute_tools → double_rag → assemble → END"
    )
    return compiled
