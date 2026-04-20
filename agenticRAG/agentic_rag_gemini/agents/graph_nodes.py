"""LangGraph node functions for the Agentic RAG orchestrator.

Each function is a stateless node that reads from AgentState and returns
a partial state update dict. Dependencies (tools, LLM, config) are injected
via closures in ``orchestrator_graph.py``.
"""

import json
import re
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Any, Dict, List, Optional

from langchain_core.messages import HumanMessage, SystemMessage

from agents.graph_state import AgentState
from utils.logger import get_logger

logger = get_logger(__name__)


# ──────────────────────────────────────────────────────────────────────────────
# Helpers
# ──────────────────────────────────────────────────────────────────────────────


def _clean_json_response(text: str) -> str:
    """Strip markdown fences and isolate the JSON object."""
    text = re.sub(r'```json\s*', '', text)
    text = re.sub(r'```\s*', '', text)
    text = text.strip()
    match = re.search(r'\{[^{}]*(?:\{[^{}]*\}[^{}]*)*\}', text, re.DOTALL)
    return match.group(0) if match else text


# ──────────────────────────────────────────────────────────────────────────────
# System prompt (unchanged from OrchestratorAgent._UNIFIED_ANALYSIS_PROMPT)
# ──────────────────────────────────────────────────────────────────────────────

UNIFIED_ANALYSIS_PROMPT = """You are the routing brain for a physical therapy AI assistant.
Analyze the user query and return a single JSON object.

Intents:
- conversation            : general chat, greetings, or follow-ups with no exercise content
- knowledge_query         : asks for explanation, facts, or non-motion advice  
- exercise_recommendation : asks for one or more exercises, stretches, or workouts
- visualize_motion        : asks to see / animate / show a specific single movement

Actions:
- call_llm       : generate a text answer (use for knowledge_query, exercise_recommendation, conversation)
- generate_motion: produce a motion animation (use for visualize_motion; can combine with call_llm)
- hybrid         : both call_llm and generate_motion
- clarify        : ask user for more detail

Respond with valid JSON ONLY, no extra text:
{
  "intent": "<intent_value>",
  "action": "call_llm" | "generate_motion" | "hybrid" | "clarify",
  "confidence": 0.0-1.0,
  "reasoning": "one sentence",
  "exercise_name": "<exercise if visualize_motion, else null>",
  "expanded_query": "<rephrase the query with 2-3 extra keywords to improve retrieval, or repeat original>",
  "needs_rag": true | false,
  "generate_motion": true | false,
  "motion_type": "<motion type if applicable, else null>"
}"""


# ──────────────────────────────────────────────────────────────────────────────
# Node: classify
# ──────────────────────────────────────────────────────────────────────────────


def make_classify_node(llm, config):
    """Create the classification node with injected LLM + config.

    This node replaces ``OrchestratorAgent.classify_intent_and_analyze()``.
    It performs a single LLM call to classify intent + produce routing metadata.

    Returns:
        A function ``classify(state) -> dict`` suitable for StateGraph.
    """

    def classify(state: AgentState) -> dict:
        query = state["query"]
        history = state.get("conversation_history") or []

        logger.info("classify node: '%s'", query[:100])

        # Build history snippet
        history_snippet = ""
        if history:
            turns = history[-3:]
            history_snippet = "\n\nRecent conversation:\n" + "\n".join(
                f"{t['role']}: {t['content']}" for t in turns
            )

        try:
            messages = [
                SystemMessage(content=UNIFIED_ANALYSIS_PROMPT),
                HumanMessage(content=query + history_snippet),
            ]
            response = llm.invoke(messages)
            raw = response.content if hasattr(response, "content") else str(response)
            data = json.loads(_clean_json_response(raw))

            intent = data.get("intent", "knowledge_query")
            action = data.get("action", "call_llm")

            result = {
                "intent": intent,
                "action": action,
                "confidence": float(data.get("confidence", 0.8)),
                "reasoning": data.get("reasoning", ""),
                "exercise_name": data.get("exercise_name"),
                "expanded_query": data.get("expanded_query") or query,
                "needs_rag": bool(data.get("needs_rag", True)),
                "generate_motion": bool(data.get("generate_motion", False)),
                "motion_type": data.get("motion_type"),
            }

            logger.info(
                "classify → intent=%s action=%s needs_rag=%s generate_motion=%s",
                intent, action, result["needs_rag"], result["generate_motion"],
            )
            return result

        except Exception as exc:
            logger.error("classify node failed: %s", exc)
            return {
                "intent": "knowledge_query",
                "action": "call_llm",
                "confidence": 0.5,
                "reasoning": "Fallback — classification failed",
                "exercise_name": None,
                "expanded_query": query,
                "needs_rag": True,
                "generate_motion": False,
                "motion_type": None,
            }

    return classify


# ──────────────────────────────────────────────────────────────────────────────
# Node: transform_query
# ──────────────────────────────────────────────────────────────────────────────


def make_transform_query_node(query_transformer):
    """Create the query transformation node.

    Replaces the inline query transformation in OrchestratorAgent.process_query().
    Generates an expanded query + HyDE document for better retrieval.

    Args:
        query_transformer: QueryTransformer instance.

    Returns:
        A function ``transform_query(state) -> dict``.
    """

    def transform_query(state: AgentState) -> dict:
        query = state["query"]
        needs_rag = state.get("needs_rag", True)
        action = state.get("action", "call_llm")

        if not needs_rag and action != "generate_motion":
            logger.info("transform_query: skipped (no RAG needed)")
            return {
                "expanded_query": state.get("expanded_query", query),
                "hyde_document": query,
            }

        if query_transformer is None:
            logger.info("transform_query: no transformer configured, using original query")
            return {
                "expanded_query": state.get("expanded_query", query),
                "hyde_document": query,
            }

        try:
            result = query_transformer.transform_query(query)
            expanded = result.get("expanded_query", query)
            hyde = result.get("hyde_document", query)
            logger.info("transform_query: expanded=%d chars, hyde=%d chars", len(expanded), len(hyde))
            return {
                "expanded_query": expanded,
                "hyde_document": hyde,
            }
        except Exception as exc:
            logger.warning("transform_query failed: %s", exc)
            return {
                "expanded_query": state.get("expanded_query", query),
                "hyde_document": query,
            }

    return transform_query


# ──────────────────────────────────────────────────────────────────────────────
# Node: execute_tools
# ──────────────────────────────────────────────────────────────────────────────


def make_execute_tools_node(memory_tool, document_tool, web_search_tool):
    """Create the concurrent tool execution node.

    Replaces ``OrchestratorAgent._select_tools()`` + ``_run_tools()``.

    Args:
        memory_tool:      MemoryTool instance (or None).
        document_tool:    DocumentRetrievalTool instance (or None).
        web_search_tool:  WebSearchTool instance (or None).

    Returns:
        A function ``execute_tools(state) -> dict``.
    """

    # Intent sets that skip tools (matching original logic)
    _SKIP_ALL = {"conversation"}
    _SKIP_CONTENT = {"visualize_motion"}

    def execute_tools(state: AgentState) -> dict:
        intent = state.get("intent", "knowledge_query")
        action = state.get("action", "call_llm")
        expanded_query = state.get("expanded_query", state["query"])
        user_id = state["user_id"]

        # Determine which tools to run based on intent
        if intent in _SKIP_ALL:
            logger.info("execute_tools: skipping all (conversation intent)")
            return {"tool_results": {}, "selected_tools": []}

        # Build candidate tool list based on action type
        action_tool_map = {
            "call_llm": ["memory", "documents", "web_search"],
            "generate_motion": ["memory"],
            "hybrid": ["memory", "documents", "web_search"],
            "retrieve_memory": ["memory"],
            "clarify": [],
        }
        candidates = action_tool_map.get(action, ["memory", "documents", "web_search"])

        if intent in _SKIP_CONTENT:
            candidates = [t for t in candidates if t == "memory"]

        # Filter to available tools
        available = {
            "memory": memory_tool is not None,
            "documents": document_tool is not None,
            "web_search": web_search_tool is not None,
        }
        selected = [t for t in candidates if available.get(t, False)]

        if not selected:
            return {"tool_results": {}, "selected_tools": []}

        # Concurrent execution
        tool_fns = {
            "memory": lambda: memory_tool.retrieve_memory(user_id=user_id, query=expanded_query),
            "documents": lambda: document_tool.search_documents(query=expanded_query, user_id=user_id),
            "web_search": lambda: web_search_tool.search_web(query=expanded_query),
        }

        results: Dict[str, Any] = {}
        with ThreadPoolExecutor(max_workers=len(selected)) as executor:
            futures = {
                executor.submit(tool_fns[t]): t
                for t in selected
                if t in tool_fns
            }
            for future in as_completed(futures):
                name = futures[future]
                try:
                    results[name] = future.result()
                    logger.debug("Tool '%s' completed", name)
                except Exception as exc:
                    logger.error("Tool '%s' failed: %s", name, exc)

        logger.info("execute_tools: ran %s", list(results.keys()))
        return {"tool_results": results, "selected_tools": selected}

    return execute_tools


# ──────────────────────────────────────────────────────────────────────────────
# Node: double_rag
# ──────────────────────────────────────────────────────────────────────────────


def make_double_rag_node(llm, document_tool, motion_retriever, config):
    """Create the Double-RAG node (clinical dispatch → constraint extraction → conditioned motion).

    Replaces the inline Double-RAG logic in OrchestratorAgent.process_query().

    Args:
        llm:              LangChain ChatModel for constraint extraction.
        document_tool:    DocumentRetrievalTool instance.
        motion_retriever: MotionCandidateRetriever instance.
        config:           Orchestrator config.

    Returns:
        A function ``double_rag(state) -> dict``.
    """

    def double_rag(state: AgentState) -> dict:
        needs_rag = state.get("needs_rag", False)
        action = state.get("action", "call_llm")
        expanded_query = state.get("expanded_query", state["query"])
        hyde_document = state.get("hyde_document", state["query"])

        if not (needs_rag or action == "generate_motion"):
            return {
                "clinical_docs": [],
                "constraints": "",
                "motion_candidates": [],
                "double_rag_meta": {},
            }

        if document_tool is None:
            return {
                "clinical_docs": [],
                "constraints": "",
                "motion_candidates": [],
                "double_rag_meta": {},
            }

        try:
            # Step 1: Clinical dispatch
            clinical_docs = document_tool.search_documents(expanded_query)

            # Step 2: Constraint extraction via LLM
            constraints = _extract_constraints(llm, clinical_docs, config)
            logger.info("Double-RAG constraints: %s", constraints[:100])

            # Step 3: Conditioned motion search
            motion_candidates = []
            if motion_retriever is not None:
                conditioned_query = f"{hyde_document} Constraints: {constraints}"
                motion_candidates = motion_retriever.retrieve_top_k(conditioned_query, k=1)
                if motion_candidates:
                    hyde_document = motion_candidates[0].text_description

            # Merge clinical docs into tool_results
            existing_results = dict(state.get("tool_results", {}))
            existing_results["documents"] = clinical_docs

            return {
                "clinical_docs": clinical_docs,
                "constraints": constraints,
                "motion_candidates": motion_candidates,
                "hyde_document": hyde_document,
                "tool_results": existing_results,
                "double_rag_meta": {"constraints": constraints},
            }

        except Exception as exc:
            logger.error("double_rag node failed: %s", exc)
            return {
                "clinical_docs": [],
                "constraints": "General safe range of motion.",
                "motion_candidates": [],
                "double_rag_meta": {},
            }

    return double_rag


def _extract_constraints(llm, clinical_docs: list, config) -> str:
    """Extract safety constraints from clinical documents via LLM."""
    if not clinical_docs:
        return "No specific constraints."

    clinical_text = "\n".join([doc.get("document", "") for doc in clinical_docs[:3]])

    prompt = (
        "You are a clinical expert. Extract the key physical constraints, "
        "safety warnings, and targeted muscles from the provided clinical text.\n"
        "Ensure the output is extremely brief, no more than 2 sentences. "
        "Focus ONLY on biomechanical rules and limitations.\n\n"
        f"Clinical Text:\n{clinical_text}"
    )

    try:
        response = llm.invoke([HumanMessage(content=prompt)])
        text = response.content if hasattr(response, "content") else str(response)
        return text.strip()
    except Exception as e:
        logger.error("Constraint extraction failed: %s", e)
        return "General safe range of motion."


# ──────────────────────────────────────────────────────────────────────────────
# Node: assemble
# ──────────────────────────────────────────────────────────────────────────────


def make_assemble_node():
    """Create the final assembly node that builds the action_plan dict.

    This is a pure data-transformation node (no I/O, no LLM calls).

    Returns:
        A function ``assemble(state) -> dict``.
    """

    def assemble(state: AgentState) -> dict:
        intent = state.get("intent", "knowledge_query")
        action = state.get("action", "call_llm")
        tool_results = state.get("tool_results", {})

        action_plan = {
            "user_id": state["user_id"],
            "query": state["query"],
            "intent": intent,
            "action": action,
            "decision": {
                "action": action,
                "confidence": state.get("confidence", 0.8),
                "reasoning": state.get("reasoning", ""),
                "parameters": {
                    "use_memory": True,
                    "use_documents": state.get("needs_rag", True),
                    "generate_motion": state.get("generate_motion", False),
                    "motion_type": state.get("motion_type"),
                },
            },
            "actions": {
                "retrieve_memory": "memory" in (state.get("selected_tools") or []),
                "call_llm": action in ("call_llm", "hybrid", "retrieve_memory"),
                "generate_motion": (
                    state.get("generate_motion", False)
                    or intent == "visualize_motion"
                ),
            },
            "parameters": {
                "use_memory": True,
                "use_documents": state.get("needs_rag", True),
                "generate_motion": state.get("generate_motion", False),
                "motion_type": state.get("motion_type"),
            },
            "expanded_query": state.get("expanded_query", state["query"]),
            "hyde_document": state.get("hyde_document", state["query"]),
            "double_rag_meta": state.get("double_rag_meta", {}),
            "needs_rag": state.get("needs_rag", True),
            "exercise_name": state.get("exercise_name"),
            "confidence": state.get("confidence", 0.8),
            "tool_results": tool_results,
        }

        logger.info(
            "assemble → intent=%s action=%s needs_rag=%s tools=%s",
            intent, action, state.get("needs_rag"), list(tool_results.keys()),
        )
        return {"action_plan": action_plan}

    return assemble
