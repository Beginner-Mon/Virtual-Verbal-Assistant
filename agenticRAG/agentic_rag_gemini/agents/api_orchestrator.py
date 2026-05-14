"""Orchestrator Agent — the brain of the Agentic RAG system.

This module implements the orchestrator agent that:
1. Uses an LLM to analyze the user query and decide which tools to invoke
2. Runs selected tools concurrently via ThreadPoolExecutor
3. Returns the decision + pre-fetched tool results to the caller

Architecture (Agent + Tools pattern):
    OrchestratorAgent
        ├── analyze_query()   ← LLM decision
        ├── _select_tools()   ← maps decision → tool names
        ├── _run_tools()      ← concurrent execution via ThreadPoolExecutor
        └── process_query()   ← combines the above and returns full action_plan

Tools (injected via __init__):
    MemoryTool             → retrieve_memory(user_id, query)
    DocumentRetrievalTool  → search_documents(query)
    WebSearchTool          → search_web(query)
"""

import json
import re
from concurrent.futures import as_completed
from typing import Dict, List, Optional, Any

from config import get_config
from utils.logger import get_logger
from utils.prompt_templates import ORCHESTRATOR_PROMPTS
from utils.gemini_client import GeminiClientWrapper
from agents.tools.memory_tool import MemoryTool
from agents.tools.document_retrieval_tool import DocumentRetrievalTool
from agents.tools.web_search_tool import WebSearchTool
from agents.tools.motion_candidate_retriever import MotionCandidateRetriever
from agents.query_transform import QueryTransformer
from agents.intents import (
    ACTION_DEFAULT_TOOLS,
    INTENT_SKIP_CONTENT_TOOLS,
    ActionType,
    IntentType,
    parse_action,
    parse_intent,
)
from agents.double_rag import DoubleRAGAgent
from core.tracing import AgentTrace, TraceStage
from core.resource_guard import shared_tool_executor


logger = get_logger(__name__)


def clean_json_response(response_text: str) -> str:
    """Clean and normalize JSON response from LLM.

    Removes markdown code blocks, fixes common formatting issues.

    Args:
        response_text: Raw LLM response

    Returns:
        Cleaned JSON string
    """
    response_text = re.sub(r'```json\s*', '', response_text)
    response_text = re.sub(r'```\s*', '', response_text)
    response_text = response_text.strip()

    json_match = re.search(r'\{[^{}]*(?:\{[^{}]*\}[^{}]*)*\}', response_text, re.DOTALL)
    if json_match:
        response_text = json_match.group(0)

    return response_text


# IntentType / ActionType are imported from agents.intents (single source of truth).
# They are re-exported here for backward compatibility with callers and tests that
# do ``from agents.api_orchestrator import IntentType, ActionType``.


# System prompt for the UNIFIED intent + analysis call (replaces two separate LLM calls).
_UNIFIED_ANALYSIS_PROMPT = """You are the routing brain for a physical therapy AI assistant.
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
  "motion_type": "stretch" | "exercise" | "posture" | null
}

Rules:
- needs_rag=false for conversation and visualize_motion (no document/web retrieval needed)
- needs_rag=true for knowledge_query and exercise_recommendation
- generate_motion=true only for visualize_motion
- expanded_query should add anatomical/physiotherapy synonyms when relevant"""

# kept for backward compat — used only in the legacy classify_intent() method
_INTENT_CLASSIFIER_PROMPT = """You are an intent classifier for a physical therapy assistant.
Classify the user query into exactly one of these intents:

- conversation            : general chat or follow-up
- knowledge_query         : asks for knowledge, explanation, or facts
- exercise_recommendation : asks for one or more exercises / stretches / workouts
- visualize_motion        : asks to see, visualize, or animate a specific movement

Respond with valid JSON only, no extra text:
{"intent": "<intent_value>"}"""


# Action → tool list and intent skip-set come from agents.intents (single source of truth).
_ACTION_TOOL_MAP: Dict[ActionType, List[str]] = {
    action: list(tools) for action, tools in ACTION_DEFAULT_TOOLS.items()
}

# CONVERSATION skips heavy documents/web but still needs memory
# for cross-session context (e.g. "Remember yesterday?").
_INTENT_SKIP_ALL_TOOLS: set = set()

# VISUALIZE_MOTION and CONVERSATION skip doc/web retrieval but still benefit
# from memory context (user history, preferences).
_INTENT_SKIP_CONTENT_TOOLS: set = set(INTENT_SKIP_CONTENT_TOOLS)


class OrchestratorDecision:
    """Represents a decision made by the orchestrator."""

    def __init__(
        self,
        action: ActionType,
        confidence: float,
        reasoning: str,
        parameters: Optional[Dict[str, Any]] = None
    ):
        self.action = action
        self.confidence = confidence
        self.reasoning = reasoning
        self.parameters = parameters or {}

    def to_dict(self) -> Dict[str, Any]:
        """Convert decision to dictionary."""
        return {
            "action": self.action.value,
            "confidence": self.confidence,
            "reasoning": self.reasoning,
            "parameters": self.parameters,
        }


class OrchestratorAgent:
    """LLM-driven orchestrator that routes queries and executes tools.

    Implements the Agent + Tools pattern:
      - Only this class calls an LLM (for routing decisions).
      - All data retrieval is delegated to injected tool objects.
      - Tools are executed concurrently via ThreadPoolExecutor.
    """

    def __init__(
        self,
        memory_tool: Optional[MemoryTool] = None,
        document_tool: Optional[DocumentRetrievalTool] = None,
        web_search_tool: Optional[WebSearchTool] = None,
        config=None,
        client=None,
        double_rag: Optional[DoubleRAGAgent] = None,
    ):
        """Initialize the orchestrator agent.

        Args:
            memory_tool:      MemoryTool wrapping MemoryManager.
            document_tool:    DocumentRetrievalTool wrapping DocumentStore.
            web_search_tool:  WebSearchTool wrapping WebSearchService.
            config:           Configuration object. Loads default if None.
            client:           Shared GeminiClientWrapper. Creates new if None.
            double_rag:       Optional pre-built DoubleRAGAgent. If None and a
                              ``document_tool`` is provided, a default agent is
                              constructed lazily from the document tool.
        """
        self.config = config or get_config().orchestrator
        self.client = client or GeminiClientWrapper()

        # Injected tools — callers that don't supply tools get None-safe no-ops
        self._memory_tool = memory_tool
        self._document_tool = document_tool
        self._web_search_tool = web_search_tool
        self._query_transformer = QueryTransformer(use_cache=True)

        # Double-RAG agent: lazily wired when a document tool is available so the
        # orchestrator only needs to call self._double_rag.run(...).
        self._double_rag: Optional[DoubleRAGAgent] = double_rag
        if self._double_rag is None and document_tool is not None:
            self._double_rag = DoubleRAGAgent(
                document_tool=document_tool,
                motion_retriever=MotionCandidateRetriever(),
                client=self.client,
            )

        logger.info(
            f"Initialized OrchestratorAgent | model={self.config.model} | "
            f"tools=[memory={'✓' if memory_tool else '✗'}, "
            f"docs={'✓' if document_tool else '✗'}, "
            f"web={'✓' if web_search_tool else '✗'}, "
            f"double_rag={'✓' if self._double_rag else '✗'}]"
        )

    @staticmethod
    def _query_needs_personal_memory(query: str) -> bool:
        """Detect whether memory retrieval is needed for personal-context advice."""
        q = (query or "").strip().lower()
        if not q:
            return False

        personal_markers = (
            "my ", "for me", "for my", "my history", "my record", "my context",
            "my condition", "my injury", "my pain", "remember", "last time",
            "previously", "based on my", "given my", "tôi", "của tôi", "lần trước", "nhớ",
        )
        if any(marker in q for marker in personal_markers):
            return True

        return bool(
            re.search(r"\b(i have|i had|i feel|my|i'm|i am)\b", q)
            and re.search(r"\b(pain|injury|exercise|workout|rehab|stretch|mobility|posture)\b", q)
        )

    def _compute_intent_gates(
        self,
        intent: IntentType,
        query: str,
        requested_actions: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, bool]:
        """Compute canonical tool/action gates based on intent and query."""
        requested_actions = requested_actions or {}

        use_documents = intent in {IntentType.KNOWLEDGE_QUERY, IntentType.EXERCISE_RECOMMENDATION}
        use_memory = (
            intent in {
                IntentType.CONVERSATION,
                IntentType.KNOWLEDGE_QUERY,
                IntentType.EXERCISE_RECOMMENDATION,
            }
            and self._query_needs_personal_memory(query)
        )
        generate_motion = intent == IntentType.VISUALIZE_MOTION
        use_web_search = bool(requested_actions.get("use_web_search", False)) and use_documents

        return {
            "use_memory": use_memory,
            "use_documents": use_documents,
            "use_web_search": use_web_search,
            "generate_motion": generate_motion,
            "needs_rag": use_documents,
        }

    # ──────────────────────────────────────────────────────────────────────
    # Intent classification
    # ──────────────────────────────────────────────────────────────────────

    def classify_intent_and_analyze(
        self,
        query: str,
        conversation_history: Optional[List[Dict[str, str]]] = None,
    ) -> Dict[str, Any]:
        """Single LLM call that simultaneously classifies intent AND produces all
        routing metadata needed to drive the pipeline.

        Replaces the previous two-call approach (classify_intent + analyze_query)
        with a single, combined prompt — saving one full LLM round-trip per request.

        Args:
            query:                User query string.
            conversation_history: Last N turns for context.

        Returns:
            Dict with keys:
              intent          : IntentType enum
              action          : ActionType enum
              confidence      : float
              reasoning       : str
              exercise_name   : str | None
              expanded_query  : str  (enriched query for retrieval)
              needs_rag       : bool
              generate_motion : bool
              motion_type     : str | None
        """
        logger.info(f"classify_intent_and_analyze: '{query[:100]}'")

        # Add last 3 turns of history to the user message for follow-up awareness
        history_snippet = ""
        if conversation_history:
            turns = conversation_history[-3:]
            history_snippet = "\n\nRecent conversation:\n" + "\n".join(
                f"{t['role']}: {t['content']}" for t in turns
            )

        try:
            response = self.client.chat.completions.create(
                model=self.config.model,
                messages=[
                    {"role": "system", "content": _UNIFIED_ANALYSIS_PROMPT},
                    {"role": "user",   "content": query + history_snippet},
                ],
                temperature=0.0,
                max_tokens=1024,   # gemini-2.5-flash is a thinking model; 256 was too small
                response_format={"type": "json_object"},
            )
            raw  = response.choices[0].message.content
            data = json.loads(clean_json_response(raw))

            intent = parse_intent(data.get("intent"), default=IntentType.KNOWLEDGE_QUERY)
            action = parse_action(data.get("action"), default=ActionType.CALL_LLM)

            result = {
                "intent":          intent,
                "action":          action,
                "confidence":      float(data.get("confidence", 0.8)),
                "reasoning":       data.get("reasoning", ""),
                "exercise_name":   data.get("exercise_name"),
                "expanded_query":  data.get("expanded_query") or query,
                "needs_rag":       bool(data.get("needs_rag", True)),
                "generate_motion": bool(data.get("generate_motion", False)),
                "motion_type":     data.get("motion_type"),
            }
            logger.info(
                f"Unified analysis → intent={intent.value} action={action.value} "
                f"needs_rag={result['needs_rag']} generate_motion={result['generate_motion']}"
            )
            return result

        except Exception as exc:
            logger.error(f"classify_intent_and_analyze failed: {exc}")
            return {
                "intent":          IntentType.KNOWLEDGE_QUERY,
                "action":          ActionType.CALL_LLM,
                "confidence":      0.5,
                "reasoning":       "Fallback — unified analysis failed",
                "exercise_name":   None,
                "expanded_query":  query,
                "needs_rag":       True,
                "generate_motion": False,
                "motion_type":     None,
            }

    # ── Legacy single-purpose methods (kept for ui.py / tests) ──────────────

    def classify_intent(self, query: str) -> IntentType:
        """Classify intent only (legacy path). Prefer classify_intent_and_analyze()."""
        logger.debug(f"classify_intent: '{query[:80]}...' ")
        try:
            response = self.client.chat.completions.create(
                model=self.config.model,
                messages=[
                    {"role": "system", "content": _INTENT_CLASSIFIER_PROMPT},
                    {"role": "user",   "content": query},
                ],
                temperature=0.0,
                max_tokens=32,
                response_format={"type": "json_object"},
            )
            raw = response.choices[0].message.content
            data = json.loads(clean_json_response(raw))
            intent_str = data.get("intent", "knowledge_query")
            return IntentType(intent_str)
        except (json.JSONDecodeError, ValueError) as exc:
            logger.warning(f"classify_intent parse error ({exc}), defaulting to knowledge_query")
        except Exception as exc:
            logger.error(f"classify_intent failed: {type(exc).__name__}: {exc}")
        return IntentType.KNOWLEDGE_QUERY

    # ──────────────────────────────────────────────────────────────────────
    # Public API
    # ──────────────────────────────────────────────────────────────────────

    def analyze_query(
        self,
        query: str,
        conversation_history: Optional[List[Dict[str, str]]] = None,
        user_context: Optional[Dict[str, Any]] = None,
    ) -> OrchestratorDecision:
        """Analyze user query with an LLM and return a routing decision.

        Args:
            query:                User's query text.
            conversation_history: Previous conversation turns.
            user_context:         Additional context about the user.

        Returns:
            OrchestratorDecision with action and reasoning.
        """
        logger.info(f"Analyzing query: {query[:100]}...")

        prompt = self._build_analysis_prompt(query, conversation_history, user_context)

        try:
            response = self.client.chat.completions.create(
                model=self.config.model,
                messages=[
                    {"role": "system", "content": self.config.system_prompt},
                    {"role": "user", "content": prompt},
                ],
                temperature=self.config.temperature,
                max_tokens=self.config.max_tokens,
                response_format={"type": "json_object"},
            )

            response_text = response.choices[0].message.content
            cleaned_json = clean_json_response(response_text)
            decision_data = json.loads(cleaned_json)
            decision = self._parse_decision(decision_data)

            logger.info(f"Decision: {decision.action.value} (confidence: {decision.confidence:.2f})")
            return decision

        except json.JSONDecodeError as exc:
            raw = response_text[:200] if "response_text" in dir() else "N/A"
            logger.error(f"Failed to parse orchestrator JSON: {exc}. Raw: {raw}")
        except Exception as exc:
            logger.error(f"Error in query analysis: {type(exc).__name__}: {exc}", exc_info=True)

        # Fallback
        return OrchestratorDecision(
            action=ActionType.CALL_LLM,
            confidence=0.5,
            reasoning="Fallback due to orchestrator error",
            parameters={"use_memory": False},
        )

    def process_query(
        self,
        query: str,
        user_id: str,
        conversation_history: Optional[List[Dict[str, str]]] = None,
        user_context: Optional[Dict[str, Any]] = None,
        trace: Optional[AgentTrace] = None,
    ) -> Dict[str, Any]:
        """Single LLM-call routing → Double-RAG → parallel tool exec → action plan.

        Pipeline (≤ 2 LLM calls inside this method):
            1. classify_intent_and_analyze (1 LLM call)
            2. Query transform (HyDE) — local, no LLM
            3. DoubleRAGAgent.run (0–1 LLM call for constraint extraction)
            4. _run_tools concurrently — I/O only, no LLM

        Args:
            query:                User query text.
            user_id:              Unique user identifier.
            conversation_history: Previous conversation turns.
            user_context:         Extra user context dict (currently unused).
            trace:                Optional ``AgentTrace`` for telemetry.

        Returns:
            Action-plan dict consumed by ``api_server_pkg.app``. Keys:
            ``user_id, query, intent, action, actions, parameters,
            expanded_query, hyde_document, double_rag_meta, needs_rag,
            exercise_name, tool_results``.
        """
        # ── Step 1: Single unified LLM call ────────────────────────────────
        with TraceStage(trace, "orchestrator.classify_and_analyze"):
            analysis = self.classify_intent_and_analyze(query, conversation_history)
        if trace is not None:
            trace.increment_llm_calls(1)
            trace.add_decision(
                "orchestrator.classify_and_analyze",
                intent=analysis["intent"].value,
                action=analysis["action"].value,
                confidence=analysis["confidence"],
                needs_rag=analysis["needs_rag"],
            )

        intent = analysis["intent"]
        action = analysis["action"]

        gates = self._compute_intent_gates(
            intent=intent,
            query=query,
            requested_actions={
                "use_web_search": bool(analysis.get("needs_rag", True)),
            },
        )
        analysis["needs_rag"] = gates["needs_rag"]
        analysis["generate_motion"] = gates["generate_motion"]

        if action in (ActionType.GENERATE_MOTION, ActionType.HYBRID) and not gates["generate_motion"]:
            action = ActionType.CALL_LLM

        # ── Step 1.5: Query Transformation (HyDE) ──────────────────────────
        expanded_query = analysis["expanded_query"]
        hyde_document = query
        if analysis["needs_rag"] or action == ActionType.GENERATE_MOTION:
            with TraceStage(trace, "orchestrator.query_transform"):
                transform_res = self._query_transformer.transform_query(query)
            expanded_query = transform_res.get("expanded_query", query)
            hyde_document = transform_res.get("hyde_document", query)

        # ── Step 2: Double-RAG (clinical → constraints → motion match) ────
        double_rag_constraints = ""
        clinical_docs: List[Dict[str, Any]] = []
        if self._double_rag is not None and (
            action == ActionType.GENERATE_MOTION or analysis["needs_rag"]
        ):
            with TraceStage(trace, "orchestrator.double_rag"):
                dr_result = self._double_rag.run(
                    expanded_query=expanded_query,
                    hyde_document=hyde_document,
                    user_id=user_id,
                    trace=trace,
                )
            clinical_docs = dr_result.clinical_docs
            double_rag_constraints = dr_result.constraints
            if dr_result.refined_hyde_document:
                hyde_document = dr_result.refined_hyde_document

        # ── Step 3: Select and run remaining tools concurrently ───────────
        decision = OrchestratorDecision(
            action=action,
            confidence=analysis["confidence"],
            reasoning=analysis["reasoning"],
            parameters={
                "use_memory":      gates["use_memory"],
                "use_documents":   gates["use_documents"],
                "use_web_search":  gates["use_web_search"],
                "generate_motion": gates["generate_motion"],
                "motion_type":     analysis["motion_type"],
            },
        )
        selected_tools = self._select_tools(decision, intent)
        # Drop document_tool from concurrent run if Double-RAG already retrieved.
        if "documents" in selected_tools and clinical_docs:
            selected_tools.remove("documents")

        with TraceStage(trace, "orchestrator.run_tools"):
            tool_results = self._run_tools(selected_tools, expanded_query, user_id)
        if trace is not None:
            for _name in tool_results.keys():
                trace.add_decision("orchestrator.tool_used", tool=_name)

        # Merge clinical documents back into tool_results for the RAG pipeline.
        if clinical_docs:
            tool_results["documents"] = clinical_docs

        # ── Step 4: Assemble action plan ──────────────────────────────────
        action_plan = {
            "user_id":         user_id,
            "query":           query,
            "intent":          intent.value if hasattr(intent, "value") else str(intent),
            "action":          action.value if hasattr(action, "value") else str(action),
            "decision":        decision.to_dict() if hasattr(decision, "to_dict") else {},
            "actions": {
                "retrieve_memory": "memory" in selected_tools,
                "call_llm":        action in (ActionType.CALL_LLM, ActionType.HYBRID, ActionType.RETRIEVE_MEMORY),
                "generate_motion": gates["generate_motion"],
                "use_memory":     "memory" in selected_tools,
                "use_documents":  gates["use_documents"],
                "use_web_search": "web_search" in selected_tools,
            },
            "parameters":      getattr(decision, "parameters", {}),
            "expanded_query":  expanded_query,
            "hyde_document":   hyde_document,
            "double_rag_meta": {
                "constraints": double_rag_constraints,
            },
            "needs_rag":       gates["needs_rag"],
            "exercise_name":   analysis.get("exercise_name"),
            "tool_results":    tool_results,
        }

        if trace is not None:
            trace.finish(intent=action_plan["intent"], action=action_plan["action"])

        logger.info(
            f"Action plan ready | intent={intent.value} | "
            f"action={action.value} | needs_rag={analysis['needs_rag']} | "
            f"tools_run={list(tool_results.keys())}"
        )
        return action_plan

    def _select_tools(
        self,
        decision: OrchestratorDecision,
        intent: Optional[IntentType] = None,
    ) -> List[str]:
        """Return list of tool names to execute.

        Tool selection is based on the routing decision AND optionally the intent:
        - ``CONVERSATION`` and ``VISUALIZE_MOTION`` intents skip document/web tools
          because they don't need heavy retrieval.

        Args:
            decision: The routing decision from analyze_query().
            intent:   Optional intent from classify_intent().

        Returns:
            List of tool name strings (subset of: "memory", "documents", "web_search").
        """
        candidates = _ACTION_TOOL_MAP.get(decision.action, [])

        if intent in _INTENT_SKIP_ALL_TOOLS:
            # Conversation/greeetings: skip ALL tools including memory.
            # No ChromaDB lookup needed for a simple "Hello" — saves ~200ms.
            logger.debug(
                f"_select_tools: intent={intent.value if intent else 'None'} → "
                "skipping ALL tools (fast-path conversation)"
            )
            return []
        elif intent in _INTENT_SKIP_CONTENT_TOOLS:
            # visualize_motion: skip doc/web search but keep memory context.
            candidates = [t for t in candidates if t == "memory"]

        params = decision.parameters or {}
        if not bool(params.get("use_memory", False)):
            candidates = [t for t in candidates if t != "memory"]
        if not bool(params.get("use_documents", True)):
            candidates = [t for t in candidates if t != "documents"]
        if not bool(params.get("use_web_search", False)):
            candidates = [t for t in candidates if t != "web_search"]

        available = {
            "memory":     self._memory_tool is not None,
            "documents":  self._document_tool is not None,
            "web_search": self._web_search_tool is not None,
        }
        selected = [t for t in candidates if available.get(t, False)]
        logger.debug(
            f"_select_tools: action={decision.action.value} "
            f"intent={intent.value if intent else 'None'} → {selected}"
        )
        return selected

    # ──────────────────────────────────────────────────────────────────────
    # Concurrent tool execution
    # ──────────────────────────────────────────────────────────────────────

    def _run_tools(
        self,
        selected_tools: List[str],
        query: str,
        user_id: str,
    ) -> Dict[str, Any]:
        """Execute selected tools concurrently using ThreadPoolExecutor.

        Only tools in `selected_tools` are started. Results are collected
        as futures complete; any individual tool failure is logged and
        excluded from the result dict (does not abort the others).

        Args:
            selected_tools: List of tool names to run.
            query:          The current user query.
            user_id:        The current user's identifier.

        Returns:
            Dict mapping tool name → result. Keys present only for tools
            that succeeded: memory → List[Dict], documents → List[Dict],
            web_search → str.
        """
        if not selected_tools:
            return {}

        # Use the process-wide bounded executor instead of spawning one per request.
        executor = shared_tool_executor()
        futures: Dict = {}

        if "memory" in selected_tools and self._memory_tool is not None:
            futures[executor.submit(
                self._memory_tool.retrieve_memory,
                user_id=user_id,
                query=query,
            )] = "memory"

        if "documents" in selected_tools and self._document_tool is not None:
            futures[executor.submit(
                self._document_tool.search_documents,
                query=query,
                user_id=user_id,
            )] = "documents"

        if "web_search" in selected_tools and self._web_search_tool is not None:
            futures[executor.submit(
                self._web_search_tool.search_web,
                query=query,
            )] = "web_search"

        results: Dict[str, Any] = {}
        for future in as_completed(futures):
            tool_name = futures[future]
            try:
                results[tool_name] = future.result()
                logger.debug(f"Tool '{tool_name}' completed successfully")
            except Exception as exc:
                logger.error(f"Tool '{tool_name}' failed: {exc}")
                # Exclude failed tool from results (don't propagate exception)

        return results

    # ──────────────────────────────────────────────────────────────────────
    # Decision helpers (unchanged from previous version)
    # ──────────────────────────────────────────────────────────────────────

    def should_retrieve_memory(self, decision: OrchestratorDecision) -> bool:
        return (
            decision.action == ActionType.RETRIEVE_MEMORY
            or decision.action == ActionType.HYBRID
            or decision.parameters.get("use_memory", False)
        )

    def should_call_llm(self, decision: OrchestratorDecision) -> bool:
        return (
            decision.action == ActionType.CALL_LLM
            or decision.action == ActionType.HYBRID
            or decision.action == ActionType.CLARIFY
        )

    def should_generate_motion(self, decision: OrchestratorDecision) -> bool:
        return (
            decision.action == ActionType.GENERATE_MOTION
            or decision.action == ActionType.HYBRID
            or decision.parameters.get("generate_motion", False)
        )

    # ──────────────────────────────────────────────────────────────────────
    # Private helpers
    # ──────────────────────────────────────────────────────────────────────

    def _build_analysis_prompt(
        self,
        query: str,
        conversation_history: Optional[List[Dict[str, str]]],
        user_context: Optional[Dict[str, Any]],
    ) -> str:
        prompt_parts = [
            "Analyze the following user query and decide the best action.",
            f"\nUser Query: {query}",
        ]

        if conversation_history:
            history_text = "\n".join(
                f"{turn['role']}: {turn['content']}"
                for turn in conversation_history[-5:]
            )
            prompt_parts.append(f"\nRecent Conversation:\n{history_text}")

        if user_context:
            prompt_parts.append(f"\nUser Context:\n{json.dumps(user_context, indent=2)}")

        prompt_parts.append(ORCHESTRATOR_PROMPTS["decision_format"])
        return "\n".join(prompt_parts)

    def _parse_decision(self, decision_data: Dict[str, Any]) -> OrchestratorDecision:
        action = parse_action(decision_data.get("action"), default=ActionType.CALL_LLM)
        return OrchestratorDecision(
            action=action,
            confidence=float(decision_data.get("confidence", 0.7)),
            reasoning=decision_data.get("reasoning", "No reasoning provided"),
            parameters=decision_data.get("parameters", {}),
        )


if __name__ == "__main__":
    orchestrator = OrchestratorAgent()  # no tools injected — routing only
    result = orchestrator.process_query(
        query="I have neck pain from sitting at my desk all day",
        user_id="test_user_123",
    )
    print(json.dumps(result, indent=2, default=str))
