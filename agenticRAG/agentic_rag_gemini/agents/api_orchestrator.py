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
from concurrent.futures import ThreadPoolExecutor, as_completed
from enum import Enum
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


class ActionType(Enum):
    """Types of actions the orchestrator can take."""
    RETRIEVE_MEMORY = "retrieve_memory"
    CALL_LLM = "call_llm"
    GENERATE_MOTION = "generate_motion"
    HYBRID = "hybrid"
    CLARIFY = "clarify"


class IntentType(Enum):
    """High-level user intent classification.

    Values
    ------
    CONVERSATION
        General chat or follow-up (e.g. "Thanks!", "Tell me more").
    KNOWLEDGE_QUERY
        User asks for an explanation or factual answer
        (e.g. "What muscles does chin tuck work?").
    EXERCISE_RECOMMENDATION
        User asks for a list or set of exercises
        (e.g. "Give me some stretches for neck pain").
    VISUALIZE_MOTION
        User wants to see a specific exercise animated
        (e.g. "Visualize chin tuck", "Show me how to do a squat").
    """
    CONVERSATION           = "conversation"
    KNOWLEDGE_QUERY        = "knowledge_query"
    EXERCISE_RECOMMENDATION = "exercise_recommendation"
    VISUALIZE_MOTION       = "visualize_motion"


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


# Maps each ActionType to the set of tool names that should be invoked.
# NOTE: tool selection is also gated by intent via _select_tools().
_ACTION_TOOL_MAP: Dict[ActionType, List[str]] = {
    ActionType.RETRIEVE_MEMORY: ["memory"],
    ActionType.CALL_LLM:        ["memory", "documents", "web_search"],
    ActionType.GENERATE_MOTION: [],
    ActionType.HYBRID:          ["memory", "documents", "web_search"],
    ActionType.CLARIFY:         [],
}

# CONVERSATION needs NO tool retrieval — not even memory.
# A greeting/follow-up has no benefit from a ChromaDB vector lookup.
_INTENT_SKIP_ALL_TOOLS: set = {
    IntentType.CONVERSATION,
}

# VISUALIZE_MOTION skips doc/web retrieval but still benefits from memory
# context (user history, preferences).
_INTENT_SKIP_CONTENT_TOOLS: set = {
    IntentType.VISUALIZE_MOTION,
}


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
    ):
        """Initialize the orchestrator agent.

        Args:
            memory_tool:      MemoryTool wrapping MemoryManager.
            document_tool:    DocumentRetrievalTool wrapping DocumentStore.
            web_search_tool:  WebSearchTool wrapping WebSearchService.
            config:           Configuration object. Loads default if None.
            client:           Shared GeminiClientWrapper. Creates new if None.
        """
        self.config = config or get_config().orchestrator
        self.client = client or GeminiClientWrapper()

        # Injected tools — callers that don't supply tools get None-safe no-ops
        self._memory_tool = memory_tool
        self._document_tool = document_tool
        self._web_search_tool = web_search_tool
        self._motion_retriever = MotionCandidateRetriever()
        self._query_transformer = QueryTransformer(use_cache=True)

        logger.info(
            f"Initialized OrchestratorAgent | model={self.config.model} | "
            f"tools=[memory={'✓' if memory_tool else '✗'}, "
            f"docs={'✓' if document_tool else '✗'}, "
            f"web={'✓' if web_search_tool else '✗'}]"
        )

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

            intent_str = data.get("intent", "knowledge_query")
            try:
                intent = IntentType(intent_str)
            except ValueError:
                intent = IntentType.KNOWLEDGE_QUERY

            action_str = data.get("action", "call_llm")
            action_map = {
                "call_llm":        ActionType.CALL_LLM,
                "generate_motion": ActionType.GENERATE_MOTION,
                "hybrid":          ActionType.HYBRID,
                "clarify":         ActionType.CLARIFY,
                "retrieve_memory": ActionType.RETRIEVE_MEMORY,
            }
            action = action_map.get(action_str, ActionType.CALL_LLM)

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
    ) -> Dict[str, Any]:
        """Main entry point: single unified LLM call → parallel tool execution → action plan.

        New pipeline (2 LLM calls max, down from 7):
        1. classify_intent_and_analyze()  — single LLM call (was 2)
        2. _run_tools() concurrently      — I/O only, no LLM
        3. RAGPipeline (caller)           — 1 LLM call for answer generation
        Optional voice_prompt             — no LLM (keyword matching)

        Args:
            query:                User query text.
            user_id:              Unique user identifier.
            conversation_history: Previous conversation turns.
            user_context:         Extra user context dict.

        Returns:
            Dict with keys:
              user_id, query, intent, decision, actions, parameters,
              expanded_query, needs_rag, tool_results
        """
        # ── Step 1: Single unified LLM call ─────────────────────────────────
        analysis = self.classify_intent_and_analyze(query, conversation_history)
        intent   = analysis["intent"]
        action   = analysis["action"]

        # ── Step 1.5: Query Transformation (Double-RAG Engine) ──────────────
        expanded_query = analysis["expanded_query"]
        hyde_document = query
        if analysis["needs_rag"] or action == ActionType.GENERATE_MOTION:
            transform_res = self._query_transformer.transform_query(query)
            expanded_query = transform_res.get("expanded_query", query)
            hyde_document  = transform_res.get("hyde_document", query)

        # ── Step 2: Multi-Stage Orchestration (Double-RAG) ───────────────────
        # If we need clinical knowledge and motion execution, execute Double-RAG
        double_rag_results = {}
        if self._document_tool and (action == ActionType.GENERATE_MOTION or analysis["needs_rag"]):
            # 1. Clinical Dispatch
            clinical_docs = self._document_tool.search_documents(expanded_query)
            
            # 2. Constraint Extraction
            constraints = self._extract_constraints(clinical_docs)
            logger.info(f"Extracted Constraints: {constraints}")
            
            # 3. Conditioned Motion Search
            conditioned_query = f"{hyde_document} Constraints: {constraints}"
            motion_candidates = self._motion_retriever.retrieve_top_k(conditioned_query, k=1)
            
            if motion_candidates:
                hyde_document = motion_candidates[0].text_description
            
            double_rag_results = {
                "clinical_docs": clinical_docs,
                "constraints": constraints,
                "motion_candidates": motion_candidates
            }

        # ── Step 3: Select and run remaining tools concurrently ──────────────
        decision = OrchestratorDecision(
            action=action,
            confidence=analysis["confidence"],
            reasoning=analysis["reasoning"],
            parameters={
                "use_memory":      True,        # always try memory
                "use_documents":   analysis["needs_rag"],
                "generate_motion": analysis["generate_motion"],
                "motion_type":     analysis["motion_type"],
            },
        )
        selected_tools = self._select_tools(decision, intent)
        # We drop document_tool from concurrent tool run if we already did Double-RAG
        if "documents" in selected_tools and double_rag_results:
            selected_tools.remove("documents")
            
        tool_results = self._run_tools(selected_tools, expanded_query, user_id)
        
        # Merge clinical documents back into tool_results for RAG pipeline
        if double_rag_results.get("clinical_docs"):
            tool_results["documents"] = double_rag_results["clinical_docs"]

        # ── Step 4: Assemble action plan ─────────────────────────────────────
        action_plan = {
            "user_id":         user_id,
            "query":           query,
            "intent":          intent.value if hasattr(intent, "value") else str(intent),
            "action":          action.value if hasattr(action, "value") else str(action),
            "decision":        decision.to_dict() if hasattr(decision, 'to_dict') else {},
            "actions": {
                "retrieve_memory": "memory" in selected_tools,
                "call_llm":        action in (ActionType.CALL_LLM, ActionType.HYBRID, ActionType.RETRIEVE_MEMORY),
                "generate_motion": analysis.get("generate_motion", False) or intent == IntentType.VISUALIZE_MOTION,
            },
            "parameters":      getattr(decision, "parameters", {}),
            "expanded_query":  expanded_query,
            "hyde_document":   hyde_document,  # Expose to api_server for DART mapping
            "double_rag_meta": {
                "constraints": double_rag_results.get("constraints", "")
            },
            "needs_rag":       analysis.get("needs_rag", True),
            "exercise_name":   analysis.get("exercise_name"),
            "tool_results":    tool_results,
        }

        logger.info(
            f"Action plan ready | intent={intent.value} | "
            f"action={action.value} | needs_rag={analysis['needs_rag']} | "
            f"tools_run={list(tool_results.keys())}"
        )
        return action_plan

    def _extract_constraints(self, clinical_docs: List[Dict[str, Any]]) -> str:
        """Extract safety constraints and targeted muscles from clinical documents."""
        if not clinical_docs:
            return "No specific constraints."
            
        clincal_text = "\\n".join([doc.get("document", "") for doc in clinical_docs[:3]])
        
        prompt = f'''You are a clinical expert. Extract the key physical constraints, safety warnings, and targeted muscles from the provided clinical text.
Ensure the output is extremely brief, no more than 2 sentences. Focus ONLY on biomechanical rules and limitations.

Clinical Text:
{clincal_text}'''

        try:
            response = self.client.chat.completions.create(
                model=self.config.model,
                messages=[{"role": "user", "content": prompt}],
                temperature=0.1,
                max_tokens=150
            )
            return response.choices[0].message.content.strip()
        except Exception as e:
            logger.error(f"Failed to extract constraints: {e}")
            return "General safe range of motion."



    # ──────────────────────────────────────────────────────────────────────
    # Tool selection
    # ──────────────────────────────────────────────────────────────────────

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

        # Build future → tool-name mapping
        futures: Dict = {}
        with ThreadPoolExecutor(max_workers=len(selected_tools)) as executor:
            if "memory" in selected_tools:
                futures[executor.submit(
                    self._memory_tool.retrieve_memory,
                    user_id=user_id,
                    query=query,
                )] = "memory"

            if "documents" in selected_tools:
                futures[executor.submit(
                    self._document_tool.search_documents,
                    query=query,
                    user_id=user_id,
                )] = "documents"

            if "web_search" in selected_tools:
                futures[executor.submit(
                    self._web_search_tool.search_web,
                    query=query,
                )] = "web_search"

            # Collect results as they complete
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
        action_map = {
            "retrieve_memory": ActionType.RETRIEVE_MEMORY,
            "call_llm":        ActionType.CALL_LLM,
            "generate_motion": ActionType.GENERATE_MOTION,
            "hybrid":          ActionType.HYBRID,
            "clarify":         ActionType.CLARIFY,
        }
        action = action_map.get(
            decision_data.get("action", "call_llm"),
            ActionType.CALL_LLM,
        )
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
