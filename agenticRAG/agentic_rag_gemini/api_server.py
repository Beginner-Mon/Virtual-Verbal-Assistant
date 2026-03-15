#!/usr/bin/env python3
"""REST API server for AgenticRAG.

This module exposes the AgenticRAG system as a FastAPI REST service,
allowing external systems to query the RAG pipeline and receive structured
responses including motion and voice prompts.
"""

import logging
from typing import Optional, List, Dict, Any
from contextlib import asynccontextmanager

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from models import VoicePrompt, MotionPrompt  # shared models — single source of truth
import uvicorn

from agents.api_orchestrator import OrchestratorAgent
from agents.local_orchestrator import LocalOrchestrator
from agents.tools import MemoryTool, DocumentRetrievalTool, WebSearchTool
from agents.tools.motion_generation_tool import MotionGenerationTool
from memory.memory_manager import MemoryManager
from memory.document_store import DocumentStore
from memory.vector_store import VectorStore
from memory.embedding_service import EmbeddingService
from retrieval.rag_pipeline import RAGPipeline
from config import get_config
from utils.logger import get_logger
from utils.web_search import get_web_search_service
from utils.exercise_detector import get_exercise_detector
from utils.gemini_client import GeminiClientWrapper
from utils.prompt_templates import LLM_PROMPTS
from agents.response_templates import ResponseTemplateGenerator

# Initialize logger
logger = get_logger(__name__)

# ===========================
# Request/Response Models
# ===========================


class ConversationTurn(BaseModel):
    """Single conversation turn."""

    role: str = Field(..., description="Role: 'user' or 'assistant'")
    content: str = Field(..., description="Message content")


class QueryRequest(BaseModel):
    """Request to process a query."""

    query: str = Field(..., description="User's query")
    user_id: str = Field(..., description="User identifier")
    conversation_history: Optional[List[ConversationTurn]] = Field(
        None, description="Previous conversation turns"
    )


class OrchestratorDecision(BaseModel):
    """Orchestrator decision details."""

    action: str = Field(..., description="Action type: retrieve_memory, call_llm, generate_motion, hybrid, clarify")
    intent: Optional[str] = Field(None, description="Intent detected: conversation, knowledge_query, exercise_recommendation, visualize_motion, etc")
    confidence: float = Field(..., description="Confidence score 0.0-1.0")
    reasoning: str = Field(..., description="Reasoning for decision")
    parameters: Dict[str, Any] = Field(default_factory=dict, description="Action parameters")
    tools_selected: List[str] = Field(
        default_factory=list,
        description="Tools selected by orchestrator: memory, documents, web_search, motion"
    )
    tools_executed: List[str] = Field(
        default_factory=list,
        description="Tools actually executed (may differ from selected if skipped)"
    )
    tools_failed: List[str] = Field(
        default_factory=list,
        description="Tools that failed during execution"
    )
    execution_time_ms: float = Field(
        0.0,
        description="Total orchestrator execution time in milliseconds"
    )
    debug_info: Dict[str, Any] = Field(
        default_factory=dict,
        description="Debug info: memory_results count, documents_results count, web_results count, llm_calls, errors"
    )



class MotionMetadata(BaseModel):
    """Motion generation result from DART/MotionGenerationTool."""

    motion_file: str = Field(..., description="Generated .npz motion file name")
    frames:      int = Field(..., description="Total frame count")
    fps:         int = Field(..., description="Frames per second")


class QueryResponse(BaseModel):
    """Response from query processing."""

    query: str = Field(..., description="Original query")
    user_id: str = Field(..., description="User ID")
    text_answer: str = Field(..., description="Generated text answer")
    exercises: List[Dict[str, str]] = Field(
        default_factory=list,
        description="Recommended exercises (name only). Empty list when none recommended."
    )
    exercise_motion_prompt: Optional[str] = Field(
        None,
        description=(
            "Exercise name to generate motion for, selected from the exercises list. "
            "Set only when the query implies visualization (show/demonstrate/animate). "
            "Intended for MotionGenerationTool once DART is integrated."
        ),
    )
    motion: Optional[MotionMetadata] = Field(
        None, description="Motion generation result from DART (set when exercise_motion_prompt is not null)"
    )
    orchestrator_decision: OrchestratorDecision = Field(
        ..., description="Orchestrator decision and reasoning"
    )
    motion_prompt: Optional[MotionPrompt] = Field(
        None, description="Motion generation prompt if applicable"
    )
    voice_prompt: Optional[VoicePrompt] = Field(
        None, description="Voice synthesis prompt if applicable"
    )
    metadata: Dict[str, Any] = Field(default_factory=dict, description="Additional metadata")
    
    # NEW DEBUG FIELDS
    pipeline_trace: Dict[str, Any] = Field(
        default_factory=dict,
        description=(
            "Execution trace: shows which path was taken and what was executed. "
            "Keys: orchestrator_type, intent, tools_invoked, memory_results_count, "
            "documents_results_count, web_results_count, llm_calls_count, rag_used, errors"
        )
    )
    performance: Dict[str, float] = Field(
        default_factory=dict,
        description="Timing info: orchestrator_ms, tools_ms, rag_ms, motion_ms, total_ms"
    )


# ===========================
# API Initialization
# ===========================


class AgenticRAGAPI:
    """AgenticRAG API wrapper."""

    def __init__(self):
        """Initialize all components and wire tools into the orchestrator."""
        logger.info("Initializing AgenticRAG API...")

        try:
            # Shared infrastructure
            from memory.vector_store import VectorStore
            from memory.embedding_service import EmbeddingService

            vector_store = VectorStore()
            embedding_service = EmbeddingService()

            # Core services
            memory_manager = MemoryManager(
                vector_store=vector_store,
                embedding_service=embedding_service,
            )
            document_store = DocumentStore(
                vector_store=vector_store,
                embedding_service=embedding_service,
            )
            web_service = get_web_search_service()

            # Tool layer — thin wrappers with clean single-method interfaces
            memory_tool     = MemoryTool(memory_manager)
            document_tool   = DocumentRetrievalTool(document_store)
            web_search_tool = WebSearchTool(web_service)

            # Save tool references so that _get_orchestrator_decision() can
            # build a temporary OrchestratorAgent when LocalOrchestrator times out.
            self._memory_tool_ref     = memory_tool
            self._document_tool_ref   = document_tool
            self._web_search_tool_ref = web_search_tool

            # MotionGenerationTool — calls DART server; instantiated here so it
            # can be called later in process_query() without repeated construction.
            self.motion_tool = MotionGenerationTool()  # defaults: localhost:5001, 30s timeout

            # Orchestrator — try local first, fallback to API
            logger.info("Attempting to initialize LocalOrchestrator...")
            try:
                self.orchestrator = LocalOrchestrator()
                logger.info("SUCCESS: Using local Qwen2.5-3B orchestrator")
                logger.info(f"Local orchestrator type: {type(self.orchestrator)}")
                logger.info(f"Local orchestrator model: {getattr(self.orchestrator, 'model_name', 'Unknown')}")
            except Exception as e:
                logger.error(f"FAILED: Local orchestrator initialization failed: {e}")
                logger.warning(f"Falling back to API orchestrator due to: {type(e).__name__}")
                self.orchestrator = OrchestratorAgent(
                    memory_tool=memory_tool,
                    document_tool=document_tool,
                    web_search_tool=web_search_tool,
                )
                logger.info("FALLBACK: Using API orchestrator")
                logger.info(f"API orchestrator type: {type(self.orchestrator)}")

            # Warm up the local model so the first real request is never cold.
            if isinstance(self.orchestrator, LocalOrchestrator):
                self.orchestrator.warmup()

            # RAGPipeline — kept for LLM response generation
            self.rag_pipeline = RAGPipeline(memory_manager=memory_manager)
            self.template_generator = ResponseTemplateGenerator()

            # Shared LLM client for the conversation fast-path
            self._conv_client = GeminiClientWrapper()
            
            # Exercise Detector for hybrid entity extraction
            self.exercise_detector = get_exercise_detector()
            logger.info(f"ExerciseDetector initialized with {self.exercise_detector.get_exercise_count()} exercises")

            logger.info("AgenticRAG API initialized successfully")
        except Exception as e:
            logger.error(f"Failed to initialize AgenticRAG API: {e}")
            raise

    # Maps LocalOrchestrator-specific intent strings to the canonical set
    # used in process_query() branching.  Without this, 'greeting' falls
    # through to the else-branch → full RAG pipeline (2 extra LLM calls).
    _INTENT_CANONICAL_MAP: dict = {
        "greeting":                 "conversation",
        "followup_question":        "conversation",
        "resume_conversation":      "conversation",
        "ask_exercise_info":        "knowledge_query",
        "general_fitness_question": "knowledge_query",
        "visualize_motion":         "visualize_motion",
        "unknown":                  "knowledge_query",
        # Pass-through (already canonical):
        "conversation":             "conversation",
        "knowledge_query":          "knowledge_query",
        "exercise_recommendation":  "exercise_recommendation",
    }

    def _get_orchestrator_decision(
        self,
        query: str,
        user_id: str,
        conversation_history: Optional[List[Dict[str, str]]] = None,
    ) -> Dict[str, Any]:
        """Get decision from orchestrator (local or API).
        
        Returns standardized decision format for both local and API orchestrators.
        """
        logger.info(f"_get_orchestrator_decision called with orchestrator type: {type(self.orchestrator).__name__}")
        
        # Step 1: Detect exercise using ExerciseDetector (Hybrid Entity Extraction)
        logger.info(f"Detecting exercise in query: {query}")
        detected_exercise = self.exercise_detector.detect_exercise(query)
        
        if detected_exercise:
            logger.info(f"Exercise detected: '{detected_exercise}'")
        else:
            logger.info("No exercise detected in query")
        
        if isinstance(self.orchestrator, LocalOrchestrator):
            logger.info("USING LOCAL ORCHESTRATOR path")
            logger.info(f"About to call LocalOrchestrator.analyze_query with: {query}, detected_exercise: {detected_exercise}")
            decision = self.orchestrator.analyze_query(query, conversation_history, detected_exercise)
            logger.info(f"Local orchestrator returned: {decision}")

            # ── Detect fallback response (Ollama timed out or JSON invalid) ──
            # The fallback always has confidence=0.1 and intent="unknown".
            # When this happens, LocalOrchestrator is not useful for this request;
            # immediately fall back to the Gemini API orchestrator so the user
            # gets a fast, correct routing decision instead of waiting 30s for
            # a wrong "knowledge_query" classification.
            _is_fallback = (
                decision.get("confidence", 1.0) <= 0.1
                and decision.get("intent") == "unknown"
            )
            if _is_fallback:
                logger.warning(
                    "[LocalOrchestrator] Fallback response detected (confidence=0.1, intent=unknown). "
                    "Ollama likely timed out — routing this request through Gemini API orchestrator."
                )
                # Build a temporary API orchestrator and use it for this one request.
                # (We keep self.orchestrator as LocalOrchestrator for future requests
                # so that warm subsequent calls benefit from the local model.)
                _api_orch = OrchestratorAgent(
                    memory_tool=getattr(self, "_memory_tool_ref", None),
                    document_tool=getattr(self, "_document_tool_ref", None),
                    web_search_tool=getattr(self, "_web_search_tool_ref", None),
                )
                return _api_orch.process_query(query, user_id, conversation_history)

            # Normalize LocalOrchestrator-specific intent strings so that
            # process_query() branching works correctly for greetings/follow-ups.
            raw_intent = decision.get("intent", "unknown")
            canonical_intent = self._INTENT_CANONICAL_MAP.get(raw_intent, "knowledge_query")
            if canonical_intent != raw_intent:
                logger.info(
                    f"Intent normalized: '{raw_intent}' → '{canonical_intent}'"
                )
            decision["intent"] = canonical_intent
            
            # Convert to format expected by existing code
            result = {
                "intent": decision["intent"],
                "actions": {
                    "generate_motion": decision["needs_motion"],
                    "use_memory": "memory_agent" in decision["agents"],
                    "use_documents": decision["needs_retrieval"],
                    "use_web_search": decision["needs_web_search"]
                },
                "tool_results": {},  # Will be populated later
                "expanded_query": query,
                "needs_rag": decision["needs_retrieval"],
                "exercise": decision["exercise"],
                "agents": decision["agents"],
                "confidence": decision["confidence"]
            }
            logger.info(f"Converted result for API compatibility: {result}")
            return result
        else:
            logger.info("USING API ORCHESTRATOR path")
            # API orchestrator returns the existing format
            # Note: API orchestrator doesn't support detected_exercise parameter yet
            return self.orchestrator.process_query(query, user_id, conversation_history)

    def process_query(
        self,
        query: str,
        user_id: str,
        conversation_history: Optional[List[Dict[str, str]]] = None,
    ) -> QueryResponse:
        """Process a user query through the intent-branched pipeline.

        Pipeline (down from 7 LLM calls to 2):
        1. classify_intent_and_analyze()  — 1 LLM call (orchestrator)
        2. Parallel tool retrieval        — no LLM
        3. Branch by intent:
             conversation       → lightweight LLM with memory only (no RAG)
             visualize_motion   → skip RAG, return motion prompt
             knowledge_query /
             exercise_recommendation → RAGPipeline with expanded_query,
                                       skip_web_search=True, skip_reflection=True
        4. voice_prompt from text_answer  — no LLM (keyword matching only)
        """
        import time
        start_time = time.time()
        
        logger.info(f"Processing query for user {user_id}: {query[:100]}...")

        # Initialize execution trace
        trace = {
            "orchestrator_type": None,
            "intent": None,
            "tools_selected": [],
            "tools_executed": [],
            "tools_failed": [],
            "memory_results_count": 0,
            "documents_results_count": 0,
            "web_results_count": 0,
            "llm_calls_count": 0,
            "rag_used": False,
            "path_taken": None,
            "errors": [],
        }
        perf = {
            "orchestrator_ms": 0.0,
            "tools_ms": 0.0,
            "rag_ms": 0.0,
            "motion_ms": 0.0,
            "total_ms": 0.0,
        }

        try:
            # ── Step 1: Get orchestrator decision (local or API) ─────────────
            orch_start = time.time()
            logger.info(f"Using orchestrator: {type(self.orchestrator).__name__}")
            trace["orchestrator_type"] = type(self.orchestrator).__name__
            
            action_plan = self._get_orchestrator_decision(
                query=query,
                user_id=user_id,
                conversation_history=conversation_history,
            )
            orch_time = time.time() - orch_start
            perf["orchestrator_ms"] = orch_time * 1000
            logger.info(f"Orchestrator decision type: {type(action_plan)}")

            intent         = action_plan["intent"]           # str value
            generate_motion = action_plan["actions"]["generate_motion"]
            tool_results    = action_plan.get("tool_results", {})
            expanded_query  = action_plan.get("expanded_query") or query
            needs_rag       = action_plan.get("needs_rag", True)
            
            trace["intent"] = intent
            trace["tools_selected"] = action_plan.get("agents", [])
            trace["tools_executed"] = action_plan.get("agents", [])
            
            # Count tool results
            trace["memory_results_count"] = len(tool_results.get("memory", []))
            trace["documents_results_count"] = len(tool_results.get("documents", []))
            trace["web_results_count"] = len(tool_results.get("web_search", []))

            logger.info(
                f"Orchestrator: intent={intent} needs_rag={needs_rag} "
                f"generate_motion={generate_motion} | "
                f"memory={trace['memory_results_count']}, "
                f"docs={trace['documents_results_count']}, "
                f"web={trace['web_results_count']}"
            )

            # ── Step 2: Branch by intent ─────────────────────────────────────
            text_answer = ""
            exercises   = []   # populated only through the structured RAG path
            exercise_motion_prompt: Optional[str] = None  # set below if query implies visualization

            if intent == "conversation":
                # Fast path: one LLM call with memory context only, no RAG.
                logger.info("Intent=conversation → lightweight LLM (no RAG)")
                trace["path_taken"] = "conversation_fast_path"
                trace["llm_calls_count"] = 1
                
                memory_ctx = tool_results.get("memory") or []
                mem_text   = "\n".join(
                    str(m.get("document", m)) for m in memory_ctx
                ) if memory_ctx else ""
                conv_prompt = (
                    f"Memory context:\n{mem_text}\n\n" if mem_text else ""
                ) + f"User: {query}"
                _resp = self._conv_client.chat.completions.create(
                    model=self.rag_pipeline.llm_config.model,
                    messages=[
                        {"role": "system", "content": LLM_PROMPTS["system"]},
                        {"role": "user",   "content": conv_prompt},
                    ],
                    temperature=0.7,
                    max_tokens=512,
                )
                text_answer = _resp.choices[0].message.content.strip()

            elif intent == "visualize_motion":
                # Generate a text description + request motion animation from DART.
                # The text answer is always shown; DART animation is a bonus.
                logger.info("Intent=visualize_motion → LLM description + motion prompt")
                trace["path_taken"] = "visualize_motion_path"
                trace["llm_calls_count"] = 1
                
                exercise = action_plan.get("exercise_name") or query
                
                # Normalize the extraction for DART (e.g., "chin tuck exercise" -> "chin tuck")
                exercise_motion_prompt = exercise.lower().replace(" exercise", "").strip()
                
                memory_ctx = tool_results.get("memory") or []
                mem_text   = "\n".join(
                    str(m.get("document", m)) for m in memory_ctx
                ) if memory_ctx else ""
                motion_desc_prompt = (
                    f"{'Memory context:\n' + mem_text + chr(10) + chr(10) if mem_text else ''}"
                    f"The user wants to see how to perform: {exercise}.\n"
                    f"Provide a clear, concise step-by-step description of how to do this "
                    f"exercise/movement correctly (starting position, movement, key tips). "
                    f"Keep it practical and brief (3-6 steps)."
                )
                _resp = self._conv_client.chat.completions.create(
                    model=self.rag_pipeline.llm_config.model,
                    messages=[
                        {"role": "system", "content": LLM_PROMPTS["system"]},
                        {"role": "user",   "content": motion_desc_prompt},
                    ],
                    temperature=0.5,
                    max_tokens=512,
                )
                text_answer = _resp.choices[0].message.content.strip()

            else:
                # knowledge_query or exercise_recommendation → full RAG with structured output
                logger.info(f"Intent={intent} → RAGPipeline with expanded_query (structured=True)")
                trace["path_taken"] = "rag_path"
                trace["rag_used"] = True
                trace["llm_calls_count"] = 2  # Query expansion + response generation
                
                rag_start = time.time()
                rag_result = self.rag_pipeline.generate_response(
                    query=query,
                    user_id=user_id,
                    conversation_history=conversation_history,
                    memory_context=tool_results.get("memory"),
                    document_context=tool_results.get("documents"),
                    web_context=tool_results.get("web_search"),
                    # Orchestrator already decided on retrieval — skip duplicates:
                    skip_web_search=True,
                    expanded_query=expanded_query,
                    skip_reflection=True,
                    structured=True,     # returns JSON {text_answer, exercises}
                )
                rag_time = time.time() - rag_start
                perf["rag_ms"] = rag_time * 1000
                
                text_answer = rag_result["response"]
                exercises   = rag_result.get("exercises", [])

                # ── Visualization intent detection ────────────────────────────
                # If the query implies the user wants to SEE the exercise performed,
                # select the first exercise as the motion target.  MotionGenerationTool
                # will use this in a later integration step — for now we just prepare the field.
                _VIZ_KEYWORDS = (
                    "show", "visualize", "visualise", "demonstrate",
                    "animation", "animate", "how to do", "how do i",
                )
                _q_lower = query.lower()
                if exercises and any(kw in _q_lower for kw in _VIZ_KEYWORDS):
                    exercise_motion_prompt = exercises[0]["name"].lower()
                    logger.info(
                        f"Visualization query detected → exercise_motion_prompt={exercise_motion_prompt!r}"
                    )

            # ── Step 2b: Call MotionGenerationTool if motion was requested ────
            motion_metadata: Optional[MotionMetadata] = None
            if exercise_motion_prompt:
                logger.info(
                    f"Calling MotionGenerationTool for prompt={exercise_motion_prompt!r}"
                )
                motion_start = time.time()
                try:
                    raw_motion = self.motion_tool.generate_motion(exercise_motion_prompt)
                    motion_time = time.time() - motion_start
                    perf["motion_ms"] = motion_time * 1000
                    
                    if "error" not in raw_motion and raw_motion.get("motion_file"):
                        motion_metadata = MotionMetadata(
                            motion_file=raw_motion["motion_file"],
                            frames=raw_motion["frames"],
                            fps=raw_motion["fps"],
                        )
                        logger.info(
                            f"Motion generated: {motion_metadata.motion_file} "
                            f"({motion_metadata.frames} frames @ {motion_metadata.fps} fps)"
                        )
                    else:
                        logger.warning(
                            f"MotionGenerationTool returned error or missing fields: {raw_motion}"
                        )
                        trace["tools_failed"].append("motion_generation_tool")
                except Exception as _me:
                    logger.error(
                        f"MotionGenerationTool failed for {exercise_motion_prompt!r}: {_me}"
                    )
                    trace["tools_failed"].append("motion_generation_tool")
                    trace["errors"].append(f"Motion generation error: {str(_me)}")
                    # Motion failure is non-fatal — continue with text-only response

            # ── Step 3: Motion prompt (if needed) ────────────────────────────
            motion_prompt = None
            if generate_motion:
                logger.info("Generating motion prompt")
                motion_prompt = self.template_generator.generate_motion_prompt(
                    query=query,
                    action_plan=action_plan,
                    response=text_answer,
                )

            # ── Step 4: Voice prompt — keyword-based, no LLM call ────────────
            voice_prompt = None
            if text_answer:
                voice_prompt = self.template_generator.generate_voice_prompt(
                    text=text_answer,
                    query=query,
                    user_id=user_id,
                    action_plan=action_plan,
                )

            # ── Assemble response ─────────────────────────────────────────────
            total_time = time.time() - start_time
            perf["total_ms"] = total_time * 1000
            
            # Create enhanced orchestrator decision with debug info
            orch_decision = OrchestratorDecision(
                action=action_plan.get("action", "unknown"),
                intent=intent,
                confidence=action_plan.get("confidence", 0.5),
                reasoning=action_plan.get("reasoning", "Orchestrator decision"),
                parameters=action_plan.get("parameters", {}),
                tools_selected=trace["tools_selected"],
                tools_executed=trace["tools_executed"],
                tools_failed=trace["tools_failed"],
                execution_time_ms=perf["orchestrator_ms"],
                debug_info={
                    "memory_results": trace["memory_results_count"],
                    "documents_results": trace["documents_results_count"],
                    "web_results": trace["web_results_count"],
                    "llm_calls": trace["llm_calls_count"],
                    "rag_used": trace["rag_used"],
                    "path": trace["path_taken"],
                },
            )
            
            response = QueryResponse(
                query=query,
                user_id=user_id,
                text_answer=text_answer,
                exercises=exercises,
                exercise_motion_prompt=exercise_motion_prompt,
                motion=motion_metadata,
                orchestrator_decision=orch_decision,
                motion_prompt=MotionPrompt(**motion_prompt) if motion_prompt else None,
                voice_prompt=VoicePrompt(**voice_prompt) if voice_prompt else None,
                metadata=action_plan.get("metadata", {}),
                pipeline_trace=trace,
                performance=perf,
            )

            logger.info(
                f"✓ Query processed successfully | "
                f"intent={intent} | "
                f"path={trace['path_taken']} | "
                f"llm_calls={trace['llm_calls_count']} | "
                f"memory={trace['memory_results_count']} | "
                f"docs={trace['documents_results_count']} | "
                f"web={trace['web_results_count']} | "
                f"total_time={total_time*1000:.0f}ms"
            )
            return response

        except Exception as e:
            logger.error(f"Error processing query: {e}", exc_info=True)
            trace["errors"].append(str(e))
            raise HTTPException(status_code=500, detail=str(e))


# ===========================
# FastAPI Application
# ===========================

# Global API instance
api_instance: Optional[AgenticRAGAPI] = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    """FastAPI lifespan context manager for startup/shutdown."""
    global api_instance
    logger.info("Starting AgenticRAG API server...")
    api_instance = AgenticRAGAPI()
    yield
    logger.info("Shutting down AgenticRAG API server...")


# Create FastAPI application
app = FastAPI(
    title="AgenticRAG API",
    description="REST API for Agentic Retrieval-Augmented Generation",
    version="1.0.0",
    lifespan=lifespan,
)

# Enable CORS for test UI
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ===========================
# API Routes
# ===========================


@app.post("/query", response_model=QueryResponse, summary="Process a query")
async def process_query(request: QueryRequest) -> QueryResponse:
    """Process a user query through the AgenticRAG pipeline.

    Args:
        request: Query request with user query and optional history

    Returns:
        Query response with text answer, decisions, and downstream prompts
    """
    if api_instance is None:
        raise HTTPException(status_code=500, detail="API not initialized")

    # Convert history to dict format if provided
    history = None
    if request.conversation_history:
        history = [turn.dict() for turn in request.conversation_history]

    # IMPORTANT: api_instance.process_query() is synchronous and contains
    # blocking I/O (requests.post → Ollama, ChromaDB queries, Gemini API).
    # Calling it directly from an async handler freezes the uvicorn event loop,
    # making all concurrent requests stall until the blocking call completes.
    # asyncio.to_thread() offloads it to a worker thread so the event loop
    # stays responsive throughout.
    import asyncio
    response = await asyncio.to_thread(
        api_instance.process_query,
        request.query,
        request.user_id,
        history,
    )

    return response


@app.get("/health", summary="Health check")
async def health_check() -> Dict[str, str]:
    """Health check endpoint.

    Returns:
        Status dictionary
    """
    return {"status": "healthy", "service": "agenticrag"}


@app.get("/info", summary="Get service info")
async def get_info() -> Dict[str, Any]:
    """Get service information.

    Returns:
        Service information dictionary
    """
    return {
        "service": "AgenticRAG API",
        "version": "1.0.0",
        "description": "REST API for Agentic Retrieval-Augmented Generation",
        "endpoints": {
            "POST /query": "Process a user query",
            "GET /health": "Health check",
            "GET /info": "Service information",
        },
    }


# ===========================
# Main
# ===========================

if __name__ == "__main__":
    logger.info("Starting AgenticRAG REST API server...")
    uvicorn.run(
        app,
        host="0.0.0.0",
        port=8000,
        log_level="info",
    )
