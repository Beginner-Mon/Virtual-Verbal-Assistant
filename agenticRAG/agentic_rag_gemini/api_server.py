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

from agents.orchestrator import OrchestratorAgent
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
    confidence: float = Field(..., description="Confidence score 0.0-1.0")
    reasoning: str = Field(..., description="Reasoning for decision")
    parameters: Dict[str, Any] = Field(default_factory=dict, description="Action parameters")



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

            # MotionGenerationTool — calls DART server; instantiated here so it
            # can be called later in process_query() without repeated construction.
            self.motion_tool = MotionGenerationTool()  # defaults: localhost:5001, 30s timeout

            # OrchestratorAgent — receives tools via dependency injection
            self.orchestrator = OrchestratorAgent(
                memory_tool=memory_tool,
                document_tool=document_tool,
                web_search_tool=web_search_tool,
            )

            # RAGPipeline — kept for LLM response generation
            self.rag_pipeline = RAGPipeline(memory_manager=memory_manager)
            self.template_generator = ResponseTemplateGenerator()

            # Shared LLM client for the conversation fast-path
            self._conv_client = GeminiClientWrapper()

            logger.info("AgenticRAG API initialized successfully")
        except Exception as e:
            logger.error(f"Failed to initialize AgenticRAG API: {e}")
            raise

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
        logger.info(f"Processing query for user {user_id}: {query[:100]}...")

        try:
            # ── Step 1: Single unified LLM call via orchestrator ─────────────
            action_plan = self.orchestrator.process_query(
                query=query,
                user_id=user_id,
                conversation_history=conversation_history,
            )

            intent         = action_plan["intent"]           # str value
            generate_motion = action_plan["actions"]["generate_motion"]
            tool_results    = action_plan.get("tool_results", {})
            expanded_query  = action_plan.get("expanded_query") or query
            needs_rag       = action_plan.get("needs_rag", True)

            logger.info(
                f"Orchestrator: intent={intent} needs_rag={needs_rag} "
                f"generate_motion={generate_motion}"
            )

            # ── Step 2: Branch by intent ─────────────────────────────────────
            text_answer = ""
            exercises   = []   # populated only through the structured RAG path
            exercise_motion_prompt: Optional[str] = None  # set below if query implies visualization

            if intent == "conversation":
                # Fast path: one LLM call with memory context only, no RAG.
                logger.info("Intent=conversation → lightweight LLM (no RAG)")
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
                exercise = action_plan.get("exercise_name") or query
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
                try:
                    raw_motion = self.motion_tool.generate_motion(exercise_motion_prompt)
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
                except Exception as _me:
                    logger.error(
                        f"MotionGenerationTool failed for {exercise_motion_prompt!r}: {_me}"
                    )
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
            response = QueryResponse(
                query=query,
                user_id=user_id,
                text_answer=text_answer,
                exercises=exercises,
                exercise_motion_prompt=exercise_motion_prompt,
                motion=motion_metadata,
                orchestrator_decision=OrchestratorDecision(
                    action=action_plan["decision"]["action"],
                    confidence=action_plan["decision"]["confidence"],
                    reasoning=action_plan["decision"]["reasoning"],
                    parameters=action_plan["parameters"],
                ),
                motion_prompt=MotionPrompt(**motion_prompt) if motion_prompt else None,
                voice_prompt=VoicePrompt(**voice_prompt) if voice_prompt else None,
                metadata=action_plan.get("metadata", {}),
            )

            logger.info(f"Query processed successfully for user {user_id}")
            return response

        except Exception as e:
            logger.error(f"Error processing query: {e}")
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

    response = api_instance.process_query(
        query=request.query,
        user_id=request.user_id,
        conversation_history=history,
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
