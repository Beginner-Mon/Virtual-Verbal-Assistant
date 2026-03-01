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
import uvicorn

from agents.orchestrator import OrchestratorAgent
from memory.memory_manager import MemoryManager
from retrieval.rag_pipeline import RAGPipeline
from config import get_config
from utils.logger import get_logger
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


class VoicePrompt(BaseModel):
    """Voice synthesis prompt."""

    text: str = Field(..., description="Text to synthesize")
    emotion: Optional[str] = Field(None, description="Detected or requested emotion")
    duration_estimate_seconds: float = Field(5.0, description="Estimated audio duration")


class MotionPrompt(BaseModel):
    """Motion generation prompt."""

    description: str = Field(..., description="Natural language motion description")
    primitive_sequence: str = Field(
        ...,
        description='Primitive action sequence (e.g., "walk*20,turn_left*10")',
    )
    num_frames: int = Field(160, description="Number of frames to generate")
    fps: int = Field(30, description="Frames per second")


class QueryResponse(BaseModel):
    """Response from query processing."""

    query: str = Field(..., description="Original query")
    user_id: str = Field(..., description="User ID")
    text_answer: str = Field(..., description="Generated text answer")
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
        """Initialize all components."""
        logger.info("Initializing AgenticRAG API...")

        try:
            # Initialize core components
            self.memory_manager = MemoryManager()
            self.rag_pipeline = RAGPipeline(memory_manager=self.memory_manager)
            self.orchestrator = OrchestratorAgent()
            self.template_generator = ResponseTemplateGenerator()

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
        """Process a user query through the RAG pipeline.

        Args:
            query: User's query text
            user_id: User identifier
            conversation_history: Previous conversation turns

        Returns:
            QueryResponse with text answer, decision, and prompts for downstream services
        """
        logger.info(f"Processing query for user {user_id}: {query[:100]}...")

        try:
            # Step 1: Orchestrator analyzes query and makes decision
            action_plan = self.orchestrator.process_query(
                query=query,
                user_id=user_id,
                conversation_history=conversation_history,
            )

            logger.info(f"Orchestrator decision: {action_plan['decision']['action']}")

            # Step 2: Decide if we use memory and if we call LLM
            use_memory = action_plan["actions"]["retrieve_memory"]
            call_llm = action_plan["actions"]["call_llm"]
            generate_motion = action_plan["actions"]["generate_motion"]

            # Step 3: Generate response if needed
            text_answer = ""
            if call_llm:
                logger.info("Calling RAG pipeline for response generation")

                rag_result = self.rag_pipeline.generate_response(
                    query=query,
                    user_id=user_id,
                    conversation_history=conversation_history,
                    use_memory=use_memory,
                )

                text_answer = rag_result["response"]

            else:
                # If no LLM call, provide clarification or use existing response
                text_answer = action_plan["decision"].get(
                    "response", "Could you clarify your request?"
                )

            # Step 4: Generate motion prompt if needed
            motion_prompt = None
            voice_prompt = None

            if generate_motion:
                logger.info("Generating motion prompt")
                motion_prompt = self.template_generator.generate_motion_prompt(
                    query=query,
                    action_plan=action_plan,
                    response=text_answer,
                )

            # Step 5: Generate voice prompt if we have a response
            if text_answer:
                logger.info("Generating voice prompt")
                voice_prompt = self.template_generator.generate_voice_prompt(
                    text=text_answer,
                    query=query,
                    user_id=user_id,
                    action_plan=action_plan,
                )

            # Prepare response
            response = QueryResponse(
                query=query,
                user_id=user_id,
                text_answer=text_answer,
                orchestrator_decision=OrchestratorDecision(
                    action=action_plan["decision"]["action"],
                    confidence=action_plan["decision"]["confidence"],
                    reasoning=action_plan["decision"]["reasoning"],
                    parameters=action_plan["parameters"],
                ),
                motion_prompt=motion_prompt,
                voice_prompt=voice_prompt,
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
