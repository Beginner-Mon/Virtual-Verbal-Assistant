#!/usr/bin/env python3
"""REST API server for AgenticRAG.

This module exposes the AgenticRAG system as a FastAPI REST service,
allowing external systems to query the RAG pipeline and receive structured
responses including motion and voice prompts.
"""

import logging
import os
import time
import re
import uuid
import json
from concurrent.futures import ThreadPoolExecutor, TimeoutError as FuturesTimeoutError
from enum import Enum
from typing import Optional, List, Dict, Any
from contextlib import asynccontextmanager

from fastapi import FastAPI, HTTPException, Query, Request
from fastapi.responses import StreamingResponse, Response, RedirectResponse, JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field
from models import VoicePrompt, MotionPrompt  # shared models — single source of truth
import uvicorn
import httpx

from agents.api_orchestrator import OrchestratorAgent
from agents.local_orchestrator import LocalOrchestrator
from agents.safety_filter import SafetyFilter
from agents.query_transform import QueryTransformer
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
from motion_jobs import MotionJobManager
import asyncio
from agents.response_templates import ResponseTemplateGenerator

# Initialize logger
logger = get_logger(__name__)

# In-memory task store for unified /process_query -> /tasks/{id} polling flow.
TASK_STORE: Dict[str, Dict[str, Any]] = {}
TASK_STORE_LOCK = asyncio.Lock()
CHAT_HISTORY_DIR = os.getenv("CHAT_HISTORY_DIR", "./memory/chat_history")
CHAT_HISTORY_LOCK = asyncio.Lock()


def _env_int(name: str, default: int) -> int:
    value = os.getenv(name)
    if value is None:
        return default
    try:
        return int(value)
    except ValueError:
        logger.warning("Invalid integer for %s=%r; using default %d", name, value, default)
        return default


def _env_float(name: str, default: float) -> float:
    value = os.getenv(name)
    if value is None:
        return default
    try:
        return float(value)
    except ValueError:
        logger.warning("Invalid float for %s=%r; using default %.2f", name, value, default)
        return default


MAIN_API_PROXY_BASE_URL = os.getenv("MAIN_API_PROXY_BASE_URL", "http://127.0.0.1:8080").rstrip("/")
MAIN_API_PROXY_TIMEOUT_SECONDS = _env_float("MAIN_API_PROXY_TIMEOUT_SECONDS", 120.0)


_DURATION_PATTERN = re.compile(r"(\d+(?:\.\d+)?)\s*(?:s|sec|secs|second|seconds)\b", re.IGNORECASE)


def _coerce_duration_seconds(raw: Any, min_seconds: float, max_seconds: float) -> Optional[float]:
    """Convert a duration-like value to a bounded float in seconds."""
    if raw is None:
        return None
    try:
        value = float(raw)
    except (TypeError, ValueError):
        return None
    if value <= 0:
        return None
    return max(min_seconds, min(value, max_seconds))


def _extract_duration_seconds_from_text(text: str) -> Optional[float]:
    """Extract an explicit duration value from free text such as '3.2s' or '5 seconds'."""
    if not text:
        return None
    match = _DURATION_PATTERN.search(text)
    if not match:
        return None
    try:
        return float(match.group(1))
    except (TypeError, ValueError):
        return None


def _detect_query_language(query: str) -> str:
    """Detect query language with lightweight heuristics.

    Returns one of: en, vi, jp, other.
    """
    text = (query or "").strip()
    if not text:
        return "other"

    # Japanese scripts (hiragana, katakana, kanji)
    if re.search(r"[\u3040-\u30ff\u31f0-\u31ff\u4e00-\u9fff]", text):
        return "jp"

    # Vietnamese diacritics and common words
    vi_chars_pattern = r"[ăâđêôơưĂÂĐÊÔƠƯáàảãạắằẳẵặấầẩẫậéèẻẽẹếềểễệíìỉĩịóòỏõọốồổỗộớờởỡợúùủũụứừửữựýỳỷỹỵ]"
    if re.search(vi_chars_pattern, text):
        return "vi"

    lowered = f" {text.lower()} "
    vi_keywords = (
        " bạn ", " tôi ", " chúng tôi ", " xin chào ", " cảm ơn ", " bài tập ",
        " đau ", " không ", " giúp ", " như thế nào ", " thế nào ",
    )
    if any(k in lowered for k in vi_keywords):
        return "vi"

    # Default latin-script queries to English unless another non-latin script appears.
    if re.search(r"[a-zA-Z]", text):
        if not re.search(r"[\u0400-\u04FF\u0600-\u06FF\u0590-\u05FF\u0900-\u097F\u0E00-\u0E7F]", text):
            return "en"

    return "other"

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
    user_id: str = Field("guest", description="User identifier (optional, defaults to guest)")
    conversation_history: Optional[List[ConversationTurn]] = Field(
        None, description="Previous conversation turns"
    )
    motion_duration_seconds: Optional[float] = Field(
        None,
        description="Requested motion clip duration in seconds for strict duration mode",
    )
    stream: bool = Field(False, description="Enable streaming response")


class OrchestratorDecision(BaseModel):
    """Orchestrator decision details."""

    action: str = Field(..., description="Action type: retrieve_memory, call_llm, generate_motion, hybrid, clarify")
    intent: Optional[str] = Field(None, description="Intent detected: conversation, knowledge_query, exercise_recommendation, visualize_motion, etc")
    confidence: float = Field(..., description="Confidence score 0.0-1.0")
    language: str = Field("other", description="Detected language code: en | vi | jp | other")
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


class MotionJobStatus(BaseModel):
    """Async motion job status payload."""

    job_id: str = Field(..., description="Celery job identifier")
    status: str = Field(..., description="queued | processing | completed | failed")
    motion_file_url: Optional[str] = Field(None, description="Absolute GLB URL when completed")
    video_url: Optional[str] = Field(None, description="Rendered video or artifact URL when completed")
    error: Optional[str] = Field(None, description="Error details for failed jobs")


class QueryResponse(BaseModel):
    """Response from query processing."""

    query: str = Field(..., description="Original query")
    user_id: str = Field(..., description="User ID")
    language: str = Field("other", description="Detected language code: en | vi | jp | other")
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
    motion_job: Optional[MotionJobStatus] = Field(
        None,
        description="Async motion job handle when motion queue mode is enabled",
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


class UnifiedTaskResponse(BaseModel):
    """Strict response contract for ECA UI 2.0 polling flow."""

    task_id: str = Field(..., description="Unique task identifier")
    status: str = Field(..., description="processing | completed | failed")
    progress_stage: str = Field(
        ..., description="queued | text_ready | motion_generation | completed | failed"
    )
    result: Optional[Dict[str, Any]] = Field(
        None,
        description="Task result payload including absolute motion_file_url when available",
    )
    error: Optional[str] = Field(None, description="Error details when task fails")


class ChatHistoryResponse(BaseModel):
    """File-backed chat history contract used by Official UI."""

    user_id: str
    messages: List[Dict[str, Any]] = Field(default_factory=list)


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

            # Save tool references for orchestrator integrations and future extensions.
            self._memory_tool_ref     = memory_tool
            self._document_tool_ref   = document_tool
            self._web_search_tool_ref = web_search_tool

            # MotionGenerationTool — calls DART server; instantiated here so it
            # can be called later in process_query() without repeated construction.
            self.motion_tool = MotionGenerationTool()  # defaults: localhost:5001, 30s timeout
            self.motion_job_manager = MotionJobManager()
            self.motion_async_enabled = os.getenv("MOTION_ASYNC_ENABLED", "false").lower() in {
                "1", "true", "yes", "on"
            }
            self.motion_default_duration_seconds = _env_float("MOTION_DEFAULT_DURATION_SECONDS", 12.0)
            self.motion_min_duration_seconds = _env_float("MOTION_MIN_DURATION_SECONDS", 1.0)
            self.motion_max_duration_seconds = _env_float("MOTION_MAX_DURATION_SECONDS", 120.0)
            if self.motion_max_duration_seconds < self.motion_min_duration_seconds:
                self.motion_min_duration_seconds, self.motion_max_duration_seconds = (
                    self.motion_max_duration_seconds,
                    self.motion_min_duration_seconds,
                )
            self.motion_duration_policy = os.getenv("MOTION_DURATION_POLICY", "strict").strip().lower() or "strict"

            # Safety Filter (Red Flag Screening via SLM)
            self.safety_filter = SafetyFilter()
            
            # Query Transformation Engine (Double-RAG Semantic Router)
            self.query_transformer = QueryTransformer(use_cache=True)

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

            # Lightweight in-memory cache for orchestrator routing decisions.
            self._orchestrator_cache: Dict[str, Dict[str, Any]] = {}
            self._orchestrator_cache_ttl_sec = _env_int("ORCHESTRATOR_CACHE_TTL_SEC", 120)
            self.orchestrator_decision_timeout_seconds = max(
                0.1,
                _env_float("ORCHESTRATOR_DECISION_TIMEOUT_SECONDS", 2.0),
            )
            self.enable_fast_motion_bypass = os.getenv(
                "ORCHESTRATOR_FAST_MOTION_BYPASS", "true"
            ).lower() in {"1", "true", "yes", "on"}
            fast_markers_raw = os.getenv(
                "ORCHESTRATOR_FAST_MOTION_MARKERS",
                "show me how,show me,demonstrate,visualize,visualise,animate,how to do",
            )
            self.fast_motion_markers = tuple(
                marker.strip().lower()
                for marker in fast_markers_raw.split(",")
                if marker.strip()
            )

            logger.info("AgenticRAG API initialized successfully")
        except Exception as e:
            logger.error(f"Failed to initialize AgenticRAG API: {e}")
            raise

    def get_motion_job_status(self, job_id: str, request_base_url: Optional[str] = None) -> Dict[str, Any]:
        """Get async motion job status by job_id."""
        return self.motion_job_manager.get_status(job_id, request_base_url=request_base_url)

    def _orchestrator_cache_key(self, user_id: str, query: str) -> str:
        return f"{user_id}::{query.strip().lower()}"

    def _looks_like_conversation_query(self, query: str) -> bool:
        q = query.strip().lower()
        if not q:
            return True
        conversation_markers = {
            "hi", "hello", "hey", "thanks", "thank you", "ok", "okay",
            "how are you", "good morning", "good afternoon", "good evening",
        }
        if q in conversation_markers:
            return True
        if len(q) <= 20 and any(marker in q for marker in conversation_markers):
            return True
        motion_markers = ("exercise", "stretch", "workout", "pain", "show", "visualize", "demonstrate")
        return len(q.split()) <= 5 and not any(m in q for m in motion_markers)

    def _resolve_motion_duration_seconds(
        self,
        query: str,
        action_plan: Dict[str, Any],
        request_duration_seconds: Optional[float] = None,
    ) -> float:
        """Resolve motion duration with strict precedence and bounded clamping."""
        min_s = self.motion_min_duration_seconds
        max_s = self.motion_max_duration_seconds

        candidates: List[Any] = []
        if request_duration_seconds is not None:
            candidates.append(request_duration_seconds)

        if isinstance(action_plan, dict):
            parameters = action_plan.get("parameters")
            double_rag_meta = action_plan.get("double_rag_meta")
            motion_prompt = action_plan.get("motion_prompt")

            candidates.append(action_plan.get("duration_seconds"))
            if isinstance(parameters, dict):
                candidates.append(parameters.get("duration_seconds"))
            if isinstance(double_rag_meta, dict):
                candidates.append(double_rag_meta.get("duration_seconds"))
                constraints = double_rag_meta.get("constraints")
                if constraints:
                    candidates.append(_extract_duration_seconds_from_text(str(constraints)))
            if isinstance(motion_prompt, dict):
                candidates.append(motion_prompt.get("duration_seconds"))
                candidates.append(motion_prompt.get("duration_estimate_seconds"))

        # In strict policy, explicit duration mentions in query should be honored.
        if self.motion_duration_policy == "strict":
            candidates.append(_extract_duration_seconds_from_text(query))

        for raw in candidates:
            resolved = _coerce_duration_seconds(raw, min_s, max_s)
            if resolved is not None:
                return resolved

        return _coerce_duration_seconds(self.motion_default_duration_seconds, min_s, max_s) or self.motion_default_duration_seconds

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

    def _get_token_limit(self, intent: str) -> int:
        """Get the max_tokens limit for a given intent.
        
        Args:
            intent: Intent type (conversation, visualize_motion, knowledge_query, exercise_recommendation)
            
        Returns:
            Token limit for this intent, or fallback if intent not found
        """
        config = get_config()
        intent_limits = config.intent_token_limits
        
        # Try to get limit for this specific intent using getattr (fallback if not found)
        limit = getattr(intent_limits, intent, None)
        
        if limit is not None:
            logger.debug(f"Token limit for intent '{intent}': {limit}")
            return limit
        
        # Fallback to default limit
        fallback_limit = intent_limits.fallback
        logger.debug(f"Intent '{intent}' not found in config, using fallback: {fallback_limit}")
        return fallback_limit

    def _is_multi_activity_request(self, query: str) -> bool:
        """Heuristic: detect requests asking for a list of multiple exercises."""
        q = (query or "").lower()
        multi_markers = (
            "exercises", "exercise list", "list", "suggest", "recommend",
            "top ", "best ", "routine", "workout plan",
        )
        explicit_count_markers = (
            " 2 ", " 3 ", " 4 ", " 5 ", " 6 ", " 7 ", " 8 ", " 9 ", " 10 ",
            "two", "three", "four", "five", "six", "seven", "eight", "nine", "ten",
        )
        asks_multiple = any(m in q for m in multi_markers)
        asks_count = any(m in f" {q} " for m in explicit_count_markers)
        return asks_multiple or asks_count

    def _is_direct_motion_command(self, query: str) -> bool:
        """Detect direct motion-demo commands that should bypass slow routing."""
        q = (query or "").strip().lower()
        if not q:
            return False
        if self._is_multi_activity_request(q):
            return False
        markers = self.fast_motion_markers
        if not markers:
            return False
        return any(q.startswith(marker) or f" {marker} " in f" {q} " for marker in markers)

    def _build_fast_fallback_action_plan(
        self,
        query: str,
        detected_exercise: Optional[str],
        reason: str,
    ) -> Dict[str, Any]:
        """Build a deterministic action plan when orchestrator calls time out/fail."""
        direct_motion = self._is_direct_motion_command(query)
        is_conversation = self._looks_like_conversation_query(query)

        if direct_motion:
            intent = "visualize_motion"
            generate_motion = True
            needs_rag = False
            agents = ["memory_agent"]
        elif is_conversation:
            intent = "conversation"
            generate_motion = False
            needs_rag = False
            agents = ["memory_agent"]
        else:
            intent = "knowledge_query"
            generate_motion = False
            needs_rag = True
            agents = ["memory_agent", "retrieval_agent"]

        return {
            "intent": intent,
            "actions": {
                "generate_motion": generate_motion,
                "use_memory": True,
                "use_documents": needs_rag,
                "use_web_search": False,
            },
            "tool_results": {},
            "expanded_query": query,
            "needs_rag": needs_rag,
            "exercise": detected_exercise,
            "exercise_name": detected_exercise,
            "agents": agents,
            "confidence": 0.4,
            "fallback_reason": reason,
        }

    def _run_with_timeout(
        self,
        call,
        timeout_seconds: float,
        label: str,
    ) -> Optional[Dict[str, Any]]:
        """Run a blocking call with a hard timeout and return None on timeout/error."""
        if timeout_seconds <= 0:
            try:
                return call()
            except Exception as exc:
                logger.warning("%s failed: %s", label, exc)
                return None

        executor = ThreadPoolExecutor(max_workers=1, thread_name_prefix="orchestrator-timebox")
        future = executor.submit(call)
        try:
            return future.result(timeout=timeout_seconds)
        except FuturesTimeoutError:
            future.cancel()
            logger.warning("%s timed out after %.2fs", label, timeout_seconds)
            return None
        except Exception as exc:
            logger.warning("%s failed: %s", label, exc)
            return None
        finally:
            executor.shutdown(wait=False, cancel_futures=True)

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

        # Cache only when there is no conversation history context attached.
        use_cache = not conversation_history
        cache_key = self._orchestrator_cache_key(user_id, query)
        if use_cache:
            cached = self._orchestrator_cache.get(cache_key)
            if cached:
                age = time.time() - cached["ts"]
                if age <= self._orchestrator_cache_ttl_sec:
                    logger.info(f"Orchestrator cache hit (age={age:.1f}s)")
                    return cached["decision"]
                self._orchestrator_cache.pop(cache_key, None)
        
        # Step 1: Detect exercise using ExerciseDetector (Hybrid Entity Extraction)
        detected_exercise = None
        if self._looks_like_conversation_query(query):
            logger.info("Skipping exercise detection for conversation-like query")
        else:
            logger.info(f"Detecting exercise in query: {query}")
            detected_exercise = self.exercise_detector.detect_exercise(query)
        
        if detected_exercise:
            logger.info(f"Exercise detected: '{detected_exercise}'")
        else:
            logger.info("No exercise detected in query")

        if self.enable_fast_motion_bypass and self._is_direct_motion_command(query):
            logger.info("Fast motion bypass hit; skipping orchestrator model call")
            fast_result = self._build_fast_fallback_action_plan(
                query=query,
                detected_exercise=detected_exercise,
                reason="fast_motion_bypass",
            )
            if use_cache:
                self._orchestrator_cache[cache_key] = {"ts": time.time(), "decision": fast_result}
            return fast_result
        
        if isinstance(self.orchestrator, LocalOrchestrator):
            logger.info("USING LOCAL ORCHESTRATOR path")
            logger.info(f"About to call LocalOrchestrator.analyze_query with: {query}, user_id: {user_id}, detected_exercise: {detected_exercise}")
            decision = self._run_with_timeout(
                lambda: self.orchestrator.analyze_query(query, user_id, conversation_history, detected_exercise),
                timeout_seconds=self.orchestrator_decision_timeout_seconds,
                label="LocalOrchestrator.analyze_query",
            )
            if not isinstance(decision, dict):
                timeout_result = self._build_fast_fallback_action_plan(
                    query=query,
                    detected_exercise=detected_exercise,
                    reason="local_orchestrator_timeout_or_error",
                )
                if use_cache:
                    self._orchestrator_cache[cache_key] = {"ts": time.time(), "decision": timeout_result}
                return timeout_result
            logger.info(f"Local orchestrator returned: {decision}")

            # ── Detect fallback response (Ollama timed out or JSON invalid) ──
            # The fallback always has confidence=0.1 and intent="unknown".
            # When this happens, LocalOrchestrator is not useful for this request;
            # immediately use deterministic fallback routing so the user gets
            # a fast, bounded decision instead of stalling.
            _is_fallback = (
                decision.get("confidence", 1.0) <= 0.1
                and decision.get("intent") == "unknown"
            )
            if _is_fallback:
                logger.warning(
                    "[LocalOrchestrator] Fallback response detected (confidence=0.1, intent=unknown). "
                    "Using fast deterministic fallback action plan."
                )
                fallback_result = self._build_fast_fallback_action_plan(
                    query=query,
                    detected_exercise=detected_exercise,
                    reason="local_orchestrator_returned_fallback",
                )
                if use_cache:
                    self._orchestrator_cache[cache_key] = {"ts": time.time(), "decision": fallback_result}
                return fallback_result

            # Normalize LocalOrchestrator-specific intent strings so that
            # process_query() branching works correctly for greetings/follow-ups.
            raw_intent = decision.get("intent", "unknown")
            canonical_intent = self._INTENT_CANONICAL_MAP.get(raw_intent, "knowledge_query")
            if canonical_intent != raw_intent:
                logger.info(
                    f"Intent normalized: '{raw_intent}' → '{canonical_intent}'"
                )
            decision["intent"] = canonical_intent

            # ── Fix B: Low-confidence visualization override ────────────────
            # If Qwen is unsure (confidence < 0.6) AND the user uses explicit
            # visualization verbs, force intent to visualize_motion so DART
            # generates the motion instead of RAG returning a text-only answer.
            _VIZ_KEYWORDS = ("show me how", "how to do", "visualize", "demonstrate")
            _q = query.lower()
            if (
                decision.get("confidence", 1.0) < 0.6
                and decision["intent"] != "visualize_motion"
                and any(kw in _q for kw in _VIZ_KEYWORDS)
                and not self._is_multi_activity_request(query)
            ):
                logger.info(
                    f"Low-confidence override: intent='{decision['intent']}' "
                    f"(conf={decision.get('confidence')}) → visualize_motion"
                )
                decision["intent"] = "visualize_motion"
                decision["needs_motion"] = True

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
            if use_cache:
                self._orchestrator_cache[cache_key] = {"ts": time.time(), "decision": result}
            return result
        else:
            logger.info("USING API ORCHESTRATOR path")
            # API orchestrator returns the existing format
            # Note: API orchestrator doesn't support detected_exercise parameter yet
            api_result = self._run_with_timeout(
                lambda: self.orchestrator.process_query(query, user_id, conversation_history),
                timeout_seconds=self.orchestrator_decision_timeout_seconds,
                label="APIOrchestrator.process_query",
            )
            if not isinstance(api_result, dict):
                api_result = self._build_fast_fallback_action_plan(
                    query=query,
                    detected_exercise=detected_exercise,
                    reason="api_orchestrator_timeout_or_error",
                )
            if use_cache:
                self._orchestrator_cache[cache_key] = {"ts": time.time(), "decision": api_result}
            return api_result

    def process_query(
        self,
        query: str,
        user_id: str,
        conversation_history: Optional[List[Dict[str, str]]] = None,
        stream: bool = False,
        motion_duration_seconds: Optional[float] = None,
    ) -> Any:
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
        
        If stream=True, returns a generator yielding text chunks.
        """
        import time
        start_time = time.time()

        user_id = (user_id or "guest").strip() or "guest"
        
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
            # ── Step 0: Red Flag Safety Screening (First Gate) ───────────────
            safety_start = time.time()
            safety_result = self.safety_filter.check_query_safety(query)
            if not safety_result["is_safe"]:
                safety_ms = (time.time() - safety_start) * 1000
                logger.warning(f"Query rejected by SafetyFilter: {safety_result['reason']}")
                # Fast return rejecting the unsafe query
                return {
                    "query": query,
                    "user_id": user_id,
                    "language": _detect_query_language(query),
                    "text_answer": f"⚠️ Safety Alert: {safety_result['reason']}",
                    "exercises": [],
                    "exercise_motion_prompt": None,
                    "motion": None,
                    "motion_job": None,
                    "orchestrator_decision": OrchestratorDecision(
                        action="clarify",
                        intent="unknown",
                        confidence=1.0,
                        language="en",
                        reasoning="Safety filter rejection",
                        parameters={"reason": safety_result["reason"]}
                    ),
                    "motion_prompt": None,
                    "voice_prompt": None,
                    "metadata": {"safety_rejected": True},
                    "pipeline_trace": {"errors": [safety_result["reason"]]},
                    "performance": {"safety_ms": safety_ms, "total_ms": safety_ms}
                }

            # ── Step 1: Get orchestrator decision (local or API) ─────────────
            orch_start = time.time()
            logger.info(f"Using orchestrator: {type(self.orchestrator).__name__}")
            trace["orchestrator_type"] = type(self.orchestrator).__name__
            query_language = _detect_query_language(query)
            
            action_plan = self._get_orchestrator_decision(
                query=query,
                user_id=user_id,
                conversation_history=conversation_history,
            )
            orch_time = time.time() - orch_start
            perf["orchestrator_ms"] = orch_time * 1000
            logger.info(f"Orchestrator decision type: {type(action_plan)}")

            actions = action_plan.get("actions") or {}
            intent = action_plan.get("intent", "knowledge_query")
            if isinstance(intent, Enum):
                intent = intent.value
            generate_motion = bool(actions.get("generate_motion", False))
            tool_results = action_plan.get("tool_results") or {}
            expanded_query  = action_plan.get("expanded_query") or query
            needs_rag = bool(action_plan.get("needs_rag", True))
            
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

            resolved_motion_duration_seconds = self._resolve_motion_duration_seconds(
                query=query,
                action_plan=action_plan,
                request_duration_seconds=motion_duration_seconds,
            )

            # ── Step 1.5: Query Transformation results (from Double-RAG Engine) ──
            hyde_document = action_plan.get("hyde_document", query)
            logger.info(f"Using Double-RAG engine results. HyDE length: {len(hyde_document)}")
            
            # The orchestrator already executed Phase 1 & 2 logic internally and 
            # mapped constraints if this was a motion intent.
            double_rag_meta = action_plan.get("double_rag_meta", {})
            constraints = double_rag_meta.get("constraints", "")
            if constraints:
                logger.info(f"Clinical Constraints extracted by orchestrator: {constraints}")

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
                    max_tokens=self._get_token_limit("conversation"),
                    stream=stream,
                )
                if stream:
                    return (chunk.choices[0].message.content for chunk in _resp if chunk.choices[0].message.content)
                text_answer = _resp.choices[0].message.content.strip()

            elif intent == "visualize_motion":
                # Generate a text description + request motion animation from DART.
                # The text answer is always shown; DART animation is a bonus.
                logger.info("Intent=visualize_motion → LLM description + motion prompt")
                trace["path_taken"] = "visualize_motion_path"
                trace["llm_calls_count"] = 1
                
                exercise = action_plan.get("exercise_name") or query
                
                # Double-RAG flow guarantees hyde_document is the technical motion caption
                exercise_motion_prompt = hyde_document
                if constraints:
                    logger.info("Applying clinical constraints to visualize_motion path.")
                
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
                    max_tokens=self._get_token_limit("visualize_motion"),
                    stream=stream,
                )
                if stream:
                    return (chunk.choices[0].message.content for chunk in _resp if chunk.choices[0].message.content)
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
                    structured=not stream,     # cannot stream structured JSON easily
                    stream=stream,
                    max_tokens=self._get_token_limit(intent),
                )
                if stream:
                    return rag_result
                
                perf["rag_ms"] = (time.time() - rag_start) * 1000
                
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
                is_multi_activity = self._is_multi_activity_request(query)
                if exercises and any(kw in _q_lower for kw in _VIZ_KEYWORDS) and not is_multi_activity:
                    # Double-RAG engine provides the exact motion prompt via hyde_document
                    exercise_motion_prompt = hyde_document
                    logger.info(
                        f"Visualization query detected → exercise_motion_prompt={exercise_motion_prompt!r}"
                    )
                elif exercises and any(kw in _q_lower for kw in _VIZ_KEYWORDS) and is_multi_activity:
                    logger.info(
                        "Visualization keyword found, but query is multi-activity/list request; "
                        "returning exercise list without auto-generating motion."
                    )

            # ── Step 2b: Call MotionGenerationTool if motion was requested ────
            motion_metadata: Optional[MotionMetadata] = None
            motion_job: Optional[MotionJobStatus] = None
            if exercise_motion_prompt:
                motion_start = time.time()
                if self.motion_async_enabled:
                    try:
                        job_id = self.motion_job_manager.enqueue(
                            user_query=query,
                            motion_prompt=exercise_motion_prompt,
                            user_id=user_id,
                            duration_seconds=resolved_motion_duration_seconds,
                        )
                        motion_job = MotionJobStatus(
                            job_id=job_id,
                            status="queued",
                            video_url=None,
                            error=None,
                        )
                        perf["motion_ms"] = (time.time() - motion_start) * 1000
                        logger.info("Async motion job queued: %s", job_id)
                    except Exception as _qe:
                        logger.error("Async queue failed, falling back sync motion: %s", _qe)
                        trace["errors"].append(f"Motion queue error: {str(_qe)}")

                if not self.motion_async_enabled or (self.motion_async_enabled and motion_job is None):
                    # For kinematic motion search, we now pass the hyde_document instead of a vague prompt string
                    motion_target = hyde_document if intent != "conversation" else exercise_motion_prompt
                    if constraints:
                        motion_target = f"{motion_target}. Clinical constraints: {constraints}"

                    logger.info(
                        f"Calling MotionGenerationTool for motion_target (HyDE)= {motion_target[:50]}..."
                    )
                    try:
                        raw_motion = self.motion_tool.generate_motion(motion_target)
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
            action_value = action_plan.get("action", "unknown")
            intent_value = intent
            if isinstance(action_value, Enum):
                action_value = action_value.value
            if isinstance(intent_value, Enum):
                intent_value = intent_value.value

            orch_decision = OrchestratorDecision(
                action=str(action_value),
                intent=str(intent_value),
                confidence=action_plan.get("confidence", 0.5),
                language=query_language,
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
                language=query_language,
                text_answer=text_answer,
                exercises=exercises,
                exercise_motion_prompt=exercise_motion_prompt,
                motion=motion_metadata,
                motion_job=motion_job,
                orchestrator_decision=orch_decision,
                motion_prompt=MotionPrompt(**motion_prompt) if motion_prompt else None,
                voice_prompt=VoicePrompt(**voice_prompt) if voice_prompt else None,
                metadata=action_plan.get("metadata", {}),
                pipeline_trace=trace,
                performance=perf,
            )

            response.metadata = dict(response.metadata or {})
            response.metadata["motion_duration_seconds"] = resolved_motion_duration_seconds
            response.metadata["motion_duration_policy"] = self.motion_duration_policy

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

def _normalize_enums(value: Any) -> Any:
    """Recursively convert Enum values to .value to guarantee JSON-safe strings."""
    if isinstance(value, Enum):
        return value.value
    if isinstance(value, dict):
        return {str(k): _normalize_enums(v) for k, v in value.items()}
    if isinstance(value, list):
        return [_normalize_enums(v) for v in value]
    return value


def _model_to_dict(value: Any) -> Dict[str, Any]:
    """Convert pydantic model (v1/v2) or mapping payload to a dict."""
    if value is None:
        return {}
    if isinstance(value, dict):
        return value
    if hasattr(value, "model_dump"):
        return value.model_dump()
    if hasattr(value, "dict"):
        return value.dict()
    return dict(value)


def _get_request_base_url(request: Request) -> str:
    """Build base URL using reverse-proxy headers so Ngrok URLs remain valid."""
    proto = request.headers.get("x-forwarded-proto") or request.url.scheme
    host = request.headers.get("x-forwarded-host") or request.headers.get("host") or request.url.netloc
    if not host:
        return str(request.base_url).rstrip("/")
    return f"{proto}://{host}".rstrip("/")


def _to_absolute_url(url_or_path: Optional[str], request: Request) -> Optional[str]:
    if not url_or_path:
        return None
    if url_or_path.startswith("http://") or url_or_path.startswith("https://"):
        return url_or_path
    base = _get_request_base_url(request)
    path = url_or_path if url_or_path.startswith("/") else f"/{url_or_path}"
    return f"{base}{path}"


def _to_dart_download_url(motion_ref: Optional[str], request: Request) -> Optional[str]:
    """Build absolute DART download URLs for sync motion payloads."""
    if not motion_ref:
        return None

    if motion_ref.startswith("http://") or motion_ref.startswith("https://"):
        return motion_ref

    # Force all motion artifact URLs through the 8000 gateway.
    base = _get_request_base_url(request)

    if motion_ref.startswith("/static/"):
        path = motion_ref
    elif motion_ref.startswith("/"):
        path = motion_ref
    elif motion_ref.startswith("download/"):
        path = f"/{motion_ref}"
    else:
        # Legacy payloads only include the artifact filename.
        path = f"/download/{motion_ref}"

    return f"{base}{path}"


async def _proxy_main_api_request(
    method: str,
    path: str,
    *,
    json_body: Optional[Dict[str, Any]] = None,
) -> Response:
    """Forward request to the orchestrator API running on port 8080."""
    upstream_url = f"{MAIN_API_PROXY_BASE_URL}{path}"
    timeout_seconds = max(0.1, float(MAIN_API_PROXY_TIMEOUT_SECONDS))

    try:
        async with httpx.AsyncClient(
            timeout=httpx.Timeout(timeout_seconds, connect=min(timeout_seconds, 10.0))
        ) as client:
            upstream = await client.request(method, upstream_url, json=json_body)
    except Exception as exc:
        logger.error("Main API proxy failed for %s %s: %s", method, path, exc)
        raise HTTPException(status_code=502, detail="Unable to reach orchestrator API on port 8080")

    content_type = upstream.headers.get("content-type", "")
    if "application/json" in content_type.lower():
        try:
            payload = upstream.json()
        except Exception:
            payload = {"detail": upstream.text or "Invalid JSON response from orchestrator"}
        return JSONResponse(status_code=upstream.status_code, content=payload)

    return Response(
        content=upstream.content,
        status_code=upstream.status_code,
        media_type=content_type or "application/octet-stream",
    )


def _build_unified_result(query_payload: Dict[str, Any], request: Request) -> Dict[str, Any]:
    """Project QueryResponse into the stable result payload used by Official UI."""
    orchestrator = query_payload.get("orchestrator_decision") or {}
    motion = query_payload.get("motion") or {}
    motion_job = query_payload.get("motion_job") or {}

    motion_file_url = None

    if isinstance(motion, dict):
        motion_file_url = motion.get("motion_file_url")

    if not motion_file_url and isinstance(motion_job, dict):
        motion_file_url = motion_job.get("motion_file_url") or motion_job.get("video_url")

    if isinstance(motion, dict) and motion_file_url:
        motion_file_url = _to_dart_download_url(motion_file_url, request)

    if not motion_file_url and isinstance(motion, dict):
        # Legacy best-effort path where only motion_file exists.
        motion_file = motion.get("motion_file")
        motion_file_url = _to_dart_download_url(motion_file, request)

    return {
        "query": query_payload.get("query", ""),
        "user_id": query_payload.get("user_id", "guest"),
        "language": query_payload.get("language", "other"),
        "text_answer": query_payload.get("text_answer", ""),
        "exercises": query_payload.get("exercises", []),
        "exercise_motion_prompt": query_payload.get("exercise_motion_prompt"),
        "motion_duration_seconds": (
            (query_payload.get("metadata") or {}).get("motion_duration_seconds")
            if isinstance(query_payload.get("metadata"), dict)
            else None
        ),
        "orchestrator": {
            "action": orchestrator.get("action", "unknown"),
            "intent": orchestrator.get("intent", "unknown"),
            "confidence": orchestrator.get("confidence", 0.0),
        },
        "motion_file_url": _to_absolute_url(motion_file_url, request),
        "motion": motion,
        "motion_job": motion_job,
    }


def _history_file_path(user_id: str) -> str:
    safe_user = re.sub(r"[^a-zA-Z0-9._-]", "_", (user_id or "guest")).strip("._-") or "guest"
    os.makedirs(CHAT_HISTORY_DIR, exist_ok=True)
    return os.path.join(CHAT_HISTORY_DIR, f"{safe_user}.json")


async def _read_chat_history(user_id: str) -> List[Dict[str, Any]]:
    file_path = _history_file_path(user_id)
    if not os.path.exists(file_path):
        return []

    async with CHAT_HISTORY_LOCK:
        try:
            with open(file_path, "r", encoding="utf-8") as f:
                payload = json.load(f)
            if isinstance(payload, list):
                return payload
        except Exception as exc:
            logger.warning("Failed to read chat history %s: %s", file_path, exc)
    return []


async def _append_chat_history(user_id: str, entries: List[Dict[str, Any]]) -> None:
    if not entries:
        return

    file_path = _history_file_path(user_id)
    async with CHAT_HISTORY_LOCK:
        history: List[Dict[str, Any]] = []
        if os.path.exists(file_path):
            try:
                with open(file_path, "r", encoding="utf-8") as f:
                    payload = json.load(f)
                if isinstance(payload, list):
                    history = payload
            except Exception:
                history = []

        history.extend(entries)
        # Keep file size bounded for UI load performance.
        history = history[-200:]

        with open(file_path, "w", encoding="utf-8") as f:
            json.dump(history, f, ensure_ascii=False, indent=2)


def _allowed_cors_origins() -> List[str]:
    """Build explicit CORS allowlist for localhost UI and active ngrok origin."""
    origins = {
        "http://localhost:3000",
        "http://127.0.0.1:3000",
        "http://localhost:8501",
        "http://127.0.0.1:8501",
        # file:// pages send Origin: null in browsers.
        "null",
    }

    configured = os.getenv("CORS_ALLOWED_ORIGINS", "")
    if configured:
        for item in configured.split(","):
            origin = item.strip()
            if origin:
                origins.add(origin)

    ngrok_origin = os.getenv("NGROK_ORIGIN", "").strip()
    if ngrok_origin:
        origins.add(ngrok_origin)

    return sorted(origins)


_allowed_origins = _allowed_cors_origins()

app.add_middleware(
    CORSMiddleware,
    allow_origins=_allowed_origins,
    allow_origin_regex=r"^https://[a-z0-9-]+\.ngrok(?:-free)?\.app$",
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Expose local static video directory for async worker artifacts.
os.makedirs("./static/videos", exist_ok=True)
app.mount("/static", StaticFiles(directory="./static"), name="static")

# Serve test-ui frontend at /ui/ so it's accessible through Ngrok
_ui_dir = os.path.join(os.path.dirname(__file__), "..", "..", "test-ui")
if os.path.isdir(_ui_dir):
    app.mount("/ui", StaticFiles(directory=_ui_dir, html=True), name="test-ui")

# Serve Official ECA UI at /eca/ so one port-8000 tunnel can host UI + API.
_eca_ui_dir = os.path.join(os.path.dirname(__file__), "..", "..", "ECA_UI")
if os.path.isdir(_eca_ui_dir):
    app.mount("/eca", StaticFiles(directory=_eca_ui_dir, html=True), name="eca-ui")


# ===========================
# API Routes
# ===========================


@app.post("/query", summary="Process a query")
async def process_query(request: QueryRequest):
    """Process a user query through the AgenticRAG pipeline.
    
    Returns:
        - If streaming (request.stream=True): StreamingResponse with text chunks (text/plain)
        - Otherwise: QueryResponse with full metadata (application/json)
    
    Args:
        request: Query request with user query and optional history
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
    
    # If streaming is requested, return StreamingResponse with text chunks
    request_user_id = (request.user_id or "guest").strip() or "guest"

    if request.stream:
        # For streaming, we run it in a thread and wrap the generator.
        # Since the generator yielded by process_query is sync, 
        # StreamingResponse can handle it directly.
        # NOTE: Streaming returns plain text chunks, not JSON.
        generator = api_instance.process_query(
            request.query,
            request_user_id,
            history,
            stream=True,
            motion_duration_seconds=request.motion_duration_seconds,
        )
        return StreamingResponse(generator, media_type="text/plain")

    # For non-streaming requests, return full QueryResponse with all metadata
    response = await asyncio.to_thread(
        api_instance.process_query,
        request.query,
        request_user_id,
        history,
        False,
        request.motion_duration_seconds,
    )

    return _normalize_enums(_model_to_dict(response))


async def _run_query_task(task_id: str, request_payload: QueryRequest, request: Request) -> None:
    """Background worker for unified task flow."""
    if api_instance is None:
        async with TASK_STORE_LOCK:
            TASK_STORE[task_id] = {
                "task_id": task_id,
                "status": "failed",
                "progress_stage": "failed",
                "result": None,
                "error": "API not initialized",
            }
        return

    history = None
    if request_payload.conversation_history:
        history = [turn.dict() for turn in request_payload.conversation_history]

    request_user_id = (request_payload.user_id or "guest").strip() or "guest"

    try:
        request_base = _get_request_base_url(request)
        async with TASK_STORE_LOCK:
            state = TASK_STORE.get(task_id) or {}
            state["status"] = "processing"
            state["progress_stage"] = "queued"
            state["error"] = None
            TASK_STORE[task_id] = state

        raw_response = await asyncio.to_thread(
            api_instance.process_query,
            request_payload.query,
            request_user_id,
            history,
            False,
            request_payload.motion_duration_seconds,
        )

        response_payload = _normalize_enums(_model_to_dict(raw_response))
        unified_result = _build_unified_result(response_payload, request)

        async with TASK_STORE_LOCK:
            TASK_STORE[task_id] = {
                "task_id": task_id,
                "status": "processing",
                "progress_stage": "text_ready",
                "result": unified_result,
                "error": None,
            }

        motion_job = unified_result.get("motion_job") or {}
        motion_job_id = motion_job.get("job_id") if isinstance(motion_job, dict) else None

        # Force async enrichment for unified UI polling even when /query path returned sync motion.
        if not motion_job_id and api_instance is not None:
            exercise_motion_prompt = response_payload.get("exercise_motion_prompt")
            orchestrator_intent = (
                ((response_payload.get("orchestrator_decision") or {}).get("intent"))
                if isinstance(response_payload.get("orchestrator_decision"), dict)
                else None
            )
            should_enqueue_motion = bool(exercise_motion_prompt) or orchestrator_intent == "visualize_motion"

            if should_enqueue_motion:
                try:
                    resolved_duration = _coerce_duration_seconds(
                        (unified_result.get("motion_duration_seconds") or request_payload.motion_duration_seconds),
                        api_instance.motion_min_duration_seconds,
                        api_instance.motion_max_duration_seconds,
                    ) or api_instance.motion_default_duration_seconds

                    queued_job_id = api_instance.motion_job_manager.enqueue(
                        user_query=request_payload.query,
                        motion_prompt=exercise_motion_prompt or request_payload.query,
                        user_id=request_user_id,
                        duration_seconds=resolved_duration,
                    )
                    motion_job_id = queued_job_id
                    unified_result["motion_job"] = {
                        "job_id": queued_job_id,
                        "status": "queued",
                        "motion_file_url": None,
                        "video_url": None,
                        "error": None,
                    }

                    async with TASK_STORE_LOCK:
                        TASK_STORE[task_id] = {
                            "task_id": task_id,
                            "status": "processing",
                            "progress_stage": "motion_generation",
                            "result": unified_result,
                            "error": None,
                        }
                except Exception as queue_exc:
                    logger.error(
                        "Unified async motion enqueue failed for task %s: %s",
                        task_id,
                        queue_exc,
                    )

        final_status = "completed"
        final_stage = "completed"
        final_error = None

        if motion_job_id and api_instance is not None:
            # Keep task open while async motion artifact is being generated.
            while True:
                motion_state = api_instance.get_motion_job_status(
                    motion_job_id,
                    request_base_url=request_base,
                )
                motion_status = (motion_state.get("status") or "processing").lower()
                motion_stage = motion_state.get("stage") or "motion_generation"
                normalized_stage = motion_stage
                if motion_status in {"queued", "processing"}:
                    normalized_stage = "motion_generation"
                elif motion_status == "completed":
                    normalized_stage = "completed"
                elif motion_status == "failed":
                    normalized_stage = "failed"

                unified_result["motion_job"] = motion_state
                if motion_state.get("motion_file_url"):
                    unified_result["motion_file_url"] = motion_state.get("motion_file_url")
                if motion_state.get("video_url"):
                    unified_result["video_url"] = motion_state.get("video_url")

                async with TASK_STORE_LOCK:
                    next_status = "processing"
                    if motion_status == "failed":
                        next_status = "failed"
                    elif motion_status == "completed":
                        next_status = "completed"

                    TASK_STORE[task_id] = {
                        "task_id": task_id,
                        "status": next_status,
                        "progress_stage": normalized_stage,
                        "result": unified_result,
                        "error": motion_state.get("error"),
                    }

                if motion_status in {"completed", "failed"}:
                    final_status = "failed" if motion_status == "failed" else "completed"
                    final_stage = "failed" if motion_status == "failed" else "completed"
                    final_error = motion_state.get("error")
                    break

                await asyncio.sleep(1.5)

            final_motion_error = (unified_result.get("motion_job") or {}).get("error")
            if final_motion_error:
                logger.warning("Motion job %s completed with error: %s", motion_job_id, final_motion_error)

        if not motion_job_id:
            final_status = "completed"
            final_stage = "completed"

        async with TASK_STORE_LOCK:
            TASK_STORE[task_id] = {
                "task_id": task_id,
                "status": final_status,
                "progress_stage": final_stage,
                "result": unified_result,
                "error": final_error,
            }

        if final_status == "completed":
            motion_payload = unified_result.get("motion") or {}
            motion_prompt = unified_result.get("exercise_motion_prompt")
            motion_url = unified_result.get("motion_file_url")
            now_label = time.strftime("%H:%M")

            await _append_chat_history(
                request_user_id,
                [
                    {
                        "id": int(time.time() * 1000),
                        "role": "user",
                        "text": request_payload.query,
                        "motion": None,
                        "time": now_label,
                    },
                    {
                        "id": int(time.time() * 1000) + 1,
                        "role": "assistant",
                        "text": unified_result.get("text_answer", ""),
                        "motion": {
                            "label": motion_prompt or "Motion generated",
                            "motion_file_url": motion_url,
                            "prompt": motion_prompt,
                            "frames": motion_payload.get("frames"),
                            "fps": motion_payload.get("fps"),
                        }
                        if motion_url
                        else None,
                        "time": now_label,
                    },
                ],
            )
    except Exception as exc:
        logger.error("Unified task failed: %s", exc, exc_info=True)
        async with TASK_STORE_LOCK:
            TASK_STORE[task_id] = {
                "task_id": task_id,
                "status": "failed",
                "progress_stage": "failed",
                "result": None,
                "error": str(exc),
            }


@app.post("/process_query", response_model=UnifiedTaskResponse, summary="Submit unified query task")
async def process_query_unified(request_payload: QueryRequest, request: Request) -> UnifiedTaskResponse:
    """Create a query task and return the unified polling envelope."""
    task_id = str(uuid.uuid4())

    async with TASK_STORE_LOCK:
        TASK_STORE[task_id] = {
            "task_id": task_id,
            "status": "processing",
            "progress_stage": "queued",
            "result": None,
            "error": None,
        }

    asyncio.create_task(_run_query_task(task_id, request_payload, request))

    return UnifiedTaskResponse(
        task_id=task_id,
        status="processing",
        progress_stage="queued",
        result=None,
        error=None,
    )


@app.get("/tasks/{task_id}", response_model=UnifiedTaskResponse, summary="Get unified task status")
async def get_task_status(task_id: str, request: Request) -> UnifiedTaskResponse:
    """Poll unified task status with 2-3 defensive retries before 404."""
    state: Optional[Dict[str, Any]] = None
    for _ in range(3):
        async with TASK_STORE_LOCK:
            state = TASK_STORE.get(task_id)
        if state is not None:
            break
        await asyncio.sleep(0.2)

    if state is None:
        raise HTTPException(status_code=404, detail=f"Task not found: {task_id}")

    result = _normalize_enums(state.get("result")) if state.get("result") else None
    if isinstance(result, dict):
        result["motion_file_url"] = _to_absolute_url(result.get("motion_file_url"), request)

    return UnifiedTaskResponse(
        task_id=state.get("task_id", task_id),
        status=state.get("status", "processing"),
        progress_stage=state.get("progress_stage", "processing"),
        result=result,
        error=state.get("error"),
    )


@app.get("/history/{user_id}", response_model=ChatHistoryResponse, summary="Get file-backed chat history")
async def get_chat_history(user_id: str) -> ChatHistoryResponse:
    messages = await _read_chat_history(user_id)
    return ChatHistoryResponse(user_id=user_id, messages=messages)


@app.get("/download/{filename:path}", summary="Proxy DART artifacts through port 8000")
async def proxy_dart_download(filename: str) -> Response:
    dart_base = os.getenv("DART_PROXY_BASE_URL", "http://127.0.0.1:5001").rstrip("/")
    upstream_url = f"{dart_base}/download/{filename}"

    try:
        async with httpx.AsyncClient(timeout=httpx.Timeout(300.0, connect=10.0)) as client:
            upstream = await client.get(upstream_url)
    except Exception as exc:
        logger.error("DART download proxy failed for %s: %s", filename, exc)
        raise HTTPException(status_code=502, detail="Unable to reach DART download service")

    if upstream.status_code == 404:
        raise HTTPException(status_code=404, detail=f"Artifact not found: {filename}")
    if upstream.status_code >= 400:
        raise HTTPException(status_code=502, detail=f"DART returned {upstream.status_code}")

    forward_headers: Dict[str, str] = {}
    for key in ("content-disposition", "cache-control", "etag", "last-modified"):
        value = upstream.headers.get(key)
        if value:
            forward_headers[key] = value

    media_type = upstream.headers.get("content-type", "application/octet-stream")
    return Response(content=upstream.content, media_type=media_type, headers=forward_headers)


@app.post("/answer", summary="Proxy full-pipeline answer endpoint to orchestrator")
async def proxy_answer(request: Request) -> Response:
    """Accept /answer on port 8000 and forward to orchestrator /answer on 8080."""
    try:
        payload = await request.json()
    except Exception:
        raise HTTPException(status_code=400, detail="Invalid JSON body")

    if not isinstance(payload, dict):
        raise HTTPException(status_code=400, detail="Request body must be a JSON object")

    return await _proxy_main_api_request("POST", "/answer", json_body=payload)


@app.get("/answer/status/{request_id}", summary="Proxy full-pipeline status endpoint to orchestrator")
async def proxy_answer_status(request_id: str) -> Response:
    """Accept /answer/status on port 8000 and forward to orchestrator on 8080."""
    return await _proxy_main_api_request("GET", f"/answer/status/{request_id}")


@app.get("/job-status/{job_id}", response_model=MotionJobStatus, summary="Get async motion job status")
async def get_job_status(job_id: str, request: Request):
    """Return queue status for a motion job."""
    if api_instance is None:
        raise HTTPException(status_code=500, detail="API not initialized")

    status = api_instance.get_motion_job_status(job_id, _get_request_base_url(request))

    # Rewrite video_url to use the public base URL from the incoming request
    # so that URLs are correct when accessed through Ngrok or other reverse proxies.
    if status.get("video_url"):
        base = _get_request_base_url(request)
        relative = status["video_url"]
        # Strip any existing localhost base to get the relative path
        for prefix in ("http://localhost:8000", "http://127.0.0.1:8000"):
            if relative.startswith(prefix):
                relative = relative[len(prefix):]
                break
        status["video_url"] = f"{base}{relative}"

    if status.get("motion_file_url"):
        status["motion_file_url"] = _to_absolute_url(status.get("motion_file_url"), request)

    return status


@app.get("/health", summary="Health check with infrastructure status")
async def health_check() -> Dict[str, Any]:
    """Check connectivity of all infrastructure dependencies through Port 8000.

    Probes: Redis, ChromaDB, Celery workers, DART (5001), Orchestrator (8080).
    """
    import socket as _socket

    checks: Dict[str, Any] = {}

    # 1. Redis
    try:
        import redis as _redis
        r = _redis.Redis(
            host=os.getenv("REDIS_HOST", "localhost"),
            port=int(os.getenv("REDIS_PORT", "6379")),
            socket_connect_timeout=2,
        )
        r.ping()
        checks["redis"] = {"status": "ok"}
    except Exception as exc:
        checks["redis"] = {"status": "unreachable", "error": str(exc)}

    # 2. ChromaDB
    try:
        import chromadb
        client = chromadb.HttpClient(
            host=os.getenv("CHROMA_HOST", "localhost"),
            port=int(os.getenv("CHROMA_PORT", "8100")),
        )
        client.heartbeat()
        checks["chromadb"] = {"status": "ok"}
    except Exception:
        # ChromaDB may be running in-process (PersistentClient) which is fine
        if api_instance and hasattr(api_instance, "rag_pipeline"):
            checks["chromadb"] = {"status": "ok (in-process)"}
        else:
            checks["chromadb"] = {"status": "unreachable"}

    # 3. Celery Worker
    try:
        from celery_app import celery_app as _celery
        if _celery is not None:
            inspector = _celery.control.inspect(timeout=0.2)
            active = inspector.active_queues()
            checks["celery"] = {
                "status": "ok" if active else "no_workers",
                "workers": len(active) if active else 0,
            }
        else:
            checks["celery"] = {"status": "disabled"}
    except Exception as exc:
        checks["celery"] = {"status": "unreachable", "error": str(exc)}

    # 4. Internal services via TCP probe
    for name, host, port in [
        ("dart", "127.0.0.1", 5001),
        ("orchestrator", "127.0.0.1", 8080),
    ]:
        try:
            with _socket.socket(_socket.AF_INET, _socket.SOCK_STREAM) as s:
                s.settimeout(1.0)
                s.connect((host, port))
            checks[name] = {"status": "ok"}
        except Exception:
            checks[name] = {"status": "unreachable"}

    overall = "healthy" if all(
        c.get("status") in ("ok", "ok (in-process)", "disabled")
        for c in checks.values()
    ) else "degraded"

    return {"status": overall, "service": "agenticrag", "checks": checks}


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
            "POST /process_query": "Submit async query task and return task envelope",
            "GET /tasks/{task_id}": "Poll unified task state",
            "POST /answer": "Proxy to orchestrator /answer on port 8080",
            "GET /answer/status/{request_id}": "Proxy to orchestrator async status on port 8080",
            "GET /health": "Health check",
            "GET /info": "Service information",
        },
        "cors_allowed_origins": _allowed_origins,
    }


@app.get("/", include_in_schema=False)
async def root_redirect() -> RedirectResponse:
    """Open Official UI by default when available."""
    target = "/eca/" if os.path.isdir(_eca_ui_dir) else "/info"
    return RedirectResponse(url=target, status_code=307)


# ===========================
# Main
# ===========================

if __name__ == "__main__":
    host = os.getenv("AGENTICRAG_HOST", "0.0.0.0")
    port = _env_int("AGENTICRAG_PORT", 8000)
    logger.info("Starting AgenticRAG REST API server...")
    logger.info(f"Host: {host}  Port: {port}")
    uvicorn.run(
        app,
        host=host,
        port=port,
        log_level="info",
    )
