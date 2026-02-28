"""Central pipeline orchestrator for coordinating three-service integration.

This module orchestrates the complete pipeline:
1. AgenticRAG processes the query and extracts motion/voice prompts
2. Parallel requests to SpeechLLm and DART
3. Aggregates results into unified response
"""

import asyncio
import logging
import time
from typing import Optional, Dict, Any
from dataclasses import dataclass
from pathlib import Path

import httpx

from utils.logger import get_logger

logger = get_logger(__name__)


# ===========================
# Data Models
# ===========================


@dataclass
class PipelineResult:
    """Complete pipeline result."""

    text_answer: str
    voice_file: Optional[str] = None
    voice_duration: Optional[float] = None
    motion_file: Optional[str] = None
    motion_frames: Optional[int] = None
    motion_fps: Optional[int] = None
    generation_time_ms: float = 0.0
    errors: Dict[str, str] = None  # Service errors if any

    def __post_init__(self):
        if self.errors is None:
            self.errors = {}


# ===========================
# Configuration
# ===========================

DEFAULT_CONFIG = {
    "agenticrag_url": "http://localhost:8000",
    "speechllm_url": "http://localhost:5000",
    "dart_url": "http://localhost:5001",
    "timeout_seconds": 5.0,
    "service_timeout_seconds": 4.0,  # Max 4s per service, leaving 1s for aggregation
    "retry_count": 1,
}


# ===========================
# Pipeline Orchestrator
# ===========================


class PipelineOrchestrator:
    """Orchestrates the three-service pipeline."""

    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """Initialize pipeline orchestrator.

        Args:
            config: Configuration dictionary with service URLs and timeouts
        """
        self.config = {**DEFAULT_CONFIG, **(config or {})}
        logger.info(
            f"Pipeline Orchestrator initialized with config: {self._sanitize_config()}"
        )

    def _sanitize_config(self) -> Dict[str, Any]:
        """Return config without sensitive data."""
        return {k: v for k, v in self.config.items() if k != "api_keys"}

    async def process_query(
        self,
        query: str,
        user_id: str = "default",
        conversation_history: Optional[list] = None,
    ) -> PipelineResult:
        """Process a user query through the complete pipeline.

        Args:
            query: User query text
            user_id: User identifier
            conversation_history: Previous conversation turns

        Returns:
            PipelineResult with all outputs
        """
        start_time = time.time()
        logger.info(f"[{user_id}] Starting pipeline for query: {query[:100]}...")

        result = PipelineResult(text_answer="")

        try:
            async with httpx.AsyncClient(timeout=self.config["timeout_seconds"]) as client:
                # Step 1: Call AgenticRAG
                logger.info(f"[{user_id}] Step 1: Calling AgenticRAG...")
                rag_response = await self._call_agenticrag(client, query, user_id, conversation_history)

                if not rag_response:
                    result.text_answer = "Error: Could not process query"
                    result.generation_time_ms = (time.time() - start_time) * 1000
                    return result

                result.text_answer = rag_response.get("text_answer", "")
                motion_prompt = rag_response.get("motion_prompt")
                voice_prompt = rag_response.get("voice_prompt")

                logger.info(f"[{user_id}] Step 1 complete: {result.text_answer[:100]}...")

                # Step 2: Parallel calls to SpeechLLm and DART
                logger.info(f"[{user_id}] Step 2: Calling SpeechLLm and DART in parallel...")

                tasks = []

                # Add SpeechLLm task if we have voice prompt
                if voice_prompt:
                    tasks.append(
                        self._call_speechllm(
                            client,
                            voice_prompt,
                        )
                    )
                else:
                    tasks.append(asyncio.sleep(0))

                # Add DART task if we have motion prompt
                if motion_prompt:
                    tasks.append(
                        self._call_dart(
                            client,
                            motion_prompt,
                        )
                    )
                else:
                    tasks.append(asyncio.sleep(0))

                # Execute tasks in parallel with timeout
                try:
                    responses = await asyncio.wait_for(
                        asyncio.gather(*tasks, return_exceptions=True),
                        timeout=self.config["service_timeout_seconds"],
                    )

                    # Process SpeechLLm response
                    if voice_prompt and responses[0] and not isinstance(responses[0], Exception):
                        tts_response = responses[0]
                        result.voice_file = tts_response.get("audio_file")
                        result.voice_duration = tts_response.get("duration_seconds")
                        logger.info(f"[{user_id}] SpeechLLm complete: {result.voice_file}")
                    elif responses[0] and isinstance(responses[0], Exception):
                        result.errors["speechllm"] = str(responses[0])
                        logger.warning(f"[{user_id}] SpeechLLm error: {responses[0]}")

                    # Process DART response
                    if motion_prompt and responses[1] and not isinstance(responses[1], Exception):
                        motion_response = responses[1]
                        result.motion_file = motion_response.get("motion_file")
                        result.motion_frames = motion_response.get("num_frames")
                        result.motion_fps = motion_response.get("fps", 30)
                        logger.info(f"[{user_id}] DART complete: {result.motion_file}")
                    elif responses[1] and isinstance(responses[1], Exception):
                        result.errors["dart"] = str(responses[1])
                        logger.warning(f"[{user_id}] DART error: {responses[1]}")

                except asyncio.TimeoutError:
                    logger.warning(
                        f"[{user_id}] Parallel services timed out after {self.config['service_timeout_seconds']}s"
                    )
                    result.errors["timeout"] = "Parallel service execution timed out"

        except Exception as e:
            logger.error(f"[{user_id}] Pipeline error: {e}")
            result.errors["pipeline"] = str(e)

        # Calculate total time
        result.generation_time_ms = (time.time() - start_time) * 1000

        logger.info(
            f"[{user_id}] Pipeline complete in {result.generation_time_ms:.1f}ms. "
            f"Text: ✓ Voice: {'✓' if result.voice_file else '✗'} Motion: {'✓' if result.motion_file else '✗'}"
        )

        return result

    async def _call_agenticrag(
        self,
        client: httpx.AsyncClient,
        query: str,
        user_id: str,
        conversation_history: Optional[list],
    ) -> Optional[Dict[str, Any]]:
        """Call AgenticRAG API.

        Args:
            client: AsyncHTTP client
            query: User query
            user_id: User ID
            conversation_history: Conversation history

        Returns:
            Response dictionary or None on error
        """
        try:
            url = f"{self.config['agenticrag_url']}/query"

            payload = {
                "query": query,
                "user_id": user_id,
                "conversation_history": conversation_history or [],
            }

            logger.debug(f"Calling AgenticRAG: POST {url}")

            response = await client.post(url, json=payload)
            response.raise_for_status()

            data = response.json()
            logger.debug(f"AgenticRAG response: {list(data.keys())}")

            return data

        except httpx.TimeoutException:
            logger.error("AgenticRAG request timed out")
            return None
        except Exception as e:
            logger.error(f"AgenticRAG request failed: {e}")
            return None

    async def _call_speechllm(
        self,
        client: httpx.AsyncClient,
        voice_prompt: Dict[str, Any],
    ) -> Optional[Dict[str, Any]]:
        """Call SpeechLLm API.

        Args:
            client: AsyncHTTP client
            voice_prompt: Voice prompt from AgenticRAG

        Returns:
            Response dictionary or None on error
        """
        try:
            url = f"{self.config['speechllm_url']}/synthesize"

            payload = {
                "text": voice_prompt.get("text", ""),
                "emotion": voice_prompt.get("emotion"),
                "user_id": "pipeline",
            }

            logger.debug(f"Calling SpeechLLm: POST {url}")

            response = await client.post(
                url,
                json=payload,
                timeout=self.config["service_timeout_seconds"],
            )
            response.raise_for_status()

            data = response.json()
            logger.debug(f"SpeechLLm response: {list(data.keys())}")

            return data

        except httpx.TimeoutException:
            logger.error("SpeechLLm request timed out")
            raise
        except Exception as e:
            logger.error(f"SpeechLLm request failed: {e}")
            raise

    async def _call_dart(
        self,
        client: httpx.AsyncClient,
        motion_prompt: Dict[str, Any],
    ) -> Optional[Dict[str, Any]]:
        """Call DART API.

        Args:
            client: AsyncHTTP client
            motion_prompt: Motion prompt from AgenticRAG

        Returns:
            Response dictionary or None on error
        """
        try:
            url = f"{self.config['dart_url']}/generate_motion"

            # Parse primitive sequence to extract num_primitives
            primitive_seq = motion_prompt.get("primitive_sequence", "walk*20")
            num_primitives = motion_prompt.get("num_frames", 160) // 8

            payload = {
                "text_prompt": motion_prompt.get("description", "walk"),
                "num_primitives": num_primitives,
                "guidance_scale": 5.0,
                "num_steps": 10,
            }

            logger.debug(f"Calling DART: POST {url}")

            response = await client.post(
                url,
                json=payload,
                timeout=self.config["service_timeout_seconds"],
            )
            response.raise_for_status()

            data = response.json()
            logger.debug(f"DART response: {list(data.keys())}")

            return data

        except httpx.TimeoutException:
            logger.error("DART request timed out")
            raise
        except Exception as e:
            logger.error(f"DART request failed: {e}")
            raise

    def process_query_sync(
        self,
        query: str,
        user_id: str = "default",
        conversation_history: Optional[list] = None,
    ) -> PipelineResult:
        """Synchronous wrapper for process_query.

        Use this if not in an async context.

        Args:
            query: User query text
            user_id: User identifier
            conversation_history: Previous conversation turns

        Returns:
            PipelineResult with all outputs
        """
        try:
            loop = asyncio.get_event_loop()
            if loop.is_running():
                # Already in async context, need to create new loop or use run_in_executor
                import concurrent.futures
                with concurrent.futures.ThreadPoolExecutor() as executor:
                    future = executor.submit(
                        asyncio.run,
                        self.process_query(query, user_id, conversation_history),
                    )
                    return future.result()
            else:
                return asyncio.run(self.process_query(query, user_id, conversation_history))
        except RuntimeError:
            # No event loop, create one
            return asyncio.run(self.process_query(query, user_id, conversation_history))


# ===========================
# Utility Functions
# ===========================


def format_pipeline_result(result: PipelineResult) -> Dict[str, Any]:
    """Format pipeline result for JSON response.

    Args:
        result: PipelineResult object

    Returns:
        JSON-serializable dictionary
    """
    return {
        "text_answer": result.text_answer,
        "voice": (
            {
                "file": result.voice_file,
                "duration_seconds": result.voice_duration,
            }
            if result.voice_file
            else None
        ),
        "motion": (
            {
                "file": result.motion_file,
                "num_frames": result.motion_frames,
                "fps": result.motion_fps,
            }
            if result.motion_file
            else None
        ),
        "generation_time_ms": result.generation_time_ms,
        "errors": result.errors if result.errors else None,
    }
