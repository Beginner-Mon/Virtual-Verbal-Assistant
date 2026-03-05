#!/usr/bin/env python3
"""RAG (Retrieval-Augmented Generation) Pipeline.

This module implements the complete RAG pipeline for context-aware response generation.

Changes vs original:
  - RateLimiter class (sliding window) throttles LLM calls
  - _assess_context_quality: weighted scoring by source_type instead of plain avg
  - _retrieve_context: error-isolated per-source retrieval (no single failure kills all)
  - _retry_generation: iterative loop instead of recursion (stack-safe)
  - Removed duplicate `from config import get_config` inside _retrieve_context
  - generate_response: retrieval + speculative web search run in parallel (ThreadPoolExecutor)
"""

import re
import json
import time
import threading
import concurrent.futures
from typing import List, Dict, Any, Optional

from memory.memory_manager import MemoryManager
from memory.embedding_service import EmbeddingService
from memory.document_store import DocumentStore
from memory.vector_store import VectorStore
from config import get_config
from utils.logger import get_logger
from utils.validators import ResponseValidator
from utils.gemini_client import GeminiClientWrapper
from utils.web_search import get_web_search_service
from utils.prompt_templates import QUERY_REFORMULATION_PROMPTS, REFLECTION_PROMPTS


logger = get_logger(__name__)


# ---------------------------------------------------------------------------
# Sliding-window rate limiter (ported + improved from index.ts RateLimiter)
# Thread-safe via Lock so it works if RAGPipeline is ever used concurrently.
# ---------------------------------------------------------------------------

class RateLimiter:
    """Sliding-window rate limiter for LLM API calls.

    Inspired by the RateLimiter in web-scout-mcp/src/index.ts.
    Original used JS Date objects; here we use time.monotonic() for precision.

    Args:
        requests_per_minute: Max calls allowed in any 60-second window.
    """

    def __init__(self, requests_per_minute: int = 20) -> None:
        self.requests_per_minute = requests_per_minute
        self._timestamps: List[float] = []
        self._lock = threading.Lock()

    def acquire(self) -> None:
        """Block until a request slot is available.

        Sleeps OUTSIDE the lock so other threads are not blocked while waiting.
        """
        while True:
            with self._lock:
                now = time.monotonic()
                # Evict timestamps older than 60 s
                self._timestamps = [t for t in self._timestamps if now - t < 60.0]

                if len(self._timestamps) < self.requests_per_minute:
                    # Slot available — claim it and return
                    self._timestamps.append(time.monotonic())
                    return

                # Calculate how long until the oldest slot expires
                oldest = self._timestamps[0]
                wait_seconds = 60.0 - (now - oldest) + 0.01  # small buffer

            # Sleep OUTSIDE the lock so other threads can proceed independently
            logger.debug(f"RateLimiter: sleeping {wait_seconds:.1f}s")
            time.sleep(wait_seconds)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

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


# ---------------------------------------------------------------------------
# Source-type weights for context quality scoring
# Documents are highest-trust; web_search injected at fixed 0.7 similarity
# so we down-weight it slightly; session summaries are aggregated / lossy.
# ---------------------------------------------------------------------------
_SOURCE_WEIGHTS: Dict[str, float] = {
    "document": 1.0,
    "conversation": 0.9,
    "web_search": 0.75,
    "session_summary": 0.8,
}


# ---------------------------------------------------------------------------
# RAGPipeline
# ---------------------------------------------------------------------------

class RAGPipeline:
    """Complete RAG pipeline for retrieval-augmented generation.

    Pipeline steps:
    1. Query processing and expansion
    2. Context retrieval from memory
    3. Context ranking and filtering
    4. Prompt construction with retrieved context
    5. LLM response generation
    6. Response validation
    """

    def __init__(
        self,
        memory_manager: MemoryManager = None,
        config=None,
        vector_store=None,
        embedding_service=None,
        client=None,
    ):
        """Initialize RAG pipeline.

        Args:
            memory_manager: MemoryManager instance. Creates new if None.
            config: Configuration object. If None, loads from default config.
            vector_store: Shared VectorStore instance. Creates new if None.
            embedding_service: Shared EmbeddingService instance. Creates new if None.
            client: Shared GeminiClientWrapper instance. Creates new if None.
        """
        # Load configs once — no repeated get_config() calls in methods
        _global_config = get_config()
        self.config = config or _global_config.rag
        self.llm_config = _global_config.llm
        self._rag_config = _global_config.rag  # for max_chunks_per_document etc.

        self.memory_manager = memory_manager or MemoryManager()
        self.embedding_service = embedding_service or EmbeddingService(_global_config.embedding)
        self.vector_store = vector_store or VectorStore(_global_config.vector_database)
        self.document_store = DocumentStore(self.vector_store, self.embedding_service)
        self.validator = ResponseValidator()

        # Web search service for fallback
        self.web_search = get_web_search_service(
            max_results=getattr(self.config, 'max_web_results', 5),
            timeout=getattr(self.config, 'web_search_timeout', 10),
        )
        self.enable_web_search = getattr(self.config, 'enable_web_search', True)
        self.min_context_threshold = getattr(self.config, 'min_context_threshold', 2)
        self.client = client or GeminiClientWrapper()

        # Rate limiter — protects Gemini API from burst calls across all LLM
        # methods (expansion, reformulation, generation, reflection).
        # Default 20 rpm matches Gemini free-tier guidance.
        self._rate_limiter = RateLimiter(
            requests_per_minute=getattr(self.llm_config, 'requests_per_minute', 20)
        )
        self._last_error: Optional[Exception] = None  # Set by _generate_llm_response on failure

        logger.info("Initialized RAGPipeline with Gemini and hybrid retrieval")

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def generate_response(
        self,
        query: str,
        user_id: str,
        conversation_history: Optional[List[Dict[str, str]]] = None,
        use_memory: bool = True,
    ) -> Dict[str, Any]:
        """Generate a response using RAG pipeline.

        Pipeline steps:
        1. Process / expand query
        2. Retrieve context (with query reformulation loop)
        3. Web search fallback
        4. Build prompt & generate LLM response
        5. Iterative reflection (self-correction against context)
        6. Validate safety/length

        Args:
            query: User query
            user_id: User identifier
            conversation_history: Previous conversation turns
            use_memory: Whether to retrieve memory context

        Returns:
            Dictionary with response and metadata
        """
        logger.info(f"Generating response for query: {query[:100]}...")

        # Step 1: Process query (expansion if enabled)
        processed_query = self._process_query(query)

        # Step 2: Retrieve context with query reformulation loop.
        # Simultaneously launch a speculative web search in the background so
        # that its network I/O overlaps with local retrieval (both are I/O-bound
        # and share no mutable state, so ThreadPoolExecutor is safe here).
        retrieved_context: List[Dict[str, Any]] = []
        web_search_used = False
        reformulation_count = 0
        current_query = processed_query
        ws_quality_threshold = getattr(self.config, 'web_search_quality_threshold', 0.65)

        # --- Speculative web search future (started immediately, used only if needed) ---
        _web_future: Optional[concurrent.futures.Future] = None
        _executor: Optional[concurrent.futures.ThreadPoolExecutor] = None
        if self.enable_web_search and self.web_search.is_available():
            _executor = concurrent.futures.ThreadPoolExecutor(max_workers=1)
            _web_future = _executor.submit(
                self.web_search.search_and_summarize, current_query
            )
            logger.debug("Speculative web search started in background")

        try:
            if use_memory:
                max_attempts = (
                    self.config.max_reformulation_attempts + 1
                    if self.config.enable_query_reformulation
                    else 1
                )

                for attempt in range(max_attempts):
                    retrieved_context = self._retrieve_context(
                        user_id=user_id,
                        query=current_query,
                    )

                    quality = self._assess_context_quality(retrieved_context)

                    if (quality >= self.config.reformulation_quality_threshold
                            or not self.config.enable_query_reformulation):
                        break

                    if attempt < self.config.max_reformulation_attempts:
                        new_query = self._reformulate_query(
                            original_query=query,
                            retrieved_context=retrieved_context,
                            attempt=attempt,
                        )
                        if new_query and new_query != current_query:
                            current_query = new_query
                            reformulation_count += 1
                            logger.info(
                                f"Reformulated query (attempt {attempt + 1}): {current_query[:100]}"
                            )
                        else:
                            break

            # Step 2.5: Decide whether to use speculative web search result.
            if _web_future is not None:
                context_quality = self._assess_context_quality(retrieved_context)
                too_few = len(retrieved_context) < self.min_context_threshold
                too_poor = context_quality < ws_quality_threshold and len(retrieved_context) > 0
                logger.info(
                    f"Web search check: quality={context_quality:.2f}, "
                    f"threshold={ws_quality_threshold}, count={len(retrieved_context)}, "
                    f"too_few={too_few}, too_poor={too_poor}"
                )

                if too_few or too_poor:
                    reason = (
                        f"too few results ({len(retrieved_context)}/{self.min_context_threshold})"
                        if too_few
                        else f"low quality (avg similarity {context_quality:.2f} < {ws_quality_threshold})"
                    )
                    logger.info(f"Context insufficient — {reason}, collecting web search result...")
                    try:
                        # Result is already computed (ran in parallel), so this
                        # .result() call returns almost immediately.
                        web_context = _web_future.result(timeout=getattr(self.config, 'web_search_timeout', 5))
                    except concurrent.futures.TimeoutError:
                        logger.warning("Speculative web search timed out — skipping")
                        web_context = ""
                    except Exception as exc:
                        logger.warning(f"Speculative web search failed: {exc}")
                        web_context = ""

                    if web_context:
                        retrieved_context.append({
                            "document": web_context,
                            "source_type": "web_search",
                            "similarity": 0.7,
                            "metadata": {"source": "DuckDuckGo"},
                        })
                        web_search_used = True
                        logger.info("Added parallel web search results to context")
                    else:
                        logger.info("Web search result empty — skipping")
                else:
                    # Context is good enough — cancel / discard the web search future.
                    _web_future.cancel()
                    logger.debug("Context sufficient — web search result discarded")

        finally:
            # Always shut down the executor to free the background thread.
            if _executor is not None:
                _executor.shutdown(wait=False)

        # Step 3: Build prompt with context
        prompt = self._build_prompt(
            query=query,
            retrieved_context=retrieved_context,
            conversation_history=conversation_history,
        )

        # Step 4: Generate response
        response_text = self._generate_llm_response(prompt)

        # Step 4a: Iterative reflection — self-correct against retrieved facts
        reflection_used = False
        if (self.config.enable_iterative_reflection
                and retrieved_context
                and not response_text.startswith("[ERROR]")):
            for iteration in range(self.config.max_reflection_iterations):
                reflection = self._reflect_on_response(
                    query=query,
                    retrieved_context=retrieved_context,
                    draft_response=response_text,
                    iteration=iteration,
                )
                if reflection["is_grounded"]:
                    logger.info(f"Reflection pass {iteration + 1}: response is grounded")
                    break
                logger.info(
                    f"Reflection pass {iteration + 1}: self-correcting ({reflection['issues']})"
                )
                response_text = reflection["revised_answer"]
                reflection_used = True

        # Step 5: Validate response
        validation_result = self.validator.validate_response(
            response=response_text,
            query=query,
        )

        # Step 6: Handle validation failures
        if not validation_result["is_valid"] and self.llm_config.enable_validation:
            logger.warning(f"Response failed validation: {validation_result['issues']}")
            if self.llm_config.max_retries > 0:
                response_text = self._retry_generation(
                    prompt=prompt,
                    validation_issues=validation_result["issues"],
                )

        result = {
            "response": response_text,
            "metadata": {
                "retrieved_context_count": len(retrieved_context),
                "validation": validation_result,
                "query_processed": processed_query != query,
                "web_search_used": web_search_used,
                "reformulation_count": reformulation_count,
                "reflection_used": reflection_used,
            },
        }

        # NOTE: Interaction storage is handled by the UI layer (process_user_query)
        # which stores with richer metadata (active documents, source info).
        # Do NOT store here to avoid duplicate entries in the vector DB.

        logger.info("Response generation complete")
        return result

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _process_query(self, query: str) -> str:
        """Process and optionally expand query.

        Args:
            query: Original query

        Returns:
            Processed query
        """
        if not self.config.enable_query_expansion:
            return query

        if self.config.query_expansion_method == "llm":
            expansion_prompt = (
                f"Given this query, suggest 2-3 related keywords or phrases that would help "
                f"find relevant information:\n\nQuery: {query}\n\n"
                f"Return only the expanded keywords, comma-separated."
            )
            try:
                # Throttle before every LLM call
                self._rate_limiter.acquire()
                response = self.client.chat.completions.create(
                    model=self.llm_config.model,
                    messages=[{"role": "user", "content": expansion_prompt}],
                    temperature=0.3,
                    max_tokens=50,
                )
                expanded_terms = response.choices[0].message.content.strip()
                processed = f"{query} {expanded_terms}"
                logger.debug(f"Expanded query: {processed}")
                return processed
            except Exception as e:
                logger.warning(f"Query expansion failed: {e}")
                return query

        return query

    def _retrieve_context(
        self,
        user_id: str,
        query: str,
    ) -> List[Dict[str, Any]]:
        """Retrieve relevant context — hybrid: memory + documents + session summaries.

        Improvement over original: each source is fetched independently with its own
        try/except so a failure in one source never silently kills the others.
        (Inspired by per-URL error isolation in index.ts fetchMultipleUrls.)

        Args:
            user_id: User identifier
            query: Query text

        Returns:
            List of relevant context items sorted by weighted similarity
        """
        all_results: List[Dict[str, Any]] = []

        # --- Conversation memory ---
        try:
            memory_results = self.memory_manager.retrieve_relevant_memory(
                user_id=user_id,
                query=query,
                top_k=self.config.top_k_documents,
            )
            for r in memory_results:
                r["source_type"] = "conversation"
            all_results.extend(memory_results)
        except Exception as exc:
            logger.warning(f"Conversation memory retrieval skipped: {exc}")

        # --- Document store ---
        try:
            max_chunks_per_doc = getattr(self._rag_config, 'max_chunks_per_document', 3)
            document_results = self.document_store.search_documents(
                query=query,
                user_id=user_id,
                top_k=self.config.top_k_documents,
                max_chunks_per_document=max_chunks_per_doc,
            )
            for r in document_results:
                r["source_type"] = "document"
                if "metadata" in r:
                    r["filename"] = r["metadata"].get("filename", "unknown")
            all_results.extend(document_results)
        except Exception as exc:
            logger.warning(f"Document store retrieval skipped: {exc}")

        # --- Session summaries ---
        try:
            session_results = self.memory_manager.retrieve_session_context(
                user_id=user_id,
                query=query,
                top_k=3,
            )
            for r in session_results:
                r["source_type"] = "session_summary"
            all_results.extend(session_results)
        except Exception as exc:
            logger.warning(f"Session summary retrieval skipped: {exc}")

        # Filter by similarity threshold
        filtered = [
            r for r in all_results
            if r.get("similarity", 0) >= self.config.similarity_threshold
        ]

        # Sort by weighted similarity (source quality × cosine similarity)
        filtered.sort(
            key=lambda x: _SOURCE_WEIGHTS.get(x.get("source_type", ""), 1.0)
                          * x.get("similarity", 0),
            reverse=True,
        )

        filtered = filtered[:self.config.top_k_documents]

        doc_count     = sum(1 for r in filtered if r.get("source_type") == "document")
        conv_count    = sum(1 for r in filtered if r.get("source_type") == "conversation")
        session_count = sum(1 for r in filtered if r.get("source_type") == "session_summary")
        logger.info(
            f"Retrieved {len(filtered)} items: {doc_count} documents, "
            f"{conv_count} conversations, {session_count} session summaries"
        )
        return filtered

    def _assess_context_quality(self, retrieved_context: List[Dict[str, Any]]) -> float:
        """Compute weighted average similarity of retrieved context.

        Improvement over original: plain avg treated all sources equally.
        Documents are higher-trust than web results, so they should count more
        toward the "is this good enough?" decision.

        Args:
            retrieved_context: List of retrieved items with 'similarity' scores

        Returns:
            Weighted average similarity score (0.0 if no context)
        """
        if not retrieved_context:
            return 0.0

        total_weight = 0.0
        weighted_sum = 0.0
        for item in retrieved_context:
            w = _SOURCE_WEIGHTS.get(item.get("source_type", ""), 1.0)
            weighted_sum += w * item.get("similarity", 0.0)
            total_weight += w

        score = weighted_sum / total_weight if total_weight > 0 else 0.0
        logger.debug(
            f"Context quality (weighted): {score:.3f} over {len(retrieved_context)} items"
        )
        return score

    def _reformulate_query(
        self,
        original_query: str,
        retrieved_context: List[Dict[str, Any]],
        attempt: int,
    ) -> str:
        """Use LLM to rewrite the query for better retrieval.

        Args:
            original_query: The user's original question
            retrieved_context: Current (poor) retrieval results
            attempt: Current reformulation attempt number

        Returns:
            Reformulated query string
        """
        context_lines = []
        for item in retrieved_context[:5]:
            text_preview = item.get("document", "")[:150]
            sim = item.get("similarity", 0.0)
            context_lines.append(f"  - (sim={sim:.2f}) {text_preview}")
        context_summary = "\n".join(context_lines) if context_lines else "  (no results)"

        avg_similarity = self._assess_context_quality(retrieved_context)

        prompt = QUERY_REFORMULATION_PROMPTS["reformulate"].format(
            query=original_query,
            context_summary=context_summary,
            avg_similarity=avg_similarity,
        )

        try:
            self._rate_limiter.acquire()
            response = self.client.chat.completions.create(
                model=self.llm_config.model,
                messages=[{"role": "user", "content": prompt}],
                temperature=0.4,
                max_tokens=200,
            )
            reformulated = response.choices[0].message.content.strip().strip('"').strip("'")
            logger.info(
                f"Query reformulation attempt {attempt + 1}: "
                f"'{original_query[:60]}' -> '{reformulated[:60]}'"
            )
            return reformulated
        except Exception as e:
            logger.warning(f"Query reformulation failed: {e}")
            return original_query

    def _reflect_on_response(
        self,
        query: str,
        retrieved_context: List[Dict[str, Any]],
        draft_response: str,
        iteration: int,
    ) -> Dict[str, Any]:
        """Use LLM to check if the draft response is grounded in retrieved context.

        Args:
            query: Original user question
            retrieved_context: Retrieved context items
            draft_response: The generated response to verify
            iteration: Current reflection iteration

        Returns:
            Dict with 'is_grounded', 'issues', and 'revised_answer'
        """
        context_parts = []
        for item in retrieved_context:
            source = item.get("source_type", "unknown")
            text = item.get("document", "")[:500]
            sim = item.get("similarity", 0.0)
            filename = item.get("filename", "")
            label = f"[{source}" + (f": {filename}" if filename else "") + f", sim={sim:.2f}]"
            context_parts.append(f"{label}\n{text}")
        context_text = "\n\n".join(context_parts)

        prompt = REFLECTION_PROMPTS["reflect"].format(
            query=query,
            context=context_text,
            draft_answer=draft_response,
        )

        try:
            self._rate_limiter.acquire()
            response = self.client.chat.completions.create(
                model=self.llm_config.model,
                messages=[{"role": "user", "content": prompt}],
                temperature=0.1,
                max_tokens=self.llm_config.max_tokens,
                response_format={"type": "json_object"},
            )
            response_text = response.choices[0].message.content.strip()
            cleaned = clean_json_response(response_text)
            result = json.loads(cleaned)
            return {
                "is_grounded": result.get("is_grounded", True),
                "issues": result.get("issues", []),
                "revised_answer": result.get("revised_answer", draft_response),
            }
        except Exception as e:
            logger.warning(f"Reflection failed (iteration {iteration}): {e}")
            return {
                "is_grounded": True,
                "issues": [],
                "revised_answer": draft_response,
            }

    def _build_prompt(
        self,
        query: str,
        retrieved_context: List[Dict[str, Any]],
        conversation_history: Optional[List[Dict[str, str]]],
    ) -> str:
        """Build prompt with retrieved context and conversation history.

        Args:
            query: User query
            retrieved_context: Retrieved context items
            conversation_history: Previous conversation

        Returns:
            Complete prompt string
        """
        prompt_parts = []

        if retrieved_context:
            if self.config.include_metadata:
                documents = [i for i in retrieved_context if i.get("source_type") == "document"]
                conversations = [i for i in retrieved_context if i.get("source_type") == "conversation"]
                web_results = [i for i in retrieved_context if i.get("source_type") == "web_search"]
                session_summaries = [i for i in retrieved_context if i.get("source_type") == "session_summary"]

                context_sections = []

                if documents:
                    doc_content = []
                    doc_groups: Dict[str, list] = {}
                    for item in documents:
                        fname = item.get("filename", "document")
                        doc_groups.setdefault(fname, []).append(item)

                    for filename, chunks in doc_groups.items():
                        chunks.sort(key=lambda x: (
                            x.get("metadata", {}).get("start_position", 0),
                            -x.get("similarity", 0),
                        ))
                        chunk_info = chunks[0]
                        similarity = chunk_info.get("similarity", 0)
                        total_chunks = chunk_info.get("metadata", {}).get("total_chunks", 1)

                        doc_header = f"📄 [{filename}] (relevance: {similarity:.2f}"
                        if total_chunks > 1:
                            doc_header += f", {len(chunks)} chunks"
                        doc_header += "):"
                        doc_content.append(doc_header)

                        all_chunk_text = []
                        for item in chunks:
                            text = item.get("document", "")
                            query_words = set(query.lower().split())
                            text_sentences = text.split('. ')
                            relevant_sentences = [
                                s for s in text_sentences
                                if query_words.intersection(set(s.lower().split()))
                            ]
                            if relevant_sentences:
                                relevant_text = '. '.join(relevant_sentences[:3])
                            else:
                                relevant_text = text[:800]
                            all_chunk_text.append(relevant_text.strip())

                        doc_content.append(f"  {' [...] '.join(all_chunk_text)}")
                    context_sections.append("\n".join(doc_content))

                if conversations:
                    conv_content = [
                        f"  • {item.get('document', '')[:200]}..."
                        for item in conversations
                    ]
                    context_sections.append("💬 CONVERSATION HISTORY:\n" + "\n".join(conv_content))

                if web_results:
                    web_content = [f"  {item.get('document', '')}" for item in web_results]
                    context_sections.append("🌐 WEB SEARCH RESULTS:\n" + "\n".join(web_content))
                    context_sections.append(
                        "⚠️ INSTRUCTION: The above web search results are REAL and RELEVANT. "
                        "You MUST use them to build a comprehensive, helpful answer. "
                        "Include source URLs in your response. "
                        "Do NOT say you don't have information — the web results ARE your information source."
                    )

                if session_summaries:
                    summary_content = [
                        f"  • {item.get('document', '')[:300]}"
                        for item in session_summaries
                    ]
                    context_sections.append("🧠 PAST SESSION CONTEXT:\n" + "\n".join(summary_content))

                if context_sections:
                    context_text = "\n\n".join(context_sections)
                    prompt_parts.append(f"=== RETRIEVED CONTEXT ===\n{context_text}")
            else:
                context_text = "\n\n".join(
                    f"• {item.get('document', '')}" for item in retrieved_context
                )
                prompt_parts.append(f"Relevant context:\n{context_text}")

        if conversation_history:
            history_text = "\n".join(
                f"{turn['role'].capitalize()}: {turn['content']}"
                for turn in conversation_history[-5:]
            )
            prompt_parts.append(f"\n=== CONVERSATION HISTORY ===\n{history_text}")

        prompt_parts.append(f"\n=== CURRENT QUESTION ===\n{query}")
        return "\n".join(prompt_parts)

    def _generate_llm_response(self, prompt: str) -> str:
        """Generate response using LLM.

        Args:
            prompt: Complete prompt with context

        Returns:
            Generated response text
        """
        try:
            self._rate_limiter.acquire()
            response = self.client.chat.completions.create(
                model=self.llm_config.model,
                messages=[
                    {"role": "system", "content": self.llm_config.system_prompt},
                    {"role": "user", "content": prompt},
                ],
                temperature=self.llm_config.temperature,
                max_tokens=self.llm_config.max_tokens,
            )
            response_text = response.choices[0].message.content.strip()
            logger.debug(f"Generated response: {response_text[:100]}...")
            return response_text

        except Exception as e:
            logger.error(
                f"Error generating LLM response: {type(e).__name__}: {e}", exc_info=True
            )
            self._last_error = e
            return (
                f"[ERROR] Failed to generate response. Details: {type(e).__name__}: {e}. "
                f"Please check your Gemini API key and internet connection."
            )

    def _retry_generation(
        self,
        prompt: str,
        validation_issues: List[str],
    ) -> str:
        """Retry response generation with validation feedback (iterative, not recursive).

        Original used recursion — converted to a loop to avoid stack overflow when
        max_retries is large (same motivation as index.ts batch loop over URLs).

        Args:
            prompt: Original prompt
            validation_issues: List of validation issues from first attempt

        Returns:
            Best response text after retries
        """
        current_issues = validation_issues

        for retry_count in range(self.llm_config.max_retries):
            feedback = "\n".join(f"- {issue}" for issue in current_issues)
            retry_prompt = (
                f"{prompt}\n\nPrevious response had these issues:\n{feedback}\n\n"
                f"Please provide a new response that addresses these concerns."
            )

            response_text = self._generate_llm_response(retry_prompt)
            validation = self.validator.validate_response(response_text, prompt)

            if validation["is_valid"]:
                return response_text

            current_issues = validation["issues"]
            logger.debug(f"Retry {retry_count + 1}/{self.llm_config.max_retries} still invalid")

        # All retries exhausted
        logger.warning("Max retries reached")
        if hasattr(self, '_last_error') and "rate limit" in str(self._last_error).lower():
            return (
                "[RATE_LIMITED] The Gemini API rate limit has been exceeded. "
                "This system is free-tier and has a limit of 20 requests per day. "
                "Please try again tomorrow or upgrade to a paid plan at https://ai.google.dev"
            )
        return "I apologize, but I'm having difficulty providing an appropriate response. Could you rephrase your question?"


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    rag = RAGPipeline()

    user_id = "test_user_123"

    rag.memory_manager.store_user_context(
        user_id=user_id,
        context_type="physical_context",
        content="User has neck pain from desk work, prefers exercises without equipment",
    )

    query = "What exercises can help my neck pain?"
    result = rag.generate_response(
        query=query,
        user_id=user_id,
        use_memory=True,
    )

    print("Response:", result["response"])
    print("\nMetadata:", result["metadata"])