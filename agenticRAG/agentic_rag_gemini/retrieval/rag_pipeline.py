"""RAG (Retrieval-Augmented Generation) Pipeline.

This module implements the complete RAG pipeline for context-aware response generation.
"""

import re
import json
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


def clean_json_response(response_text: str) -> str:
    """Clean and normalize JSON response from LLM.
    
    Removes markdown code blocks, fixes common formatting issues.
    
    Args:
        response_text: Raw LLM response
        
    Returns:
        Cleaned JSON string
    """
    # Remove markdown code blocks
    response_text = re.sub(r'```json\s*', '', response_text)
    response_text = re.sub(r'```\s*', '', response_text)
    
    # Strip whitespace
    response_text = response_text.strip()
    
    # Try to extract JSON if wrapped in text
    json_match = re.search(r'\{[^{}]*(?:\{[^{}]*\}[^{}]*)*\}', response_text, re.DOTALL)
    if json_match:
        response_text = json_match.group(0)
    
    return response_text


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
    
    def __init__(self, memory_manager: MemoryManager = None, config=None,
                 vector_store=None, embedding_service=None, client=None):
        """Initialize RAG pipeline.
        
        Args:
            memory_manager: MemoryManager instance. Creates new if None.
            config: Configuration object. If None, loads from default config.
            vector_store: Shared VectorStore instance. Creates new if None.
            embedding_service: Shared EmbeddingService instance. Creates new if None.
            client: Shared GeminiClientWrapper instance. Creates new if None.
        """
        self.config = config or get_config().rag
        self.llm_config = get_config().llm
        
        self.memory_manager = memory_manager or MemoryManager()
        self.embedding_service = embedding_service or EmbeddingService(get_config().embedding)
        self.vector_store = vector_store or VectorStore(get_config().vector_database)
        self.document_store = DocumentStore(self.vector_store, self.embedding_service)
        self.validator = ResponseValidator()
        
        # Web search service for fallback
        self.web_search = get_web_search_service(
            max_results=getattr(self.config, 'max_web_results', 5),
            timeout=getattr(self.config, 'web_search_timeout', 10)
        )
        self.enable_web_search = getattr(self.config, 'enable_web_search', True)
        self.min_context_threshold = getattr(self.config, 'min_context_threshold', 2)
        self.client = client or GeminiClientWrapper()
        
        logger.info("Initialized RAGPipeline with Gemini and hybrid retrieval")
    
    def generate_response(
        self,
        query: str,
        user_id: str,
        conversation_history: Optional[List[Dict[str, str]]] = None,
        use_memory: bool = True
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
        
        # Step 2: Retrieve context with query reformulation loop
        retrieved_context = []
        web_search_used = False
        reformulation_count = 0
        current_query = processed_query
        
        if use_memory:
            max_attempts = (
                self.config.max_reformulation_attempts + 1
                if self.config.enable_query_reformulation
                else 1
            )
            
            for attempt in range(max_attempts):
                retrieved_context = self._retrieve_context(
                    user_id=user_id,
                    query=current_query
                )
                
                quality = self._assess_context_quality(retrieved_context)
                
                # Good enough or reformulation disabled — stop
                if (quality >= self.config.reformulation_quality_threshold
                        or not self.config.enable_query_reformulation):
                    break
                
                # Still have reformulation budget — rewrite query
                if attempt < self.config.max_reformulation_attempts:
                    new_query = self._reformulate_query(
                        original_query=query,
                        retrieved_context=retrieved_context,
                        attempt=attempt
                    )
                    if new_query and new_query != current_query:
                        current_query = new_query
                        reformulation_count += 1
                        logger.info(f"Reformulated query (attempt {attempt + 1}): {current_query[:100]}")
                    else:
                        break  # reformulation returned same text, no point retrying
        
        # Step 2.5: Web search fallback if context insufficient OR low quality
        if self.enable_web_search and self.web_search.is_available():
            context_quality = self._assess_context_quality(retrieved_context)
            ws_quality_threshold = getattr(self.config, 'web_search_quality_threshold', 0.65)
            too_few = len(retrieved_context) < self.min_context_threshold
            too_poor = context_quality < ws_quality_threshold and len(retrieved_context) > 0
            logger.info(f"Web search check: quality={context_quality:.2f}, threshold={ws_quality_threshold}, count={len(retrieved_context)}, too_few={too_few}, too_poor={too_poor}")

            if too_few or too_poor:
                reason = (
                    f"too few results ({len(retrieved_context)}/{self.min_context_threshold})"
                    if too_few
                    else f"low quality (avg similarity {context_quality:.2f} < {ws_quality_threshold})"
                )
                logger.info(f"Context insufficient — {reason}, trying web search...")
                web_context = self.web_search.search_and_summarize(current_query)
                if web_context:
                    retrieved_context.append({
                        "document": web_context,
                        "source_type": "web_search",
                        "similarity": 0.7,
                        "metadata": {"source": "DuckDuckGo"}
                    })
                    web_search_used = True
                    logger.info("Added web search results to context")
        
        # Step 3: Build prompt with context
        prompt = self._build_prompt(
            query=query,
            retrieved_context=retrieved_context,
            conversation_history=conversation_history
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
                    iteration=iteration
                )
                if reflection["is_grounded"]:
                    logger.info(f"Reflection pass {iteration + 1}: response is grounded")
                    break
                logger.info(f"Reflection pass {iteration + 1}: self-correcting ({reflection['issues']})")
                response_text = reflection["revised_answer"]
                reflection_used = True
        
        # Step 5: Validate response (safety / length)
        validation_result = self.validator.validate_response(
            response=response_text,
            query=query
        )
        
        # Step 6: Handle validation failures
        if not validation_result["is_valid"] and self.llm_config.enable_validation:
            logger.warning(f"Response failed validation: {validation_result['issues']}")
            if self.llm_config.max_retries > 0:
                response_text = self._retry_generation(
                    prompt=prompt,
                    validation_issues=validation_result["issues"]
                )
        
        # Build result
        result = {
            "response": response_text,
            "metadata": {
                "retrieved_context_count": len(retrieved_context),
                "validation": validation_result,
                "query_processed": processed_query != query,
                "web_search_used": web_search_used,
                "reformulation_count": reformulation_count,
                "reflection_used": reflection_used
            }
        }
        
        # NOTE: Interaction storage is handled by the UI layer (process_user_query)
        # which stores with richer metadata (active documents, source info).
        # Do NOT store here to avoid duplicate entries in the vector DB.
        
        logger.info("Response generation complete")
        return result
    
    def _process_query(self, query: str) -> str:
        """Process and optionally expand query.
        
        Args:
            query: Original query
            
        Returns:
            Processed query
        """
        if not self.config.enable_query_expansion:
            return query
        
        # Simple query expansion (in production, use more sophisticated methods)
        # This could use LLM, WordNet, or embedding-based expansion
        
        if self.config.query_expansion_method == "llm":
            # Use LLM to expand query
            expansion_prompt = f"""Given this query, suggest 2-3 related keywords or phrases that would help find relevant information:

Query: {query}

Return only the expanded keywords, comma-separated."""
            
            try:
                response = self.client.chat.completions.create(
                    model=self.llm_config.model,
                    messages=[{"role": "user", "content": expansion_prompt}],
                    temperature=0.3,
                    max_tokens=50
                )
                
                expanded_terms = response.choices[0].message.content.strip()
                processed = f"{query} {expanded_terms}"
                
                logger.debug(f"Expanded query: {processed}")
                return processed
                
            except Exception as e:
                logger.warning(f"Query expansion failed: {str(e)}")
                return query
        
        return query
    
    def _retrieve_context(
        self,
        user_id: str,
        query: str
    ) -> List[Dict[str, Any]]:
        """Retrieve relevant context from memory, documents, AND chat session summaries (hybrid retrieval).
        
        Args:
            user_id: User identifier
            query: Query text
            
        Returns:
            List of relevant context items (conversations + documents + session summaries combined)
        """
        all_results = []
        
        # Retrieve from conversation memory
        memory_results = self.memory_manager.retrieve_relevant_memory(
            user_id=user_id,
            query=query,
            top_k=self.config.top_k_documents
        )
        
        # Add source type to memory results
        for result in memory_results:
            result["source_type"] = "conversation"
        all_results.extend(memory_results)
        
        # Retrieve from documents (if available)
        # Get max chunks per document from config
        from config import get_config
        config = get_config()
        max_chunks_per_doc = getattr(config.rag, 'max_chunks_per_document', 3)
        document_results = self.document_store.search_documents(
            query=query,
            user_id=user_id,
            top_k=self.config.top_k_documents,
            max_chunks_per_document=max_chunks_per_doc
        )
        
        # Add source type to document results
        for result in document_results:
            result["source_type"] = "document"
            # Extract filename from metadata if available
            if "metadata" in result:
                result["filename"] = result["metadata"].get("filename", "unknown")
        all_results.extend(document_results)
        
        # Retrieve from chat session summaries (past conversation memory)
        try:
            session_results = self.memory_manager.retrieve_session_context(
                user_id=user_id,
                query=query,
                top_k=3  # Keep small to avoid overwhelming context
            )
            for result in session_results:
                result["source_type"] = "session_summary"
            all_results.extend(session_results)
        except Exception as exc:
            logger.warning(f"Session summary retrieval skipped: {exc}")
        
        # Filter by similarity threshold
        filtered_results = [
            r for r in all_results
            if r.get("similarity", 0) >= self.config.similarity_threshold
        ]
        
        # Sort by similarity score (descending)
        filtered_results.sort(key=lambda x: x.get("similarity", 0), reverse=True)
        
        # Limit to top_k combined results
        filtered_results = filtered_results[:self.config.top_k_documents]
        
        # Separate conversation and document sources for logging
        doc_count = sum(1 for r in filtered_results if r["source_type"] == "document")
        conv_count = sum(1 for r in filtered_results if r["source_type"] == "conversation")
        session_count = sum(1 for r in filtered_results if r["source_type"] == "session_summary")
        
        logger.info(f"Retrieved {len(filtered_results)} items: {doc_count} documents, {conv_count} conversations, {session_count} session summaries")
        return filtered_results
    
    def _assess_context_quality(self, retrieved_context: List[Dict[str, Any]]) -> float:
        """Compute average similarity of retrieved context.
        
        Args:
            retrieved_context: List of retrieved items with 'similarity' scores
            
        Returns:
            Average similarity score (0.0 if no context)
        """
        if not retrieved_context:
            return 0.0
        scores = [item.get("similarity", 0.0) for item in retrieved_context]
        avg = sum(scores) / len(scores)
        logger.debug(f"Context quality: avg_similarity={avg:.3f} over {len(scores)} items")
        return avg
    
    def _reformulate_query(
        self,
        original_query: str,
        retrieved_context: List[Dict[str, Any]],
        attempt: int
    ) -> str:
        """Use LLM to rewrite the query for better retrieval.
        
        Args:
            original_query: The user's original question
            retrieved_context: Current (poor) retrieval results
            attempt: Current reformulation attempt number
            
        Returns:
            Reformulated query string
        """
        # Build a summary of what was retrieved
        context_lines = []
        for item in retrieved_context[:5]:  # top 5 for brevity
            text_preview = item.get("document", "")[:150]
            sim = item.get("similarity", 0.0)
            context_lines.append(f"  - (sim={sim:.2f}) {text_preview}")
        context_summary = "\n".join(context_lines) if context_lines else "  (no results)"
        
        avg_similarity = self._assess_context_quality(retrieved_context)
        
        prompt = QUERY_REFORMULATION_PROMPTS["reformulate"].format(
            query=original_query,
            context_summary=context_summary,
            avg_similarity=avg_similarity
        )
        
        try:
            response = self.client.chat.completions.create(
                model=self.llm_config.model,
                messages=[{"role": "user", "content": prompt}],
                temperature=0.4,
                max_tokens=200
            )
            reformulated = response.choices[0].message.content.strip()
            # Strip quotes if the LLM wrapped the query
            reformulated = reformulated.strip('"').strip("'")
            logger.info(f"Query reformulation attempt {attempt + 1}: '{original_query[:60]}' -> '{reformulated[:60]}'")
            return reformulated
        except Exception as e:
            logger.warning(f"Query reformulation failed: {e}")
            return original_query
    
    def _reflect_on_response(
        self,
        query: str,
        retrieved_context: List[Dict[str, Any]],
        draft_response: str,
        iteration: int
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
        # Build context text for the reflection prompt
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
            draft_answer=draft_response
        )
        
        try:
            response = self.client.chat.completions.create(
                model=self.llm_config.model,
                messages=[{"role": "user", "content": prompt}],
                temperature=0.1,
                max_tokens=self.llm_config.max_tokens,
                response_format={"type": "json_object"}
            )
            
            response_text = response.choices[0].message.content.strip()
            cleaned = clean_json_response(response_text)
            result = json.loads(cleaned)
            
            return {
                "is_grounded": result.get("is_grounded", True),
                "issues": result.get("issues", []),
                "revised_answer": result.get("revised_answer", draft_response)
            }
        except (json.JSONDecodeError, Exception) as e:
            logger.warning(f"Reflection failed (iteration {iteration}): {e}")
            # On failure, assume grounded to avoid blocking the user
            return {
                "is_grounded": True,
                "issues": [],
                "revised_answer": draft_response
            }
    
    def _build_prompt(
        self,
        query: str,
        retrieved_context: List[Dict[str, Any]],
        conversation_history: Optional[List[Dict[str, str]]]
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
        
        # Add retrieved context if available with enhanced metadata
        if retrieved_context:
            if self.config.include_metadata:
                # Separate documents, conversations, and web search results
                documents = [item for item in retrieved_context if item.get("source_type") == "document"]
                conversations = [item for item in retrieved_context if item.get("source_type") == "conversation"]
                web_results = [item for item in retrieved_context if item.get("source_type") == "web_search"]
                session_summaries = [item for item in retrieved_context if item.get("source_type") == "session_summary"]
                
                context_sections = []
                
                # Add documents section if available
                if documents:
                    doc_content = []
                    
                    # Group documents by filename for better organization
                    doc_groups = {}
                    for item in documents:
                        filename = item.get("filename", "document")
                        if filename not in doc_groups:
                            doc_groups[filename] = []
                        doc_groups[filename].append(item)
                    
                    # Process each document's chunks
                    for filename, chunks in doc_groups.items():
                        # Sort chunks by position if available, otherwise by similarity
                        chunks.sort(key=lambda x: (
                            x.get("metadata", {}).get("start_position", 0),
                            -x.get("similarity", 0)
                        ))
                        
                        # Add document header
                        chunk_info = chunks[0]  # Best chunk for metadata
                        similarity = chunk_info.get("similarity", 0)
                        total_chunks = chunk_info.get("metadata", {}).get("total_chunks", 1)
                        
                        doc_header = f"📄 [{filename}] (relevance: {similarity:.2f}"
                        if total_chunks > 1:
                            doc_header += f", {len(chunks)} chunks"
                        doc_header += "):"
                        
                        doc_content.append(doc_header)
                        
                        # Add content from chunks with better extraction
                        all_chunk_text = []
                        for item in chunks:
                            text = item.get("document", "")
                            
                            # Extract more relevant content based on query keywords
                            query_words = set(query.lower().split())
                            text_sentences = text.split('. ')
                            
                            # Find sentences with query keywords for better relevance
                            relevant_sentences = []
                            for sentence in text_sentences:
                                sentence_words = set(sentence.lower().split())
                                if query_words.intersection(sentence_words):
                                    relevant_sentences.append(sentence)
                            
                            # Use keyword-rich sentences or fallback to longer excerpt
                            if relevant_sentences:
                                relevant_text = '. '.join(relevant_sentences[:3])  # Top 3 relevant sentences
                            else:
                                relevant_text = text[:800]  # Increased from 300 to 800 chars
                            
                            all_chunk_text.append(relevant_text.strip())
                        
                        # Combine chunks with separator
                        combined_content = " [...] ".join(all_chunk_text)
                        doc_content.append(f"  {combined_content}")
                    
                    context_sections.append("\n".join(doc_content))
                
                # Add conversation context section if available
                if conversations:
                    conv_content = []
                    for item in conversations:
                        text = item.get("document", "")[:200]
                        conv_content.append(f"  • {text}...")
                    
                    context_sections.append(f"💬 CONVERSATION HISTORY:\n" + "\n".join(conv_content))
                
                # Add web search results section
                if web_results:
                    web_content = []
                    for item in web_results:
                        text = item.get("document", "")
                        web_content.append(f"  {text}")
                    context_sections.append(f"🌐 WEB SEARCH RESULTS:\n" + "\n".join(web_content))
                
                # Add session summary context
                if session_summaries:
                    summary_content = []
                    for item in session_summaries:
                        text = item.get("document", "")[:300]
                        summary_content.append(f"  • {text}")
                    context_sections.append(f"🧠 PAST SESSION CONTEXT:\n" + "\n".join(summary_content))
                
                if context_sections:
                    context_text = "\n\n".join(context_sections)
                    prompt_parts.append(f"=== RETRIEVED CONTEXT ===\n{context_text}")
            else:
                # Simple context without source attribution
                context_text = "\n\n".join([
                    f"• {item['document']}"
                    for item in retrieved_context
                ])
                prompt_parts.append(f"Relevant context:\n{context_text}")

        
        # Add conversation history if available
        if conversation_history:
            history_text = "\n".join([
                f"{turn['role'].capitalize()}: {turn['content']}"
                for turn in conversation_history[-5:]  # Last 5 turns
            ])
            prompt_parts.append(f"\n=== CONVERSATION HISTORY ===\n{history_text}")
        
        # Add current query
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
            response = self.client.chat.completions.create(
                model=self.llm_config.model,
                messages=[
                    {"role": "system", "content": self.llm_config.system_prompt},
                    {"role": "user", "content": prompt}
                ],
                temperature=self.llm_config.temperature,
                max_tokens=self.llm_config.max_tokens
            )
            
            response_text = response.choices[0].message.content.strip()
            
            logger.debug(f"Generated response: {response_text[:100]}...")
            return response_text
            
        except Exception as e:
            logger.error(f"Error generating LLM response: {type(e).__name__}: {str(e)}", exc_info=True)
            self._last_error = e  # Store for later use
            error_response = (
                f"[ERROR] Failed to generate response. Details: {type(e).__name__}: {str(e)}. "
                f"Please check your Gemini API key and internet connection."
            )
            logger.error(f"Full error response: {error_response}")
            return error_response
    
    def _retry_generation(
        self,
        prompt: str,
        validation_issues: List[str],
        retry_count: int = 0
    ) -> str:
        """Retry response generation with validation feedback.
        
        Args:
            prompt: Original prompt
            validation_issues: List of validation issues
            retry_count: Current retry attempt
            
        Returns:
            New response text
        """
        if retry_count >= self.llm_config.max_retries:
            logger.warning("Max retries reached")
            # Check if error was rate limiting
            if hasattr(self, '_last_error') and "rate limit" in str(self._last_error).lower():
                return "[RATE_LIMITED] The Gemini API rate limit has been exceeded. This system is free-tier and has a limit of 20 requests per day. Please try again tomorrow or upgrade to a paid plan at https://ai.google.dev"
            return "I apologize, but I'm having difficulty providing an appropriate response. Could you rephrase your question?"
        
        # Add validation feedback to prompt
        feedback = "\n".join([f"- {issue}" for issue in validation_issues])
        retry_prompt = f"""{prompt}

Previous response had these issues:
{feedback}

Please provide a new response that addresses these concerns."""
        
        # Generate new response
        response_text = self._generate_llm_response(retry_prompt)
        
        # Validate again
        validation = self.validator.validate_response(response_text, prompt)
        
        if not validation["is_valid"]:
            # Retry recursively
            return self._retry_generation(
                prompt=prompt,
                validation_issues=validation["issues"],
                retry_count=retry_count + 1
            )
        
        return response_text


if __name__ == "__main__":
    # Example usage
    rag = RAGPipeline()
    
    user_id = "test_user_123"
    
    # First, store some context
    rag.memory_manager.store_user_context(
        user_id=user_id,
        context_type="physical_context",
        content="User has neck pain from desk work, prefers exercises without equipment"
    )
    
    # Generate response
    query = "What exercises can help my neck pain?"
    result = rag.generate_response(
        query=query,
        user_id=user_id,
        use_memory=True
    )
    
    print("Response:", result["response"])
    print("\nMetadata:", result["metadata"])
