"""Gemini API client wrapper for Agentic RAG system.

This module provides a wrapper around Google's Gemini API with an interface
compatible with the rest of the system.
"""

import os
import json
from typing import List, Dict, Any, Optional

import google.generativeai as genai
from google.generativeai import types

from utils.logger import get_logger
from utils.api_key_manager import get_api_key_manager, APIKeyManager


logger = get_logger(__name__)


class GeminiClient:
    """Wrapper for Gemini API providing chat completion and embeddings.
    
    This class provides methods compatible with the system's expectations
    while using Google's Gemini API under the hood.
    """
    
    def __init__(self, api_key: str = None, key_manager: APIKeyManager = None):
        """Initialize Gemini client.
        
        Args:
            api_key: Gemini API key. If None, uses APIKeyManager.
            key_manager: Optional APIKeyManager instance. If None, uses singleton.
        """
        self.key_manager = key_manager or get_api_key_manager()
        self.api_key = api_key or self.key_manager.get_current_key()
        
        if not self.api_key:
            raise ValueError("No Gemini API key found. Set GEMINI_API_KEYS or GEMINI_API_KEY in environment.")
        
        # Configure Gemini with current key
        genai.configure(api_key=self.api_key)
        
        logger.info(f"Initialized GeminiClient with API key {self.key_manager.get_current_key_index() + 1}/{self.key_manager.get_total_keys()}")
    
    def chat_completion(
        self,
        messages: List[Dict[str, str]],
        model: str = "gemini-2.5-flash",
        temperature: float = 0.7,
        max_tokens: int = 1000,
        response_format: Optional[Dict[str, str]] = None
    ) -> str:
        """Generate chat completion using Gemini.
        
        Args:
            messages: List of message dicts with 'role' and 'content'
            model: Gemini model name
            temperature: Sampling temperature (0.0 to 2.0)
            max_tokens: Maximum tokens to generate
            response_format: Optional format spec, e.g. {"type": "json_object"}
            
        Returns:
            Generated text response
        """
        try:
            # Convert messages to Gemini format
            gemini_contents = self._convert_messages_to_gemini(messages)
            
            # Create model instance
            model_instance = genai.GenerativeModel(model_name=model)
            
            # Try to use generation_config if available (newer versions)
            # Fall back to simple call for older versions
            try:
                # Newer API (0.4.0+)
                config = types.GenerateContentConfig(
                    temperature=temperature,
                    max_output_tokens=max_tokens
                )
                response = model_instance.generate_content(
                    contents=gemini_contents,
                    generation_config=config
                )
            except (AttributeError, TypeError):
                # Older API (0.3.x) - use simpler parameters
                logger.debug("Using older google-generativeai API (0.3.x)")
                response = model_instance.generate_content(
                    contents=gemini_contents,
                    generation_config=genai.types.GenerationConfig(
                        temperature=temperature,
                        candidate_count=1,
                        max_output_tokens=max_tokens,
                        top_p=0.95,
                        top_k=40
                    )
                )
            
            # Check if response was blocked or invalid
            if hasattr(response, 'prompt_feedback') and response.prompt_feedback:
                if hasattr(response.prompt_feedback, 'block_reason') and response.prompt_feedback.block_reason:
                    block_reason = response.prompt_feedback.block_reason
                    error_msg = f"Gemini API blocked response: {block_reason}"
                    logger.error(error_msg)
                    raise RuntimeError(error_msg)
            
            if not hasattr(response, 'candidates') or not response.candidates or len(response.candidates) == 0:
                error_msg = "Gemini API returned no candidates"
                logger.error(error_msg)
                raise RuntimeError(error_msg)
            
            candidate = response.candidates[0]
            
            # Check finish reason
            if hasattr(candidate, 'finish_reason'):
                finish_reason = candidate.finish_reason
                if finish_reason != 1 and finish_reason != "STOP":  # 1 or "STOP" = normal completion
                    error_msg = f"Gemini response incomplete. Finish reason: {finish_reason}"
                    logger.error(error_msg)
                    raise RuntimeError(error_msg)
            
            # Extract text
            if not hasattr(candidate, 'content') or not candidate.content:
                error_msg = "Gemini response has no content"
                logger.error(error_msg)
                raise RuntimeError(error_msg)
                
            if not hasattr(candidate.content, 'parts') or len(candidate.content.parts) == 0:
                error_msg = "Gemini response has no content parts"
                logger.error(error_msg)
                raise RuntimeError(error_msg)
            
            result = candidate.content.parts[0].text
            
            # Ensure result is a string
            if isinstance(result, dict):
                logger.warning("Gemini returned dict instead of text, converting to JSON string")
                result = json.dumps(result)
            elif not isinstance(result, str):
                logger.warning(f"Gemini returned {type(result).__name__}, converting to string")
                result = str(result)
            
            logger.debug(f"Gemini response generated: {len(result)} chars")
            return result
            
        except (AttributeError, KeyError) as e:
            error_msg = f"Error accessing Gemini response: {str(e)}. Response structure may be invalid."
            logger.error(error_msg)
            logger.error(f"Response object type: {type(response) if 'response' in locals() else 'N/A'}")
            raise RuntimeError(error_msg) from e
        except Exception as e:
            # Check if this is a rate limit error
            error_str = str(e)
            if "429" in error_str or "quota" in error_str.lower() or "exceeded" in error_str.lower():
                logger.warning(f"Gemini API quota exceeded for key {self.key_manager.get_current_key_index() + 1}")
                
                # Try to rotate to next key
                if self._handle_quota_error(e):
                    # Retry with new key
                    logger.info("Retrying with rotated API key...")
                    return self.chat_completion(
                        messages=messages,
                        model=model,
                        temperature=temperature,
                        max_tokens=max_tokens,
                        response_format=response_format
                    )
                else:
                    # All keys exhausted
                    error_msg = f"All API keys exhausted. {self.key_manager.get_last_error()}"
                    logger.error(error_msg)
                    self.key_manager.set_last_error(error_msg)
                    raise RuntimeError(error_msg) from e
            
            logger.error(f"Error in Gemini chat completion: {type(e).__name__}: {str(e)}", exc_info=True)
            raise
    
    def _handle_quota_error(self, error: Exception) -> bool:
        """Handle quota error by rotating to next key.
        
        Args:
            error: The exception that occurred
            
        Returns:
            True if rotation succeeded and retry should be attempted
        """
        if self.key_manager.rotate_to_next_key():
            self.api_key = self.key_manager.get_current_key()
            genai.configure(api_key=self.api_key)
            logger.info(f"Switched to API key {self.key_manager.get_current_key_index() + 1}/{self.key_manager.get_total_keys()}")
            return True
        return False
    
    def get_key_status(self) -> dict:
        """Get the current API key manager status.
        
        Returns:
            dict: Status information about API keys
        """
        return self.key_manager.get_status()
    
    def _convert_messages_to_gemini(
        self,
        messages: List[Dict[str, str]]
    ) -> List[Dict[str, Any]]:
        """Convert OpenAI-style messages to Gemini format.
        
        Args:
            messages: List of {"role": "user/assistant/system", "content": "..."}
            
        Returns:
            Gemini-compatible contents list
        """
        gemini_contents = []
        system_instruction = None
        
        for msg in messages:
            role = msg.get("role")
            content = msg.get("content", "")
            
            if role == "system":
                # Gemini uses system instruction separately
                # We'll prepend it to the first user message
                system_instruction = content
            elif role == "user":
                # If we have system instruction, prepend it
                if system_instruction:
                    content = f"{system_instruction}\n\n{content}"
                    system_instruction = None  # Only use once
                
                gemini_contents.append({
                    "role": "user",
                    "parts": [{"text": content}]
                })
            elif role == "assistant":
                gemini_contents.append({
                    "role": "model",
                    "parts": [{"text": content}]
                })
        
        return gemini_contents
    
    def embeddings_create(
        self,
        texts: List[str],
        model: str = "models/text-embedding-004"
    ) -> List[List[float]]:
        """Generate embeddings using Gemini.
        
        Args:
            texts: List of texts to embed
            model: Embedding model name
            
        Returns:
            List of embedding vectors
        """
        try:
            embeddings = []
            
            for text in texts:
                result = genai.embed_content(
                    model=model,
                    content=text,
                    task_type="retrieval_document"
                )
                embeddings.append(result['embedding'])
            
            logger.debug(f"Generated {len(embeddings)} embeddings")
            return embeddings
            
        except Exception as e:
            logger.error(f"Error generating Gemini embeddings: {type(e).__name__}: {str(e)}", exc_info=True)
            raise


class GeminiChatCompletion:
    """Mock class to mimic OpenAI's client.chat.completions structure."""
    
    def __init__(self, gemini_client: GeminiClient):
        self.gemini_client = gemini_client
    
    def create(
        self,
        model: str,
        messages: List[Dict[str, str]],
        temperature: float = 0.7,
        max_tokens: int = 1000,
        response_format: Optional[Dict[str, str]] = None,
        **kwargs
    ) -> 'GeminiResponse':
        """Create chat completion (OpenAI-compatible interface).
        
        Returns:
            GeminiResponse object with OpenAI-like structure
        """
        response_text = self.gemini_client.chat_completion(
            messages=messages,
            model=model,
            temperature=temperature,
            max_tokens=max_tokens,
            response_format=response_format
        )
        
        return GeminiResponse(response_text)


class GeminiResponse:
    """Mock response object to mimic OpenAI's response structure."""
    
    def __init__(self, text: str):
        try:
            # Handle case where text might be None or not a string
            if text is None:
                text = ""
            elif isinstance(text, dict):
                # If it's a dict, convert to JSON string
                text = json.dumps(text)
            elif not isinstance(text, str):
                text = str(text)
            
            self.choices = [GeminiChoice(text)]
        except Exception as e:
            logger.error(f"Error creating GeminiResponse: {type(e).__name__}: {str(e)}")
            # Fall back to error message
            self.choices = [GeminiChoice(f"[ERROR] Failed to parse response: {str(e)}")]


class GeminiChoice:
    """Mock choice object."""
    
    def __init__(self, text: str):
        self.message = GeminiMessage(text)


class GeminiMessage:
    """Mock message object."""
    
    def __init__(self, text: str):
        self.content = text


class GeminiEmbeddings:
    """Mock class to mimic OpenAI's client.embeddings structure."""
    
    def __init__(self, gemini_client: GeminiClient):
        self.gemini_client = gemini_client
    
    def create(
        self,
        model: str,
        input: List[str],
        **kwargs
    ) -> 'GeminiEmbeddingResponse':
        """Create embeddings (OpenAI-compatible interface).
        
        Returns:
            GeminiEmbeddingResponse object
        """
        embeddings = self.gemini_client.embeddings_create(
            texts=input,
            model=model
        )
        
        return GeminiEmbeddingResponse(embeddings)


class GeminiEmbeddingResponse:
    """Mock embedding response object."""
    
    def __init__(self, embeddings: List[List[float]]):
        self.data = [
            GeminiEmbeddingData(embedding)
            for embedding in embeddings
        ]


class GeminiEmbeddingData:
    """Mock embedding data object."""
    
    def __init__(self, embedding: List[float]):
        self.embedding = embedding


class GeminiClientWrapper:
    """Full wrapper that mimics OpenAI client structure.
    
    This allows drop-in replacement:
    from openai import OpenAI
    client = OpenAI()
    
    →
    
    from utils.gemini_client import GeminiClientWrapper
    client = GeminiClientWrapper()
    """
    
    def __init__(self, api_key: str = None):
        """Initialize wrapper.
        
        Args:
            api_key: Gemini API key
        """
        self._gemini_client = GeminiClient(api_key)
        self.chat = GeminiChat(self._gemini_client)
        self.embeddings = GeminiEmbeddings(self._gemini_client)


class GeminiChat:
    """Chat namespace."""
    
    def __init__(self, gemini_client: GeminiClient):
        self.completions = GeminiChatCompletion(gemini_client)


if __name__ == "__main__":
    # Example usage
    import os
    
    # Set API key
    os.environ["GEMINI_API_KEY"] = "your-api-key-here"
    
    # Initialize client (OpenAI-compatible interface)
    client = GeminiClientWrapper()
    
    # Test chat completion
    response = client.chat.completions.create(
        model="gemini-2.5-flash",
        messages=[
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": "What is 2+2?"}
        ],
        temperature=0.7,
        max_tokens=100
    )
    
    print("Response:", response.choices[0].message.content)
    
    # Test JSON mode
    response = client.chat.completions.create(
        model="gemini-2.5-flash",
        messages=[
            {"role": "user", "content": "Return a JSON with keys 'name' and 'age' for a person named John who is 30."}
        ],
        response_format={"type": "json_object"}
    )
    
    print("JSON Response:", response.choices[0].message.content)
    
    # Test embeddings
    response = client.embeddings.create(
        model="models/text-embedding-004",
        input=["Hello world", "Gemini API is great"]
    )
    
    print(f"Embeddings: {len(response.data)} vectors of dimension {len(response.data[0].embedding)}")
