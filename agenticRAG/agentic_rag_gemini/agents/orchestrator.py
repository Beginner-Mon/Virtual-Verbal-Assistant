"""Orchestrator Agent - The brain of the Agentic RAG system.

This module implements the orchestrator agent that decides how to handle user queries.
It uses a Small Language Model (SLM) to analyze queries and route them to appropriate modules.
"""

import json
import re
from enum import Enum
from typing import Dict, List, Optional, Any

from config import get_config
from utils.logger import get_logger
from utils.prompt_templates import ORCHESTRATOR_PROMPTS
from utils.gemini_client import GeminiClientWrapper


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


class ActionType(Enum):
    """Types of actions the orchestrator can take."""
    RETRIEVE_MEMORY = "retrieve_memory"
    CALL_LLM = "call_llm"
    GENERATE_MOTION = "generate_motion"
    HYBRID = "hybrid"
    CLARIFY = "clarify"


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
            "parameters": self.parameters
        }


class OrchestratorAgent:
    """Orchestrator agent that routes queries to appropriate modules.
    
    The orchestrator uses a Small Language Model to analyze user queries and decide:
    1. Whether to retrieve relevant memory/context
    2. Whether to call the LLM for response generation
    3. Whether to trigger motion generation
    4. Or a combination of the above (hybrid approach)
    
    This implements the ReAct pattern: Reasoning + Acting.
    """
    
    def __init__(self, config=None, client=None):
        """Initialize the orchestrator agent.
        
        Args:
            config: Configuration object. If None, loads from default config.
            client: Shared GeminiClientWrapper instance. Creates new if None.
        """
        self.config = config or get_config().orchestrator
        self.client = client or GeminiClientWrapper()
        
        logger.info(f"Initialized OrchestratorAgent with Gemini model: {self.config.model}")
    
    def analyze_query(
        self,
        query: str,
        conversation_history: Optional[List[Dict[str, str]]] = None,
        user_context: Optional[Dict[str, Any]] = None
    ) -> OrchestratorDecision:
        """Analyze user query and decide on appropriate action.
        
        Args:
            query: User's query text
            conversation_history: Previous conversation turns
            user_context: Additional context about the user
            
        Returns:
            OrchestratorDecision object with action and reasoning
        """
        logger.info(f"Analyzing query: {query[:100]}...")
        
        # Build prompt with context
        prompt = self._build_analysis_prompt(query, conversation_history, user_context)
        
        try:
            # Call SLM for decision
            response = self.client.chat.completions.create(
                model=self.config.model,
                messages=[
                    {"role": "system", "content": self.config.system_prompt},
                    {"role": "user", "content": prompt}
                ],
                temperature=self.config.temperature,
                max_tokens=self.config.max_tokens,
                response_format={"type": "json_object"}
            )
            
            # Clean and parse decision
            response_text = response.choices[0].message.content
            cleaned_json = clean_json_response(response_text)
            decision_data = json.loads(cleaned_json)
            decision = self._parse_decision(decision_data)
            
            logger.info(f"Decision: {decision.action.value} (confidence: {decision.confidence:.2f})")
            
            return decision
            
        except json.JSONDecodeError as e:
            logger.error(f"Failed to parse orchestrator JSON response: {str(e)}. Raw response: {response_text[:200] if 'response_text' in locals() else 'N/A'}")
            # Fallback to safe default
            return OrchestratorDecision(
                action=ActionType.CALL_LLM,
                confidence=0.5,
                reasoning="Fallback due to JSON parsing error in orchestrator",
                parameters={"use_memory": False}
            )
        except Exception as e:
            logger.error(f"Error in query analysis: {type(e).__name__}: {str(e)}", exc_info=True)
            # Fallback to safe default
            return OrchestratorDecision(
                action=ActionType.CALL_LLM,
                confidence=0.5,
                reasoning=f"Fallback due to analysis error: {type(e).__name__}",
                parameters={"use_memory": False}
            )
    
    def _build_analysis_prompt(
        self,
        query: str,
        conversation_history: Optional[List[Dict[str, str]]],
        user_context: Optional[Dict[str, Any]]
    ) -> str:
        """Build the analysis prompt for the orchestrator.
        
        Args:
            query: User query
            conversation_history: Previous conversation
            user_context: User context information
            
        Returns:
            Formatted prompt string
        """
        prompt_parts = [
            "Analyze the following user query and decide the best action.",
            f"\nUser Query: {query}",
        ]
        
        # Add conversation history if available
        if conversation_history:
            history_text = "\n".join([
                f"{turn['role']}: {turn['content']}"
                for turn in conversation_history[-5:]  # Last 5 turns
            ])
            prompt_parts.append(f"\nRecent Conversation:\n{history_text}")
        
        # Add user context if available
        if user_context:
            context_text = json.dumps(user_context, indent=2)
            prompt_parts.append(f"\nUser Context:\n{context_text}")
        
        # Add decision format instructions
        prompt_parts.append(ORCHESTRATOR_PROMPTS["decision_format"])
        
        return "\n".join(prompt_parts)
    
    def _parse_decision(self, decision_data: Dict[str, Any]) -> OrchestratorDecision:
        """Parse decision from LLM response.
        
        Args:
            decision_data: Dictionary from LLM JSON response
            
        Returns:
            OrchestratorDecision object
        """
        action_str = decision_data.get("action", "call_llm")
        
        # Map string to ActionType enum
        action_map = {
            "retrieve_memory": ActionType.RETRIEVE_MEMORY,
            "call_llm": ActionType.CALL_LLM,
            "generate_motion": ActionType.GENERATE_MOTION,
            "hybrid": ActionType.HYBRID,
            "clarify": ActionType.CLARIFY
        }
        
        action = action_map.get(action_str, ActionType.CALL_LLM)
        confidence = float(decision_data.get("confidence", 0.7))
        reasoning = decision_data.get("reasoning", "No reasoning provided")
        parameters = decision_data.get("parameters", {})
        
        return OrchestratorDecision(
            action=action,
            confidence=confidence,
            reasoning=reasoning,
            parameters=parameters
        )
    
    def should_retrieve_memory(self, decision: OrchestratorDecision) -> bool:
        """Determine if memory retrieval is needed based on decision.
        
        Args:
            decision: Orchestrator decision
            
        Returns:
            True if memory should be retrieved
        """
        return (
            decision.action == ActionType.RETRIEVE_MEMORY or
            decision.action == ActionType.HYBRID or
            decision.parameters.get("use_memory", False)
        )
    
    def should_call_llm(self, decision: OrchestratorDecision) -> bool:
        """Determine if LLM should be called based on decision.
        
        Args:
            decision: Orchestrator decision
            
        Returns:
            True if LLM should be called
        """
        return (
            decision.action == ActionType.CALL_LLM or
            decision.action == ActionType.HYBRID or
            decision.action == ActionType.CLARIFY
        )
    
    def should_generate_motion(self, decision: OrchestratorDecision) -> bool:
        """Determine if motion generation is needed based on decision.
        
        Args:
            decision: Orchestrator decision
            
        Returns:
            True if motion should be generated
        """
        return (
            decision.action == ActionType.GENERATE_MOTION or
            decision.action == ActionType.HYBRID or
            decision.parameters.get("generate_motion", False)
        )
    
    def process_query(
        self,
        query: str,
        user_id: str,
        conversation_history: Optional[List[Dict[str, str]]] = None,
        user_context: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """Main entry point for processing a user query.
        
        This is a high-level method that:
        1. Analyzes the query
        2. Makes routing decisions
        3. Returns action plan
        
        Args:
            query: User query text
            user_id: Unique user identifier
            conversation_history: Previous conversation
            user_context: User context information
            
        Returns:
            Dictionary with action plan and decision details
        """
        # Analyze query
        decision = self.analyze_query(query, conversation_history, user_context)
        
        # Build action plan
        action_plan = {
            "user_id": user_id,
            "query": query,
            "decision": decision.to_dict(),
            "actions": {
                "retrieve_memory": self.should_retrieve_memory(decision),
                "call_llm": self.should_call_llm(decision),
                "generate_motion": self.should_generate_motion(decision)
            },
            "parameters": decision.parameters
        }
        
        logger.info(f"Action plan: {json.dumps(action_plan, indent=2)}")
        
        return action_plan


if __name__ == "__main__":
    # Example usage
    orchestrator = OrchestratorAgent()
    
    # Test query
    test_query = "I have neck pain from sitting at my desk all day"
    result = orchestrator.process_query(
        query=test_query,
        user_id="test_user_123"
    )
    
    print(json.dumps(result, indent=2))
