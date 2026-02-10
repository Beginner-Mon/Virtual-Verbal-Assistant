"""Tests for Orchestrator Agent."""

import pytest
from unittest.mock import Mock, patch
import json

from agents.orchestrator import OrchestratorAgent, ActionType, OrchestratorDecision


class TestOrchestratorAgent:
    """Test suite for OrchestratorAgent."""
    
    @pytest.fixture
    def orchestrator(self):
        """Create orchestrator instance for testing."""
        return OrchestratorAgent()
    
    def test_initialization(self, orchestrator):
        """Test orchestrator initializes correctly."""
        assert orchestrator is not None
        assert orchestrator.client is not None
        assert orchestrator.config is not None
    
    @patch('agents.orchestrator.OpenAI')
    def test_analyze_query_memory_retrieval(self, mock_openai, orchestrator):
        """Test that orchestrator correctly identifies memory retrieval needs."""
        # Mock LLM response
        mock_response = Mock()
        mock_response.choices[0].message.content = json.dumps({
            "action": "retrieve_memory",
            "confidence": 0.9,
            "reasoning": "User references past conversation",
            "parameters": {"use_memory": True}
        })
        
        orchestrator.client.chat.completions.create = Mock(return_value=mock_response)
        
        # Test query
        query = "What did we discuss last time about my neck pain?"
        decision = orchestrator.analyze_query(query)
        
        assert decision.action == ActionType.RETRIEVE_MEMORY
        assert decision.confidence >= 0.8
    
    @patch('agents.orchestrator.OpenAI')
    def test_analyze_query_motion_generation(self, mock_openai, orchestrator):
        """Test that orchestrator identifies when motion generation is needed."""
        # Mock LLM response
        mock_response = Mock()
        mock_response.choices[0].message.content = json.dumps({
            "action": "generate_motion",
            "confidence": 0.85,
            "reasoning": "User needs visual demonstration of exercise",
            "parameters": {
                "generate_motion": True,
                "motion_type": "stretch"
            }
        })
        
        orchestrator.client.chat.completions.create = Mock(return_value=mock_response)
        
        query = "Show me how to do a neck stretch"
        decision = orchestrator.analyze_query(query)
        
        assert decision.action == ActionType.GENERATE_MOTION
        assert decision.parameters.get("motion_type") == "stretch"
    
    @patch('agents.orchestrator.OpenAI')
    def test_analyze_query_hybrid(self, mock_openai, orchestrator):
        """Test hybrid action for complex queries."""
        mock_response = Mock()
        mock_response.choices[0].message.content = json.dumps({
            "action": "hybrid",
            "confidence": 0.8,
            "reasoning": "User needs both advice and visual demonstration",
            "parameters": {
                "use_memory": True,
                "generate_motion": True
            }
        })
        
        orchestrator.client.chat.completions.create = Mock(return_value=mock_response)
        
        query = "My neck hurts from sitting all day, what exercises can help?"
        decision = orchestrator.analyze_query(query)
        
        assert decision.action == ActionType.HYBRID
        assert orchestrator.should_retrieve_memory(decision)
        assert orchestrator.should_call_llm(decision)
        assert orchestrator.should_generate_motion(decision)
    
    def test_should_retrieve_memory(self, orchestrator):
        """Test memory retrieval decision logic."""
        # Should retrieve for RETRIEVE_MEMORY action
        decision1 = OrchestratorDecision(
            action=ActionType.RETRIEVE_MEMORY,
            confidence=0.9,
            reasoning="Test"
        )
        assert orchestrator.should_retrieve_memory(decision1)
        
        # Should retrieve for HYBRID action
        decision2 = OrchestratorDecision(
            action=ActionType.HYBRID,
            confidence=0.8,
            reasoning="Test"
        )
        assert orchestrator.should_retrieve_memory(decision2)
        
        # Should not retrieve for CALL_LLM without memory parameter
        decision3 = OrchestratorDecision(
            action=ActionType.CALL_LLM,
            confidence=0.7,
            reasoning="Test",
            parameters={"use_memory": False}
        )
        assert not orchestrator.should_retrieve_memory(decision3)
    
    def test_process_query(self, orchestrator):
        """Test full query processing pipeline."""
        query = "I need help with shoulder pain"
        user_id = "test_user_123"
        
        with patch.object(orchestrator, 'analyze_query') as mock_analyze:
            mock_decision = OrchestratorDecision(
                action=ActionType.CALL_LLM,
                confidence=0.8,
                reasoning="User needs advice",
                parameters={"use_memory": True}
            )
            mock_analyze.return_value = mock_decision
            
            result = orchestrator.process_query(query, user_id)
            
            assert "decision" in result
            assert "actions" in result
            assert result["actions"]["call_llm"] is True
            assert result["user_id"] == user_id
    
    def test_fallback_on_error(self, orchestrator):
        """Test that orchestrator falls back gracefully on errors."""
        with patch.object(orchestrator.client.chat.completions, 'create') as mock_create:
            mock_create.side_effect = Exception("API Error")
            
            query = "Test query"
            decision = orchestrator.analyze_query(query)
            
            # Should fall back to CALL_LLM with lower confidence
            assert decision.action == ActionType.CALL_LLM
            assert decision.confidence <= 0.5


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
