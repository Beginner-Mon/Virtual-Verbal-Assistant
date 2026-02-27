"""Main entry point for Agentic RAG system.

This module provides a complete example of how to use the Agentic RAG system.
"""

import argparse
from typing import Optional, List, Dict, Any

from agents.orchestrator import OrchestratorAgent
from memory.memory_manager import MemoryManager
from retrieval.rag_pipeline import RAGPipeline
from config import get_config
from utils.logger import get_logger


logger = get_logger(__name__)


class AgenticRAGSystem:
    """Complete Agentic RAG system orchestrating all components."""
    
    def __init__(self):
        """Initialize the complete system."""
        logger.info("Initializing Agentic RAG System...")
        
        # Initialize components
        self.memory_manager = MemoryManager()
        self.rag_pipeline = RAGPipeline(memory_manager=self.memory_manager)
        self.orchestrator = OrchestratorAgent()
        
        logger.info("Agentic RAG System initialized successfully")
    
    def process_query(
        self,
        query: str,
        user_id: str,
        conversation_history: Optional[List[Dict[str, str]]] = None
    ) -> Dict[str, Any]:
        """Process a user query through the complete pipeline.
        
        Args:
            query: User's query text
            user_id: User identifier
            conversation_history: Previous conversation turns
            
        Returns:
            Dictionary with response and metadata
        """
        logger.info(f"Processing query for user {user_id}: {query[:100]}...")
        
        # Step 1: Orchestrator analyzes query and makes decision
        action_plan = self.orchestrator.process_query(
            query=query,
            user_id=user_id,
            conversation_history=conversation_history
        )
        
        logger.info(f"Orchestrator decision: {action_plan['decision']['action']}")
        
        # Step 2: Execute based on decision
        response_data = {
            "query": query,
            "user_id": user_id,
            "orchestrator_decision": action_plan["decision"],
            "response": None,
            "metadata": {}
        }
        
        # Check if we should use memory retrieval
        use_memory = action_plan["actions"]["retrieve_memory"]
        
        # Check if we need LLM response
        if action_plan["actions"]["call_llm"]:
            logger.info("Calling RAG pipeline for response generation")
            
            rag_result = self.rag_pipeline.generate_response(
                query=query,
                user_id=user_id,
                conversation_history=conversation_history,
                use_memory=use_memory
            )
            
            response_data["response"] = rag_result["response"]
            response_data["metadata"].update(rag_result["metadata"])
        
        # Check if we need motion generation
        if action_plan["actions"]["generate_motion"]:
            logger.info("Motion generation requested")
            response_data["metadata"]["motion_requested"] = True
            response_data["metadata"]["motion_type"] = action_plan["parameters"].get("motion_type")
            
            # Note: Actual motion generation would be handled by text-to-motion module
            # This is a placeholder for integration
            if response_data["response"]:
                response_data["response"] += "\n\n[Visual demonstration will be generated]"
            else:
                response_data["response"] = "[Visual demonstration will be generated]"
        
        # Fallback if no response generated
        if not response_data["response"]:
            response_data["response"] = "I understand your query. Could you provide more details so I can assist you better?"
        
        logger.info("Query processing complete")
        return response_data
    
    def interactive_mode(self, user_id: str = "demo_user"):
        """Run interactive chat mode for testing.
        
        Args:
            user_id: User identifier for the session
        """
        print("\n" + "="*60)
        print("Agentic RAG System - Interactive Mode")
        print("="*60)
        print(f"User ID: {user_id}")
        print("Type 'quit' or 'exit' to end the session")
        print("Type 'history' to see conversation history")
        print("="*60 + "\n")
        
        conversation_history = []
        
        while True:
            try:
                # Get user input
                user_input = input("You: ").strip()
                
                if not user_input:
                    continue
                
                # Check for commands
                if user_input.lower() in ['quit', 'exit']:
                    print("\nGoodbye! Take care of yourself.")
                    break
                
                if user_input.lower() == 'history':
                    print("\nConversation History:")
                    for i, turn in enumerate(conversation_history, 1):
                        print(f"{i}. {turn['role'].capitalize()}: {turn['content'][:100]}...")
                    print()
                    continue
                
                # Process query
                result = self.process_query(
                    query=user_input,
                    user_id=user_id,
                    conversation_history=conversation_history
                )
                
                # Display response
                print(f"\nAssistant: {result['response']}\n")
                
                # Show metadata in debug mode
                if logger.level == "DEBUG":
                    print(f"[Debug] Decision: {result['orchestrator_decision']['action']}")
                    print(f"[Debug] Confidence: {result['orchestrator_decision']['confidence']:.2f}")
                    print()
                
                # Update conversation history
                conversation_history.append({
                    "role": "user",
                    "content": user_input
                })
                conversation_history.append({
                    "role": "assistant",
                    "content": result['response']
                })
                
            except KeyboardInterrupt:
                print("\n\nSession interrupted. Goodbye!")
                break
            except Exception as e:
                logger.error(f"Error in interactive mode: {str(e)}")
                print(f"\nError: {str(e)}\n")


def main():
    """Main function with CLI interface."""
    parser = argparse.ArgumentParser(description="Agentic RAG System")
    
    parser.add_argument(
        "--mode",
        choices=["interactive", "single"],
        default="interactive",
        help="Running mode: interactive chat or single query"
    )
    
    parser.add_argument(
        "--query",
        type=str,
        help="Single query to process (for single mode)"
    )
    
    parser.add_argument(
        "--user-id",
        type=str,
        default="demo_user",
        help="User identifier"
    )
    
    parser.add_argument(
        "--debug",
        action="store_true",
        help="Enable debug logging"
    )
    
    args = parser.parse_args()
    
    # Setup logging level
    if args.debug:
        from utils.logger import setup_logging
        setup_logging(log_level="DEBUG")
    
    # Initialize system
    system = AgenticRAGSystem()
    
    if args.mode == "interactive":
        # Run interactive mode
        system.interactive_mode(user_id=args.user_id)
    
    elif args.mode == "single":
        # Process single query
        if not args.query:
            print("Error: --query is required for single mode")
            return
        
        result = system.process_query(
            query=args.query,
            user_id=args.user_id
        )
        
        print("\nQuery:", args.query)
        print("\nResponse:", result['response'])
        print("\nDecision:", result['orchestrator_decision']['action'])
        print("Confidence:", result['orchestrator_decision']['confidence'])


if __name__ == "__main__":
    main()
