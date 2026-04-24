import os
import pytest
from unittest.mock import patch, MagicMock

from main import AgenticRAGSystem


@pytest.fixture
def mock_pinecone_index():
    """Provides a mock for the Pinecone Index to intercept VectorStore queries."""
    with patch("pinecone.Pinecone") as MockPinecone:
        mock_client = MagicMock()
        mock_index = MagicMock()
        
        MockPinecone.return_value = mock_client
        mock_client.Index.return_value = mock_index
        
        # When vector store queries Pinecone, return an empty result
        # so the LLM falls back to web search or its own knowledge.
        mock_index.query.return_value = {"matches": []}
        
        yield mock_index


@pytest.fixture
def mock_firestore():
    """Provides a mock for Firebase Firestore to intercept SessionStore calls."""
    with patch("firebase_admin.firestore.client") as mock_firestore_client:
        mock_db = MagicMock()
        mock_firestore_client.return_value = mock_db
        yield mock_db


@pytest.mark.asyncio
class TestAgenticRAGIntegration:
    """Integration tests for the core AgenticRAGSystem."""
    
    @pytest.fixture(autouse=True)
    def setup_env(self):
        """Ensure necessary environment variables are set for the integration test."""
        # Check if we have a real Gemini API key
        if not os.environ.get("GEMINI_API_KEY"):
            pytest.skip("GEMINI_API_KEY is not set. Skipping integration test.")
            
        # Set dummy Pinecone keys so VectorStore initializes
        with patch.dict(os.environ, {
            "PINECONE_API_KEY": "dummy_key",
            "PINECONE_INDEX_NAME": "dummy_index"
        }):
            yield
    
    def test_end_to_end_query_processing(self, mock_pinecone_index):
        """
        Tests the full AgenticRAG pipeline:
        1. Instantiates AgenticRAGSystem
        2. Sends a real query (uses the real GEMINI_API_KEY)
        3. Asserts the Orchestrator works and the RAGPipeline returns a coherent response
        4. Asserts that Pinecone was queried
        """
        system = AgenticRAGSystem()
        
        query = "What are three good stretches for lower back pain?"
        user_id = "integration_test_user"
        
        # Act
        result = system.process_query(query=query, user_id=user_id)
        
        # Assert - Logic Verification
        assert "orchestrator_decision" in result
        assert "response" in result
        
        decision = result["orchestrator_decision"]
        
        # The LLM should recognize this as a knowledge query or exercise recommendation
        valid_intents = ["knowledge_query", "exercise_recommendation"]
        assert decision.get("action") in valid_intents, f"Unexpected intent: {decision.get('action')}"
        
        # The response should be a non-empty string
        assert isinstance(result["response"], str)
        assert len(result["response"]) > 50
        
        # Assert - Usage Verification (Pinecone)
        # Verify that MemoryManager's retrieve_relevant_memory triggered a Pinecone query
        mock_pinecone_index.query.assert_called()
        
        # Check that we queried the expected namespaces
        # Memory retrieval usually queries 'documents' and 'conversations'
        called_namespaces = [
            call.kwargs.get("namespace") for call in mock_pinecone_index.query.call_args_list
        ]
        
        # Depending on the system config, it might query one or more.
        # Just assert it queried something!
        assert len(called_namespaces) > 0


class TestSessionStoreIntegration:
    """Tests for the SessionStore Firebase integration."""
    
    @patch.dict(os.environ, {
        "GOOGLE_APPLICATION_CREDENTIALS": "/dummy.json",
        "FIRESTORE_PROJECT_ID": "dummy_project"
    })
    @patch("os.path.exists", return_value=True)
    @patch("firebase_admin.credentials.Certificate")
    @patch("firebase_admin.initialize_app")
    def test_session_store_writes_to_firestore(
        self,
        mock_initialize_app,
        mock_certificate,
        mock_path_exists,
        mock_firestore
    ):
        """Verifies SessionStore writes correctly to Firestore."""
        # Ensure clean state
        import memory.session_store as ss_module
        ss_module._firestore_init_attempted = False
        ss_module._firestore_db = None
        
        from memory.session_store import SessionStore
        
        store = SessionStore(user_id="test_user")
        
        # Mock the document reference
        mock_doc_ref = MagicMock()
        mock_collection = mock_firestore.collection.return_value.document.return_value.collection.return_value
        mock_collection.document.return_value = mock_doc_ref
        
        # Act
        store.create_session(first_message="Hello there")
        
        # Assert Usage
        # SessionStore uses: db.collection("users").document(user_id).collection("sessions").document(session_id).set()
        mock_doc_ref.set.assert_called()
        
        # Verify what was written
        written_data = mock_doc_ref.set.call_args[0][0]
        assert written_data["user_id"] == "test_user"
        assert written_data["title"] == "Hello there"
        assert written_data["messages"] == []
