import os
from pathlib import Path
from unittest.mock import patch, MagicMock
import pytest
from dotenv import load_dotenv

from config import Config, VectorDatabaseConfig

# Explicitly load the real .env file
ENV_PATH = Path("agenticRAG/agentic_rag_gemini/.env")
load_dotenv(ENV_PATH)


@pytest.fixture
def mock_config():
    """Provides a dummy config for VectorDB initialization."""
    mock_conf = MagicMock(spec=Config)
    mock_conf.vector_database = VectorDatabaseConfig(
        type="pinecone",
        pinecone={"api_key": "", "index_name": "", "index_host": ""}
    )
    return mock_conf


class TestPineconeConfiguration:
    """Verifies that VectorStore correctly reads Pinecone configuration from the REAL .env file."""

    @patch("pinecone.Pinecone")
    def test_vector_store_loads_real_pinecone_env(self, mock_pinecone_class, mock_config):
        real_api_key = os.environ.get("PINECONE_API_KEY")
        
        if not real_api_key:
            pytest.skip("PINECONE_API_KEY is empty in .env - skipping real Pinecone config test.")

        from memory.vector_store import VectorStore
        
        mock_pinecone_instance = MagicMock()
        mock_pinecone_class.return_value = mock_pinecone_instance
        
        # Initialize VectorStore. Because we aren't mocking os.environ,
        # it must find the real API key loaded from .env.
        store = VectorStore(config=mock_config.vector_database)
        
        # Assert Pinecone was initialized with the REAL API key
        mock_pinecone_class.assert_called_once_with(api_key=real_api_key)
        
        # Assert the Index was initialized with the REAL name
        real_index_name = os.environ.get("PINECONE_INDEX_NAME", "kinetichat")
        mock_pinecone_instance.Index.assert_called_once()
        assert mock_pinecone_instance.Index.call_args[1]["name"] == real_index_name


class TestFirebaseConfiguration:
    """Verifies that SessionStore reads Firebase configuration from the REAL .env file."""

    @patch("firebase_admin.credentials.Certificate")
    @patch("firebase_admin.initialize_app")
    @patch("firebase_admin.firestore.client")
    def test_session_store_loads_real_firebase_env(
        self, 
        mock_firestore_client, 
        mock_initialize_app, 
        mock_certificate
    ):
        real_cred_path = os.environ.get("GOOGLE_APPLICATION_CREDENTIALS")
        if not real_cred_path:
            pytest.skip("GOOGLE_APPLICATION_CREDENTIALS is empty in .env - skipping real Firebase config test.")
            
        import firebase_admin
        import memory.session_store as ss_module
        
        # Reset internal state to force initialization
        ss_module._firestore_init_attempted = False
        ss_module._firestore_db = None
        
        # Force firebase_admin._apps to be empty so it runs the initialize_app block
        with patch.object(firebase_admin, "_apps", {}):
            dummy_cert = MagicMock()
            mock_certificate.return_value = dummy_cert
            
            from memory.session_store import SessionStore
            store = SessionStore(user_id="test_user")
            
            # We are NOT mocking os.path.isfile anymore! 
            # If the JSON file doesn't actually exist on disk, this will fail, 
            # proving the configuration is invalid.
            assert store._use_firestore is True, (
                f"Failed to initialize Firestore! Does the file at '{real_cred_path}' "
                "actually exist on your machine?"
            )
            
            # Assert Certificate was loaded
            mock_certificate.assert_called_once()
            
            # Assert Firebase app was initialized
            mock_initialize_app.assert_called_once()
