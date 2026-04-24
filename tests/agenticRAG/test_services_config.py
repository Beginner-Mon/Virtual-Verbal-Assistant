import os
from unittest.mock import patch, MagicMock
import pytest

from config import Config, VectorDatabaseConfig


@pytest.fixture
def mock_config():
    """Provides a dummy config where VectorDB is set to pinecone."""
    mock_conf = MagicMock(spec=Config)
    mock_conf.vector_database = VectorDatabaseConfig(
        type="pinecone",
        pinecone={"api_key": "", "index_name": "default_idx", "index_host": ""}
    )
    return mock_conf


class TestPineconeConfiguration:
    """Verifies that VectorStore reads Pinecone configuration correctly from .env"""

    @patch.dict(os.environ, {
        "PINECONE_API_KEY": "test-pinecone-key-123",
        "PINECONE_INDEX_NAME": "test-index",
        "PINECONE_INDEX_HOST": "https://test-host.pinecone.io"
    }, clear=True)
    @patch("memory.vector_store.Pinecone")
    def test_vector_store_loads_pinecone_env_vars(self, mock_pinecone_class, mock_config):
        # We must import VectorStore AFTER patching the environment 
        # so that it uses our patched os.environ during init
        from memory.vector_store import VectorStore
        
        mock_pinecone_instance = MagicMock()
        mock_pinecone_class.return_value = mock_pinecone_instance
        
        # Initialize VectorStore
        # It should read the environment variables and initialize Pinecone
        store = VectorStore(config=mock_config)
        
        # Assert Pinecone was initialized with the exact API key from the environment
        mock_pinecone_class.assert_called_once_with(api_key="test-pinecone-key-123")
        
        # Assert the Index was initialized with the correct name and host
        mock_pinecone_instance.Index.assert_called_once_with(
            name="test-index", 
            host="https://test-host.pinecone.io"
        )


class TestFirebaseConfiguration:
    """Verifies that SessionStore reads Firebase configuration correctly from .env"""

    @patch.dict(os.environ, {
        "GOOGLE_APPLICATION_CREDENTIALS": "/fake/path/to/firebase-service-account.json",
        "FIRESTORE_PROJECT_ID": "test-project-id"
    }, clear=True)
    @patch("os.path.exists", return_value=True)  # Pretend the key file exists
    @patch("firebase_admin.credentials.Certificate")
    @patch("firebase_admin.initialize_app")
    @patch("firebase_admin.firestore.client")
    def test_session_store_loads_firebase_env_vars(
        self, 
        mock_firestore_client, 
        mock_initialize_app, 
        mock_certificate, 
        mock_path_exists
    ):
        # Ensure we don't use a cached _firestore_db from previous tests
        import memory.session_store as ss_module
        ss_module._firestore_init_attempted = False
        ss_module._firestore_db = None
        
        # Create dummy cert object
        dummy_cert = MagicMock()
        mock_certificate.return_value = dummy_cert
        
        # Initialize SessionStore
        from memory.session_store import SessionStore
        store = SessionStore(user_id="test_user")
        
        # The store should have attempted to initialize Firestore
        assert store._use_firestore is True
        
        # Assert Certificate was loaded from the exact path in the environment
        mock_certificate.assert_called_once_with("/fake/path/to/firebase-service-account.json")
        
        # Assert Firebase app was initialized with the right project ID
        mock_initialize_app.assert_called_once_with(
            dummy_cert, 
            {"projectId": "test-project-id"}
        )
