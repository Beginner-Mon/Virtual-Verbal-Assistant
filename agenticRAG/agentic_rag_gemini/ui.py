"""Streamlit UI for AgenticRAG System with Document Loading and Chat.

Run with: streamlit run ui.py
"""

import streamlit as st
from streamlit_chat import message as st_message
from pathlib import Path
import tempfile
import os
import time
from typing import List, Dict, Any
import json
from datetime import datetime

# Add parent to path
import sys
sys.path.insert(0, str(Path(__file__).parent))

from agents.orchestrator import OrchestratorAgent
from agents.summarize_agent import SummarizeAgent
from memory.memory_manager import MemoryManager
from memory.document_store import DocumentStore
from memory.session_store import SessionStore
from memory.vector_store import VectorStore
from memory.embedding_service import EmbeddingService
from retrieval.rag_pipeline import RAGPipeline
from utils.document_loader import DocumentLoader
from utils.logger import get_logger
from utils.api_key_manager import get_api_key_manager
from utils.gemini_client import GeminiClientWrapper
from config import get_config

logger = get_logger(__name__)

# Page config
st.set_page_config(
    page_title="AgenticRAG - Conversational AI with Document Memory",
    page_icon="🤖",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
<style>
    .main {
        padding: 1rem;
    }
    .stTabs [data-baseweb="tab-list"] {
        gap: 2rem;
    }
    .doc-count {
        font-size: 0.9rem;
        color: #666;
        margin: 0.5rem 0;
    }
    .status-badge {
        display: inline-block;
        padding: 0.25rem 0.75rem;
        border-radius: 1rem;
        font-size: 0.85rem;
        font-weight: bold;
    }
    .status-active {
        background-color: #d4edda;
        color: #155724;
    }
    .status-inactive {
        background-color: #f8d7da;
        color: #721c24;
    }
</style>
""", unsafe_allow_html=True)


# =====================
# CACHED RESOURCE LOADERS
# =====================
# These use @st.cache_resource to load heavy objects ONCE
# and share them across all components (survives Streamlit reruns)

@st.cache_resource
def load_config():
    """Load configuration once."""
    return get_config()

@st.cache_resource
def load_embedding_service():
    """Load the embedding model once (sentence-transformers ~100MB)."""
    config = load_config()
    return EmbeddingService(config.embedding)

@st.cache_resource
def load_vector_store():
    """Initialize ChromaDB connection once."""
    config = load_config()
    return VectorStore(config.vector_database)

@st.cache_resource
def load_gemini_client():
    """Initialize Gemini API client once."""
    return GeminiClientWrapper()

@st.cache_resource
def load_document_loader():
    """Initialize document loader once."""
    return DocumentLoader()


# Initialize session state
def init_session_state():
    """Initialize Streamlit session state with shared cached resources."""
    if "initialized" not in st.session_state:
        # Load shared resources (cached - only created once)
        config = load_config()
        vector_store = load_vector_store()
        embedding_service = load_embedding_service()
        gemini_client = load_gemini_client()
        document_loader = load_document_loader()

        # System components - pass shared instances to avoid duplicates
        st.session_state.memory_manager = MemoryManager(
            vector_store=vector_store,
            embedding_service=embedding_service,
            document_loader=document_loader
        )

        st.session_state.vector_store = vector_store
        st.session_state.embedding_service = embedding_service
        st.session_state.document_store = None  # Created lazily when needed

        st.session_state.rag_pipeline = RAGPipeline(
            memory_manager=st.session_state.memory_manager,
            vector_store=vector_store,
            embedding_service=embedding_service,
            client=gemini_client
        )
        st.session_state.orchestrator = OrchestratorAgent(client=gemini_client)
        st.session_state.document_loader = document_loader

        # Chat state
        st.session_state.messages = []
        st.session_state.current_user = "web_user"
        st.session_state.user_id = "web_user"

        # Session management
        st.session_state.session_store = SessionStore(user_id="web_user")
        st.session_state.current_session_id = None
        st.session_state.summarize_agent = SummarizeAgent(
            client=gemini_client,
            vector_store=vector_store,
            embedding_service=embedding_service,
        )

        # UI state
        st.session_state.loaded_documents = {}
        st.session_state.active_documents = set()

        # Quota error state for retry functionality
        st.session_state.quota_error = False
        st.session_state.last_failed_message = ""

        st.session_state.initialized = True

        logger.info("Streamlit session initialized with shared cached resources")


init_session_state()


def add_message_to_chat(role: str, content: str, metadata: Dict = None):
    """Add message to chat history and persist to session file."""
    msg = {
        "role": role,
        "content": content,
        "timestamp": datetime.now().isoformat(),
        "metadata": metadata or {}
    }
    st.session_state.messages.append(msg)

    # Persist to session JSON
    sid = st.session_state.current_session_id
    if sid:
        try:
            st.session_state.session_store.save_turn(
                session_id=sid,
                role=role,
                content=content,
                metadata=metadata,
            )
        except Exception as exc:
            logger.warning(f"Failed to persist turn: {exc}")


def _ensure_session():
    """Make sure there is an active session. Create one if needed."""
    if not st.session_state.current_session_id:
        sid = st.session_state.session_store.create_session()
        st.session_state.current_session_id = sid
        st.session_state.messages = []


def _switch_to_session(session_id: str):
    """Load an existing session into the Chat tab."""
    # Summarize outgoing session first
    _summarize_current_session()

    # Load target session
    data = st.session_state.session_store.load_session(session_id)
    if data is None:
        st.error("Session not found.")
        return

    st.session_state.current_session_id = session_id
    st.session_state.messages = data.get("messages", [])


def _start_new_session():
    """Start a brand-new chat session."""
    _summarize_current_session()

    # Prune old sessions based on config
    config = get_config()
    max_sessions = getattr(config.memory, "max_chat_sessions", 5)
    st.session_state.session_store.delete_oldest_sessions(keep=max_sessions)

    sid = st.session_state.session_store.create_session()
    st.session_state.current_session_id = sid
    st.session_state.messages = []


def _summarize_current_session():
    """Summarize the current session if it qualifies (≥2 turns, not yet summarized)."""
    sid = st.session_state.current_session_id
    if not sid:
        return
    data = st.session_state.session_store.get_unsummarized_session(sid)
    if data is None:
        return
    try:
        summary = st.session_state.summarize_agent.summarize_and_store(
            user_id=st.session_state.user_id,
            session_id=sid,
            messages=data["messages"],
            session_title=data.get("title", ""),
        )
        if summary:
            st.session_state.session_store.mark_summarized(sid, summary)
    except Exception as exc:
        logger.error(f"Session summarization failed: {exc}")


def process_user_query(query: str) -> str:
    """Process user query through the system."""
    try:
        logger.info(f"Processing query: {query[:100]}...")
        
        # Get conversation history
        history = [
            {"role": m["role"], "content": m["content"]}
            for m in st.session_state.messages[-6:]  # Last 3 turns
        ]
        
        # Get active documents info
        active_docs_count = len(st.session_state.active_documents)
        total_docs = len(st.session_state.loaded_documents)
        
        # Process through orchestrator and RAG
        result = st.session_state.rag_pipeline.generate_response(
            query=query,
            user_id=st.session_state.user_id,
            conversation_history=history,
            use_memory=True
        )
        
        # Store interaction with active documents info
        st.session_state.memory_manager.store_interaction(
            user_id=st.session_state.user_id,
            user_message=query,
            assistant_response=result.get("response", ""),
            metadata={
                "source": "web_ui",
                "active_documents": ", ".join(st.session_state.active_documents) if st.session_state.active_documents else "none",
                "document_count": active_docs_count,
                "total_documents": total_docs,
                "retrieved_context": len(result.get("context", []))
            }
        )
        
        return result.get("response", "No response generated")
        
    except Exception as e:
        logger.error(f"Error processing query: {e}")
        error_str = str(e).lower()
        
        # Check if this is a quota/rate limit error
        if "quota" in error_str or "rate" in error_str or "429" in error_str or "exhausted" in error_str:
            st.session_state.quota_error = True
            st.session_state.last_failed_message = query
            return f"⚠️ API quota exceeded. Please click 'Try Again with Different Key' to retry."
        
        return f"❌ Error: {str(e)}"


def load_document_file(file_path: str, context_type: str = "document", filename: str = ""):
    """Load a document file and store in DocumentStore."""
    try:
        # Lazy initialize DocumentStore if needed
        if st.session_state.document_store is None:
            st.session_state.document_store = DocumentStore(
                st.session_state.vector_store,
                st.session_state.embedding_service
            )
        
        # Load document content
        document_content = st.session_state.document_loader.load_file(file_path)
        
        # Store in DocumentStore (separate from conversations)
        doc_id = st.session_state.document_store.store_document(
            user_id=st.session_state.user_id,
            document_content=document_content,
            filename=filename or Path(file_path).name,
            context_type=context_type
        )
        
        logger.info(f"Document stored with ID: {doc_id}")
        return True
    except Exception as e:
        logger.error(f"Error loading document: {e}")
        st.error(f"Failed to load document: {e}")
        return False
        return False


# =====================
# SIDEBAR - Controls
# =====================

with st.sidebar:
    st.title("🤖 AgenticRAG Control Panel")
    
    # User settings
    st.subheader("👤 User Settings")
    user_id = st.text_input(
        "User ID",
        value=st.session_state.user_id,
        key="user_id_input"
    )
    if user_id != st.session_state.user_id:
        st.session_state.user_id = user_id
        st.session_state.messages = []  # Reset chat for new user
        st.rerun()
    
    # System status
    st.subheader("📊 System Status")
    col1, col2 = st.columns(2)
    with col1:
        st.markdown(
            '<span class="status-badge status-active">✓ Active</span>',
            unsafe_allow_html=True
        )
    with col2:
        st.caption("All systems operational")
    
    # Config info
    config = get_config()
    st.subheader("⚙️ Configuration")
    with st.expander("View Config", expanded=False):
        st.info(f"""
        **Orchestrator Model:** {config.orchestrator.model}
        **LLM Model:** {config.llm.model}
        **Embedding:** {config.embedding.model}
        **RAG Top-K:** {config.rag.top_k_documents}
        """)
    
    # Clear chat — wipes all sessions from History tab and starts fresh
    if st.button("🗑️ Clear Chat History", use_container_width=True):
        st.session_state.session_store.clear_all_sessions()
        st.session_state.current_session_id = None
        _start_new_session()
        st.rerun()
    
    # Clear all data with confirmation
    st.markdown("---")
    if st.button("🔴 Clear All Data", use_container_width=True, help="Delete all conversations and documents from vector database"):
        # Show confirmation dialog
        if "show_clear_confirmation" not in st.session_state:
            st.session_state.show_clear_confirmation = True
        
        if st.session_state.show_clear_confirmation:
            st.error("⚠️ **WARNING: This will permanently delete ALL data including:")
            st.error("- All conversation history")
            st.error("- All uploaded documents")
            st.error("- All vector embeddings")
            
            col1, col2 = st.columns(2)
            with col1:
                if st.button("✅ Yes, Delete Everything", type="primary", use_container_width=True):
                    try:
                        with st.spinner("Clearing all data from vector store..."):
                            # Reset collections to fix schema issues, then clear data
                            reset_success = st.session_state.vector_store.reset_collections()
                            clear_success = st.session_state.vector_store.clear_all_data()
                            
                        if reset_success or clear_success:
                            # Clear session state
                            st.session_state.messages = []
                            st.session_state.loaded_documents = {}
                            st.session_state.active_documents = set()
                            st.session_state.show_clear_confirmation = False
                            
                            # Clear in-memory state from MemoryManager
                            st.session_state.memory_manager.clear_memory()

                            # Clear session files on disk and reset session state
                            st.session_state.session_store.clear_all_sessions()
                            st.session_state.current_session_id = None
                            _ensure_session()

                            # Reinitialize RAG pipeline's DocumentStore with fresh references
                            st.session_state.rag_pipeline.document_store = DocumentStore(
                                st.session_state.vector_store,
                                st.session_state.embedding_service
                            )

                            # Reinitialize document store to clear any cached data
                            if st.session_state.document_store is not None:
                                st.session_state.document_store = None

                            # Reset quota error state
                            st.session_state.quota_error = False
                            st.session_state.last_failed_message = ""

                            st.success("✅ All data has been cleared successfully!")
                            st.balloons()
                            time.sleep(2)
                            st.rerun()
                        else:
                            st.error("❌ Failed to clear data. Please check the logs for details.")
                            
                    except Exception as e:
                        st.error(f"❌ Error clearing data: {str(e)}")
                        logger.error(f"UI clear data error: {e}")
            
            with col2:
                if st.button("❌ Cancel", use_container_width=True):
                    st.session_state.show_clear_confirmation = False
                    st.rerun()
    
    # Documentation
    st.divider()
    st.subheader("📚 Documentation")
    st.markdown("[📖 Quick Start](QUICKSTART.md)")
    st.markdown("[👨‍💻 Developer Guide](README_DEVELOPERS.md)")
    st.markdown("[📄 Document Loading](DOCUMENT_LOADING.md)")


# =====================
# MAIN CONTENT
# =====================

st.title("🤖 AgenticRAG - Intelligent Conversational AI")
st.markdown("_Powered by Google Gemini with Document Memory_")

# Ensure there is always an active session
_ensure_session()

# Create tabs
tab1, tab2, tab3, tab4 = st.tabs(["💬 Chat", "📄 Documents", "📜 History", "⚙️ Settings"])

# =====================
# TAB 1 - CHAT
# =====================

with tab1:
    # Header with New Chat button
    header_col1, header_col2 = st.columns([0.8, 0.2])
    with header_col1:
        st.subheader("💬 Conversation with Memory")
    with header_col2:
        if st.button("➕ New Chat", use_container_width=True, type="primary"):
            _start_new_session()
            st.rerun()

    st.markdown("Ask questions and the system will remember previous conversations and loaded documents.")
    
    # Chat display
    chat_container = st.container()
    
    with chat_container:
        for idx, msg in enumerate(st.session_state.messages):
            if msg["role"] == "user":
                st_message(msg["content"], is_user=True, key=f"msg_user_{idx}")
            else:
                st_message(msg["content"], is_user=False, key=f"msg_assistant_{idx}")
    
    # (input handled below, outside tabs)
    
    # Quota error retry button
    if st.session_state.quota_error:
        st.divider()
        
        # Get key status
        key_manager = get_api_key_manager()
        key_status = key_manager.get_status()
        
        st.warning(f"⚠️ **API quota exceeded** - Key {key_status['current_key_index']}/{key_status['total_keys']} hit rate limit.")
        
        if key_status['has_available_keys']:
            st.info(f"🔑 {key_status['available_keys_count']} API key(s) still available. Click below to try with a different key.")
            
            col1, col2 = st.columns([0.5, 0.5])
            with col1:
                if st.button("🔄 Try Again with Different Key", type="primary", use_container_width=True):
                    # Rotate to next key
                    if key_manager.rotate_to_next_key():
                        st.session_state.quota_error = False
                        
                        # Re-process last message
                        with st.spinner("🔄 Retrying with different API key..."):
                            response = process_user_query(st.session_state.last_failed_message)
                            
                            # Update last assistant message
                            if st.session_state.messages and st.session_state.messages[-1]["role"] == "assistant":
                                st.session_state.messages[-1]["content"] = response
                            
                        st.rerun()
                    else:
                        st.error("❌ Failed to rotate to next key.")
            
            with col2:
                if st.button("❌ Cancel", use_container_width=True):
                    st.session_state.quota_error = False
                    st.session_state.last_failed_message = ""
                    st.rerun()
        else:
            st.error(f"""❌ **All API keys exhausted!**
            
All {key_status['total_keys']} API key(s) have exceeded their quota. Please:
- Wait for quota to refresh (usually resets daily)
- Add more API keys to `.env` file as `GEMINI_API_KEYS=key1,key2,key3`
- Upgrade to a paid Gemini API plan""")
            
            if st.button("🔄 Reset All Keys", help="Reset all keys and try again (use if quotas have refreshed)"):
                key_manager.reset_failed_keys()
                st.session_state.quota_error = False
                st.success("All keys reset! You can now try again.")
                st.rerun()


# =====================
# TAB 2 - DOCUMENTS
# =====================

with tab2:
    st.subheader("📄 Document Management")
    st.markdown("Upload PDF, Word documents, or images to add to your knowledge base. They will be automatically embedded and searchable.")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.info("""
        **Supported Formats:**
        - 📕 PDF (text + OCR for scanned)
        - 📗 Word (.docx)
        - 📄 Text (.txt)
        - 🖼️ Images (.png, .jpg, .gif)
        """)
    
    with col2:
        st.info("""
        **Features:**
        - Automatic text extraction
        - OCR for scanned documents
        - Table extraction from Word docs
        - Semantic search
        - Memory integration
        """)
    
    st.divider()
    
    # Document upload
    st.subheader("⬆️ Upload Documents")
    
    uploaded_files = st.file_uploader(
        "Choose files to upload",
        type=["pdf", "docx", "doc", "txt", "png", "jpg", "jpeg", "gif", "bmp"],
        accept_multiple_files=True,
        label_visibility="collapsed"
    )
    
    # Track current uploaded files
    current_uploaded = {f.name for f in uploaded_files} if uploaded_files else set()
    previously_loaded = set(st.session_state.loaded_documents.keys())
    
    # NOTE: Removed automatic removal logic to prevent documents from disappearing
    # when file_uploader clears (which happens after each upload in Streamlit)
    # Users can manually clear documents using the "Clear All Data" button
    
    if uploaded_files:
        progress_bar = st.progress(0)
        status_text = st.empty()
        
        for idx, uploaded_file in enumerate(uploaded_files):
            # Skip if already loaded
            if uploaded_file.name in previously_loaded:
                progress_bar.progress((idx + 1) / len(uploaded_files))
                continue
            
            status_text.text(f"Processing {uploaded_file.name}...")
            
            try:
                # Save temporarily
                with tempfile.NamedTemporaryFile(delete=False, suffix=Path(uploaded_file.name).suffix) as tmp:
                    tmp.write(uploaded_file.getbuffer())
                    tmp_path = tmp.name
                
                # Load document using DocumentStore
                load_document_file(
                    tmp_path,
                    context_type="uploaded_document",
                    filename=uploaded_file.name
                )
                
                st.session_state.loaded_documents[uploaded_file.name] = {
                    "size": uploaded_file.size,
                    "uploaded_at": datetime.now().isoformat()
                }
                
                # AUTO-ENABLE newly uploaded documents for RAG
                st.session_state.active_documents.add(uploaded_file.name)
                
                st.success(f"✓ Loaded: {uploaded_file.name} (enabled for RAG)")
                
                # Cleanup
                os.unlink(tmp_path)
                
            except Exception as e:
                st.error(f"✗ Failed to load {uploaded_file.name}: {e}")
            
            progress_bar.progress((idx + 1) / len(uploaded_files))
        
        status_text.empty()
        progress_bar.empty()
    
    # Loaded documents list with enable/disable toggles
    if st.session_state.loaded_documents:
        st.subheader("✓ Loaded Documents")
        st.markdown("**Enable/Disable documents to control which ones are used for RAG queries**")
        st.divider()
        
        doc_names = list(st.session_state.loaded_documents.keys())
        
        for name in doc_names:
            info = st.session_state.loaded_documents[name]
            is_active = name in st.session_state.active_documents
            
            col1, col2, col3, col4, col5, col6 = st.columns([0.08, 0.4, 0.15, 0.12, 0.12, 0.13])
            
            with col1:
                # Toggle checkbox
                new_state = st.checkbox(
                    "enabled",
                    value=is_active,
                    key=f"toggle_{name}",
                    label_visibility="collapsed"
                )
                # Update active documents set
                if new_state and not is_active:
                    st.session_state.active_documents.add(name)
                elif not new_state and is_active:
                    st.session_state.active_documents.discard(name)
            
            with col2:
                st.caption(f"📄 {name}")
            
            with col3:
                st.caption(f"📊 {info['size'] / 1024:.1f} KB")
            
            with col4:
                st.caption(f"📅 {info['uploaded_at'][:10]}")
            
            with col5:
                status_text = "✓ Active" if new_state else "✗ Disabled"
                status_color = "🟢" if new_state else "🔴"
                st.caption(f"{status_color} {status_text}")
            
            with col6:
                if st.button("🗑️", key=f"remove_{name}", help="Remove document", use_container_width=True):
                    # Remove from session state
                    st.session_state.loaded_documents.pop(name, None)
                    st.session_state.active_documents.discard(name)
                    st.success(f"Removed: {name}")
                    st.rerun()
        
        st.divider()
        active_count = len(st.session_state.active_documents)
        total_count = len(st.session_state.loaded_documents)
        
        if active_count == 0:
            st.warning(f"⚠️ No documents enabled! ({active_count}/{total_count} documents selected)")
        else:
            st.success(f"✓ Active documents: {active_count}/{total_count} (will be used for RAG)")


# =====================
# TAB 3 - CHAT HISTORY
# =====================

with tab3:
    st.subheader("📜 Chat History")
    st.markdown("Browse and resume previous conversations. The system remembers context from past sessions.")

    sessions = st.session_state.session_store.list_sessions(limit=10)

    if not sessions:
        st.info("No chat history yet. Start a conversation in the Chat tab!")
    else:
        for idx, meta in enumerate(sessions):
            is_current = meta.session_id == st.session_state.current_session_id
            badge = "  🟢 _active_" if is_current else ""

            with st.expander(
                f"**{meta.title}**{badge}  —  {meta.message_count} messages",
                expanded=False,
            ):
                col1, col2, col3 = st.columns([0.4, 0.3, 0.3])
                with col1:
                    st.caption(f"📅 {meta.created_at[:16]}")
                with col2:
                    st.caption(f"🔄 {meta.updated_at[:16]}")
                with col3:
                    summarized_icon = "✅ Summarized" if meta.is_summarized else "⏳ Not summarized"
                    st.caption(summarized_icon)

                # Preview first few messages
                data = st.session_state.session_store.load_session(meta.session_id)
                if data and data.get("messages"):
                    preview_msgs = data["messages"][:4]
                    for pm in preview_msgs:
                        role_icon = "👤" if pm["role"] == "user" else "🤖"
                        st.markdown(f"{role_icon} {pm['content'][:120]}{'…' if len(pm['content']) > 120 else ''}")
                    if len(data["messages"]) > 4:
                        st.caption(f"… and {len(data['messages']) - 4} more messages")

                # Show summary if available
                if data and data.get("summary"):
                    st.info(f"📝 **Summary:** {data['summary']}")

                btn_col1, btn_col2 = st.columns(2)
                with btn_col1:
                    if not is_current:
                        if st.button("💬 Resume", key=f"resume_{meta.session_id}", use_container_width=True):
                            _switch_to_session(meta.session_id)
                            st.rerun()
                    else:
                        st.button("💬 Current", key=f"current_{meta.session_id}", disabled=True, use_container_width=True)
                with btn_col2:
                    if st.button("🗑️ Delete", key=f"del_{meta.session_id}", use_container_width=True):
                        st.session_state.session_store.delete_session(meta.session_id)
                        if is_current:
                            _start_new_session()
                        st.rerun()


# =====================
# TAB 4 - SETTINGS
# =====================

with tab4:
    st.subheader("⚙️ System Settings")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("🎯 System Configuration")
        
        with st.expander("LLM Settings", expanded=True):
            st.caption("Temperature (creativity)")
            st.slider(
                "LLM Temperature",
                min_value=0.0,
                max_value=1.0,
                value=0.7,
                step=0.1,
                help="Lower = more deterministic, Higher = more creative"
            )
            
            st.caption("Response Length")
            st.slider(
                "Max Tokens",
                min_value=100,
                max_value=2000,
                value=1000,
                step=100,
                help="Maximum tokens in response"
            )
        
        with st.expander("Memory Settings", expanded=True):
            st.caption("Retrieval Count")
            st.slider(
                "Top-K Documents",
                min_value=1,
                max_value=20,
                value=5,
                help="Number of documents to retrieve"
            )
            
            st.caption("Relevance Threshold")
            st.slider(
                "Min Similarity",
                min_value=0.0,
                max_value=1.0,
                value=0.7,
                step=0.05,
                help="Minimum similarity score for retrieval"
            )

            st.caption("Chat History Limit")
            max_sessions = st.slider(
                "Max Chat Sessions",
                min_value=1,
                max_value=10,
                value=config.memory.max_chat_sessions,
                help="Number of chat sessions to keep in history (1-10)",
                key="max_chat_sessions_slider",
            )
    
    with col2:
        st.subheader("📊 System Information")
        
        info_text = f"""
        **System Status:** ✅ Online
        
        **Components:**
        - 🤖 Orchestrator Agent: Ready
        - 📚 Memory Manager: Ready
        - 🔍 RAG Pipeline: Ready
        - 💾 Vector Store: Ready
        
        **Current User:** {st.session_state.user_id}
        **Chat Messages:** {len(st.session_state.messages)}
        **Loaded Documents:** {len(st.session_state.loaded_documents)}
        
        **Models:**
        - Orchestrator: {config.orchestrator.model}
        - LLM: {config.llm.model}
        - Embedding: {config.embedding.model.split('/')[-1]}
        
        **Vector Database:** ChromaDB
        """
        
        st.info(info_text)
    
    st.divider()
    
    # Advanced settings
    with st.expander("🔧 Advanced Settings"):
        col1, col2 = st.columns(2)
        
        with col1:
            enable_validation = st.checkbox(
                "Enable Response Validation",
                value=True,
                help="Validate responses for quality"
            )
            
            enable_memory = st.checkbox(
                "Enable Memory Storage",
                value=True,
                help="Store interactions for future retrieval"
            )
        
        with col2:
            query_expansion = st.checkbox(
                "Enable Query Expansion",
                value=True,
                help="Expand queries for better retrieval"
            )
            
            debug_mode = st.checkbox(
                "Debug Mode",
                value=False,
                help="Show detailed logs"
            )
    
    st.divider()
    
    # Help section
    st.subheader("❓ Help & Support")
    with st.expander("How to use this system"):
        st.markdown("""
        1. **Chat Tab:** Ask questions naturally. The system learns from conversations.
        2. **Documents Tab:** Upload PDFs, Word docs, or images to build knowledge base.
        3. **Search Tab:** Search through all documents and past conversations.
        4. **Settings Tab:** Adjust system parameters and view configuration.
        
        **Tips:**
        - Ask follow-up questions - the system remembers context
        - Upload reference materials to improve answers
        - Use specific queries for better search results
        - The system automatically extracts text from images and PDFs
        """)
    
    with st.expander("Troubleshooting"):
        st.markdown("""
        **Issue:** Slow responses
        - Reduce max tokens
        - Reduce top-K retrieval count
        
        **Issue:** Poor search results
        - Use more specific keywords
        - Upload more relevant documents
        - Adjust minimum similarity threshold
        
        **Issue:** Memory not working
        - Ensure "Enable Memory Storage" is checked
        - Check system logs for errors
        """)


# =====================
# CHAT INPUT (top-level — required by Streamlit, cannot be inside tabs/columns/expanders)
# =====================

user_input = st.chat_input("Ask me anything...")

if user_input:
    with st.spinner("💭 Thinking..."):
        add_message_to_chat("user", user_input)
        response = process_user_query(user_input)
        add_message_to_chat("assistant", response)
        st.rerun()

# =====================
# FOOTER
# =====================

st.divider()
col1, col2, col3 = st.columns(3)

with col1:
    st.caption("🤖 AgenticRAG System")

with col2:
    st.caption("Powered by Google Gemini AI")

with col3:
    st.caption("💾 Vector Database: ChromaDB")
