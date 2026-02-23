# AgenticRAG - System Overview for Developers

## What This System Does

**AgenticRAG** is an intelligent conversational system powered by Google's Gemini AI that:
- 🤖 Understands user queries intelligently using an Orchestrator Agent
- 💾 Maintains conversation memory with semantic search capabilities
- 📚 Retrieves relevant context from uploaded documents and past interactions (RAG - Retrieval Augmented Generation)
- 🎯 Routes queries to appropriate modules (memory retrieval, document search, LLM response)
- ✅ Validates responses for quality and relevance
- 📄 Supports document upload (PDF, Word, Images) with OCR capabilities

**Example Use Case**: Upload course documents (PDFs, Word files) and ask questions about their content. The system remembers past conversations and provides accurate, document-based answers.

---

## 🧠 Vector Database & Chunking Architecture

### **Document Processing Pipeline**

```
Document Upload → Text Extraction → Chunking → Embedding Generation → ChromaDB Storage
     ↓                ↓              ↓              ↓                    ↓
  PDF/Word/Img    pypdf/OCR    Overlapping    Sentence-Transformers   Individual Chunks
   Files         Extract      Chunks         (384-dim vectors)      with Metadata
```

### **🔪 Chunking System**

**Location**: [`memory/document_store.py`](memory/document_store.py) & [`utils/document_loader.py`](utils/document_loader.py)

#### **Chunking Algorithm**:
```python
def _chunk_text(text: str, chunk_size: int = 1500, overlap: int = 300) -> List[str]:
    chunks = []
    start = 0
    
    while start < len(text):
        end = start + chunk_size
        chunks.append(text[start:end])
        start = end - overlap  # Overlap ensures context continuity
    
    return chunks
```

#### **Current Configuration** (`config/config.yaml`):
```yaml
chunking:
  enable_chunking: true              # Enable document chunking
  chunk_size: 1500                  # Characters per chunk
  chunk_overlap: 300                 # Overlap between chunks (preserves context)
  min_chunk_size: 300               # Minimum chunk size to store
  chunk_search_multiplier: 3        # Search multiplier for deduplication
```

#### **Chunk Storage Strategy**:
1. **Small Documents** (< 300 chars): Stored as single chunk
2. **Large Documents**: Split into overlapping chunks
3. **Each Chunk Gets**:
   - Individual vector embedding (384 dimensions)
   - Rich metadata (filename, chunk_number, position, etc.)
   - Separate storage in ChromaDB documents collection

#### **Metadata per Chunk**:
```python
chunk_metadata = {
    "user_id": user_id,
    "filename": filename,
    "chunk_number": i,
    "total_chunks": len(chunks),
    "chunk_type": "chunked",  # vs "single"
    "start_position": start_pos,
    "end_position": end_pos,
    "content_length": len(chunk),
    "timestamp": datetime.now().isoformat(),
    "document_type": "uploaded_knowledge"
}
```

### **🔍 Search & Retrieval System**

#### **Search Process**:
1. **Query Embedding**: Convert user query to 384-dim vector
2. **Vector Search**: Find similar chunks in ChromaDB
3. **Chunk Deduplication**: Group chunks by document, keep best ones
4. **Context Building**: Combine selected chunks for LLM

#### **Deduplication Algorithm**:
```python
def _deduplicate_chunks(results, max_chunks_per_document=3):
    # Group chunks by document
    document_groups = {}
    
    for result in results:
        filename = result.get("metadata", {}).get("filename")
        doc_key = f"{filename}_chunked"
        
        if doc_key not in document_groups:
            document_groups[doc_key] = []
        document_groups[doc_key].append(result)
    
    # Select best chunks from each document
    deduplicated_results = []
    for doc_key, chunks in document_groups.items():
        # Sort by similarity score (descending)
        chunks.sort(key=lambda x: x.get("similarity", 0), reverse=True)
        # Take top chunks for this document
        best_chunks = chunks[:max_chunks_per_document]
        deduplicated_results.extend(best_chunks)
    
    # Sort all results by similarity
    deduplicated_results.sort(key=lambda x: x.get("similarity", 0), reverse=True)
    return deduplicated_results
```

#### **Search Configuration**:
```yaml
rag:
  top_k_documents: 8              # Initial search results
  similarity_threshold: 0.1       # Minimum similarity score
  max_chunks_per_document: 3       # Max chunks per doc after dedup
```

### **💾 ChromaDB Architecture**

#### **Collections**:
- **`kinetichat_memory`**: Conversation history and summaries
- **`kinetichat_memory_documents`**: Uploaded document chunks

#### **Vector Storage**:
- **Embedding Dimension**: 384 (sentence-transformers/all-MiniLM-L6-v2)
- **Distance Metric**: Squared Euclidean (converted to cosine similarity)
- **Persistence**: `data/vector_store/` directory
- **Similarity Calculation**: `max(0.0, 1.0 - (distance / 2.0))`

#### **Schema Management**:
- **Collections auto-created** on first use
- **Metadata fields** indexed for filtering
- **Reset capability**: `reset_collections()` for schema fixes

### **🎯 Why This Architecture Works**

#### **Chunking Benefits**:
- **Context Preservation**: 300-char overlap ensures complete thoughts aren't split
- **Granular Search**: Smaller chunks allow more precise matching
- **Flexible Retrieval**: Can retrieve specific sections vs entire documents
- **Memory Efficiency**: Only relevant chunks loaded into context

#### **Deduplication Benefits**:
- **Diverse Results**: Prevents one document from dominating results
- **Quality Focus**: Keeps highest similarity chunks per document
- **Context Balance**: Mix of multiple sources for comprehensive answers

#### **Vector Database Benefits**:
- **Semantic Search**: Finds content by meaning, not just keywords
- **Scalable Storage**: Efficient for large document collections
- **Fast Retrieval**: Optimized for similarity search operations
- **Persistent Storage**: Data survives application restarts

---

## Core System Architecture

```
User Query (with optional uploaded documents)
    ↓
1. ORCHESTRATOR AGENT (gemini-2.5-flash)
   └─ Analyzes query → decides action (retrieve memory? search documents? call LLM?)
    ↓
2. MEMORY RETRIEVAL (if needed)
   ├─ Embedding Service (sentence-transformers/all-MiniLM-L6-v2)
   ├─ Vector Store (ChromaDB with PersistentClient)
   └─ Retrieves relevant past interactions
    ↓
3. DOCUMENT SEARCH (if documents uploaded)
   ├─ Document Store (ChromaDB with proper similarity calculation)
   ├─ Document Loader (PDF, Word, Images with OCR)
   └─ Retrieves relevant document content
    ↓
4. RAG PIPELINE (gemini-2.5-flash)
   ├─ Query Processing
   ├─ Context Building (with retrieved memory + documents)
   ├─ LLM Response Generation
   └─ Response Validation
    ↓
5. MEMORY STORAGE
   └─ Stores new interaction for future retrieval
    ↓
User Response
```

---

## Main Code Components

### **1. Orchestrator Agent** 
**Location**: [`agents/orchestrator.py`](agents/orchestrator.py)  
**What it does**: Analyzes incoming queries and decides which modules to activate
- **Key method**: `analyze_query()` - Uses LLM to parse query intent
- **Key method**: `process_query()` - Routes to appropriate action
- **Config**: `config.orchestrator` in `config/config.yaml`

**If you need to modify routing logic**: Edit the `_build_analysis_prompt()` method

---

### **2. RAG Pipeline** 
**Location**: [`retrieval/rag_pipeline.py`](retrieval/rag_pipeline.py)  
**What it does**: Generates context-aware responses using retrieved memory and documents
- **Key method**: `generate_response()` - Main RAG orchestration
- **Key method**: `_retrieve_context()` - Fetches relevant past interactions AND documents
- **Key method**: `_build_prompt()` - Constructs full prompt for LLM with keyword extraction
- **Key method**: `_generate_llm_response()` - Calls Gemini API
- **Config**: `config.rag` and `config.llm` in `config/config.yaml`

**If you need to modify response generation**: Edit `_generate_llm_response()` or `_build_prompt()`

---

### **3. Memory Manager** 
**Location**: [`memory/memory_manager.py`](memory/memory_manager.py)  
**What it does**: Manages conversation memory and interactions
- **Key method**: `store_interaction()` - Saves user query + assistant response
- **Key method**: `get_user_memory()` - Retrieves stored interactions for a user
- **Key method**: `search_memory()` - Semantic search in memory
- **Key method**: `load_documents_from_file()` - Load and index uploaded documents
- **Config**: `config.memory` in `config/config.yaml`

**If you need to modify memory behavior**: Edit methods here

---

### **4. Document Store** 
**Location**: [`memory/document_store.py`](memory/document_store.py)  
**What it does**: Advanced document storage with intelligent chunking and retrieval
- **Key method**: `store_document()` - Intelligent chunking based on document size
- **Key method**: `search_documents()` - Semantic search with chunk deduplication
- **Key method**: `_store_chunked_document()` - Handles large document chunking
- **Key method**: `_deduplicate_chunks()` - Prevents document domination in results
- **Integration**: Works with DocumentLoader for multi-format file processing
- **Storage**: Uses ChromaDB with separate collections for conversations and documents
- **Chunking Config**: `config.chunking` in `config/config.yaml`

**Chunking Strategy**:
- **< 300 chars**: Single chunk storage
- **≥ 300 chars**: Overlapping chunks (1500 chars, 300 overlap)
- **Metadata**: Rich chunk information (position, number, parent doc, etc.)

**If you need to modify document storage**: Edit chunking parameters in `config.yaml` or methods in `document_store.py`

---

### **5. Vector Store** 
**Location**: [`memory/vector_store.py`](memory/vector_store.py)  
**What it does**: Advanced vector storage with dual collections and schema management
- **Backend**: ChromaDB with PersistentClient for data persistence
- **Collections**: Separate storage for conversations and document chunks
- **Key method**: `add_documents()` - Stores embeddings with metadata
- **Key method**: `search()` - Semantic search with proper similarity calculation
- **Key method**: `clear_all_data()` - Enhanced data clearing with verification
- **Key method**: `reset_collections()` - Complete collection reset for schema fixes
- **Key fix**: Similarity calculation fixed from `1 - distance` to `max(0.0, 1.0 - (distance / 2.0))`
- **Storage**: `data/vector_store/` directory (persistent across restarts)

**Collections**:
- **`kinetichat_memory`**: Conversation history and summaries
- **`kinetichat_memory_documents`**: Document chunks with rich metadata

**Schema Management**:
- **Auto-creation**: Collections created on first use
- **Reset capability**: Full schema reset for corruption recovery
- **Metadata indexing**: Optimized for filtering by user_id, filename, etc.

**If you need to switch backends**: Edit `_init_chromadb()` or `_init_qdrant()`

---

### **6. Embedding Service** 
**Location**: [`memory/embedding_service.py`](memory/embedding_service.py)  
**What it does**: Converts text to embeddings for semantic search
- **Model**: `sentence-transformers/all-MiniLM-L6-v2` (384-dim)
- **Key method**: `embed_texts()` - Converts text to embedding vector
- **Config**: `config.embedding` in `config/config.yaml`
- **Note**: Normalization removed to work with ChromaDB's distance calculation

**If you need to change embedding model**: Edit `_init_sentence_transformer()`

---

### **7. Gemini Client** 
**Location**: [`utils/gemini_client.py`](utils/gemini_client.py)  
**What it does**: Wrapper around Google's Gemini API
- **Compatible with**: OpenAI-style interface (chat.completions.create())
- **Key class**: `GeminiClientWrapper` - Main interface
- **Key method**: `chat_completion()` - Calls Gemini API
- **Note**: Handles both old (0.3.x) and new (0.4.0+) API versions

**If you need to modify API calls**: Edit `chat_completion()` method

---

### **8. Document Loader** 
**Location**: [`utils/document_loader.py`](utils/document_loader.py)  
**What it does**: Loads and extracts text from multiple document formats
- **Supported formats**: PDF (text + OCR for scans), Word (.docx), Images (PNG/JPG with OCR), Text files
- **Key method**: `load_file()` - Loads single file
- **Key method**: `load_directory()` - Loads all files from folder
- **Dependencies**: pypdf, python-docx, pytesseract, pdf2image, pillow

**Integration**: Use via `MemoryManager.load_documents_from_file()` or `.load_documents_from_directory()`

---

### **9. Web Interface** 
**Location**: [`ui.py`](ui.py) and [`run_ui.py`](run_ui.py)  
**What it does**: Streamlit web interface for user interaction
- **Features**: File upload, chat interface, document management
- **Key method**: Handles file uploads and stores them in DocumentStore
- **User ID**: Uses `"web_user"` for all web interactions

**If you need to modify UI**: Edit `ui.py` for interface changes

---

## Configuration Reference

**File**: `config/config.yaml`

```yaml
orchestrator:
  model: "gemini-2.5-flash"        # LLM for routing decisions
  temperature: 0.1                   # Low for consistent routing
  max_tokens: 500

llm:
  model: "gemini-2.5-flash"         # LLM for response generation
  temperature: 0.7                   # Higher for more creative responses
  max_tokens: 1000
  enable_validation: false           # Disabled temporarily for debugging

embedding:
  model: "sentence-transformers/all-MiniLM-L6-v2"
  dimension: 384

rag:
  enable_query_expansion: true       # Expand queries before search
  top_k_documents: 8                 # Return top 8 document matches
  similarity_threshold: 0.3          # Min similarity for document retrieval

memory:
  max_items: 100                     # Max items to keep
  relevance_threshold: 0.7           # Min similarity for memory retrieval
```

---

## Recent Fixes & Improvements

### ✅ Fixed: ChromaDB Persistence Issue
**Problem**: Documents not persisting between sessions  
**Solution**: Changed from `chromadb.Client()` to `chromadb.PersistentClient()`  
**Location**: `memory/vector_store.py` - `_init_chromadb()` method

### ✅ Fixed: ChromaDB Query Error with Multiple Filters
**Problem**: Query failed with multiple filter conditions  
**Solution**: Used `$and` operator for multiple filter keys  
**Location**: `memory/vector_store.py` - `search()` and `search_documents()` methods

### ✅ Fixed: Similarity Calculation Issue
**Problem**: Negative similarities despite good embeddings  
**Solution**: Fixed similarity calculation from `1 - distance` to `max(0.0, 1.0 - (distance / 2.0))`  
**Location**: `memory/vector_store.py` - similarity calculation in both search methods

### ✅ Improved: Document Context Building
**Improvement**: Increased content limit from 300 to 800 characters, added keyword-based sentence extraction  
**Location**: `retrieval/rag_pipeline.py` - `_build_prompt()` method

### ✅ Optimized: Retrieval Parameters
**Improvement**: Increased top_k from 5 to 8, adjusted similarity threshold to 0.3  
**Location**: `config/config.yaml` - rag section

---

## Known Issues & Where to Fix

### ❌ Issue: Model not found (404 errors)
**Cause**: Invalid model name in config  
**Where to fix**: 
- Check available models: `python list_available_models.py`
- Update models in `config/config.yaml` (lines 36, 46)

### ❌ Issue: API rate limiting (429 errors)
**Cause**: Too many requests to Gemini API (free tier: 20 requests/day)  
**Where to fix**: 
- Reduce `max_retries` in `config/config.yaml`
- Add delays between requests in `retrieval/rag_pipeline.py`
- Consider upgrading to paid Gemini API tier

### ❌ Issue: Documents not being retrieved
**Cause**: Similarity threshold too high or embedding issues  
**Where to fix**: 
- Adjust `similarity_threshold` in `config/config.yaml`
- Check document storage with `test_upload_flow.py`

### ❌ Issue: Memory not retrieving relevant context
**Cause**: Embedding quality or vector store issue  
**Where to fix**: 
- Try different embedding model in `memory/embedding_service.py`
- Adjust similarity threshold in `memory/memory_manager.py`

---

## ⚙️ Configuration Overview

### **Key Configuration Files**
- **`config/config.yaml`**: Main system configuration
- **`.env`**: API keys and environment variables

### Test individual components
```bash
# Test embedding service
python -c "from memory.embedding_service import EmbeddingService; e = EmbeddingService(); print(len(e.embed_texts('test')))"

# Test vector store
python -c "from memory.vector_store import VectorStore; v = VectorStore(); print(v.search('test', 5))"

# Test Gemini API
python -c "from utils.gemini_client import GeminiClientWrapper; c = GeminiClientWrapper(); print(c.chat.completions.create(model='gemini-2.5-flash', messages=[{'role': 'user', 'content': 'hi'}]).choices[0].message.content)"
```

### Test document upload flow
```bash
python test_upload_flow.py
```

### List available models
```bash
python list_available_models.py
```

---

## File Structure

```
agentic_rag_gemini/
├── main.py                        # Entry point
├── run_ui.py                      # Web interface launcher
├── ui.py                          # Streamlit web interface
├── config/
│   ├── __init__.py               # Config loader
│   └── config.yaml               # Configuration (EDIT HERE FOR SETTINGS)
├── agents/
│   └── orchestrator.py           # Query routing logic
├── retrieval/
│   └── rag_pipeline.py           # Response generation pipeline
├── memory/
│   ├── memory_manager.py         # Conversation memory
│   ├── document_store.py         # Document storage and search
│   ├── vector_store.py           # Embedding storage (FIXED)
│   └── embedding_service.py      # Text-to-vector conversion
├── utils/
│   ├── gemini_client.py          # Gemini API wrapper
│   ├── validators.py             # Response validation
│   ├── prompt_templates.py       # Prompt templates
│   ├── logger.py                 # Logging setup
│   └── document_loader.py        # Document processing (PDF, Word, Images)
├── tests/
│   └── test_orchestrator.py      # Essential test file
├── data/
│   └── vector_store/             # Vector database files (PERSISTENT)
├── logs/
│   └── agentic_rag.log          # Runtime logs
├── QUICKSTART.md                 # User setup guide (KEEP THIS)
└── README_DEVELOPERS.md         # This file (KEEP THIS)
```

---

## How to Extend the System

### Add a new action type
1. Edit `agents/orchestrator.py` - add to `ActionType` enum
2. Edit `main.py` - handle new action in `process_query()`
3. Create new module if needed

### Change embedding model
1. Edit `config/config.yaml` - change `embedding.model`
2. Edit `memory/embedding_service.py` - update `_init_sentence_transformer()`

### Add response post-processing
1. Edit `retrieval/rag_pipeline.py` - add logic after `_generate_llm_response()`
2. Update response before returning

### Switch vector database
1. Edit `config/config.yaml` - set `vector_db.type: qdrant`
2. Edit `memory/vector_store.py` - update `_init_qdrant()`

---

## Testing the System

### Run interactive mode
```bash
python main.py --mode interactive
```

### Run web interface
```bash
python run_ui.py
```

### Run with test queries (add to main.py)
```python
test_queries = [
    "Hello",
    "What documents have been uploaded?",
    "What are the evaluation criteria?"
]
for query in test_queries:
    result = system.process_query(query, "web_user")
    print(result["response"])
```

---

## Loading Documents (PDF, Word, Images)

### Quick Start

```python
from memory.memory_manager import MemoryManager

mm = MemoryManager()

# Load a single PDF
mm.load_documents_from_file(
    user_id="web_user",
    file_path="documents/guide.pdf",
    context_type="uploaded_document"
)

# Search loaded documents
results = mm.search_documents(
    user_id="web_user",
    query="evaluation criteria",
    top_k=5
)
```

### Supported Formats

| Format | Support | Features |
|--------|---------|----------|
| PDF | ✅ Text + OCR | Scanned PDFs automatically detected and OCR'd |
| DOCX | ✅ Full | Extracts paragraphs and tables |
| TXT | ✅ Full | Plain text files |
| PNG/JPG | ✅ OCR | Automatic text extraction from images |
| GIF/BMP | ✅ OCR | Additional image formats |

### Setup OCR (Optional but Recommended)

For best OCR performance on scanned documents and images:

```bash
# Install system dependency (Windows)
choco install tesseract

# Install system dependency (Linux)
sudo apt-get install tesseract-ocr

# Verify installation
python -c "import pytesseract; print(pytesseract.get_tesseract_version())"
```

---

## Common Errors & Solutions

| Error | Cause | Solution |
|-------|-------|----------|
| `GEMINI_API_KEY not set` | Missing .env | Copy .env.example to .env and add your API key |
| `404 Model not found` | Invalid model name | Run `python list_available_models.py` to see available models |
| `Rate limited (429)` | Too many API calls | Wait 1-2 minutes or upgrade API quota |
| `Documents not retrieved` | Similarity threshold too high | Lower `similarity_threshold` in config.yaml |
| `ChromaDB persistence issue` | Using wrong client | Ensure PersistentClient is used (already fixed) |
| `Negative similarities` | Distance calculation error | Fixed with proper similarity calculation |

---

## For Production Deployment

1. ✅ Use environment variables for secrets (API key, model names)
2. ✅ Set `log_level: WARNING` in config
3. ✅ Increase `max_retries` for stability
4. ✅ Monitor `logs/` directory
5. ✅ Use persistent vector store (already configured)
6. ✅ Add authentication/authorization around web interface
7. ✅ Handle API rate limiting gracefully

---

## Quick Reference for Common Modifications

**Change LLM model**: `config/config.yaml` line 36  
**Change embedding model**: `config/config.yaml` line 40  
**Adjust response temperature**: `config/config.yaml` line 38  
**Increase document retrieval**: `config/config.yaml` line 118 (top_k_documents)  
**Adjust similarity threshold**: `config/config.yaml` line 119 (similarity_threshold)  
**Change validation rules**: `utils/validators.py` line ~60  
**Modify prompt template**: `utils/prompt_templates.py` or in each component  
**Add document support**: Use `memory.load_documents_from_file()` or `.load_documents_from_directory()`

---

## Contact/Questions

See individual files for detailed docstrings and method documentation.

**Start with QUICKSTART.md for user setup**  
**Start with this README_DEVELOPERS.md for development**
