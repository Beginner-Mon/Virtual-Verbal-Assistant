# AgenticRAG — Developer Documentation

> **Last updated**: 2026-03-15  
> **Python**: 3.13+ &nbsp;|&nbsp; **Local LLM**: Ollama (Qwen 0.5B) &nbsp;|&nbsp; **Cloud LLM**: Google Gemini 2.5 Flash &nbsp;|&nbsp; **Vector DB**: ChromaDB 1.5  
> **Status**: Optimization for Multi-User concurrency

---

## Table of Contents

1. [System Overview](#system-overview)
2. [High-Level Architecture](#high-level-architecture)
3. [Project Structure](#project-structure)
4. [Module Reference](#module-reference)
5. [Agentic Pipeline — Detailed Flow](#agentic-pipeline--detailed-flow)
6. [Vector Database & Chunking Architecture](#vector-database--chunking-architecture)
7. [Session & Memory Management](#session--memory-management)
8. [Web Search Fallback](#web-search-fallback)
9. [API Key Rotation System](#api-key-rotation-system)
10. [Prompt Engineering](#prompt-engineering)
11. [Configuration Reference](#configuration-reference)
12. [Environment Variables](#environment-variables)
13. [Setup & Installation](#setup--installation)
14. [Running the System](#running-the-system)
15. [Testing](#testing)
16. [How to Extend the System](#how-to-extend-the-system)
17. [Dependency Matrix](#dependency-matrix)
18. [Resolved Issues & Fixes](#resolved-issues--fixes)
19. [Known Limitations & Open Issues](#known-limitations--open-issues)
20. [Production Deployment Checklist](#production-deployment-checklist)

---

## System Overview

**AgenticRAG** (internally branded *KineticChat*) is an intelligent conversational AI system built on a **Retrieval-Augmented Generation** (RAG) architecture with **agentic decision-making**. The system:

- ⚡ **Sub-second Routing** — Uses a local **Ollama** model (Qwen 0.5B) for intent routing, reducing latency from ~15s to <1s.
- 🤖 **Hybrid Orchestration** — Local orchestrator for speed, with automatic fallback to Gemini API for high-complexity queries or when local confidence is low.
- 📚 **Multi-source knowledge** — Retrieves context from uploaded documents (PDF, Word, Images w/ OCR), conversation memory, chat session summaries, and live web search.
- 🔁 **Self-correcting pipeline** — Implements query reformulation, iterative reflection (verifies answers against sources), and web search fallback.
- 💾 **Persistent memory & Caching** — Stores state in ChromaDB and uses **Redis** for sub-millisecond embedding and retrieval caching.
- 🔑 **Multi-key API management** — Rotates across multiple Gemini API keys automatically on quota errors.
- 🌐 **Streamlit & API exposure** — Streamlit UI plus a high-performance FastAPI server (`api_server.py`) supporting concurrent requests.

### Example Use Case

Upload course documents (PDFs, Word files) and ask questions about their content. The system remembers past conversations, searches across documents semantically, and falls back to web search when local knowledge is insufficient.

---

## High-Level Architecture

```
┌─────────────────────────────────────────────────────────────────────────┐
│                         USER (Streamlit UI / CLI)                       │
│                    ui.py  ·  run_ui.py  ·  main.py                     │
└─────────────────────┬───────────────────────────────────┬───────────────┘
                      │ query + uploaded files             │ response
                      ▼                                    ▲
┌─────────────────────────────────────────────────────────────────────────┐
│  1. HYBRID ORCHESTRATOR               agents/local_orchestrator.py      │
│     ┌────────────────────────────────────────────────────┐              │
│     │ Primary: Ollama (Qwen 0.5B, local, temp=0.1)       │              │
│     │ Fallback: Gemini 2.5 Flash (via orchestrator.py)   │              │
│     │ Analyzes query → decides intent/action plan (JSON) │              │
│     │ Latency: ~200-500ms (Local) vs ~3-10s (API)        │              │
│     └────────────────────────────────────────────────────┘              │
└─────────────────────┬───────────────────────────────────────────────────┘
                      ▼
┌─────────────────────────────────────────────────────────────────────────┐
│  2. RAG PIPELINE                     retrieval/rag_pipeline.py         │
│                                                                         │
│  ┌──────────────┐  ┌──────────────┐  ┌──────────────┐  ┌────────────┐  │
│  │ Query        │→ │ Retrieve     │→ │ Quality      │→ │ Build      │  │
│  │ Expansion    │  │ Context      │  │ Assessment   │  │ Prompt     │  │
│  │ (LLM)       │  │ (hybrid)     │  │ + Reformulate│  │ + Generate │  │
│  └──────────────┘  └──────────────┘  └──────────────┘  └────────────┘  │
│                           │                 │                │          │
│                           ▼                 ▼                ▼          │
│                    ┌─────────────┐   ┌────────────┐   ┌────────────┐   │
│                    │ Web Search  │   │ Iterative  │   │ Response   │   │
│                    │ Fallback    │   │ Reflection │   │ Validation │   │
│                    │ (DuckDuckGo)│   │ (grounding)│   │ (safety)   │   │
│                    └─────────────┘   └────────────┘   └────────────┘   │
└─────────────────────┬───────────────────────────────────────────────────┘
                      ▼
┌─────────────────────────────────────────────────────────────────────────┐
│  3. MEMORY LAYER                                                        │
│                                                                         │
│  ┌─────────────────┐  ┌─────────────────┐  ┌─────────────────────────┐ │
│  │ Memory Manager  │  │ Document Store  │  │ Session Store           │ │
│  │ (conversations) │  │ (doc chunks)    │  │ (JSON-on-disk sessions) │ │
│  └────────┬────────┘  └────────┬────────┘  └────────────┬────────────┘ │
│           │                    │                         │              │
│           ▼                    ▼                         ▼              │
│  ┌─────────────────────────────────────────────────────────────────┐    │
│  │                    Vector Store (ChromaDB)                      │    │
│  │  Collections:                                                   │    │
│  │  • kinetichat_memory           (conversations)                  │    │
│  │  • kinetichat_memory_documents (document chunks)                │    │
│  │  • kinetichat_chat_summaries   (session summaries)              │    │
│  └─────────────────────────────────────────────────────────────────┘    │
│                                                                         │
│  ┌─────────────────┐  ┌──────────────────────────────────────────┐     │
│  │ Embedding       │  │ Summarize Agent                          │     │
│  │ Service         │  │ (condenses sessions → vector summaries)  │     │
│  │ (MiniLM-L6-v2)  │  │ agents/summarize_agent.py                │     │
│  └─────────────────┘  └──────────────────────────────────────────┘     │
└─────────────────────────────────────────────────────────────────────────┘
                      ▼
┌─────────────────────────────────────────────────────────────────────────┐
│  4. UTILITIES                                                           │
│                                                                         │
│  ┌───────────────┐ ┌───────────────┐ ┌──────────────┐ ┌─────────────┐  │
│  │ Gemini Client │ │ API Key Mgr   │ │ Document     │ │ Logger      │  │
│  │ (OpenAI-compat│ │ (multi-key    │ │ Loader       │ │ (loguru)    │  │
│  │  wrapper)     │ │  rotation)    │ │ (PDF/OCR)    │ │             │  │
│  └───────────────┘ └───────────────┘ └──────────────┘ └─────────────┘  │
│  ┌───────────────┐ ┌───────────────┐ ┌──────────────┐                  │
│  │ Web Search    │ │ Validators    │ │ Prompt       │                  │
│  │ (DuckDuckGo)  │ │ (safety)      │ │ Templates    │                  │
│  └───────────────┘ └───────────────┘ └──────────────┘                  │
└─────────────────────────────────────────────────────────────────────────┘
```

---

## Project Structure

```
agentic_rag_gemini/
│
├── main.py                          # CLI entry point (interactive / single-query mode)
├── ui.py                            # Streamlit web interface (951 lines)
├── run_ui.py                        # Streamlit launcher helper
├── requirements.txt                 # Python dependencies
├── setup.ps1                        # Windows PowerShell setup script
├── setup.sh                         # Linux/macOS setup script
├── .env.example                     # Environment variable template
├── .env                             # API keys (not committed)
├── .gitignore                       # Git ignore rules
├── QUICKSTART.md                    # 5-minute user setup guide
├── README_DEVELOPERS.md             # This file
│
├── config/
│   ├── __init__.py                  # Pydantic config models + YAML loader
│   └── config.yaml                  # Main configuration (226 lines)
│
├── agents/
│   ├── __init__.py
│   ├── local_orchestrator.py        # Faster Ollama-based router (309 lines)
│   ├── orchestrator.py              # Cloud Gemini-based router (340 lines)
│   └── summarize_agent.py           # Chat session summarizer (144 lines)
│
├── retrieval/
│   ├── __init__.py
│   └── rag_pipeline.py              # RAG pipeline with agentic loops (759 lines)
│
├── memory/
│   ├── __init__.py
│   ├── memory_manager.py            # Conversation memory CRUD (555 lines)
│   ├── document_store.py            # Document chunking & search (399 lines)
│   ├── vector_store.py              # ChromaDB/Qdrant wrapper (765 lines)
│   ├── embedding_service.py         # Sentence-Transformers + Gemini embeddings (261 lines)
│   └── session_store.py             # JSON-on-disk chat session persistence (252 lines)
│
├── utils/
│   ├── __init__.py
│   ├── cache_service.py             # Redis-backed multi-level cache (NEW)
│   ├── gemini_client.py             # OpenAI-compatible Gemini wrapper (517 lines)
│   ├── api_key_manager.py           # Multi-key rotation with round-robin (267 lines)
│   ├── ollama_client.py             # Simple direct Ollama API wrapper
│   ├── document_loader.py           # PDF/Word/Image/OCR loader (409 lines)
│   ├── web_search.py                # DuckDuckGo search service (232 lines)
│   ├── prompt_templates.py          # All prompt templates (430 lines)
│   └── logger.py                    # Loguru-based logging (140 lines)
│
├── tests/
│   ├── test_orchestrator.py         # Orchestrator unit tests
│   └── test_session_store.py        # Session store unit tests
│
├── data/
│   ├── vector_store/                # ChromaDB persistent storage
│   └── sessions/                    # Chat session JSON files
│
└── logs/
    └── agentic_rag.log              # Application logs (rotated at 10MB)
```

---

## Module Reference

### 1. Hybrid Orchestration — `agents/local_orchestrator.py`

The system utilizes a low-latency local model for the majority of categorizations, falling back to a powerful cloud model only when necessary.

| Component | Logic | Latency |
|---|---|---|
| **Primary (Local)** | Ollama `qwen:0.5b` with compact JSON prompt | ~300ms |
| **Fallback (Cloud)** | Gemini 2.5 Flash on low confidence / error | ~5s |

**Performance Tuning**:
- **Compact Prompts**: Orchestrator system prompt reduced from ~800 to ~160 tokens.
- **Model Warmup**: Ollama models are pre-loaded at server startup to eliminate cold-start lag.
- **JSON Enforcements**: Uses Ollama's `format: "json"` to ensure zero parsing errors.

---

### 2. Summarize Agent — `agents/summarize_agent.py`

Condenses chat session transcripts into concise summaries that are embedded into ChromaDB for cross-session recall.

| Method | Purpose |
|---|---|
| `summarize_session()` | Calls Gemini to generate a 3–5 sentence summary of a message list |
| `store_summary()` | Embeds the summary vector into the `chat_summaries` collection |
| `summarize_and_store()` | Convenience: summarize + embed in one call |

**Trigger**: Called from `ui.py._summarize_current_session()` when the user switches to a new chat session.

---

### 3. RAG Pipeline — `retrieval/rag_pipeline.py`

The core response-generation engine. Implements a multi-step agentic pipeline:

| Method | Pipeline Step |
|---|---|
| `generate_response()` | **Orchestrates the full pipeline** (see Agentic Pipeline below) |
| `_process_query()` | Step 1: Query expansion via LLM |
| `_retrieve_context()` | Step 2: Hybrid retrieval (memory + documents + session summaries) |
| `_assess_context_quality()` | Step 3: Compute average similarity score |
| `_reformulate_query()` | Step 3b: LLM-based query rewrite if quality is low |
| `_build_prompt()` | Step 4: Construct full prompt with context, history, and instructions |
| `_generate_llm_response()` | Step 5: Call Gemini to generate answer |
| `_reflect_on_response()` | Step 6: LLM self-check for grounding/hallucination |
| `_retry_generation()` | Step 7: Retry with validation feedback if needed |

---

### 4. Memory Manager — `memory/memory_manager.py`

CRUD layer for conversation-level memory.

| Method | Purpose |
|---|---|
| `store_interaction()` | Saves a user↔assistant turn as an embedded vector |
| `retrieve_relevant_memory()` | Semantic search over past interactions |
| `get_recent_interactions()` | Chronological fetch of recent turns |
| `store_user_context()` | Stores user preferences, physical state, constraints |
| `get_user_profile()` | Aggregates user info from memory |
| `load_documents_from_file()` | ⚠️ **Deprecated** — forwards to DocumentStore |
| `load_documents_from_directory()` | ⚠️ **Deprecated** — forwards to DocumentStore |
| `clear_memory()` | Resets in-memory counters and state |
| `_create_summary()` | Periodic conversation summarization |

---

### 5. Document Store — `memory/document_store.py`

Manages uploaded documents with intelligent chunking and deduplication.

| Method | Purpose |
|---|---|
| `store_document()` | Routes to single-chunk or chunked storage based on doc size |
| `_store_single_document()` | Stores small docs (< `min_chunk_size`) as one vector |
| `_store_chunked_document()` | Splits large docs into overlapping chunks with rich metadata |
| `search_documents()` | Semantic search with chunk deduplication |
| `_deduplicate_chunks()` | Keeps top-N chunks per document to prevent result domination |
| `get_user_documents()` | Lists all documents for a user |
| `delete_document()` / `delete_user_documents()` | Cleanup operations |
| `count_documents()` | Document count (optionally per user) |

---

### 6. Vector Store — `memory/vector_store.py`

Low-level ChromaDB (and Qdrant stub) wrapper managing three collections.

| Collection | Contents | Collection Name |
|---|---|---|
| Conversations | User↔assistant interaction embeddings | `kinetichat_memory` |
| Documents | Uploaded document chunk embeddings | `kinetichat_memory_documents` |
| Chat Summaries | Session summary embeddings | `kinetichat_chat_summaries` |

| Key Method | Purpose |
|---|---|
| `_init_chromadb()` | Initializes `PersistentClient` + creates/gets all 3 collections |
| `add_documents()` | Batch add to conversations or documents collection |
| `search()` | Similarity search on conversations with proper filter handling |
| `search_documents()` | Similarity search on document chunks |
| `add_chat_summary()` / `search_chat_summaries()` | CRUD for session summaries |
| `clear_all_data()` | Wipes all 3 collections with verification |
| `reset_collections()` | Delete + recreate collections (for schema fixes) |

**Similarity Calculation** (critical fix):
```python
# ChromaDB returns squared L2 distance. Conversion to cosine similarity:
similarity = max(0.0, 1.0 - (distance / 2.0))
```

---

### 7. Embedding Service — `memory/embedding_service.py`

Text → vector conversion with high-performance caching.

- **Model**: `all-MiniLM-L6-v2` (Local CPU inference ~5ms/query)
- **Caching**: **Redis integration** prevents redundant embedding calculation for identical queries across different users.

| Method | Purpose |
|---|---|
| `embed_texts()` | Main entry — checks Redis cache first |
| `compute_similarity()` | Cosine similarity between two vectors |
| `count_tokens()` / `truncate_text()` | Token management |

---

### 8. Session Store — `memory/session_store.py`

JSON-on-disk persistence for chat sessions. Each session is a JSON file under `data/sessions/{user_id}/`.

| Method | Purpose |
|---|---|
| `create_session()` | Creates a new empty session file |
| `save_turn()` | Appends a message to a session |
| `load_session()` | Reads full session data |
| `list_sessions()` | Returns sessions sorted by most-recently-updated |
| `delete_oldest_sessions()` | Prunes sessions beyond a keep limit |
| `mark_summarized()` | Stores summary text and flags session as summarized |

**Session file format**:
```json
{
  "session_id": "uuid",
  "title": "Auto-generated from first message",
  "created_at": "ISO-8601",
  "updated_at": "ISO-8601",
  "is_summarized": false,
  "summary": null,
  "messages": [
    {"role": "user", "content": "...", "timestamp": "..."},
    {"role": "assistant", "content": "...", "timestamp": "...", "metadata": {}}
  ]
}
```

---

### 9. Gemini Client — `utils/gemini_client.py`

OpenAI-compatible wrapper around Google's Gemini API.

| Class | Purpose |
|---|---|
| `GeminiClient` | Core API wrapper with auto-retry and key rotation |
| `GeminiClientWrapper` | Top-level interface exposing `GeminiClientWrapper.generate()` and `GeminiClientWrapper.chat.completions.create()` |
| `GeminiChatCompletion` | Mimics `client.chat.completions.create()` (OpenAI interface) |
| `GeminiResponse` / `GeminiChoice` / `GeminiMessage` | Response objects matching OpenAI's structure |

**Key features**:
- Automatic API key rotation on 429 (quota exceeded) errors
- Converts OpenAI-style `messages` list to Gemini's `contents` format
- System prompt extracted from messages and passed as `system_instruction`
- Supports `response_format` parameter for JSON mode

---

### 10. API Key Manager — `utils/api_key_manager.py`

Thread-safe singleton that manages multiple Gemini API keys with round-robin rotation.

```
GEMINI_API_KEYS=key1,key2,key3  # Comma-separated in .env
```

| Method | Purpose |
|---|---|
| `get_current_key()` | Returns the active API key |
| `rotate_to_next_key()` | Moves to next key (circular). Returns `False` after 2 full cycles |
| `mark_key_failed()` | Increments failure counter |
| `reset_success()` | Resets failure counter after successful call |
| `has_available_keys()` | Checks if more retries are allowed |

**Rotation logic**: Allows `2 × total_keys` consecutive failures before declaring all keys exhausted.

---

### 11. Document Loader — `utils/document_loader.py`

Multi-format document text extraction.

| Format | Method | Features |
|---|---|---|
| PDF | `_load_pdf()` | Text extraction via pypdf + OCR via pdf2image + pytesseract |
| PDF (scanned) | `_ocr_pdf()` / `_ocr_pdf_images()` | Full-page OCR for scanned documents |
| Word (.docx) | `_load_docx()` | Extracts paragraphs + tables |
| Images | `_load_image()` | OCR via pytesseract |
| Text (.txt) | `_load_text()` | Plain text reading |

**PDF Strategy**: Always extracts regular text first, then OCR-renders each page as an image to catch diagrams, screenshots, and figures. Both sources are merged.

---

### 12. Web Search — `utils/web_search.py`

DuckDuckGo-powered web search fallback (free, no API key required).

| Method | Purpose |
|---|---|
| `search()` | Raw search → list of `{title, url, snippet}` |
| `search_and_summarize()` | Formatted markdown context for RAG prompt injection |
| `search_health_topics()` | Health-focused search with enhanced keywords |

**Trigger**: Activated by the RAG pipeline when local context quality is below `web_search_quality_threshold` (default 0.65) or fewer than `min_context_threshold` items (default 2) are retrieved.

---

### 13. Validators — `utils/validators.py`

Response safety and quality validation.

| Check | Description |
|---|---|
| Safety check | Flags unsafe keywords (diagnosis, treatment plan, prescription, etc.) |
| Length check | Ensures response is between 50–1500 characters |
| Relevance check | Compares response keywords against query keywords |

**Note**: Validation is currently **disabled** (`enable_validation: false` in config) to reduce API latency during development.

---

### 14. Logger — `utils/logger.py`

Structured logging via Loguru with console + file output.

- **Console**: Colorized human-readable format
- **File**: `logs/agentic_rag.log` with 10MB rotation, 30-day retention, zip compression
- **Context manager**: `LogContext` binds `user_id`, `session_id` to log entries

---

## Agentic Pipeline — Detailed Flow

This is the complete query processing flow when a user sends a message:

```
User Query
    │
    ▼
┌───────────────────────────────────────────────────────┐
│ 1. ORCHESTRATOR ANALYSIS                               │
│    • Build analysis prompt (system + user query)       │
│    • Call Gemini (temp=0.1, max_tokens=500)             │
│    • Parse JSON decision                               │
│    • Output: action, confidence, parameters             │
└───────────────┬───────────────────────────────────────┘
                │
    ┌───────────┼──────────────┐
    │           │              │
    ▼           ▼              ▼
RETRIEVE    CALL_LLM      HYBRID
DOCUMENT/   (direct)      (retrieve +
MEMORY                     generate)
    │           │              │
    └───────────┴──────────────┘
                │
                ▼
┌───────────────────────────────────────────────────────┐
│ 2. QUERY PROCESSING                                    │
│    • If enable_query_expansion: LLM rewrites query     │
│    • Cleaned, expanded query passed to retrieval       │
└───────────────┬───────────────────────────────────────┘
                │
                ▼
┌───────────────────────────────────────────────────────┐
│ 3. HYBRID CONTEXT RETRIEVAL                            │
│    Source A: Memory (conversations collection)         │
│    Source B: Documents (document chunks collection)     │
│    Source C: Chat Summaries (session summaries)         │
│    • All 3 sources searched in parallel                 │
│    • Results merged and sorted by similarity            │
└───────────────┬───────────────────────────────────────┘
                │
                ▼
┌───────────────────────────────────────────────────────┐
│ 4. QUALITY ASSESSMENT + QUERY REFORMULATION LOOP       │
│    • Compute avg similarity of retrieved context       │
│    • If avg < reformulation_quality_threshold (0.3):   │
│      → LLM rewrites query (up to 2 attempts)           │
│      → Re-run retrieval with reformulated query        │
│    • If still poor → trigger web search fallback       │
└───────────────┬───────────────────────────────────────┘
                │
                ▼
┌───────────────────────────────────────────────────────┐
│ 5. WEB SEARCH FALLBACK (if enabled)                    │
│    • Triggered when:                                   │
│      - avg similarity < web_search_quality_threshold   │
│      - OR fewer than min_context_threshold items       │
│    • DuckDuckGo search → results injected into prompt  │
└───────────────┬───────────────────────────────────────┘
                │
                ▼
┌───────────────────────────────────────────────────────┐
│ 6. PROMPT BUILDING                                     │
│    • System prompt (from config)                       │
│    • Retrieved context (documents, memory, summaries)  │
│    • Web search results (if any)                       │
│    • Conversation history                              │
│    • User query                                        │
│    • Response guidelines + source priority instructions │
└───────────────┬───────────────────────────────────────┘
                │
                ▼
┌───────────────────────────────────────────────────────┐
│ 7. LLM RESPONSE GENERATION                            │
│    • Call Gemini (temp=0.7, max_tokens=2000)            │
│    • Full prompt with all context                      │
└───────────────┬───────────────────────────────────────┘
                │
                ▼
┌───────────────────────────────────────────────────────┐
│ 8. ITERATIVE REFLECTION (if enabled)                   │
│    • LLM checks if response is grounded in context     │
│    • Returns: is_grounded, issues, revised_answer      │
│    • If not grounded → uses revised answer             │
│    • Max 1 reflection iteration (configurable)         │
└───────────────┬───────────────────────────────────────┘
                │
                ▼
┌───────────────────────────────────────────────────────┐
│ 9. MEMORY STORAGE                                      │
│    • Store interaction (query + response) in memory    │
│    • Save turn to session JSON file                    │
│    • Periodic summarization if threshold reached       │
└───────────────┬───────────────────────────────────────┘
                │
                ▼
           User Response
```

---

## Vector Database & Chunking Architecture

### Document Processing Pipeline

```
Document Upload → Text Extraction → Chunking → Embedding → ChromaDB Storage
     ↓                ↓              ↓           ↓              ↓
  PDF/Word/Img    pypdf/OCR    Overlapping   MiniLM-L6-v2   Chunks with
   files         extraction    chunks        (384-dim)       rich metadata
```

### Chunking System

**Location**: `memory/document_store.py` & `utils/document_loader.py`

#### Algorithm
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

#### Configuration (`config/config.yaml`)
```yaml
chunking:
  enable_chunking: true
  chunk_size: 1500          # Characters per chunk
  chunk_overlap: 300        # Overlap between consecutive chunks
  min_chunk_size: 300       # Documents smaller than this → single chunk
  chunk_search_multiplier: 3  # Fetch 3× top_k for deduplication headroom
```

#### Chunk Metadata
Each chunk is stored with:
```python
{
    "user_id": "web_user",
    "filename": "document.pdf",
    "chunk_number": 0,           # 0-indexed
    "total_chunks": 5,
    "chunk_type": "chunked",     # or "single"
    "start_position": 0,
    "end_position": 1500,
    "content_length": 1500,
    "timestamp": "2026-02-28T...",
    "document_type": "uploaded_knowledge"
}
```

### Search & Deduplication

1. **Query embedding**: Convert query → 384-dim vector
2. **Vector search**: ChromaDB similarity search (top_k × chunk_search_multiplier)
3. **Deduplication**: Group chunks by document, keep top `max_chunks_per_document` (default 3) per document
4. **Context assembly**: Merge selected chunks for prompt injection

```yaml
rag:
  top_k_documents: 8
  similarity_threshold: 0.1
  max_chunks_per_document: 3
```

### ChromaDB Collections

| Collection | Purpose | Similarity Formula |
|---|---|---|
| `kinetichat_memory` | Conversation history | `max(0.0, 1.0 - (distance / 2.0))` |
| `kinetichat_memory_documents` | Document chunks | `max(0.0, 1.0 - (distance / 2.0))` |
| `kinetichat_chat_summaries` | Session summaries | `max(0.0, 1.0 - (distance / 2.0))` |

**Storage**: `data/vector_store/` (persistent across restarts via `chromadb.PersistentClient`)

---

## Session & Memory Management

### Session Flow (in `ui.py`)

```
User opens app
    │
    ├─ init_session_state() → loads cached resources (embedding model, vector store, etc.)
    │
    ├─ _ensure_session() → creates/loads session via SessionStore
    │
    ├─ User sends message
    │   ├─ process_user_query() → orchestrator → RAG pipeline → response
    │   ├─ add_message_to_chat() → saves to session JSON file
    │   └─ Store interaction in memory (vector store)
    │
    ├─ User starts new session
    │   ├─ _summarize_current_session() → SummarizeAgent condenses prior session
    │   ├─ Summary embedded in chat_summaries collection
    │   └─ _start_new_session() → creates fresh session file
    │
    └─ Session history sidebar → list/switch/delete sessions
```

### Session Pruning
```yaml
memory:
  max_chat_sessions: 5    # Oldest sessions auto-deleted beyond this limit
```

### Summarization Pipeline
1. When user switches sessions, `_summarize_current_session()` is called
2. If session has ≥ 2 messages and hasn't been summarized:
   - `SummarizeAgent.summarize_session()` calls Gemini to generate a summary
   - Summary is embedded via `SummarizeAgent.store_summary()` into `chat_summaries` collection
   - Session marked as `is_summarized = true` in JSON file
3. Future queries can retrieve relevant past session summaries via `VectorStore.search_chat_summaries()`

---

## Web Search Fallback

The system uses DuckDuckGo as a free web search fallback when local knowledge is insufficient.

### Trigger Conditions (any of):
- Average similarity of retrieved context < `web_search_quality_threshold` (0.65)
- Fewer than `min_context_threshold` (2) context items retrieved

### Configuration
```yaml
rag:
  enable_web_search: true
  web_search_quality_threshold: 0.65
  min_context_threshold: 2
  max_web_results: 5
  web_search_timeout: 10  # seconds
```

### Integration Point
In `rag_pipeline.py → generate_response()`:
```python
if web_search_triggered:
    web_context = web_search_service.search_and_summarize(query)
    # Web results injected as "🌐 WEB SEARCH RESULTS" section in prompt
```

The LLM is instructed to cite web sources with URLs when using web search results.

---

## API Key Rotation System

### Setup
```bash
# .env file — comma-separated keys
GEMINI_API_KEYS=AIza..._key1,AIza..._key2,AIza..._key3
```

### Flow
```
API Call → 200 OK → reset_success() (reset failure counter)
    │
    └→ 429 Quota Error → rotate_to_next_key()
                              │
                              ├─ Cycles < 2 × total_keys → retry with next key
                              │
                              └─ Cycles exhausted → raise error
```

### Thread Safety
- Singleton pattern with `threading.Lock()`
- All state mutations are thread-safe
- Used by `GeminiClient._rotate_and_retry()`

---

## Prompt Engineering

All prompts are centralized in `utils/prompt_templates.py`. The system uses 7 prompt categories:

| Category | Variable | Used By |
|---|---|---|
| `ORCHESTRATOR_PROMPTS` | system, decision_format | `agents/orchestrator.py` |
| `RAG_PROMPTS` | system, with_context | `retrieval/rag_pipeline.py` |
| `VALIDATION_PROMPTS` | safety_check | `utils/validators.py` |
| `SUMMARIZATION_PROMPTS` | conversation_summary, user_profile | `memory/memory_manager.py` |
| `QUERY_REFORMULATION_PROMPTS` | reformulate | `retrieval/rag_pipeline.py` |
| `REFLECTION_PROMPTS` | reflect | `retrieval/rag_pipeline.py` |
| `SESSION_SUMMARY_PROMPTS` | system, summarize | `agents/summarize_agent.py` |
| `FALLBACK_MESSAGES` | generic, memory_failure, etc. | Various |

### Source Priority (built into system prompt)
```
1. UPLOADED DOCUMENTS — Primary knowledge source
   → "Based on [filename], ..."
2. WEB SEARCH RESULTS — When documents don't have the answer
   → Always cite source URLs
3. GENERAL KNOWLEDGE — Only when neither is available
```

### Language Adaptation
The system prompt instructs the LLM to respond in the same language as the user's query (supports Vietnamese and English natively).

---

## Configuration Reference

**File**: `config/config.yaml` (226 lines)

### Orchestrator
```yaml
orchestrator:
  model: "gemini-2.5-flash"
  temperature: 0.1          # Low for consistent routing
  max_tokens: 500
  memory_retrieval_threshold: 0.6
  llm_call_threshold: 0.7
  motion_generation_threshold: 0.8
```

### LLM (Response Generation)
```yaml
llm:
  model: "gemini-2.5-flash"
  temperature: 0.7
  max_tokens: 2000
  enable_validation: false   # Disabled for debugging
  max_retries: 1
```

### Embedding
```yaml
embedding:
  model: "sentence-transformers/all-MiniLM-L6-v2"
  dimension: 384
  batch_size: 32
```

### RAG Pipeline
```yaml
rag:
  top_k_documents: 8
  similarity_threshold: 0.1
  max_chunks_per_document: 3
  enable_query_expansion: true
  enable_query_reformulation: true
  max_reformulation_attempts: 2
  reformulation_quality_threshold: 0.3
  enable_iterative_reflection: true
  max_reflection_iterations: 1
  enable_web_search: true
  web_search_quality_threshold: 0.65
  min_context_threshold: 2
  max_web_results: 5
```

### Memory
```yaml
memory:
  max_items_per_user: 100
  retention_days: 90
  top_k: 5
  similarity_threshold: 0.3
  summarization_interval: 5
  max_chat_sessions: 5
```

### Chunking
```yaml
chunking:
  enable_chunking: true
  chunk_size: 1500
  chunk_overlap: 300
  min_chunk_size: 300
  chunk_search_multiplier: 3
```

### Validation
```yaml
validation:
  enable_safety_check: true
  enable_factuality_check: false
  enable_relevance_check: true
  min_response_length: 50
  max_response_length: 1500
```

### Performance
```yaml
performance:
  enable_caching: true
  cache_ttl: 3600
  orchestrator_timeout: 5
  memory_retrieval_timeout: 3
  llm_timeout: 30
```

---

## Environment Variables

**File**: `.env` (see `.env.example` for template)

| Variable | Required | Description |
|---|---|---|
| `GEMINI_API_KEYS` | ✅ | Comma-separated Gemini API keys for rotation |
| `ORCHESTRATOR_MODEL` | ❌ | Override orchestrator model (default: from config.yaml) |
| `LLM_MODEL` | ❌ | Override LLM model (default: from config.yaml) |
| `EMBEDDING_MODEL` | ❌ | Override embedding model |
| `VECTOR_DB_TYPE` | ❌ | `chromadb` or `qdrant` |
| `VECTOR_DB_PATH` | ❌ | Path for ChromaDB persistence |
| `QDRANT_URL` | ❌ | Qdrant server URL |
| `QDRANT_API_KEY` | ❌ | Qdrant API key |
| `LOG_LEVEL` | ❌ | Logging level (DEBUG/INFO/WARNING/ERROR) |

**Priority**: Environment variables override `config.yaml` values (handled in `config/__init__.py`).

---

## Setup & Installation

### Prerequisites
- Python 3.10+ (tested with 3.13)
- Gemini API key(s) from [Google AI Studio](https://aistudio.google.com/app/apikey)
- *Optional*: Tesseract OCR for scanned document support

### Quick Setup (Windows — PowerShell)
```powershell
cd agentic_rag_gemini

# Option A: Automated
.\setup.ps1

# Option B: Manual
pip install -r requirements.txt
mkdir -Force data\vector_store, logs
cp .env.example .env
# Edit .env → add GEMINI_API_KEYS=your_key_here
```

### Quick Setup (Linux/macOS)
```bash
cd agentic_rag_gemini
pip install -r requirements.txt
mkdir -p data/vector_store logs
cp .env.example .env
# Edit .env → add GEMINI_API_KEYS=your_key_here
```

### OCR Setup (Optional)
```bash
# Windows (via Chocolatey)
choco install tesseract

# Linux
sudo apt-get install tesseract-ocr

# Verify
python -c "import pytesseract; print(pytesseract.get_tesseract_version())"
```

---

## Running the System

### Web Interface (Recommended)
```bash
python run_ui.py
# OR directly:
streamlit run ui.py
# Opens at http://localhost:8501
```

### CLI Interactive Mode
```bash
python main.py --mode interactive
```

### Single Query (Programmatic)
```python
from main import AgenticRAGSystem

system = AgenticRAGSystem()
result = system.process_query(
    query="What are the evaluation criteria?",
    user_id="web_user"
)
print(result["response"])
```

---

## Testing

### Run Tests
```bash
# Orchestrator tests
python -m pytest tests/test_orchestrator.py -v

# Session store tests
python -m pytest tests/test_session_store.py -v
```

### Test Individual Components
```bash
# Test embedding service
python -c "from memory.embedding_service import EmbeddingService; e = EmbeddingService(); print(len(e.embed_texts('test')))"

# Test vector store
python -c "from memory.vector_store import VectorStore; v = VectorStore(); print('Collections initialized')"

# Test Gemini API connection
python -c "from utils.gemini_client import GeminiClientWrapper; c = GeminiClientWrapper(); print(c.chat.completions.create(model='gemini-2.5-flash', messages=[{'role': 'user', 'content': 'hi'}]).choices[0].message.content)"

# Test document loader
python -c "from utils.document_loader import DocumentLoader; d = DocumentLoader(); print('Supported:', d.SUPPORTED_FORMATS)"

# Test web search
python -c "from utils.web_search import WebSearchService; s = WebSearchService(); print('Available:', s.is_available())"
```

---

## How to Extend the System

### Add a new orchestrator action type
1. Add to `ActionType` enum in `agents/orchestrator.py`
2. Update `OrchestratorAgent._build_analysis_prompt()` to include the new action
3. Handle the new action in `main.py:AgenticRAGSystem.process_query()`
4. Update ORCHESTRATOR_PROMPTS in `utils/prompt_templates.py`

### Change the LLM model
1. Update `config/config.yaml` → `llm.model` and/or `orchestrator.model`
2. Available models: `gemini-2.5-flash`, `gemini-2.5-pro`, `gemini-2.0-flash`, `gemini-3-pro-preview`

### Change the embedding model
1. Update `config/config.yaml` → `embedding.model` and `embedding.dimension`
2. Update `memory/embedding_service.py` → `_init_sentence_transformer()`
3. **Critical**: Reset vector store (`VectorStore.reset_collections()`) since embeddings are dimension-dependent

### Add a new document format
1. Add the extension to `DocumentLoader.SUPPORTED_FORMATS`
2. Implement `_load_<format>()` method in `utils/document_loader.py`
3. Add the routing in `DocumentLoader.load_file()`

### Switch vector database to Qdrant
1. Set `VECTOR_DB_TYPE=qdrant` in `.env`
2. Start Qdrant server: `docker run -p 6333:6333 qdrant/qdrant`
3. Complete `VectorStore._init_qdrant()` implementation (currently a stub)

### Add response post-processing
1. Edit `retrieval/rag_pipeline.py` → add logic after `_generate_llm_response()`
2. Or add a new validation rule in `utils/validators.py`

### Add a new prompt template
1. Add the template dict to `utils/prompt_templates.py`
2. Use `format_prompt()` or `get_prompt()` to retrieve it

---

## Dependency Matrix

### Core Dependencies (as of 2026-02-28)

| Package | Required Version | Installed | Purpose |
|---|---|---|---|
| streamlit | ≥ 1.28.0 | 1.54.0 | Web interface |
| streamlit-chat | ≥ 0.1.0 | ✔️ | Chat bubbles |
| pydantic | ≥ 2.0.0 | 2.12.4 | Config models |
| pydantic-settings | ≥ 2.0.0 | 2.12.0 | Settings management |
| python-dotenv | ≥ 1.0.0 | ✔️ | .env file loading |
| PyYAML | ≥ 6.0 | ✔️ | Config file parsing |
| chromadb | ≥ 0.6.0 | 1.5.1 | Vector database |
| qdrant-client | ≥ 1.6.0 | 1.17.0 | Vector DB (alt) |
| google-generativeai | ≥ 0.3.0 | 0.8.6 | Gemini API ⚠️ |
| sentence-transformers | ≥ 2.2.0 | 5.2.3 | Embeddings |
| numpy | ≥ 1.24.0 | 2.4.2 | Numeric operations |
| tiktoken | ≥ 0.5.0 | ✔️ | Token counting |
| pypdf | ≥ 3.17.0 | 6.7.4 | PDF text extraction |
| python-docx | ≥ 0.8.11 | ✔️ | Word documents |
| pytesseract | ≥ 0.3.10 | ✔️ | OCR |
| pdf2image | ≥ 1.16.0 | ✔️ | PDF → image for OCR |
| Pillow | ≥ 9.5.0 | ✔️ | Image processing |
| loguru | ≥ 0.7.0 | ✔️ | Structured logging |
| colorlog | ≥ 6.7.0 | ✔️ | Colored log output |
| ddgs | ≥ 9.0.0 | ✔️ | DuckDuckGo search |

### ⚠️ Deprecation Notice

**`google-generativeai`** (v0.8.6) is deprecated. Google has ended support for this package.
- **Action required**: Migrate to **`google-genai`** (the new official SDK)
- **Impact**: No breaking changes currently, but no future bug fixes or updates
- **Migration guide**: https://github.com/google-gemini/deprecated-generative-ai-python

### Verified Compatibility (2026-02-28)

```
✅ pydantic 2.12 + chromadb 1.5  → Compatible
✅ grpcio 1.78 + protobuf 5.29   → Compatible
✅ numpy 2.4 + sentence-transformers 5.2 → Compatible
✅ pip check → "No broken requirements found"
```

---

## Resolved Issues & Fixes

### ✅ Orchestrator Latency Fix
**Problem**: Intent routing taking 10-15s per query via Gemini.  
**Fix**: Migrated to local Ollama (`qwen:0.5b`) + prompt compression.  
**Result**: Latency dropped to <1s.

### ✅ API Concurrency
**Problem**: System blocked event loop during Ollama/Gemini calls.  
**Fix**: Implemented `asyncio.to_thread()` in `api_server.py`.

### ✅ Multi-User Cache Layer
**Problem**: Redundant RAG calls for identical queries.  
**Fix**: Implemented Redis `CacheService` with distinct Knowledge vs Memory retrieval keys.

---

## Known Limitations & Open Issues

### ⚠️ Multi-User Concurrency
**Status**: Step 1 (CacheService) implemented.  
**Action**: Continue with remaining stages of the "Multi-User Optimization Plan" (Async semaphores, Multi-worker deployment, ChromaDB Server mode).

### ⚠️ Authentication
**Status**: Web UI uses hardcoded `"web_user"` for all sessions.  
**Impact**: No multi-user separation in the UI layer yet.

### ⚠️ Single-Language Embedding Model
**Status**: `all-MiniLM-L6-v2` is English-optimized  
**Impact**: Vietnamese document retrieval may have lower similarity scores  
**Action**: Consider multilingual models (e.g., `paraphrase-multilingual-MiniLM-L12-v2`)

---

## Production Deployment Checklist

| # | Item | Status |
|---|---|---|
| 1 | Use environment variables for all secrets | ✅ Done |
| 2 | Set `log_level: WARNING` in config | ⬜ TODO |
| 3 | Enable response validation | ⬜ TODO |
| 4 | Increase `max_retries` for stability | ⬜ TODO |
| 5 | Add authentication to web interface | ⬜ TODO |
| 6 | Monitor `logs/` directory (set up alerting) | ⬜ TODO |
| 7 | Use persistent vector store | ✅ Done |
| 8 | Handle API rate limiting gracefully | ✅ Done (key rotation) |
| 9 | Migrate to `google-genai` SDK | ⬜ TODO |
| 10 | Consider multilingual embedding model | ⬜ TODO |
| 11 | Implement proper error pages in Streamlit | ⬜ TODO |
| 12 | Set up CI/CD pipeline with tests | ⬜ TODO |

---

## Quick Reference — Common Modifications

| What to Change | Where |
|---|---|
| LLM model | `config/config.yaml` → `llm.model` |
| Orchestrator model | `config/config.yaml` → `orchestrator.model` |
| Response creativity | `config/config.yaml` → `llm.temperature` |
| Embedding model | `config/config.yaml` → `embedding.model` + `embedding.dimension` |
| Number of retrieved documents | `config/config.yaml` → `rag.top_k_documents` |
| Similarity threshold | `config/config.yaml` → `rag.similarity_threshold` |
| Chunk size | `config/config.yaml` → `chunking.chunk_size` |
| Max sessions kept | `config/config.yaml` → `memory.max_chat_sessions` |
| Web search aggressiveness | `config/config.yaml` → `rag.web_search_quality_threshold` |
| Prompt templates | `utils/prompt_templates.py` |
| System persona | `config/config.yaml` → `llm.system_prompt` |
| Routing behavior | `config/config.yaml` → `orchestrator.system_prompt` |
| Safety keywords | `config/config.yaml` → `validation.unsafe_keywords` |
| API keys | `.env` → `GEMINI_API_KEYS` |
| UI layout | `ui.py` |
| Add document format | `utils/document_loader.py` → `SUPPORTED_FORMATS` + new `_load_*()` |

---

## Contact / Further Reading

- **User setup**: See [`QUICKSTART.md`](QUICKSTART.md)
- **Config details**: See [`config/config.yaml`](config/config.yaml)
- **Prompt templates**: See [`utils/prompt_templates.py`](utils/prompt_templates.py)
- **Individual modules**: Each file contains detailed docstrings and method documentation
