# RAG System File Flow Documentation

## 1. Startup and Initialization

### Files Involved:
- `app.py`: Main entry point
- `src/rag_system.py`: Core RAG system
- `.env`: Environment variables
- `requirements.txt`: Dependencies

### Flow:
1. `app.py` initializes:
   - Loads environment variables from `.env`
   - Sets up logging
   - Initializes session state
   - Creates RAG system instance

2. `rag_system.py` initializes:
   - Sets up vector database connection
   - Configures LLM client
   - Loads embedding model
   - Initializes cache

## 2. Web Interface Loading

### Files Involved:
- `templates/index.html`: Main web template
- `static/style.css`: Styling
- `static/script.js`: Client-side JavaScript
- `static/bootstrap.min.css`: Bootstrap framework
- `static/bootstrap.bundle.min.js`: Bootstrap JavaScript

### Flow:
1. `index.html` loads:
   - Loads Bootstrap CSS and JS
   - Renders UI components
   - Sets up event listeners
   - Makes initial API calls

## 3. Query Processing

### Files Involved:
- `app.py`: Request handling
- `src/rag_system.py`: Main processing
- `src/vector_db/faiss_db.py`: Vector database
- `src/retrieval/retriever.py`: Document retrieval
- `src/llm/claude_client.py`: LLM integration

### Flow:
1. User submits query:
   - `index.html` -> AJAX call
   - `app.py` -> `rag_system.py`

2. Query processing:
   - `rag_system.py` generates embedding
   - `faiss_db.py` searches for similar documents
   - `retriever.py` retrieves relevant chunks

3. Response generation:
   - `claude_client.py` generates response
   - `rag_system.py` formats response
   - `app.py` returns to frontend

## 4. Document Display

### Files Involved:
- `templates/index.html`: Response display
- `src/document_processing/document_processor.py`: Document handling
- `src/vector_db/faiss_db.py`: Document metadata

### Flow:
1. Response received:
   - `index.html` updates UI
   - Shows main response
   - Displays supporting documents

2. Document interaction:
   - `document_processor.py` handles document data
   - `faiss_db.py` provides metadata
   - `index.html` renders document snippets

## 5. Cache Management

### Files Involved:
- `src/rag_system.py`: Cache operations
- `src/cache/semantic_cache.py`: Semantic caching
- `src/cache/response_cache.py`: Response caching

### Flow:
1. Cache check:
   - `rag_system.py` checks semantic cache
   - `response_cache.py` checks for previous responses

2. Cache updates:
   - `semantic_cache.py` stores embeddings
   - `response_cache.py` stores responses
   - `rag_system.py` manages cache eviction

## 6. Session Management

### Files Involved:
- `app.py`: Session handling
- `src/rag_system.py`: Session state
- `templates/index.html`: Session UI

### Flow:
1. Session start:
   - `index.html` -> `app.py`
   - `rag_system.py` initializes session

2. Session maintenance:
   - `app.py` manages session state
   - `rag_system.py` maintains context
   - `index.html` displays session info

## 7. Performance Monitoring

### Files Involved:
- `src/rag_system.py`: Performance tracking
- `src/cache/cache_stats.py`: Cache statistics
- `templates/index.html`: Stats display

### Flow:
1. Performance tracking:
   - `rag_system.py` measures timings
   - `cache_stats.py` collects metrics

2. Stats display:
   - `index.html` shows cache hit rates
   - `index.html` displays response times
   - `index.html` shows system metrics
