# Code Organization

## 1. Directory Structure

### 1.1 Main Source Code (`src/`)
- Contains core application code
- Uses Python package structure
- All modules are importable

### 1.2 Tools (`src/tools/`)
- Core processing utilities
- Document processing
- PDF handling
- Security features
- Validation tools

### 1.3 Tests (`tests/`)
- Unit tests
- Integration tests
- Legacy tests
- Performance tests

## 2. Code Organization

### 2.1 Source Code (`src/`)
```
src/
├── __init__.py
├── rag_system.py                # Main RAG system
├── config/                     # Configuration
├── connectors/                 # External system connectors
├── conversation_manager.py     # Conversation handling
├── dashboard/                 # Dashboard components
├── database/                  # Database operations
├── document_management/       # Document management
├── document_processing/       # Document processing
├── embeddings/               # Embedding generation
├── llm/                      # LLM integration
├── preloader.py              # Model preloading
├── retrieval/               # Document retrieval
├── security/                # Security features
├── tools/                   # Processing tools
└── vector_db/              # Vector database
```

### 2.2 Tools (`src/tools/`)
```
tools/
├── __init__.py
├── abstract_document_processor.py
├── abstract_pdf_processor.py
├── document_processor.py
├── document_tracker.py
├── metadata_validator.py
├── notification_handler.py
├── pdf_processor.py
├── security_config.py
└── smart_document_processor.py
```

### 2.3 Tests (`tests/`)
```
tests/
├── README.md
├── legacy/                     # Legacy tests
│   ├── test_great_depression_dashboard.py
│   ├── test_great_depression_rag.py
│   ├── test_retrieval.py
│   ├── test_semantic_cache.py
│   └── test_semantic_cache_mock.py
├── current/                   # Current tests
│   ├── test_improved_cache.py
│   ├── test_quantized_embeddings.py
│   └── test_combined_caching.py
└── performance/               # Performance tests
```

## 3. Import Guidelines

### 3.1 Relative Imports
```python
# Correct
from .embeddings.generator import EmbeddingGenerator
from .vector_db.faiss_db import FAISSVectorDB

# Incorrect (use relative imports)
from src.embeddings.generator import EmbeddingGenerator
```

### 3.2 Tools Imports
```python
# Correct
from .tools.document_processor import DocumentProcessor

# Incorrect (use relative imports)
from src.tools.document_processor import DocumentProcessor
```

## 4. Testing Structure

### 4.1 Legacy Tests
- Located in `tests/legacy/`
- Historical test cases
- For reference only

### 4.2 Current Tests
- Located in `tests/current/`
- Active test suite
- Regularly updated

### 4.3 Performance Tests
- Located in `tests/performance/`
- Performance benchmarks
- Optimization tests
