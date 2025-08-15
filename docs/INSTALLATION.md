# RAG System Installation Guide

## 1. Prerequisites

### 1.1 Python Environment
- Python 3.10.11 or higher
- Virtual environment recommended
- Project structure follows Python package conventions

### 1.2 Required Packages
```bash
pip install -e .[all]
```

## 2. Code Organization

### 2.1 Directory Structure
```
src/
├── __init__.py
├── rag_system.py
├── tools/           # Processing utilities
├── tests/          # Test suite
└── ...             # Other modules
```

### 2.2 Import Structure
- Use relative imports within src/
- Tools are imported from .tools/
- Tests are in dedicated directories

## 3. Configuration

### 3.1 Environment Variables
Create a `.env` file with:
- ANTHROPIC_API_KEY
- ELASTICSEARCH_API_KEY (for production)
- ELASTICSEARCH_CLOUD_ID (for production)
- LOG_LEVEL

### 3.2 Vector Database Setup
For development (FAISS):
```bash
python -m src.vector_db.faiss_db init
```

For production (Elasticsearch):
```bash
python -m src.vector_db.elasticsearch_db init
```

## 4. Usage Examples

### 4.1 Basic Query
```python
from .rag_system import RAGSystem

rag = RAGSystem()
response = rag.process_query("What is the Great Depression?")
```

### 4.2 Conversational Query
```python
history = []
response = rag.process_conversational_query(
    "What caused the Great Depression?",
    conversation_history=history
)
```

## 5. Advanced Features

### 5.1 Quantized Embeddings
```python
rag = RAGSystem(
    use_quantized_embeddings=True,
    quantization_type="int8"
)
```

### 5.2 Semantic Caching
```python
rag = RAGSystem(
    semantic_similarity_threshold=0.75
)
```

## 6. Testing

### 6.1 Test Organization
- Legacy tests in `tests/legacy/`
- Current tests in `tests/current/`
- Performance tests in `tests/performance/`

### 6.2 Running Tests
```bash
# All tests
pytest tests/

# Specific test suite
pytest tests/current/test_improved_cache.py
```

## 7. Security Configuration

### 7.1 User Clearance
```python
from .security.clearance_manager import ClearanceManager

clearance_manager = ClearanceManager()
user_clearance = clearance_manager.verify_user("user1")
```

### 7.2 Document Classification
```python
# Document with classification
document = {
    "text": "...",
    "classification": "TS",
    "paragraphs": [
        {"text": "...", "classification": "U"},
        {"text": "...", "classification": "TS"}
    ]
}
```

## 8. Performance Monitoring

### 8.1 Metrics
- Response time
- Cache hit rate
- API usage
- Memory usage

### 8.2 Logging
```python
import logging
logging.basicConfig(level=logging.INFO)
```

## 9. Troubleshooting

### 9.1 Common Issues
- API key errors
- Memory issues
- Performance bottlenecks
- Import issues (use relative imports)

### 9.2 Solutions
- Check API key format
- Monitor resource usage
- Adjust thresholds
- Clear cache if needed
- Verify import paths
