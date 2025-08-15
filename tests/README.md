# RAG System Tests

This directory contains test files for the RAG system.

## Test Organization

### Legacy Tests
- test_great_depression_dashboard.py
- test_great_depression_rag.py
- test_retrieval.py
- test_semantic_cache.py
- test_semantic_cache_mock.py

### Current Tests
- test_improved_cache.py
- test_quantized_embeddings.py
- test_combined_caching.py

## Running Tests

To run all tests:
```bash
pytest tests/
```

To run specific tests:
```bash
pytest tests/test_improved_cache.py
```
