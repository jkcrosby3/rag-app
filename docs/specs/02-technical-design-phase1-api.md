# RAG System API Reference

This document provides detailed API documentation for the RAG system components. For the high-level technical design, see [02-technical-design-phase1.md](02-technical-design-phase1.md).

## Table of Contents

1. [Core Interfaces](#core-interfaces)
   - [DocumentReader](#documentreader)
   - [ElasticsearchStore](#elasticsearchstore)
   - [RAGChain](#ragchain)
2. [Configuration Reference](#configuration-reference)
   - [Vector Search Settings](#vector-search-settings)
   - [RAG Prompt Templates](#rag-prompt-templates)

## Core Interfaces

### DocumentReader

```python
class DocumentReader:
    """Handles reading and processing of various document types."""
    
    def read_file(self, file_path: Path) -> Dict[str, Any]:
        """Read and process a document file.
        
        Args:
            file_path: Path to the document file
            
        Returns:
            Dict containing:
            - text: Extracted text content
            - metadata: Document metadata
            - chunks: List of text chunks
        """
        pass
```

### ElasticsearchStore

```python
class ElasticsearchStore:
    """Vector store implementation using Elasticsearch."""
    
    def index_document(self, doc_id: str, chunks: List[Dict]) -> None:
        """Index document chunks with their embeddings."""
        pass
        
    def semantic_search(self, query: str, k: int = 3) -> List[Dict]:
        """Perform semantic search using vector similarity."""
        pass
```

### RAGChain

```python
class RAGChain:
    """Core RAG implementation combining retrieval and generation."""
    
    def generate_answer(self, question: str) -> Dict:
        """Generate answer using retrieved context."""
        pass
```

## Configuration Reference

### Vector Search Settings

```python
VECTOR_SEARCH_CONFIG = {
    "index_settings": {
        "number_of_shards": 1,
        "number_of_replicas": 0
    },
    "similarity": {
        "type": "cosine",
        "dims": 1536  # OpenAI embedding dimensions
    }
}
```

### RAG Prompt Templates

```python
PROMPT_TEMPLATES = {
    "qa": """Answer the question based on the context below.
    
Context: {context}
Question: {question}
Answer: Let me help you with that."""
}
```
