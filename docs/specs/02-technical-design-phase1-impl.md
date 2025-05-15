# RAG System Implementation Details

This document provides detailed implementation specifications for the RAG system. For the high-level technical design, see [02-technical-design-phase1.md](02-technical-design-phase1.md).

## Table of Contents

1. [Document Processing](#document-processing)
2. [Vector Store Integration](#vector-store-integration)
3. [RAG Implementation](#rag-implementation)
4. [CLI Interface](#cli-interface)
5. [Testing Strategy](#testing-strategy)

## Document Processing

### Text Chunking Strategy

```python
def chunk_text(text: str, chunk_size: int = 1000, overlap: int = 100) -> List[str]:
    """Split text into overlapping chunks."""
    chunks = []
    start = 0
    
    while start < len(text):
        end = start + chunk_size
        chunk = text[start:end]
        
        # Find natural boundary
        if end < len(text):
            last_period = chunk.rfind('.')
            if last_period > 0:
                end = start + last_period + 1
                chunk = text[start:end]
        
        chunks.append(chunk)
        start = end - overlap
        
    return chunks
```

### PDF Processing

```python
def process_pdf(file_path: Path) -> Dict[str, Any]:
    """Process PDF document with PyMuPDF."""
    with fitz.open(file_path) as doc:
        text = ""
        metadata = {
            "title": doc.metadata.get("title", ""),
            "author": doc.metadata.get("author", ""),
            "page_count": len(doc)
        }
        
        for page in doc:
            text += page.get_text()
            
        return {
            "text": text,
            "metadata": metadata,
            "chunks": chunk_text(text)
        }
```

## Vector Store Integration

### Elasticsearch Setup

```python
def setup_vector_index(es_client, index_name: str) -> None:
    """Create Elasticsearch index with vector search capabilities."""
    settings = {
        "mappings": {
            "properties": {
                "text": {"type": "text"},
                "embedding": {
                    "type": "dense_vector",
                    "dims": 1536,
                    "index": True,
                    "similarity": "cosine"
                }
            }
        }
    }
    
    es_client.indices.create(index=index_name, body=settings)
```

### Document Indexing

```python
def index_chunks(chunks: List[str], embedder: Any) -> None:
    """Index document chunks with their embeddings."""
    for chunk in chunks:
        embedding = embedder.embed_text(chunk)
        es_client.index(
            index="documents",
            body={
                "text": chunk,
                "embedding": embedding
            }
        )
```

## RAG Implementation

### Context Retrieval

```python
def get_context(question: str, k: int = 3) -> List[str]:
    """Retrieve relevant context for the question."""
    embedding = embedder.embed_text(question)
    results = es_client.search(
        index="documents",
        body={
            "query": {
                "script_score": {
                    "query": {"match_all": {}},
                    "script": {
                        "source": "cosineSimilarity(params.query_vector, 'embedding')",
                        "params": {"query_vector": embedding}
                    }
                }
            },
            "size": k
        }
    )
    
    return [hit["_source"]["text"] for hit in results["hits"]["hits"]]
```

### Answer Generation

```python
def generate_answer(question: str, context: List[str]) -> str:
    """Generate answer using LLM."""
    prompt = PROMPT_TEMPLATES["qa"].format(
        context="\n".join(context),
        question=question
    )
    
    response = openai.ChatCompletion.create(
        model="gpt-4",
        messages=[
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": prompt}
        ]
    )
    
    return response.choices[0].message.content
```

## CLI Interface

```python
def main():
    """Main CLI entry point."""
    parser = argparse.ArgumentParser(description="RAG System CLI")
    parser.add_argument("--query", type=str, help="Question to ask")
    parser.add_argument("--doc", type=str, help="Document to process")
    
    args = parser.parse_args()
    
    if args.doc:
        process_document(args.doc)
    
    if args.query:
        context = get_context(args.query)
        answer = generate_answer(args.query, context)
        print(f"Answer: {answer}")
```

## Testing Strategy

### Unit Tests

```python
def test_chunking():
    """Test text chunking functionality."""
    text = "This is a test. Another sentence. And one more."
    chunks = chunk_text(text, chunk_size=20, overlap=5)
    assert len(chunks) > 0
    assert all(len(chunk) <= 20 for chunk in chunks)

def test_pdf_processing():
    """Test PDF processing."""
    result = process_pdf(Path("test.pdf"))
    assert "text" in result
    assert "metadata" in result
    assert "chunks" in result
```

### Integration Tests

```python
def test_full_pipeline():
    """Test the complete RAG pipeline."""
    # Process document
    doc_path = Path("test.pdf")
    process_document(doc_path)
    
    # Query system
    question = "What is the main topic?"
    answer = query_system(question)
    
    assert isinstance(answer, str)
    assert len(answer) > 0
```
