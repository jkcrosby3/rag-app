# Vector Store Examples

This notebook demonstrates the vector storage and retrieval capabilities.

## Setup

First, let's import our components:

```python
from pathlib import Path
from src.vector_store import ElasticsearchStore
from src.document_processing import DocumentReader, TextChunker
from sentence_transformers import SentenceTransformer
```

## 1. Vector Store Setup
### 1.1 Basic Configuration

```python
# Initialize vector store
store = ElasticsearchStore(
    es_url="http://localhost:9200",
    index_name="documents"
)

# Load the embedding model
model = SentenceTransformer('all-MiniLM-L6-v2')
```

### 1.2 Index Creation

```python
# Check if index exists and create if needed
if not store.index_exists():
    store.create_index()
    print("Created new index")
else:
    print("Index already exists")
```

## 2. Document Processing
### 2.1 Prepare Test Documents

```python
# Create and process test documents
reader = DocumentReader()
chunker = TextChunker()

# Get test files from previous notebook
test_files = create_test_files()

# Process PDF document
pdf_doc = reader.read_file(test_files['pdf'])
pdf_chunks = chunker.create_chunks(pdf_doc['text'], pdf_doc['metadata'])

print(f"Created {len(pdf_chunks)} chunks from PDF")
```

### 2.2 Document Indexing

```python
# Index the chunks
for chunk in pdf_chunks:
    # Generate embedding
    embedding = model.encode(chunk['text'])

    # Add to vector store
    store.add_document({
        'text': chunk['text'],
        'vector': embedding,
        'metadata': chunk['metadata']
    })

print("Indexed all chunks")
```

## 3. Vector Search
### 3.1 Basic Search

```python
# Try a simple search
query = "What is this document about?"
results = store.similarity_search(query, k=3)

print("Search Results:")
for i, result in enumerate(results, 1):
    print(f"\nResult {i}:")
    print(f"Text: {result['text'][:100]}...")
    print(f"Score: {result['score']:.3f}")
```

### 3.2 Advanced Search

```python
# Search with metadata filters
filtered_results = store.similarity_search(
    query="test document",
    k=3,
    metadata_filter={
        "text_types": {"title": True}
    }
)

print("\nFiltered Results:")
for i, result in enumerate(filtered_results, 1):
    print(f"\nResult {i}:")
    print(f"Text: {result['text'][:100]}...")
    print(f"Score: {result['score']:.3f}")
    print(f"Metadata: {result['metadata']}")
```

## 4. Performance Analysis
### 4.1 Search Performance

```python
import time

def measure_search_time(query, k=3, runs=5):
    times = []
    for _ in range(runs):
        start = time.time()
        store.similarity_search(query, k=k)
        times.append(time.time() - start)
    return sum(times) / len(times)

# Test different k values
k_values = [1, 3, 5, 10]
for k in k_values:
    avg_time = measure_search_time("test query", k=k)
    print(f"Average search time (k={k}): {avg_time:.3f}s")
```

### 4.2 Index Statistics

```python
# Get index stats
stats = store.get_index_stats()
print("\nIndex Statistics:")
print(f"Total documents: {stats['doc_count']}")
print(f"Store size: {stats['store_size_bytes'] / 1024 / 1024:.2f}MB")
print(f"Vector dimensions: {stats['vector_dims']}")
```
