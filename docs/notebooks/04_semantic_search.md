# Semantic Search Examples

This notebook demonstrates semantic search capabilities and optimizations.

## Setup

First, let's import our components:

```python
from src.vector_store import ElasticsearchStore
from src.document_processing import DocumentReader, TextChunker
from sentence_transformers import SentenceTransformer
import numpy as np
```

## 1. Search Setup
### 1.1 Initialize Components

```python
# Set up vector store
store = ElasticsearchStore("http://localhost:9200")

# Initialize embedding model
model = SentenceTransformer('all-MiniLM-L6-v2')

# Load test documents (from previous notebooks)
reader = DocumentReader()
chunker = TextChunker()
test_files = create_test_files()
```

## 2. Search Features
### 2.1 Basic Search

```python
def semantic_search(query, k=3):
    # Generate query embedding
    query_vector = model.encode(query)
    
    # Search
    results = store.similarity_search(query, k=k)
    
    print(f"\nQuery: {query}")
    print(f"Found {len(results)} results:")
    for i, result in enumerate(results, 1):
        print(f"\nResult {i}:")
        print(f"Text: {result['text'][:100]}...")
        print(f"Score: {result['score']:.3f}")

# Try some searches
queries = [
    "What is a test document?",
    "Show me examples of files",
]

for query in queries:
    semantic_search(query)
```

### 2.2 Advanced Filtering

```python
def filtered_search(query, metadata_filter, k=3):
    results = store.similarity_search(
        query,
        k=k,
        metadata_filter=metadata_filter
    )
    
    print(f"\nQuery: {query}")
    print(f"Filter: {metadata_filter}")
    print(f"Found {len(results)} results:")
    for i, result in enumerate(results, 1):
        print(f"\nResult {i}:")
        print(f"Text: {result['text'][:100]}...")
        print(f"Score: {result['score']:.3f}")
        print(f"Metadata: {result['metadata']}")

# Try filtered searches
filters = [
    {"text_types": {"title": True}},
    {"hierarchy": {"section": "Introduction"}},
]

for filter_dict in filters:
    filtered_search("test", filter_dict)
```

## 3. Search Optimization
### 3.1 Score Thresholds

```python
def threshold_search(query, min_score=0.5, k=10):
    results = store.similarity_search(query, k=k)
    
    # Filter by score
    good_results = [
        r for r in results 
        if r['score'] >= min_score
    ]
    
    print(f"\nQuery: {query}")
    print(f"Results above {min_score}: {len(good_results)}/{len(results)}")
    for i, result in enumerate(good_results, 1):
        print(f"\nResult {i}:")
        print(f"Text: {result['text'][:100]}...")
        print(f"Score: {result['score']:.3f}")

# Test different thresholds
thresholds = [0.7, 0.5, 0.3]
for threshold in thresholds:
    threshold_search("test document", min_score=threshold)
```

### 3.2 Query Expansion

```python
def expand_query(query):
    # Generate related terms
    expanded = [query]
    
    # Add synonyms
    if "document" in query.lower():
        expanded.append(query.lower().replace("document", "file"))
    if "test" in query.lower():
        expanded.append(query.lower().replace("test", "sample"))
        
    return expanded

def expanded_search(query, k=3):
    queries = expand_query(query)
    all_results = []
    
    print(f"Original query: {query}")
    print(f"Expanded to: {queries}")
    
    # Search with all queries
    for q in queries:
        results = store.similarity_search(q, k=k)
        all_results.extend(results)
    
    # Deduplicate and sort by score
    unique_results = {}
    for r in all_results:
        if r['text'] not in unique_results:
            unique_results[r['text']] = r
    
    sorted_results = sorted(
        unique_results.values(),
        key=lambda x: x['score'],
        reverse=True
    )[:k]
    
    print(f"\nFound {len(sorted_results)} unique results:")
    for i, result in enumerate(sorted_results, 1):
        print(f"\nResult {i}:")
        print(f"Text: {result['text'][:100]}...")
        print(f"Score: {result['score']:.3f}")

# Try expanded search
expanded_search("test document")
expanded_search("sample file")
```

## 4. Performance Analysis
### 4.1 Search Speed vs Accuracy

```python
import time

def benchmark_search(query, k_values=[1, 3, 5, 10], runs=5):
    results = {}
    
    for k in k_values:
        times = []
        scores = []
        
        for _ in range(runs):
            start = time.time()
            search_results = store.similarity_search(query, k=k)
            times.append(time.time() - start)
            scores.extend(r['score'] for r in search_results)
        
        results[k] = {
            'avg_time': sum(times) / len(times),
            'avg_score': sum(scores) / len(scores)
        }
    
    print("\nSearch Benchmark:")
    for k, stats in results.items():
        print(f"\nk={k}:")
        print(f"Average time: {stats['avg_time']:.3f}s")
        print(f"Average score: {stats['avg_score']:.3f}")

# Run benchmark
benchmark_search("test document")
```

### 4.2 Index Analysis

```python
# Analyze index statistics
stats = store.get_index_stats()

print("Index Analysis:")
print(f"Total documents: {stats['doc_count']}")
print(f"Store size: {stats['store_size_bytes'] / 1024 / 1024:.2f}MB")
print(f"Vector dimensions: {stats['vector_dims']}")

# Analyze vector distribution
vectors = store.get_all_vectors()
magnitudes = np.linalg.norm(vectors, axis=1)

print("\nVector Analysis:")
print(f"Mean magnitude: {np.mean(magnitudes):.3f}")
print(f"Std deviation: {np.std(magnitudes):.3f}")
print(f"Min magnitude: {np.min(magnitudes):.3f}")
print(f"Max magnitude: {np.max(magnitudes):.3f}")
```
