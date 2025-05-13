# RAG Chain Examples

This notebook demonstrates the RAG (Retrieval-Augmented Generation) implementation.

## Setup

First, let's import our components:

```python
from src.rag_core import RAGChain
from src.vector_store import ElasticsearchStore
from src.document_processing import DocumentReader, TextChunker
import os
```

## 1. RAG Setup
### 1.1 Initialize Components

```python
# Set up vector store
store = ElasticsearchStore("http://localhost:9200")

# Initialize RAG chain
chain = RAGChain(store)

# Load some test documents (from previous notebooks)
reader = DocumentReader()
chunker = TextChunker()
test_files = create_test_files()

# Process and index documents
def index_document(file_path):
    doc = reader.read_file(file_path)
    chunks = chunker.create_chunks(doc['text'], doc['metadata'])
    for chunk in chunks:
        store.add_document(chunk)
    return len(chunks)

# Index test files
for file_path in test_files.values():
    num_chunks = index_document(file_path)
    print(f"Indexed {num_chunks} chunks from {file_path}")
```

## 2. Basic Q&A
### 2.1 Simple Questions

```python
# Try some basic questions
questions = [
    "What is this document about?",
    "What kind of document is it?",
]

for question in questions:
    print(f"\nQ: {question}")
    answer = chain.generate_answer(question)
    print(f"A: {answer}")
```

### 2.2 Error Handling

```python
# Test error cases
try:
    # Empty question
    chain.generate_answer("")
except ValueError as e:
    print(f"Expected error for empty question: {e}")

try:
    # Question too long
    chain.generate_answer("?" * 1000)
except ValueError as e:
    print(f"Expected error for long question: {e}")
```

## 3. Advanced Features
### 3.1 Source Attribution

```python
# Get answer with sources
def ask_with_sources(question):
    response = chain.generate_answer_with_sources(question)
    print(f"\nQ: {question}")
    print(f"A: {response['answer']}")
    print("\nSources:")
    for src in response['sources']:
        print(f"- {src['text'][:100]}...")

# Try a question
ask_with_sources("What type of files are mentioned?")
```

### 3.2 Interactive Chat

```python
# Demonstrate chat functionality
def chat_session(questions):
    history = []
    for question in questions:
        print(f"\nUser: {question}")
        response = chain.chat_with_history(question, history)
        print(f"Assistant: {response['answer']}")
        history = response['history']

# Try a conversation
chat_questions = [
    "What is this document about?",
    "Can you tell me more about that?",
    "What format is it in?"
]

chat_session(chat_questions)
```

## 4. Performance Optimization
### 4.1 Context Window

```python
# Test different context sizes
context_sizes = [1, 2, 3, 5]

question = "What is this document about?"
for size in context_sizes:
    print(f"\nTesting with {size} chunks of context:")
    response = chain.generate_answer(
        question,
        max_chunks=size
    )
    print(f"Answer: {response[:100]}...")
```

### 4.2 Response Analysis

```python
# Analyze response quality
def analyze_response(question, answer, sources):
    print(f"\nQuestion: {question}")
    print(f"Answer length: {len(answer)} chars")
    print(f"Sources used: {len(sources)}")
    
    # Check source relevance
    source_text = " ".join(s['text'] for s in sources)
    print(f"Total source context: {len(source_text)} chars")

# Test a complex question
complex_q = "Explain the main topics covered in these documents"
response = chain.generate_answer_with_sources(complex_q)
analyze_response(
    complex_q,
    response['answer'],
    response['sources']
)
```
