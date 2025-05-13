# RAG System API Reference

## Document Processing API

### DocumentReader

#### `read_file(file_path: Path) -> Dict[str, Any]`
Reads and extracts text from a document file.

**Parameters:**
- `file_path`: Path to the document file (PDF or TXT)

**Returns:**
```python
{
    "text": str,      # Extracted text content
    "metadata": {     # Document metadata
        "title": str,
        "author": str,
        "pages": int
    },
    "source": str     # Original file path
}
```

### TextChunker

#### `create_chunks(text: str) -> List[Dict[str, Any]]`
Splits text into overlapping chunks.

**Parameters:**
- `text`: Input text to chunk

**Returns:**
```python
[{
    "text": str,    # Chunk text content
    "start": int,   # Start position in original text
    "end": int      # End position in original text
}]
```

## Vector Store API

### ElasticsearchStore

#### `add_documents(documents: List[Dict[str, Any]]) -> None`
Adds document chunks to Elasticsearch.

**Parameters:**
- `documents`: List of document chunks to add
```python
[{
    "text": str,      # Document text
    "metadata": dict  # Optional metadata
}]
```

#### `similarity_search(query: str, k: int = 3) -> List[Dict]`
Searches for similar documents.

**Parameters:**
- `query`: Search query text
- `k`: Number of results to return (default: 3)

**Returns:**
```python
[{
    "text": str,      # Document text
    "metadata": dict  # Document metadata
}]
```

## RAG Chain API

### RAGChain

#### `generate_answer(question: str) -> Dict`
Generates an answer using the RAG approach.

**Parameters:**
- `question`: User's question

**Returns:**
```python
{
    "answer": str,     # Generated answer
    "sources": [{      # Source documents used
        "text": str,
        "metadata": dict
    }]
}
```

## CLI API

### RagCLI

#### `process_file(file_path: str) -> None`
Processes a document file.

**Parameters:**
- `file_path`: Path to document file

#### `ask_question(question: str) -> None`
Gets answer for a question.

**Parameters:**
- `question`: Question to answer

## Web API

### RagWebApp

#### `process_uploaded_file(uploaded_file: UploadedFile) -> None`
Processes an uploaded file.

**Parameters:**
- `uploaded_file`: Streamlit uploaded file object

#### `run() -> None`
Runs the Streamlit web application.
