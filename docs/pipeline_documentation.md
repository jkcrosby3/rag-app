# RAG System Pipeline Documentation

## 1. Document Ingestion

### Files Involved:
- `src/document_processing/batch_processor.py`
- `src/document_processing/document_processor.py`

### Key Functions:
```python
# In document_processor.py
process_file(self, file_path: Path)
process_text(self, text: str, metadata: Dict[str, Any])

# In batch_processor.py
process_documents(self, documents: List[Path])
```

### Process Flow:
1. Documents are uploaded and processed in batch
2. Each document is processed individually:
   - Extracts text content
   - Preserves metadata
   - Handles different file formats
3. Processed documents are saved as JSON files

## 2. Text Chunking

### Files Involved:
- `src/document_processing/chunker.py`
- `src/document_processing/chunk_documents.py`

### Key Functions:
```python
# In chunker.py
split_into_chunks(self, text: str, metadata: Dict[str, Any]) -> List[Dict[str, Any]]
get_chunk_size(self, text: str) -> int

# In chunk_documents.py
main()
```

### Process Flow:
1. Documents are split into smaller chunks
2. Each chunk preserves metadata and context
3. Chunks are saved with:
   - Text content
   - Chunk index
   - Token count
   - Original metadata
   - Topic information

## 3. Embedding Generation

### Files Involved:
- `src/document_processing/generator.py`
- `src/document_processing/embed_documents.py`

### Key Functions:
```python
# In generator.py
generate_embeddings(self, text: str) -> np.ndarray
process_chunk(self, chunk: Dict[str, Any]) -> Dict[str, Any]

# In embed_documents.py
main()
```

### Process Flow:
1. Chunks are processed through embedding generator
2. Uses sentence-transformers with all-MiniLM-L6-v2 model
3. Embeddings are added to each chunk
4. Embedded documents are saved in data/embedded directory

## 4. Vector Database Integration

### Files Involved:
- `src/vector_db/faiss_db.py`
- `src/vector_db/elasticsearch_db.py`
- `scripts/build_vector_db.py`

### Key Functions:
```python
# In faiss_db.py
build_vector_db_from_embedded_docs(self)
add_document(self, doc_id: str, embedding: np.ndarray, metadata: Dict[str, Any])

# In elasticsearch_db.py
build_es_vector_db_from_embedded_docs(self)
index_document(self, doc_id: str, embedding: np.ndarray, metadata: Dict[str, Any])

# In build_vector_db.py
main()
```

### Process Flow:
1. Embedded documents are loaded from data/embedded
2. Vectors are added to the database
3. Metadata is stored alongside embeddings
4. Index is built for efficient search

## 5. Query Processing

### Files Involved:
- `src/retrieval/retriever.py`
- `src/llm/claude_client.py`
- `src/rag_system.py`

### Key Functions:
```python
# In retriever.py
search(self, query: str, k: int = 5)
filter_by_topic(self, results: List[Dict[str, Any]], topic: str)

# In claude_client.py
format_context(self, chunks: List[Dict[str, Any]])
generate_response(self, query: str, context: str)

# In rag_system.py
process_query(self, query: str)
```

### Process Flow:
1. Query is received by the system
2. Vector database searches for similar chunks
3. Relevant chunks are retrieved
4. Claude LLM generates response based on context

## 6. Performance Optimization

### Files Involved:
- `src/tools/caching.py`
- `src/tools/quantized_model.py`
- `src/tools/performance_monitor.py`

### Key Functions:
```python
# In caching.py
get_cached_embedding(self, text: str)
store_embedding(self, text: str, embedding: np.ndarray)

# In quantized_model.py
load_quantized_model(self)
generate_quantized_embedding(self, text: str)

# In performance_monitor.py
track_performance(self, operation: str, duration: float)
generate_metrics_report(self)
```

### Process Flow:
1. Embeddings are cached to avoid regeneration
2. Quantized model is used for faster inference
3. Performance metrics are tracked
4. Cache hit rates and response times are monitored

## 7. Tag Management

### Files Involved:
- `src/web/models.py`
- `src/web/tracking_app.py`
- `src/tools/tag_validator.py`

### Key Functions:
```python
# In models.py
Tag.create()
Tag.update()
Tag.delete()

# In tracking_app.py
create_tag()
update_tag()
delete_tag()

# In tag_validator.py
validate_tag()
validate_synonym()
validate_hierarchy()
```

### Process Flow:
1. Tags are created and validated
2. Hierarchical relationships are maintained
3. Synonyms are managed
4. Usage statistics are tracked

## 8. Environment Configuration

### Files Involved:
- `.env.template`
- `config.py`
- `setup.py`

### Key Functions:
```python
# In config.py
load_environment()
get_api_key()
get_vector_db_config()
```

### Process Flow:
1. Environment variables are loaded
2. API keys are configured
3. Vector database settings are set
4. Logging is configured

## Error Handling and Validation

### Files Involved:
- `src/tools/validation.py`
- `src/tools/metadata_validator.py`
- `src/tools/tag_validator.py`

### Key Functions:
```python
# In validation.py
validate_document()
validate_chunk()
validate_embedding()

# In metadata_validator.py
validate_metadata_schema()
validate_required_fields()
validate_date_formats()

# In tag_validator.py
validate_tag_name()
validate_synonyms()
validate_hierarchy()
```

## Notes:
1. The system supports both FAISS (development) and Elasticsearch (production) backends
2. All text processing maintains metadata integrity
3. Performance optimizations are implemented throughout the pipeline
4. Comprehensive error handling and validation is in place
5. The system is designed for scalability and maintainability

## Dependencies:
- PyMuPDF for PDF processing
- PyPDF2 for PDF handling
- sentence-transformers for embeddings
- FAISS for vector search
- Elasticsearch for production search
- Anthropic's Claude API for LLM generation
