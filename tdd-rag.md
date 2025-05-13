# RAG System Technical Design Document (TDD)

## System Architecture

### High-Level Design
```
[Documents] → [Document Processor] → [Vector DB] → [RAG Chain] → [UI]
     ↓              ↓                    ↓            ↓          ↓
[File System]  [Text Chunks]      [Elasticsearch]   [LLM]   [Streamlit]
```

### Component Architecture
1. **Document Processor**
   - File system monitoring
   - Text extraction
   - Chunking
   - Vector generation

2. **Vector Storage**
   - Elasticsearch indices
   - Vector similarity search
   - Document metadata
   - Search optimization

3. **RAG Chain**
   - Context retrieval
   - LLM integration
   - Answer generation
   - Source tracking

4. **Web Interface**
   - Streamlit frontend
   - Real-time updates
   - File management
   - Chat interface

## Technical Stack

### Backend
- Python 3.11+
- FastAPI for API
- Elasticsearch for vectors
- LangChain for RAG
- PyPDF2 for PDFs
- sentence-transformers

### Frontend
- Streamlit
- streamlit-extras

### Testing
- pytest
- pytest-asyncio
- pytest-cov

### Dependencies
```
# Web Framework
fastapi>=0.100.0
uvicorn[standard]>=0.22.0

# Vector Storage
elasticsearch>=8.9.0

# RAG Pipeline
langchain>=0.0.300
sentence-transformers>=2.2.0

# Document Processing
pypdf2>=3.0.0
watchdog>=3.0.0
python-multipart>=0.0.6

# Frontend
streamlit>=1.24.0
streamlit-extras>=0.3.0

# Testing
pytest>=7.4.0
pytest-asyncio>=0.21.0
pytest-cov>=4.1.0
black>=23.7.0
flake8>=6.0.0
```

## Implementation Details

### Document Processing Pipeline

#### 1. Document Detection
```python
class DocumentMonitor:
    def __init__(self, raw_dir: str, supported_formats: list[str]):
        self.raw_dir = Path(raw_dir)
        self.supported_formats = supported_formats
        self.observer = Observer()
    
    def start_monitoring(self):
        event_handler = DocumentHandler()
        self.observer.schedule(event_handler, self.raw_dir, recursive=False)
        self.observer.start()
```

#### 2. Text Extraction
```python
class DocumentLoader:
    def __init__(self):
        self.loaders = {
            ".pdf": PyPDFLoader,
            ".txt": TextLoader,
            ".md": UnstructuredMarkdownLoader
        }
    
    def load(self, file_path: Path) -> Dict[str, Any]:
        loader = self.loaders.get(file_path.suffix.lower())
        documents = loader(str(file_path)).load()
        return {"content": documents, "metadata": {...}}
```

#### 3. Text Chunking
```python
class DocumentChunker:
    def __init__(self, chunk_size: int = 1000, chunk_overlap: int = 200):
        self.splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap
        )
    
    def split(self, document: Dict) -> List[Dict]:
        chunks = self.splitter.split_documents(document["content"])
        return [{"content": chunk.page_content, "metadata": {...}}]
```

### RAG Implementation

#### 1. Vector Generation
```python
class VectorGenerator:
    def __init__(self, model_name: str, es_host: str):
        self.model = SentenceTransformer(model_name)
        self.es = Elasticsearch(es_host)
    
    def generate_vectors(self, chunks: List[Dict]):
        for chunk in chunks:
            vector = self.model.encode(chunk["content"])
            yield {"vector": vector, "content": chunk["content"]}
```

#### 2. RAG Chain
```python
class RAGChain:
    def __init__(self, model_name: str, es_host: str):
        self.llm = ChatOpenAI(model_name=model_name)
        self.es = Elasticsearch(es_host)
        self.prompt = ChatPromptTemplate.from_messages([...])
    
    def generate_answer(self, question: str) -> Dict:
        docs = self.retrieve_documents(question)
        context = self.format_context(docs)
        answer = self.llm.generate(context, question)
        return {"answer": answer, "sources": docs}
```

## Security Architecture

### Authentication
- JWT-based authentication
- Role-based access control
- API key management
- Session handling

### Data Protection
- Document encryption
- Secure file storage
- Access logging
- Data backup

## Testing Strategy

### Unit Tests
```python
def test_document_processing():
    processor = DocumentProcessor()
    result = processor.process_file("test.pdf")
    assert result["status"] == "success"
    assert len(result["chunks"]) > 0

def test_rag_chain():
    chain = RAGChain()
    response = chain.generate_answer("test question")
    assert response["answer"]
    assert response["sources"]
```

### Integration Tests
```python
async def test_full_pipeline():
    # Test document upload
    doc_id = await upload_document("test.pdf")
    assert doc_id
    
    # Test processing
    status = await get_processing_status(doc_id)
    assert status == "completed"
    
    # Test search
    results = await search("test query")
    assert len(results) > 0
```

### Performance Tests
```python
async def test_search_performance():
    start_time = time.time()
    results = await search("test query")
    duration = time.time() - start_time
    assert duration < 2.0  # 2 second threshold
```

## Monitoring and Logging

### Metrics
- Document processing time
- Search latency
- LLM response time
- Error rates
- System resource usage

### Logging
```python
logger = logging.getLogger(__name__)

def log_processing(doc_id: str, status: str):
    logger.info(f"Document {doc_id}: {status}")
    metrics.increment(f"document_processing_{status}")
```

## Deployment

### Requirements
- Python 3.11+
- Elasticsearch 8.x
- 8GB+ RAM
- 4+ CPU cores
- 100GB+ storage

### Configuration
```yaml
# config.yaml
elasticsearch:
  host: localhost
  port: 9200
  index: documents

llm:
  model: gpt-4
  temperature: 0
  max_tokens: 1000

processing:
  chunk_size: 1000
  chunk_overlap: 200
  batch_size: 50
```

### Startup Script
```bash
#!/bin/bash
# start.sh
source venv/bin/activate
export CONFIG_PATH=./config.yaml
uvicorn app:app --host 0.0.0.0 --port 8000
```
