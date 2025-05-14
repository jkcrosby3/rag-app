# RAG System
# Technical Design Document: RAG System (MVP)

## 1. Overview

### Purpose
This document outlines the technical implementation for Phase 1 (MVP) of the RAG system. This is a proof-of-concept implementation focused on core functionality of first a CLI tool, then a web app if time permits.

For production features, see [`02-technical-design-phase2.md`](02-technical-design-phase2.md).

### Goals
1. Demonstrate core RAG functionality
2. Validate the technical approach
3. Gather feedback for Phase 2

### Success Criteria
- Successfully process PDF, TXT, and MD documents
- Generate relevant answers from documents
- Complete basic operations within reasonable time
- Demonstrate extensibility for Phase 2

## 2. Resources & References

### Core Dependencies
1. **Document Processing**
   - PyPDF2 (PDF processing): https://pypdf2.readthedocs.io/
   - python-magic (File type detection): https://github.com/ahupp/python-magic
   - markdown-it-py (Markdown processing): https://markdown-it-py.readthedocs.io/

2. **Vector Embeddings**
   - sentence-transformers: https://www.sbert.net/
   - HuggingFace Transformers: https://huggingface.co/docs/transformers/

3. **Vector Storage**
   - FAISS: https://github.com/facebookresearch/faiss
   - Chroma: https://www.trychroma.com/
   - Elasticsearch Vector Search:
     * Official Guide: https://www.elastic.co/guide/en/elasticsearch/reference/current/dense-vector.html
     * Text Embeddings Tutorial: https://www.elastic.co/docs/explore-analyze/machine-learning/nlp/ml-nlp-text-emb-vector-search-example
     * Semantic Search Implementation: https://www.elastic.co/blog/semantic-search-with-elasticsearch
     * RAG with Elasticsearch: https://www.elastic.co/blog/building-generative-ai-applications-elasticsearch

4. **LLM Integration**
   - LangChain: https://python.langchain.com/
   - OpenAI API: https://platform.openai.com/docs/api-reference

### Test Documents

1. **Technical Documentation**
   - Python Documentation (PDF): https://docs.python.org/3/download.html
   - Git Book (PDF, MD): https://git-scm.com/book/en/v2
   - Elasticsearch Guide (PDF): https://www.elastic.co/guide/en/elasticsearch/reference/current/elasticsearch-reference.pdf
   - Docker Overview (MD): https://github.com/docker/docs/tree/main/content

2. **Research Papers**
   - arXiv ML Papers (PDF): https://arxiv.org/list/cs.LG/recent
   - Attention Is All You Need (PDF): https://arxiv.org/pdf/1706.03762.pdf
   - LangChain Papers (PDF): https://arxiv.org/abs/2310.03722

3. **General Knowledge**
   - Wikipedia Articles (TXT): https://dumps.wikimedia.org/
   - Project Gutenberg Books (TXT): https://www.gutenberg.org/browse/scores/top
   - CommonCrawl News (TXT): https://commoncrawl.org/

4. **Sample Repositories**
   - Awesome-README Collection (MD): https://github.com/matiassingers/awesome-readme
   - OpenAI Cookbook (MD): https://github.com/openai/openai-cookbook
   - Google Style Guides (MD): https://github.com/google/styleguide

> Note: When using these documents for testing:
> - Ensure compliance with usage terms
> - Keep test datasets small (< 100MB total)
> - Include a mix of document types and content
> - Document any preprocessing steps

### RAG Implementation Guides
1. **Document Processing & Embeddings**
   - LangChain RAG Tutorial: https://python.langchain.com/docs/tutorials/rag-app/
   - Advanced RAG with HuggingFace: https://huggingface.co/learn/cookbook/advanced_rag
   - Semantic Search Implementation: https://www.sbert.net/examples/applications/semantic-search/README.html

2. **Vector Search & Retrieval**
   - FAISS Getting Started: https://github.com/facebookresearch/faiss/wiki/Getting-started
   - Chroma Quickstart: https://docs.trychroma.com/getting-started
   - Elasticsearch Semantic Search:
     * Vector Search Setup: https://www.elastic.co/guide/en/elasticsearch/reference/current/vector-search-setup.html
     * Hybrid Search Tutorial: https://www.elastic.co/blog/hybrid-search-elasticsearch-vector-text
     * Performance Tuning: https://www.elastic.co/blog/how-to-tune-elastic-vector-search-for-accuracy-speed-and-cost

3. **LLM Integration**
   - LangChain Agents: https://python.langchain.com/docs/modules/agents/
   - RAG Best Practices: https://www.pinecone.io/learn/rag-patterns/

4. **Prompt Engineering**
   - Kaggle Guide: https://www.kaggle.com/whitepaper-prompt-engineering
   - OpenAI Best Practices: https://platform.openai.com/docs/guides/prompt-engineering
   - Anthropic Prompt Design: https://docs.anthropic.com/claude/docs/prompt-engineering
   - LangChain Prompting Guide: https://python.langchain.com/docs/modules/model_io/prompts/prompt_templates
   - Microsoft Learn: https://learn.microsoft.com/en-us/semantic-kernel/prompt-engineering/
   - Brex's Prompt Engineering Guide: https://github.com/brexhq/prompt-engineering

### CLI Development
1. **Framework Options**
   - Click Tutorial: https://click.palletsprojects.com/en/8.1.x/quickstart/
   - Typer (Click-based): https://typer.tiangolo.com/
   - Rich (Terminal UI): https://rich.readthedocs.io/
   - Textual (TUI Framework): https://textual.textualize.io/

2. **Example Projects**
   - Click CLI Tutorial: https://realpython.com/python-click/
   - Typer CLI App Guide: https://realpython.com/python-typer-cli/

### Web App Development (if time permits)
1. **Streamlit**
   - Official Tutorial: https://docs.streamlit.io/get-started/tutorials
   - Best Practices: https://docs.streamlit.io/library/advanced-features/best-practices
   - Component Gallery: https://streamlit.io/components

2. **Example Projects**
   - Streamlit Chat Interface: https://docs.streamlit.io/knowledge-base/tutorials/build-conversational-apps
   - RAG with Streamlit: https://blog.streamlit.io/build-a-chatbot-with-custom-data-sources/

### Sample Code Repositories
1. **End-to-End RAG Examples**
   - LangChain RAG Template: https://github.com/langchain-ai/langchain/tree/master/templates/rag-conversation
   - HuggingFace RAG Pipeline: https://huggingface.co/spaces/Xenova/rag-pipeline

2. **CLI Examples**
   - Click Examples: https://github.com/pallets/click/tree/main/examples
   - Typer CLI Examples: https://github.com/tiangolo/typer/tree/master/examples

3. **Streamlit Examples**
   - Official Gallery: https://streamlit.io/gallery
   - Example Apps: https://github.com/streamlit/demo-self-driving

## 3. Code Organization

### Technical Design Document Code
The TDD contains essential interfaces and critical implementation details:

1. **Core Interfaces**: Basic class and method signatures that define the public API
2. **Key Data Structures**: Important configuration and data models
3. **Critical Algorithms**: Core processing logic and strategies

### Implementation Details

#### 1. Core Interfaces

##### Interface Overview
```python
class DocumentReader:
    def read_file(self, file_path: Path) -> Dict[str, Any]: pass

class ElasticsearchStore:
    def add_documents(self, documents: List[Dict]): pass
    def similarity_search(self, query: str, k: int = 5) -> List[Dict]: pass
    
    # Phase 2: Document Lifecycle Management
    # def delete_documents(self, doc_ids: List[str]): pass
    # def update_document(self, doc_id: str, document: Dict): pass
    # def bulk_delete(self, query: Dict): pass
    # def bulk_update(self, documents: List[Dict]): pass

class RAGChain:
    def generate_answer(self, question: str) -> Dict: pass
    def chat(self, message: str, history: List[Dict] = None) -> Dict: pass
```

##### Detailed API Reference

**DocumentReader**
```python
def read_file(self, file_path: Path) -> Dict[str, Any]:
    """Read and process a document file.
    
    Args:
        file_path: Path to the document (PDF, TXT, or MD)
        
    Returns:
        Dict containing:
        - content: Extracted text content
        - metadata: File metadata (type, size, etc.)
    """
```

**ElasticsearchStore**
```python
def add_documents(self, documents: List[Dict]):
    """Index documents in Elasticsearch with vector embeddings.
    
    Args:
        documents: List of documents with content and metadata
    """

def similarity_search(self, query: str, k: int = 5) -> List[Dict]:
    """Perform semantic search using vector embeddings.
    
    Args:
        query: Search query
        k: Number of results to return
        
    Returns:
        List of matching documents with scores
    """
```

**RAGChain**
```python
def generate_answer(self, question: str) -> str:
    """Generate answer using RAG pipeline (Phase 1).
    
    Args:
        question: User's question
        
    Returns:
        str: Generated response without sources
    """

def generate_answer_with_sources(self, question: str) -> Dict:
    """Generate answer using RAG pipeline with source tracking (Phase 2).
    
    Args:
        question: User's question
        
    Returns:
        Dict containing:
        - answer: Generated response
        - sources: Supporting document snippets
        - metadata: Response metadata
    """

def chat(self, message: str) -> str:
    """Simple chat without context management (Phase 1).
    
    Args:
        message: User's message
        
    Returns:
        str: Generated response
    """

def chat_with_history(self, message: str, history: List[Dict] = None) -> Dict:
    """Interactive chat with context management (Phase 2).
    
    Args:
        message: User's message
        history: Previous conversation turns
        
    Returns:
        Dict containing response and context
    """
```

#### 2. Critical Configurations

1. **Vector Search Settings**
```python
ELASTICSEARCH_CONFIG = {
    "index": {
        "mappings": {
            "properties": {
                "content": {"type": "text"},
                "embedding": {
                    "type": "dense_vector",
                    "dims": 768,
                    "index": True,
                    "similarity": "cosine"
                }
            }
        }
    }
}
```

2. **RAG Prompt Template**
```python
BASE_PROMPT = """
Answer the question based on the provided context.

Context:
{context}

Question: {question}

Answer in a helpful and informative way. If the context doesn't contain
enough information to answer the question fully, acknowledge this and
provide the best possible answer with the available information.

Answer:"""
```

#### 3. Implementation Examples

Detailed implementation examples are provided in the following notebooks:

1. **Document Processing** (`notebooks/01_document_processing.ipynb`):
   - File type handling (PDF, TXT, MD)
   - Text extraction and cleaning
   - Chunking strategies
   - Error handling

2. **Vector Store** (`notebooks/02_vector_store.ipynb`):
   - Elasticsearch setup
   - Document indexing
   - Vector search examples
   - Performance optimization

3. **RAG Implementation** (`notebooks/03_rag_chain.ipynb`):
   - End-to-end pipeline
   - Chat examples
   - Prompt engineering
   - LLM integration

4. **Semantic Search** (`notebooks/04_semantic_search.ipynb`):
   - Text embeddings generation
   - Similarity metrics comparison
   - Search optimization techniques
   - Performance benchmarking

Refer to these notebooks for complete implementation details and working examples.

## 4. System Architecture

### High-Level Design
```mermaid
graph TD
    A[Document Input] --> B[Document Processor]
    B --> C[Vector Store]
    D[User Question] --> E[RAG Chain]
    C --> E
    E --> F[Answer Output]
```

### Core Components
1. **Document Processor**
   - Reads documents (PDF, TXT)
   - Splits into chunks
   - Handles basic errors

2. **Vector Store**
   - Elasticsearch backend
   - Stores document chunks
   - Enables semantic search

3. **RAG Chain**
   - Retrieves relevant context
   - Generates answers
   - Tracks sources

### MVP Scope

#### Must Have
- Basic document processing (PDF, TXT)
- Vector search capability using Elasticsearch
- Command Line Interface (CLI)
  * Document upload and processing
  * Basic Q&A mode (answers only)
  * Basic search commands
- Local deployment

#### Nice to Have
- Markdown (MD) file support
- Source tracking in answers
  * Reference documents
  * Relevant snippets
  * Confidence scores
- Interactive chat mode
  * Chat history
  * Context management
- Streamlit Web Application
  * Document upload interface
  * Chat-like Q&A interface
  * Search results visualization
- Basic error handling
- Simple logging
- Input validation

#### Out of Scope
> All production features are documented in [`02-technical-design-phase2.md`](02-technical-design-phase2.md)
- Authentication/authorization
- Multi-user support
- High availability
- Security features
- Advanced monitoring
- Backup/recovery

### Phased Development Approach

> Note: This approach breaks down the MVP into smaller, manageable pieces that build upon each other.
> Each phase should be completed, tested, and reviewed before moving to the next.

#### Phase 1: Basic Document Processing (3-4 days)
1. **TXT File Processing** (1-2 days) - [Issue #1](https://github.com/justinlawyer/rag-app/issues/1), [Issue #2](https://github.com/justinlawyer/rag-app/issues/2)
   - [ ] Implement basic file reading
   - [ ] Simple text extraction
   - [ ] Fixed-size text chunking
   - [ ] Basic validation (file exists, not empty)
   - [ ] Unit tests for happy path

2. **PDF Support** (2 days) - [Issue #3](https://github.com/justinlawyer/rag-app/issues/3)
   - [ ] Add PDF text extraction using PyMuPDF
   - [ ] Handle basic PDF errors (file not found, corrupt file)

#### Nice-to-Have Features

1. **Web Application** - [Issue #4](https://github.com/justinlawyer/rag-app/issues/4), [Issue #5](https://github.com/justinlawyer/rag-app/issues/5)
   - Basic FastAPI setup
   - File upload endpoint
   - Basic error handling

2. **Semantic Search** - [Issue #6](https://github.com/justinlawyer/rag-app/issues/6), [Issue #7](https://github.com/justinlawyer/rag-app/issues/7)
   - Vector store integration
   - Text embedding generation
   - Basic similarity search

3. **Document Viewer** - [Issue #8](https://github.com/justinlawyer/rag-app/issues/8)
   - Web-based document viewer
   - Chunk navigation
   - Search functionality

#### Out of Scope Features

1. **Performance Optimization** - [Issue #9](https://github.com/justinlawyer/rag-app/issues/9)
   - Advanced monitoring
   - Performance optimization
   - Load testing
   - Distributed processing

2. **Security Implementation** - [Issue #10](https://github.com/justinlawyer/rag-app/issues/10)
   - Authentication
   - Authorization
   - Advanced security features
   - Penetration testing
   - Add document references
   - Include relevant snippets
   - Add confidence scores

2. **Interactive Mode**
   - Chat history management
   - Context preservation
   - Enhanced CLI interface

3. **Enhanced Search**
   - Multi-document queries
   - Ranking improvements
   - Search filters

4. **Web Interface**
   - Basic Streamlit app
   - Document upload UI
   - Chat interface

> Note: Each task should be:
> - Small enough to complete in 1-2 days
> - Have clear acceptance criteria
> - Be independently testable
> - Build towards the next phase

## 3. Development Guidelines

### Code Standards
- All classes and methods must have docstrings following Google Python Style Guide
- Type hints required for all function parameters and return values
- Maximum line length of 88 characters (Black formatter)
- Use f-strings for string formatting

## 4. Development Plan

### Timeline Overview (Junior Data Scientist)

#### Week 1-2: Setup & Core Components
- Environment setup and dependencies (2-3 days)
  * Python environment
  * Elasticsearch installation and configuration
  * Package dependencies
  * Learning curve for new tools

- Document processing (3-4 days)
  * Learning document processing libraries
  * Implementing PDF/TXT/MD readers
  * Text chunking and validation
  * Basic error handling

- Vector store setup (3-4 days)
  * Understanding vector embeddings
  * Elasticsearch vector search setup
  * Basic CRUD operations
  * Testing with sample documents

#### Week 3-4: RAG Implementation
- Core RAG chain (5-6 days)
  * Learning LangChain basics
  * Implementing retrieval logic
  * Setting up answer generation
  * Tuning prompt templates
  * Basic error handling

- CLI Development (3-4 days)
  * Learning Click/Typer
  * Basic command implementation
  * Input validation
  * Error messages

#### Week 5: Testing & Documentation
- Unit tests (2-3 days)
  * Test setup
  * Core functionality tests
  * Basic error cases

- Integration tests (2-3 days)
  * End-to-end flow tests
  * CLI command tests

- Documentation (2-3 days)
  * Code documentation
  * Usage examples
  * Setup guide

### Time Allocation Notes

1. **Learning Curve (40% of time)**
   - New technologies (Elasticsearch, LangChain)
   - Vector embeddings concepts
   - Testing frameworks
   - Best practices

2. **Development (40% of time)**
   - Core implementation
   - Debugging
   - Basic error handling

3. **Testing & Documentation (20% of time)**
   - Essential test coverage
   - Basic documentation

> Note: Timeline assumes:
> - Full-time dedication
> - Regular access to senior guidance
> - No major infrastructure issues
> - Focus on MVP features only
   - LLM integration
   - Context retrieval
   - Answer generation

5. **Command Line Interface (1 day)**
   - Basic commands
   - Error handling
   - Documentation

6. **Web Interface (1 day)**
   - Streamlit app
   - Basic UI
   - Error handling

#### Phase 3: Finalization (1 day)
7. **Documentation & Testing**
   - Code comments
   - Usage examples
   - Final tests

## 4. Development Guide

### Project Structure
```
rag-app/
├── src/
│   ├── __init__.py
│   ├── document_processing.py
│   ├── vector_store.py
│   └── rag.py
├── tests/
│   ├── __init__.py
│   ├── test_document_processing.py
│   ├── test_vector_store.py
│   └── test_rag.py
├── docs/
│   ├── specs/
│   │   ├── 01-product-requirements.md
│   │   ├── 02-technical-design-phase1.md
│   │   ├── 02-technical-design-phase2.md
│   │   ├── 04-code-examples.md
│   │   ├── 06-notebook-guide.md
│   │   └── 07-notebook-conversion.md
│   └── notebooks/
│       ├── README.md
│       ├── 01_document_processing.md
│       ├── 02_vector_store.md
│       ├── 03_rag_chain.md
│       └── 04_semantic_search.md
├── tools/
│   └── convert_notebooks.py
├── web/
│   └── app.py            # Streamlit web interface
├── config/
│   └── elasticsearch.yml  # ES configuration
└── requirements.txt
```

### Development Environment

#### 1. Prerequisites
- Python 3.11+
- 4GB+ RAM for development
- Elasticsearch (see deployment options below)
- Jupyter (for running notebooks)

#### 2. Development Setup

1. **Virtual Environment**
   ```bash
   python -m venv venv
   source venv/bin/activate
   pip install -r requirements.txt
   ```

2. **Jupyter Setup**
   ```bash
   pip install jupytext jupyter
   python tools/convert_notebooks.py docs/notebooks/*.md --to notebook
   jupyter notebook
   ```

3. **Elasticsearch Setup**
   Choose between:

   a. **Self-hosted (Docker)**
      ```bash
      docker-compose up -d elasticsearch
      ```
      - Free, open-source version
      - Good for development
      - Resources:
        * [Official Docker Image](https://www.elastic.co/guide/en/elasticsearch/reference/current/docker.html)
        * [Security Setup](https://www.elastic.co/guide/en/elasticsearch/reference/current/security-minimal-setup.html)

   b. **Managed Service**
      - [Elastic Cloud](https://www.elastic.co/cloud/)
      - [AWS Elasticsearch](https://aws.amazon.com/opensearch-service/)
      - Better for production
      - Automatic updates and maintenance

     * [Official Docker Image](https://www.elastic.co/guide/en/elasticsearch/reference/current/docker.html)
     * [Docker Compose Setup](https://www.elastic.co/guide/en/elasticsearch/reference/current/docker.html#docker-compose-file)
     * [Security Configuration](https://www.elastic.co/guide/en/elasticsearch/reference/current/security-minimal-setup.html)

2. **Managed Service**
   - Recommended for production deployments
   - Options:
     * [Elastic Cloud](https://www.elastic.co/cloud/) (Official service)
     * [AWS Elasticsearch Service](https://aws.amazon.com/opensearch-service/)
     * [Bonsai](https://bonsai.io/) (Smaller deployments)
   - Resources:
     * [Elastic Cloud Setup Guide](https://www.elastic.co/guide/en/cloud/current/ec-getting-started.html)
     * [AWS OpenSearch Migration](https://docs.aws.amazon.com/opensearch-service/latest/developerguide/migration.html)
     * [Connection Security](https://www.elastic.co/guide/en/cloud/current/ec-security.html)

**Configuration**: Update `config/elasticsearch.yml` based on your chosen deployment:
```yaml
# Self-hosted
hosts: ['localhost:9200']
ssl: false
auth: false

# Managed service
hosts: ['your-cluster-url']
ssl: true
auth:
  username: 'elastic'
  password: '${ES_PASSWORD}'  # Use environment variable
```

#### 3. Setup Steps
```bash
# 1. Clone repository
git clone <repository-url>
cd rag-app

# 2. Create virtual environment
python -m venv venv
source venv/bin/activate

# 3. Install dependencies
pip install -r requirements.txt

# 4. Install development tools
pip install pytest pytest-cov black flake8 pre-commit

# 5. Set up pre-commit hooks
pre-commit install

# 6. Start Elasticsearch
docker run -d -p 9200:9200 -e "discovery.type=single-node" elasticsearch:8.9.0

# 7. Set environment variables
export OPENAI_API_KEY=your_key_here

# 8. Run tests
python -m pytest
```

### Configuration

#### 1. Environment Variables
```bash
OPENAI_API_KEY=your_key_here
ELASTICSEARCH_URL=http://localhost:9200
CHUNK_SIZE=1000
CHUNK_OVERLAP=200
```

#### 2. Elasticsearch Settings
```yaml
# config/elasticsearch.yml
cluster.name: rag-app-app-dev
node.name: rag-app-app-node-1
network.host: 0.0.0.0
discovery.type: single-node
xpack.security.enabled: false  # Development only

# Index settings
index.number_of_shards: 1    # Development setting
index.number_of_replicas: 0  # Development setting
```

## 5. Implementation Details

### Document Processing
```python
from pathlib import Path
from typing import Dict, Any

class DocumentReader:
    def read_file(self, file_path: Path) -> Dict[str, Any]:
        """Read and extract text from a document."""
        # Implementation details in src/document_processing/reader.py
```

### Vector Store
```python
class ElasticsearchStore:
    def add_documents(self, documents: List[Dict]):
        """Add document chunks to Elasticsearch."""
        # Implementation details in src/vector_store/elasticsearch_store.py
```

### RAG Chain
```python
class RAGChain:
    def generate_answer(self, question: str) -> Dict:
        """Generate answer using RAG approach."""
        # Implementation details in src/rag_core/chain.py
```

## 6. Testing Strategy

### Unit Tests
- Document processing tests
- Vector store operations
- RAG chain functionality

### Integration Tests
- End-to-end document processing
- Search functionality
- Answer generation

### Performance Tests
- Document processing time
- Search latency
- Memory usage

## 7. Appendices

### A. API Reference
- Document Processing API
- Vector Store API
- RAG Chain API

### B. Configuration Guide
- Environment variables
- Elasticsearch settings
- Model parameters

### C. Troubleshooting Guide
- Common issues
- Debug procedures
- Error messages

## Additional Documentation

### Code Examples
For detailed implementation examples, see [`code-examples.md`](code-examples.md):
- Document Processing implementation
- Vector Store implementation
- RAG Chain implementation
- CLI and Web UI examples
- Docker configuration

### API Reference
For API details and usage, see [`api-reference.md`](api-reference.md):
- Document Processing API
- Vector Store API
- RAG Chain API
- CLI and Web interfaces

### Setup Guide
For installation and configuration, see [`setup-guide.md`](setup-guide.md):
- Local development setup
- Docker deployment
- Configuration options
- Development tools
- Troubleshooting

## System Architecture

### High-Level Overview

```mermaid
graph TB
    subgraph Client Layer
        CLI[CLI Interface]
        Web[Web UI]
        API[REST API]
    end

    subgraph Core Services
        DP[Document Processor]
        VS[Vector Store]
        RAG[RAG Engine]
        LLM[LLM Service]
    end

    subgraph Storage
        ES[(Elasticsearch)]
        FS[(File Storage)]
    end

    CLI --> DP
    Web --> DP
    API --> DP
    
    DP --> VS
    VS --> ES
    DP --> FS
    
    VS --> RAG
    RAG --> LLM
```

### Data Processing Flow

```mermaid
graph LR
    subgraph Input
        F[File Upload]
        T[Raw Text]
    end

    subgraph Processing
        R[Reader]
        C[Chunker]
        E[Embedder]
    end

    subgraph Storage
        V[(Vector Store)]
        M[(Metadata Store)]
    end

    F --> R
    T --> R
    R --> C
    C --> E
    E --> V
    R --> M
```

### Query Flow

```mermaid
sequenceDiagram
    participant U as User
    participant R as RAG Chain
    participant V as Vector Store
    participant L as LLM

    U->>R: Ask Question
    R->>V: Search Similar Docs
    V-->>R: Return Matches
    R->>L: Generate Answer
    L-->>R: Return Response
    R->>U: Formatted Answer
```

### Semantic Search Flow

```mermaid
graph TB
    subgraph Document Indexing
        D[Document] --> T[Text Extraction]
        T --> C[Chunking]
        C --> E1[Embedding Generation]
        E1 --> I[Index Storage]
    end

    subgraph Search Process
        Q[Query] --> E2[Query Embedding]
        E2 --> S[Similarity Search]
        I --> S
        S --> R1[Top K Results]
        R1 --> R2[Ranked Results]
    end

    subgraph Optimization
        direction LR
        R2 --> F[Filtering]
        F --> Re[Reranking]
        Re --> P[Post-processing]
    end

    style Document Indexing fill:#f5f5f5,stroke:#333,stroke-width:2px
    style Search Process fill:#f0f8ff,stroke:#333,stroke-width:2px
    style Optimization fill:#fff0f5,stroke:#333,stroke-width:2px
```

### Development Tips
1. **Common Issues**
   ```python
   # Issue: Chunks too large
   # Fix: Adjust chunk size
   chunker = TextChunker(
       chunk_size=500,  # Decrease from default 1000
       overlap=50      # Adjust overlap accordingly
   )

   # Issue: Slow search
   # Fix: Add index
   vector_store = ElasticsearchStore(
       index_settings={
           "index.mapping.nested_fields.limit": 100,
           "index.number_of_shards": 1
       }
   )

   # Issue: Out of memory
   # Fix: Batch processing
   for chunk_batch in chunker.create_chunks_batched(text, batch_size=10):
       vector_store.add_documents(chunk_batch)
   ```

2. **Debugging Tools**
   ```python
   # 1. Enable debug logging
   import logging
   logging.basicConfig(level=logging.DEBUG)

   # 2. Print embeddings shape
   from sentence_transformers import SentenceTransformer
   model = SentenceTransformer('all-MiniLM-L6-v2')
   embeddings = model.encode("test")
   print(f"Shape: {embeddings.shape}")

   # 3. Check Elasticsearch
   curl -X GET "localhost:9200/_cat/indices?v"
   ```

3. **Testing Strategy**
   ```python
   # 1. Start with small files
   def test_basic_functionality():
       text = "This is a test document."
       chunks = chunker.create_chunks(text)
       assert len(chunks) > 0

   # 2. Test edge cases
   def test_edge_cases():
       # Empty text
       assert len(chunker.create_chunks("")) == 0
       # Single character
       assert len(chunker.create_chunks("a")) == 1

   # 3. Test integration
   def test_full_pipeline():
       doc = reader.read_file("test.txt")
       chunks = chunker.create_chunks(doc["text"])
       vector_store.add_documents(chunks)
       results = vector_store.similarity_search("test")
       assert len(results) > 0
   ```

# RAG System Technical Design Document (TDD)

## Getting Started Guide

### 1. Development Environment Setup
```bash
# 1. Create a virtual environment
python -m venv venv

# 2. Activate the environment
# On macOS/Linux:
source venv/bin/activate
# On Windows:
# .\venv\Scripts\activate

# 3. Install basic dependencies
pip install -r requirements.txt

# 4. Install development dependencies
pip install pytest pytest-cov black flake8

# 5. Set up pre-commit hooks
pre-commit install
```

### 2. Configuration Files
└── config/
    └── elasticsearch.yml
```

### 3. Configuration Files

#### requirements.txt
```
# Core dependencies
pypdf2>=3.0.0
langchain>=0.0.300
sentence-transformers>=2.2.0
elasticsearch>=8.9.0
streamlit>=1.24.0

# Development
pytest>=7.4.0
pytest-cov>=4.1.0
black>=23.7.0
flake8>=6.0.0
pre-commit>=3.3.3
jupyter>=1.0.0
```

#### .pre-commit-config.yaml
```yaml
repos:
-   repo: https://github.com/psf/black
    rev: 23.7.0
    hooks:
    -   id: black
        language_version: python3.11
-   repo: https://github.com/pycqa/flake8
    rev: 6.0.0
    hooks:
    -   id: flake8
```

### 4. Development Workflow

1. **Start with Document Processing**
   ```bash
   # Create feature branch
   git checkout -b feature/document-processing
   
   # Implement reader and chunker
   touch src/document_processing/{reader,chunker}.py
   
   # Run tests
   pytest tests/test_document_processing.py -v
   
   # Format code
   black src/document_processing/
   ```

2. **Set up Elasticsearch**
   ```bash
   # Start Elasticsearch (Docker)
   docker run -d -p 9200:9200 -e "discovery.type=single-node" elasticsearch:8.9.0
   
   # Test connection
   curl http://localhost:9200
   ```

3. **Implement Vector Store**
   ```bash
   # Create feature branch
   git checkout -b feature/vector-store
   
   # Implement vector store
   touch src/vector_store/elasticsearch_store.py
   
   # Run tests
   pytest tests/test_vector_store.py -v
   ```

### 5. Common Tasks

```bash
# Run all tests
pytest

# Run tests with coverage
pytest --cov=src

# Format all code
black src/ tests/

# Lint code
flake8 src/ tests/

# Create documentation
pdoc src/ -o docs/
```

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

## Implementation Guide and Examples

> This guide provides production-ready implementations that correspond to the interactive examples in the notebooks.
> Each section includes references to relevant notebook examples and production code.

### 1. Document Processing

#### 1.1 Document Reader Implementation

**Production Code** (`src/document_processing/reader.py`):
```python
from pathlib import Path
from typing import Dict, Any
import fitz  # PyMuPDF

class DocumentReader:
    """Document reader with production-ready error handling and metadata extraction.
    See notebooks/01_document_processing.ipynb for interactive examples."""
    
    def read_file(self, file_path: Path) -> Dict[str, Any]:
        """Read and extract text from a document.
        
        Implements features demonstrated in notebook section 1.1 (Basic File Reading)
        and 1.2 (Error Handling).
        """
        if not file_path.exists():
            raise FileNotFoundError(f"File not found: {file_path}")
            
        text = ""
        metadata = {}
        
        if file_path.suffix.lower() == '.pdf':
            with fitz.open(file_path) as doc:
                # Extract context-enhancing metadata
                metadata = {
                    "hierarchy": self._extract_document_hierarchy(doc),
                    "text_types": self._extract_text_types(doc),
                    "page_boundaries": self._get_page_boundaries(doc)
                }
                text = "\n".join(page.get_text() for page in doc)
        elif file_path.suffix.lower() == '.txt':
            with open(file_path, 'r') as f:
                text = f.read()
                metadata = {
                    "hierarchy": {"type": "plain_text"},
                    "text_types": {"body": True},
                    "page_boundaries": None
                }
                
        return {
            "text": text,
            "metadata": metadata,
            "source": str(file_path)
        }
        
    def _extract_document_hierarchy(self, doc) -> Dict:
        """Extract document structure (headings, sections).
        See notebooks/01_document_processing.ipynb section 1.3."""
        # Implementation here
        pass
        
    def _extract_text_types(self, doc) -> Dict:
        """Identify text types (title, body, list, table).
        See notebooks/01_document_processing.ipynb section 1.4."""
        # Implementation here
        pass
        
    def _get_page_boundaries(self, doc) -> List[int]:
        """Get character positions of page boundaries.
        See notebooks/01_document_processing.ipynb section 1.5."""
        # Implementation here
        pass
```

**Example Usage** (from `notebooks/01_document_processing.ipynb`):
```python
# Initialize reader
reader = DocumentReader()

# Read different file types
pdf_doc = reader.read_file('sample.pdf')
txt_doc = reader.read_file('sample.txt')

# Access extracted metadata
print(f"Document hierarchy: {pdf_doc['metadata']['hierarchy']}")
print(f"Text types: {pdf_doc['metadata']['text_types']}")
```

#### 1.2 Text Chunker Implementation

**Production Code** (`src/document_processing/chunker.py`):
```python
from typing import List, Dict, Any

class TextChunker:
    """Text chunker with metadata-aware splitting strategies.
    See notebooks/01_document_processing.ipynb for interactive examples."""
    
    def __init__(self, chunk_size: int = 1000, chunk_overlap: int = 200):
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        
    def create_chunks(self, text: str, metadata: Dict = None) -> List[Dict[str, Any]]:
        """Split text into chunks using metadata for better boundaries.
        
        Implements features demonstrated in notebook sections 2.1 (Basic Chunking)
        and 2.2 (Chunking Strategies).
        """
        chunks = []
        start = 0
        
        # Use metadata for smart chunking if available
        if metadata and metadata.get('page_boundaries'):
            chunks.extend(self._chunk_by_pages(text, metadata['page_boundaries']))
        else:
            while start < len(text):
                end = start + self.chunk_size
                chunk_text = text[start:end]
                
                # Try to end at natural boundaries
                if metadata and metadata.get('text_types'):
                    end = self._find_natural_boundary(chunk_text, start, metadata)
                else:
                    # Fallback to sentence boundary
                    last_period = chunk_text.rfind('.')
                    if last_period != -1:
                        end = start + last_period + 1
                
                chunks.append({
                    "text": text[start:end],
                    "start": start,
                    "end": end,
                    "metadata": self._get_chunk_metadata(start, end, metadata)
                })
                
                start = end - self.chunk_overlap
                
        return chunks
        
    def _chunk_by_pages(self, text: str, page_boundaries: List[int]) -> List[Dict]:
        """Create chunks that respect page boundaries.
        See notebooks/01_document_processing.ipynb section 2.3."""
        # Implementation here
        pass
        
    def _find_natural_boundary(self, text: str, start: int, metadata: Dict) -> int:
        """Find natural chunk boundary using text type information.
        See notebooks/01_document_processing.ipynb section 2.4."""
        # Implementation here
        pass
        
    def _get_chunk_metadata(self, start: int, end: int, metadata: Dict) -> Dict:
        """Extract relevant metadata for the chunk.
        See notebooks/01_document_processing.ipynb section 2.5."""
        # Implementation here
        pass
```

**Example Usage** (from `notebooks/01_document_processing.ipynb`):
```python
# Initialize chunker
chunker = TextChunker(chunk_size=1000, overlap=200)

# Create chunks with metadata
chunks = chunker.create_chunks(pdf_doc['text'], pdf_doc['metadata'])

# Examine chunk metadata
for chunk in chunks[:2]:
    print(f"Chunk metadata: {chunk['metadata']}")
```

### Implementation-Notebook Mapping

| Production Code | Notebook Section | Description |
|-----------------|------------------|-------------|
| `DocumentReader.read_file()` | 1.1, 1.2 | Basic file reading and error handling |
| `DocumentReader._extract_document_hierarchy()` | 1.3 | Document structure extraction |
| `DocumentReader._extract_text_types()` | 1.4 | Text type identification |
| `DocumentReader._get_page_boundaries()` | 1.5 | Page boundary detection |
| `TextChunker.create_chunks()` | 2.1, 2.2 | Basic chunking and strategies |
| `TextChunker._chunk_by_pages()` | 2.3 | Page-aware chunking |
| `TextChunker._find_natural_boundary()` | 2.4 | Smart boundary detection |
| `TextChunker._get_chunk_metadata()` | 2.5 | Chunk metadata extraction |
import PyPDF2

class DocumentReader:
    def __init__(self):
        # List of supported file extensions
        self.supported_formats = {
            '.pdf': self._read_pdf,
            '.txt': self._read_txt,
            '.md': self._read_md
        }
    
    def read_file(self, file_path: str | Path) -> Dict[str, Any]:
        """Read a file and extract its text content and metadata."""
        path = Path(file_path)
        
        # Check if file exists
        if not path.exists():
            raise FileNotFoundError(f"File not found: {path}")
        
        # Check if format is supported
        if path.suffix.lower() not in self.supported_formats:
            raise ValueError(
                f"Unsupported format: {path.suffix}. "
                f"Supported formats: {list(self.supported_formats.keys())}"
            )
        
        # Read file using appropriate method
        return self.supported_formats[path.suffix.lower()](path)
    
    def _read_pdf(self, path: Path) -> Dict[str, Any]:
        """Read a PDF file using PyPDF2."""
        try:
            with path.open('rb') as file:
                reader = PyPDF2.PdfReader(file)
                text = '\n'.join(
                    page.extract_text() 
                    for page in reader.pages
                )
                metadata = {
                    'source': str(path),
                    'type': 'pdf',
                    'pages': len(reader.pages),
                    'title': path.stem,
                    **(reader.metadata if reader.metadata else {})
                }
                return {'text': text, 'metadata': metadata}
        except PyPDF2.PdfReadError as e:
            raise ValueError(f"Error reading PDF: {e}")
    
    def _read_txt(self, path: Path) -> Dict[str, Any]:
        """Read and process a text file."""
        try:
            with open(path, 'r', encoding='utf-8') as file:
                content = file.read()
            return {
                'content': content,
                'metadata': {'source': str(path)}
            }
        except Exception as e:
            raise FileProcessingError(f"Error reading text file: {e}")

    def _read_md(self, path: Path) -> Dict[str, Any]:
        """Read and process a markdown file."""
        try:
            with open(path, 'r', encoding='utf-8') as file:
                content = file.read()
            return {
                'content': content,
                'metadata': {'source': str(path)}
            }
        except Exception as e:
            raise FileProcessingError(f"Error reading markdown file: {e}")
        """Read a text file with encoding handling."""
        try:
            text = path.read_text(encoding='utf-8')
        except UnicodeDecodeError:
            text = path.read_text()
        
        metadata = {
            'source': str(path),
            'type': 'txt',
            'size': path.stat().st_size,
            'title': path.stem
        }
        return {'text': text, 'metadata': metadata}
```

### Step 2: Text Chunker Implementation
```python
# src/document_processing/chunker.py

from typing import List, Dict, Any
import re
from dataclasses import dataclass

@dataclass
class ChunkConfig:
    chunk_size: int = 1000  # Target size of each chunk
    overlap: int = 200      # Overlap between chunks
    min_size: int = 50      # Minimum chunk size

class TextChunker:
    def __init__(self, chunk_size: int = 1000, overlap: int = 200):
        self.config = ChunkConfig(
            chunk_size=chunk_size,
            overlap=overlap
        )
    
    def create_chunks(
        self, text: str, metadata: Dict[str, Any] = None
    ) -> List[Dict[str, Any]]:
        """Split text into overlapping chunks."""
        if not text.strip():
            return []
            
        # Split into sentences first
        sentences = self._split_into_sentences(text)
        
        chunks = []
        current_chunk = []
        current_size = 0
        
        for i, sentence in enumerate(sentences):
            sentence_size = len(sentence)
            
            if current_size + sentence_size <= self.config.chunk_size:
                current_chunk.append(sentence)
                current_size += sentence_size
            else:
                # Save current chunk if not empty
                if current_chunk:
                    chunks.append(self._create_chunk_dict(
                        ' '.join(current_chunk),
                        len(chunks),
                        metadata
                    ))
                
                # Start new chunk with overlap
                overlap_start = max(0, len(current_chunk) - 2)
                current_chunk = current_chunk[overlap_start:] + [sentence]
                current_size = sum(len(s) for s in current_chunk)
        
        # Add final chunk
        if current_chunk:
            chunks.append(self._create_chunk_dict(
                ' '.join(current_chunk),
                len(chunks),
                metadata
            ))
        
        return chunks
    
    def _split_into_sentences(self, text: str) -> List[str]:
        """Split text into sentences using regex."""
        # Handle common abbreviations
        abbrev = r'(?:[A-Za-z]\.){2,}|[A-Z][a-z]{1,2}\.'
        
        # Split on sentence boundaries
        pattern = f"(?<=[.!?])\s+(?<!{abbrev})(?=[A-Z0-9])"
        sentences = [s.strip() for s in re.split(pattern, text)]
        return [s for s in sentences if s]
    
    def _create_chunk_dict(
        self,
        text: str,
        chunk_id: int,
        metadata: Dict[str, Any] = None
    ) -> Dict[str, Any]:
        """Create a chunk dictionary with metadata."""
        return {
            'text': text,
            'metadata': {
                'chunk_id': chunk_id,
                'length': len(text),
                **(metadata or {})
            }
        }
```

### Step 3: Vector Store Implementation
```python
# src/vector_store/elasticsearch_store.py

from typing import List, Dict, Any
from elasticsearch import Elasticsearch
from sentence_transformers import SentenceTransformer
import numpy as np

class ElasticsearchStore:
    """Vector store using Elasticsearch.
    
    Key Features:
    - Stores text and embeddings
    - Performs similarity search
    - Handles metadata
    - Efficient batch operations
    
    Example:
        store = ElasticsearchStore()
        
        # Add documents
        docs = [{
            'text': 'Example document',
            'metadata': {'source': 'test.pdf'}
        }]
        store.add_documents(docs)
        
        # Search
        results = store.similarity_search(
            'example query',
            k=5  # Return top 5 results
        )
    """
    
    def __init__(
        self,
        es_host: str = 'http://localhost:9200',
        index_name: str = 'documents',
        model_name: str = 'all-MiniLM-L6-v2'
    ):
        # Initialize Elasticsearch client
        self.es = Elasticsearch(es_host)
        self.index_name = index_name
        
        # Load embedding model
        self.model = SentenceTransformer(model_name)
        
        # Create index if it doesn't exist
        if not self.es.indices.exists(index=index_name):
            self._create_index()
    
    def _create_index(self):
        """Create Elasticsearch index with vector search."""
        settings = {
            'mappings': {
                'properties': {
                    'text': {'type': 'text'},
                    'embedding': {
                        'type': 'dense_vector',
                        'dims': self.model.get_sentence_embedding_dimension(),
                        'index': True,
                        'similarity': 'cosine'
                    },
                    'metadata': {
                        'type': 'object',
                        'enabled': True
                    }
                }
            },
            'settings': {
                'index': {
                    'number_of_shards': 1,
                    'refresh_interval': '1s'
                }
            }
        }
        
        self.es.indices.create(
            index=self.index_name,
            body=settings
        )
    
    def add_documents(
        self,
        documents: List[Dict[str, Any]],
        batch_size: int = 100
    ) -> None:
        """Add documents to the vector store.
        
        Args:
            documents: List of dicts with 'text' and optional 'metadata'
            batch_size: Number of documents per batch
            
        Example document format:
        {
            'text': 'Document content here',
            'metadata': {
                'source': 'file.pdf',
                'page': 1,
                'chunk_id': 0
            }
        }
        """
        # Process in batches to avoid memory issues
        for i in range(0, len(documents), batch_size):
            batch = documents[i:i + batch_size]
            
            # Create embeddings for batch
            texts = [doc['text'] for doc in batch]
            embeddings = self.model.encode(texts)
            
            # Prepare bulk indexing operation
            operations = []
            for doc, embedding in zip(batch, embeddings):
                operations.extend([
                    {
                        'index': {
                            '_index': self.index_name
                        }
                    },
                    {
                        'text': doc['text'],
                        'embedding': embedding.tolist(),
                        'metadata': doc.get('metadata', {})
                    }
                ])
            
            # Execute bulk operation
            if operations:
                self.es.bulk(operations)
    
    def similarity_search(
        self,
        query: str,
        k: int = 5,
        min_score: float = 0.0
    ) -> List[Dict[str, Any]]:
        """Search for similar documents.
        
        Args:
            query: Search query
            k: Number of results to return
            min_score: Minimum similarity score (0-1)
            
        Returns:
            List of dicts with:
                - text: Document content
                - metadata: Document metadata
                - score: Similarity score
        """
        # Generate query embedding
        query_embedding = self.model.encode(query)
        
        # Prepare search query
        search_query = {
            'size': k,
            'query': {
                'script_score': {
                    'query': {'match_all': {}},
                    'script': {
                        'source': "cosineSimilarity(params.query_vector, 'embedding') + 1.0",
                        'params': {
                            'query_vector': query_embedding.tolist()
                        }
                    }
                }
            },
            '_source': ['text', 'metadata']
        }
        
        # Execute search
        response = self.es.search(
            index=self.index_name,
            body=search_query
        )
        
        # Process results
        results = []
        for hit in response['hits']['hits']:
            score = (hit['_score'] - 1.0) / 2.0  # Convert to 0-1 range
            if score >= min_score:
                results.append({
                    'text': hit['_source']['text'],
                    'metadata': hit['_source']['metadata'],
                    'score': score
                })
        
        return results
```

### Step 4: RAG Chain Implementation
```python
# src/rag_core/chain.py

from typing import Dict, Any, List
from langchain.chat_models import ChatOpenAI
from langchain.prompts import ChatPromptTemplate
from langchain.schema import AIMessage, HumanMessage
from src.vector_store.elasticsearch_store import ElasticsearchStore

class RAGChain:
    """RAG (Retrieval Augmented Generation) Chain.
    
    This class combines:
    1. Document retrieval from vector store
    2. Context formatting
    3. LLM response generation
    4. Source tracking
    
    Example:
        store = ElasticsearchStore()
        chain = RAGChain(store)
        
        response = chain.generate_answer(
            "What is machine learning?"
        )
        print(response['answer'])
        for source in response['sources']:
            print(f"Source: {source['text']}")
    """
    
    def __init__(
        self,
        vector_store: ElasticsearchStore,
        model_name: str = "gpt-3.5-turbo",
        temperature: float = 0.0,
        max_tokens: int = 500
    ):
        self.vector_store = vector_store
        
        # Initialize LLM
        self.llm = ChatOpenAI(
            model_name=model_name,
            temperature=temperature,
            max_tokens=max_tokens
        )
        
        # Create prompt template
        self.prompt = ChatPromptTemplate.from_messages([
            ("system", """You are a helpful AI assistant. 
            Answer questions based on the provided context.
            If you don't know or can't find the answer in the context, 
            say 'I don't have enough information to answer that question.'
            Always be accurate and truthful."""),
            ("human", """Question: {question}
            
Context:
{context}
            
Answer the question based on the context above.""")
        ])
    
    def generate_answer(
        self,
        question: str,
        num_sources: int = 3,
        min_score: float = 0.5
    ) -> Dict[str, Any]:
        """Generate an answer using RAG.
        
        Args:
            question: User's question
            num_sources: Number of sources to retrieve
            min_score: Minimum similarity score for sources
            
        Returns:
            Dict with:
                - answer: Generated answer
                - sources: List of source documents
        """
        # 1. Retrieve relevant documents
        sources = self.vector_store.similarity_search(
            query=question,
            k=num_sources,
            min_score=min_score
        )
        
        # If no relevant sources found
        if not sources:
            return {
                "answer": "I don't have enough information to answer that question.",
                "sources": []
            }
        
        # 2. Format context from sources
        context = "\n\n".join(
            f"Source {i+1}:\n{source['text']}"
            for i, source in enumerate(sources)
        )
        
        # 3. Generate answer using LLM
        messages = self.prompt.format_messages(
            question=question,
            context=context
        )
        
        response = self.llm.invoke(messages)
        
        return {
            "answer": response.content,
            "sources": sources
        }
    
    def chat(
        self,
        messages: List[Dict[str, str]],
        num_sources: int = 3
    ) -> Dict[str, Any]:
        """Chat with context from documents.
        
        Args:
            messages: List of message dicts with 'role' and 'content'
            num_sources: Number of sources to retrieve
            
        Returns:
            Dict with:
                - answer: Assistant's response
                - sources: List of source documents
        """
        # Convert messages to LangChain format
        chat_messages = []
        for msg in messages:
            if msg["role"] == "user":
                chat_messages.append(HumanMessage(
                    content=msg["content"]
                ))
            elif msg["role"] == "assistant":
                chat_messages.append(AIMessage(
                    content=msg["content"]
                ))
        
        # Get last user message
        last_question = messages[-1]["content"]
        
        # Generate response with sources
        return self.generate_answer(
            question=last_question,
            num_sources=num_sources
        )
```

### Example Usage

#### Document Processing
```python
# notebooks/01_document_processing.ipynb

# %% [markdown]
# # Document Processing Example
# This notebook demonstrates how to use the document processing components.

# %% [markdown]
# ## 1. Setup
# First, let's import our components and set up some test files.

# %%
from pathlib import Path
from src.document_processing.reader import DocumentReader
from src.document_processing.chunker import TextChunker

# Create test files
test_dir = Path("test_files")
test_dir.mkdir(exist_ok=True)

# Create a test PDF (using reportlab for demo)
from reportlab.pdfgen import canvas
def create_test_pdf(path: Path, text: str):
    c = canvas.Canvas(str(path))
    c.drawString(100, 750, text)
    c.save()

# Create test files
test_pdf = test_dir / "test.pdf"
create_test_pdf(test_pdf, "This is a test PDF document.")

test_txt = test_dir / "test.txt"
test_txt.write_text("This is a test text document.\nIt has multiple lines.")

# %% [markdown]
# ## 2. Reading Documents
# Now let's try reading different types of documents.

# %%
# Initialize reader
reader = DocumentReader()

# Read PDF
pdf_result = reader.read_file(test_pdf)
print("PDF Content:", pdf_result["text"])
print("PDF Metadata:", pdf_result["metadata"])

# Read TXT
txt_result = reader.read_file(test_txt)
print("\nTXT Content:", txt_result["text"])
print("TXT Metadata:", txt_result["metadata"])

# %% [markdown]
# ## 3. Chunking Text
# Let's see how the chunker works with different settings.

# %%
# Initialize chunker with different settings
default_chunker = TextChunker()
small_chunker = TextChunker(chunk_size=20, overlap=5)

# Test text
test_text = """This is a longer piece of text that will be split into chunks.
It contains multiple sentences. Each sentence should ideally stay together.
The chunker should try to break at natural boundaries."""

# Compare chunking results
default_chunks = default_chunker.create_chunks(test_text)
small_chunks = small_chunker.create_chunks(test_text)

print("Default Chunks:")
for chunk in default_chunks:
    print(f"\nChunk {chunk['metadata']['chunk_index']}:")
    print(f"Text: {chunk['text'][:50]}...")
    print(f"Length: {len(chunk['text'])}")

print("\nSmall Chunks:")
for chunk in small_chunks:
    print(f"\nChunk {chunk['metadata']['chunk_index']}:")
    print(f"Text: {chunk['text']}")
    print(f"Length: {len(chunk['text'])}")

# %% [markdown]
# ## 4. Error Handling
# Let's see how the system handles errors.

# %%
# Try reading non-existent file
try:
    reader.read_file(test_dir / "nonexistent.pdf")
except FileNotFoundError as e:
    print(f"Expected error: {e}")

# Try reading unsupported format
try:
    reader.read_file(test_dir / "test.docx")
except ValueError as e:
    print(f"Expected error: {e}")

# %% [markdown]
# ## 5. Cleanup

# %%
# Remove test files
import shutil
shutil.rmtree(test_dir)
```

## Deployment Guide

### 1. Local Development Setup
```bash
# 1. Clone repository
git clone <repository-url>
cd rag-system

# 2. Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: .\venv\Scripts\activate

# 3. Install dependencies
pip install -r requirements.txt

# 4. Set up pre-commit hooks
pre-commit install

# 5. Start Elasticsearch (using Docker)
docker run -d \
    -p 9200:9200 \
    -e "discovery.type=single-node" \
    -e "xpack.security.enabled=false" \
    elasticsearch:8.9.0

# 6. Run tests
pytest

# 7. Start Streamlit app
streamlit run src/app.py
```

### 2. Environment Variables
```bash
# .env
# OpenAI API Key
OPENAI_API_KEY=your-api-key

# Elasticsearch settings
ELASTICSEARCH_HOST=http://localhost:9200
ELASTICSEARCH_INDEX=rag-app-app-documents

# Streamlit settings
STREAMLIT_SERVER_PORT=8501
STREAMLIT_SERVER_ADDRESS=0.0.0.0
```

### 3. Docker Deployment
```dockerfile
# Dockerfile
FROM python:3.11-slim

# Set working directory
WORKDIR /app

# Copy requirements
COPY requirements.txt .

# Install dependencies
RUN pip install -r requirements.txt

# Copy application code
COPY src/ src/
COPY .streamlit/ .streamlit/

# Expose port
EXPOSE 8501

# Run the application
CMD ["streamlit", "run", "src/app.py"]
```

```yaml
# docker-compose.yml
version: '3'

services:
  elasticsearch:
    image: elasticsearch:8.9.0
    environment:
      - discovery.type=single-node
      - xpack.security.enabled=false
    ports:
      - "9200:9200"
    volumes:
      - es_data:/usr/share/elasticsearch/data
  
  rag-app:
    build: .
    environment:
      - OPENAI_API_KEY=${OPENAI_API_KEY}
      - ELASTICSEARCH_HOST=http://elasticsearch:9200
    ports:
      - "8501:8501"
    depends_on:
      - elasticsearch

volumes:
  es_data:
```

### 4. Deployment Steps

1. **Prepare Environment**
   ```bash
   # Create .env file
   cp .env.example .env
   # Edit .env with your settings
   nano .env
   ```

2. **Build and Deploy**
   ```bash
   # Build and start services
   docker-compose up -d --build
   
   # Check logs
   docker-compose logs -f
   ```

3. **Monitor Health**
   ```bash
   # Check Elasticsearch
   curl http://localhost:9200/_cluster/health
   
   # Check app
   curl http://localhost:8501
   ```

4. **Backup Data**
   ```bash
   # Create Elasticsearch snapshot
   curl -X PUT "localhost:9200/_snapshot/backup?pretty" -H 'Content-Type: application/json' -d'
   {
     "type": "fs",
     "settings": {
       "location": "backup_location"
     }
   }'
   ```

### 5. Troubleshooting Guide

1. **Elasticsearch Issues**
   ```bash
   # Check logs
   docker logs elasticsearch
   
   # Reset data
   docker-compose down -v
   docker-compose up -d
   ```

2. **Application Issues**
   ```bash
   # Check logs
   docker logs rag-app
   
   # Restart app
   docker-compose restart rag-app
   ```

3. **Common Problems**
   - Elasticsearch not starting: Check memory limits
   - App can't connect to Elasticsearch: Check network settings
   - Slow responses: Monitor resource usage

### 6. Maintenance

1. **Regular Tasks**
   ```bash
   # Update dependencies
   pip install -r requirements.txt --upgrade
   
   # Run tests
   pytest
   
   # Update containers
   docker-compose pull
   docker-compose up -d
   ```

2. **Monitoring**
   ```bash
   # Check resource usage
   docker stats
   
   # Monitor logs
   docker-compose logs -f --tail=100
   ```

### Integration Testing

#### 1. End-to-End Tests
```python
# tests/test_integration.py
import pytest
from pathlib import Path
from src.document_processing.reader import DocumentReader
from src.document_processing.chunker import TextChunker
from src.vector_store.elasticsearch_store import ElasticsearchStore
from src.rag.chain import RAGChain

@pytest.fixture(scope="module")
def test_files():
    # Create test directory
    test_dir = Path("test_files")
    test_dir.mkdir(exist_ok=True)
    
    # Create test files
    test_txt = test_dir / "test.txt"
    test_txt.write_text("""
    Artificial Intelligence (AI) is transforming industries.
    Machine learning models can perform complex tasks.
    Natural Language Processing (NLP) enables computers to understand text.
    """)
    
    yield test_dir
    
    # Cleanup
    import shutil
    shutil.rmtree(test_dir)

@pytest.fixture(scope="module")
def components(test_files):
    # Initialize components
    reader = DocumentReader()
    chunker = TextChunker()
    vector_store = ElasticsearchStore(index_name="test-integration")
    rag_chain = RAGChain(vector_store)
    
    yield reader, chunker, vector_store, rag_chain
    
    # Cleanup
    vector_store.es.indices.delete(index="test-integration", ignore=[404])

def test_full_pipeline(test_files, components):
    reader, chunker, vector_store, rag_chain = components
    
    # 1. Read document
    doc = reader.read_file(test_files / "test.txt")
    assert doc["text"] is not None
    assert "metadata" in doc
    
    # 2. Create chunks
    chunks = chunker.create_chunks(doc["text"])
    assert len(chunks) > 0
    
    # 3. Add to vector store
    docs_to_add = [
        {
            "text": chunk["text"],
            "metadata": {
                **doc["metadata"],
                **chunk["metadata"]
            }
        } for chunk in chunks
    ]
    vector_store.add_documents(docs_to_add)
    
    # 4. Test search
    results = vector_store.similarity_search("What is AI?")
    assert len(results) > 0
    assert any("AI" in result["text"] for result in results)
    
    # 5. Test RAG
    response = rag_chain.generate_answer("What is AI?")
    assert "answer" in response
    assert "sources" in response
    assert len(response["sources"]) > 0
    assert "artificial intelligence" in response["answer"].lower()

def test_error_handling(test_files, components):
    reader, chunker, vector_store, rag_chain = components
    
    # Test non-existent file
    with pytest.raises(FileNotFoundError):
        reader.read_file(test_files / "nonexistent.txt")
    
    # Test unsupported file type
    bad_file = test_files / "test.docx"
    bad_file.touch()
    with pytest.raises(ValueError):
        reader.read_file(bad_file)
    
    # Test empty query
    results = vector_store.similarity_search("")
    assert len(results) == 0
    
    # Test RAG with no relevant documents
    response = rag_chain.generate_answer("What is quantum computing?")
    assert "I don't have enough information" in response["answer"]
    assert len(response["sources"]) == 0
```

#### 2. Performance Tests
```python
# tests/test_performance.py
import pytest
import time
from pathlib import Path
from src.document_processing.reader import DocumentReader
from src.document_processing.chunker import TextChunker
from src.vector_store.elasticsearch_store import ElasticsearchStore
from src.rag.chain import RAGChain

@pytest.fixture
def large_test_file():
    # Create a large test file
    test_dir = Path("test_files")
    test_dir.mkdir(exist_ok=True)
    
    large_file = test_dir / "large.txt"
    with large_file.open("w") as f:
        # Write 1MB of text
        for i in range(1000):
            f.write(f"Test paragraph {i}. " * 20)
    
    yield large_file
    
    # Cleanup
    import shutil
    shutil.rmtree(test_dir)

def test_document_processing_performance(large_test_file):
    reader = DocumentReader()
    chunker = TextChunker()
    
    # Test reading
    start_time = time.time()
    doc = reader.read_file(large_test_file)
    read_time = time.time() - start_time
    assert read_time < 5  # Should process in under 5 seconds
    
    # Test chunking
    start_time = time.time()
    chunks = chunker.create_chunks(doc["text"])
    chunk_time = time.time() - start_time
    assert chunk_time < 2  # Should chunk in under 2 seconds

def test_vector_store_performance():
    store = ElasticsearchStore(index_name="test-performance")
    
    # Test bulk indexing
    docs = [
        {
            "text": f"Test document {i}",
            "metadata": {"id": i}
        } for i in range(1000)
    ]
    
    start_time = time.time()
    store.add_documents(docs)
    index_time = time.time() - start_time
    assert index_time < 10  # Should index in under 10 seconds
    
    # Test search latency
    start_time = time.time()
    results = store.similarity_search("test", k=5)
    search_time = time.time() - start_time
    assert search_time < 2  # Should search in under 2 seconds
    
    # Cleanup
    store.es.indices.delete(index="test-performance")

def test_rag_performance():
    store = ElasticsearchStore(index_name="test-rag-performance")
    chain = RAGChain(store)
    
    # Add test documents
    docs = [
        {
            "text": "AI is a technology that enables computers to think and learn.",
            "metadata": {"id": 1}
        },
        {
            "text": "Machine learning is a subset of AI focused on data-driven learning.",
            "metadata": {"id": 2}
        }
    ]
    store.add_documents(docs)
    
    # Test RAG latency
    start_time = time.time()
    response = chain.generate_answer("What is AI?")
    rag_time = time.time() - start_time
    assert rag_time < 5  # Should respond in under 5 seconds
    
    # Cleanup
    store.es.indices.delete(index="test-rag-performance")
```

### Week 7-8: UI Implementation

#### 1. Streamlit UI
```python
# src/app.py
import streamlit as st
from pathlib import Path
from src.document_processing.reader import DocumentReader
from src.document_processing.chunker import TextChunker
from src.vector_store.elasticsearch_store import ElasticsearchStore
from src.rag.chain import RAGChain

# Initialize components
@st.cache_resource
def init_components():
    """Initialize and cache components."""
    vector_store = ElasticsearchStore()
    reader = DocumentReader()
    chunker = TextChunker()
    rag_chain = RAGChain(vector_store)
    return reader, chunker, vector_store, rag_chain

reader, chunker, vector_store, rag_chain = init_components()

# Set page config
st.set_page_config(
    page_title="RAG System",
    page_icon="📚",
    layout="wide"
)

# Sidebar
with st.sidebar:
    st.title("📚 RAG System")
    
    # Upload documents
    st.header("Upload Documents")
    uploaded_files = st.file_uploader(
        "Choose files",
        accept_multiple_files=True,
        type=["pdf", "txt"]
    )
    
    if uploaded_files:
        with st.spinner("Processing documents..."):
            for file in uploaded_files:
                # Save file temporarily
                temp_path = Path(f"temp/{file.name}")
                temp_path.parent.mkdir(exist_ok=True)
                temp_path.write_bytes(file.getvalue())
                
                try:
                    # Read and chunk document
                    doc = reader.read_file(temp_path)
                    chunks = chunker.create_chunks(doc["text"])
                    
                    # Add chunks to vector store
                    vector_store.add_documents([
                        {
                            "text": chunk["text"],
                            "metadata": {
                                **doc["metadata"],
                                **chunk["metadata"]
                            }
                        } for chunk in chunks
                    ])
                    
                    st.success(f"Processed {file.name}")
                except Exception as e:
                    st.error(f"Error processing {file.name}: {str(e)}")
                finally:
                    # Cleanup
                    temp_path.unlink(missing_ok=True)

# Main area
tab1, tab2, tab3 = st.tabs(["💬 Chat", "🔍 Search", "ℹ️ About"])

# Chat tab
with tab1:
    st.header("Chat with Your Documents")
    
    # Initialize chat history
    if "messages" not in st.session_state:
        st.session_state.messages = []
    
    # Display chat history
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])
            
            # Show sources for assistant messages
            if message["role"] == "assistant" and "sources" in message:
                with st.expander("📚 View Sources", expanded=True):
                    for idx, source in enumerate(message["sources"], 1):
                        st.markdown(f"**Source {idx}:** {source['metadata']['source']}")
                        st.markdown(f"**Relevance:** {source['score']:.2f}")
                        with st.expander("View Content"):
                            st.markdown(f"> {source['text']}")
    
    # Chat input
    if prompt := st.chat_input("Ask a question about your documents..."):
        # Add user message to chat
        st.session_state.messages.append({"role": "user", "content": prompt})
        
        # Display user message
        with st.chat_message("user"):
            st.markdown(prompt)
        
        # Generate and display response
        with st.chat_message("assistant"):
            with st.spinner("Thinking..."):
                response = rag_chain.generate_answer(prompt)
                
                st.markdown(response["answer"])
                st.session_state.messages.append({
                    "role": "assistant",
                    "content": response["answer"],
                    "sources": response["sources"]
                })

# Search tab
with tab2:
    st.header("Search Documents")
    
    # Search input
    query = st.text_input("Search term")
    
    if query:
        with st.spinner("Searching..."):
            results = vector_store.similarity_search(query, k=5)
            
            # Display results
            for idx, result in enumerate(results, 1):
                st.markdown(f"### Result {idx}")
                st.markdown(f"**Source:** {result['metadata']['source']}")
                st.markdown(f"**Score:** {result['score']:.2f}")
                st.markdown(f"**Content:**")
                st.markdown(f"> {result['text']}")
                st.markdown("---")

# About tab
with tab3:
    st.header("About RAG System")
    
    # System Overview
    st.markdown("""
    ## Overview
    The RAG (Retrieval Augmented Generation) System is an intelligent document assistant that helps you:
    - Chat with your documents using natural language
    - Search across your document collection
    - Get accurate answers with source citations
    
    ## How It Works
    1. **Document Processing**
       - Upload your documents (PDF, TXT)
       - Documents are split into chunks
       - Text is converted into vector embeddings
    
    2. **Search & Retrieval**
       - Your questions are analyzed
       - Relevant document sections are found
       - AI generates accurate answers
    
    3. **Source Verification**
       - All answers include sources
       - View original context
       - Verify information accuracy
    
    ## Features
    - 📚 Support for PDF and TXT files
    - 💬 Natural language chat interface
    - 🔍 Semantic search capabilities
    - 📝 Source citations and verification
    - ⚡ Fast response times
    
    ## Usage Tips
    1. **For Best Results**
       - Ask specific questions
       - Include context in your queries
       - Check source citations
    
    2. **Document Guidelines**
       - Supported formats: PDF, TXT
       - Max file size: 100MB
       - Text should be machine-readable
    
    ## Version Information
    - Version: 1.0.0 (MVP)
    - Last Updated: {}
    - Framework: Streamlit
    - Backend: LangChain + Elasticsearch
    """.format(time.strftime("%Y-%m-%d")))
    
    # System Status
    st.subheader("System Status")
    col1, col2, col3 = st.columns(3)
    
    # Check Elasticsearch connection
    try:
        es_status = vector_store.es.ping()
        col1.metric("Elasticsearch", "✅ Connected" if es_status else "❌ Disconnected")
    except:
        col1.metric("Elasticsearch", "❌ Error")
    
    # Check document count
    try:
        doc_count = vector_store.es.count(index=vector_store.index_name)["count"]
        col2.metric("Documents Indexed", doc_count)
    except:
        col2.metric("Documents Indexed", "N/A")
    
    # Check API status
    try:
        openai_key = os.getenv("OPENAI_API_KEY")
        col3.metric("OpenAI API", "✅ Configured" if openai_key else "⚠️ Not Configured")
    except:
        col3.metric("OpenAI API", "❌ Error")
    
    # Help & Support
    st.subheader("Help & Support")
    st.markdown("""
    - 📧 **Support**: support@example.com
    - 📚 **Documentation**: [User Guide](https://docs.example.com)
    - 🐛 **Report Issues**: [Issue Tracker](https://github.com/example/rag-system/issues)
    """)

# Run the app
if __name__ == "__main__":
    import streamlit.cli as stcli
    import sys
    
    sys.argv = ["streamlit", "run", "src/app.py"]
    sys.exit(stcli.main())
```

#### 2. Configuration
```yaml
# .streamlit/config.toml
[theme]
primaryColor = "#FF4B4B"
backgroundColor = "#FFFFFF"
secondaryBackgroundColor = "#F0F2F6"
textColor = "#262730"

[server]
port = 8501
enableCORS = false

[browser]
serveLocalFiles = true
```

### Week 5-6: RAG Implementation

#### 1. RAG Chain Implementation
```python
# src/rag_core/chain.py
from typing import Dict, List
from langchain.chat_models import ChatOpenAI
from langchain.prompts import ChatPromptTemplate
from src.vector_store.elasticsearch_store import ElasticsearchStore

class RAGChain:
    """Simple RAG implementation for MVP.
    Focuses on basic Q&A with source tracking.
    """
    def __init__(self,
                 vector_store: ElasticsearchStore,
                 model_name: str = "gpt-3.5-turbo",
                 temperature: float = 0):
        self.vector_store = vector_store
        self.llm = ChatOpenAI(model_name=model_name, temperature=temperature)
        
        # Define prompt template
        self.prompt = ChatPromptTemplate.from_messages([
            ("system", """
            You are a helpful AI assistant. Answer questions based ONLY on the provided context.
            Format your response in this way:
            1. Give a clear, direct answer
            2. For each statement, cite the specific source
            3. If unsure, say so explicitly
            4. If no relevant information found, say 'I don't have enough information'
            
            Example:
            According to [doc1.pdf], "[exact quote]", which indicates...
            This is supported by [doc2.txt] which states "[exact quote]"...
            """),
            ("human", "Context:\n{context}\n\nQuestion: {question}")
        ])
    
    def _get_relevant_chunks(self, question: str, k: int = 4) -> List[Dict]:
        """Get relevant document chunks for the question."""
        return self.vector_store.similarity_search(question, k=k)
    
    def _format_context(self, chunks: List[Dict]) -> str:
        """Format chunks into context string."""
        return "\n\n".join([
            f"[{chunk['metadata'].get('source', 'unknown')}]:\n{chunk['text']}"
            for chunk in chunks
        ])
    
    def generate_answer(self, question: str) -> Dict:
        """Generate answer with sources."""
        # Get relevant chunks
        chunks = self._get_relevant_chunks(question)
        
        # If no chunks found
        if not chunks:
            return {
                "answer": "I don't have enough information to answer that question.",
                "sources": []
            }
        
        # Format context
        context = self._format_context(chunks)
        
        # Generate answer
        chain = self.prompt | self.llm
        response = chain.invoke({
            "context": context,
            "question": question
        })
        
        return {
            "answer": response.content,
            "sources": [{
                "text": chunk["text"],
                "metadata": chunk["metadata"],
                "score": chunk.get("score", 0)
            } for chunk in chunks]
        }
```

#### 2. RAG Chain Tests
```python
# tests/test_rag.py
import pytest
from unittest.mock import Mock, patch
from src.rag.chain import RAGChain

@pytest.fixture
def mock_vector_store():
    store = Mock()
    store.similarity_search.return_value = [
        {
            "text": "AI is transforming technology.",
            "metadata": {"source": "tech.txt"},
            "score": 0.9
        }
    ]
    return store

def test_rag_chain_initialization(mock_vector_store):
    chain = RAGChain(mock_vector_store)
    assert chain.vector_store == mock_vector_store

def test_get_relevant_chunks(mock_vector_store):
    chain = RAGChain(mock_vector_store)
    chunks = chain._get_relevant_chunks("What is AI?")
    
    assert len(chunks) == 1
    assert "AI" in chunks[0]["text"]
    mock_vector_store.similarity_search.assert_called_once()

def test_generate_answer(mock_vector_store):
    chain = RAGChain(mock_vector_store)
    result = chain.generate_answer("What is AI?")
    
    assert "answer" in result
    assert "sources" in result
    assert len(result["sources"]) == 1
    assert result["sources"][0]["metadata"]["source"] == "tech.txt"

def test_no_relevant_chunks(mock_vector_store):
    mock_vector_store.similarity_search.return_value = []
    chain = RAGChain(mock_vector_store)
    
    result = chain.generate_answer("What is AI?")
    assert "I don't have enough information" in result["answer"]
    assert result["sources"] == []
```

### Week 3-4: Vector Store Setup

#### 1. Elasticsearch Configuration
```yaml
# config/elasticsearch.yml
cluster.name: rag-app-app-cluster
node.name: rag-app-app-node

# Enable security features
xpack.security.enabled: false  # For development only

# Memory settings
indices.memory.index_buffer_size: 10%
indices.queries.cache.size: 5%

# Index settings
index.number_of_shards: 1    # Development setting
index.number_of_replicas: 0  # Development setting

# Search settings
index.max_result_window: 10000
```

#### 2. Vector Store Implementation
```python
# src/vector_store/elasticsearch_store.py
from typing import List, Dict, Optional
from elasticsearch import Elasticsearch
from sentence_transformers import SentenceTransformer
import numpy as np

class ElasticsearchStore:
    """Vector store using Elasticsearch for development.
    
    This is a simplified version for MVP that:
    1. Uses basic dense vector indexing
    2. Stores text and embeddings in the same index
    3. Uses cosine similarity for search
    """
    def __init__(self, 
                 es_host: str = "http://localhost:9200",
                 index_name: str = "rag-documents",
                 model_name: str = "all-MiniLM-L6-v2"):
        # Connect to Elasticsearch
        self.es = Elasticsearch(es_host)
        self.index_name = index_name
        
        # Load embedding model
        self.model = SentenceTransformer(model_name)
        
        # Create index if it doesn't exist
        self._create_index()
    
    def _create_index(self):
        """Create the index with vector search capabilities."""
        if not self.es.indices.exists(index=self.index_name):
            # Define index mapping
            mapping = {
                "mappings": {
                    "properties": {
                        "text": {"type": "text"},
                        "embedding": {
                            "type": "dense_vector",
                            "dims": 384,  # MiniLM dimension
                            "index": True,
                            "similarity": "cosine"
                        },
                        "metadata": {"type": "object"}
                    }
                }
            }
            
            # Create index
            self.es.indices.create(
                index=self.index_name,
                body=mapping
            )
    
    def add_documents(self, documents: List[Dict]):
        """Add documents to the vector store.
        
        Args:
            documents: List of dicts with 'text' and 'metadata' keys
        """
        # Prepare bulk indexing request
        bulk_data = []
        
        for doc in documents:
            # Generate embedding
            embedding = self.model.encode(doc["text"])
            
            # Add index operation
            bulk_data.append({
                "index": {"_index": self.index_name}
            })
            
            # Add document data
            bulk_data.append({
                "text": doc["text"],
                "embedding": embedding.tolist(),
                "metadata": doc["metadata"]
            })
        
        # Execute bulk indexing
        if bulk_data:
            self.es.bulk(operations=bulk_data, refresh=True)
    
    def similarity_search(self,
                         query: str,
                         k: int = 4,
                         filter: Optional[Dict] = None) -> List[Dict]:
        """Search for similar documents.
        
        Args:
            query: Search query
            k: Number of results to return
            filter: Optional metadata filter
        
        Returns:
            List of documents with scores
        """
        # Generate query embedding
        query_vector = self.model.encode(query)
        
        # Build search query
        search_query = {
            "query": {
                "function_score": {
                    "query": {
                        "bool": {
                            "must": [
                                {
                                    "script_score": {
                                        "query": {"match_all": {}},
                                        "script": {
                                            "source": "cosineSimilarity(params.query_vector, 'document_vector') + 1.0",
                                            "params": {"query_vector": query_vector}
                                        }
                                    }
                                }
                            ]
                        }
                    }
                }
            },
            "size": k,
            "from": 0,
            "_source": ["title", "content", "metadata", "summary"]
        }
        
        # Add filters if provided
        if filter:
            for field, value in filter.items():
                search_query["query"]["function_score"]["query"]["bool"]["must"].append(
                    {"term": {field: value}}
                )
        
        # Execute search
        results = self.es.search(
            index=self.index_name,
            body=search_query
        )
        
        # Format results
        hits = results["hits"]["hits"]
        return [
            {
                "id": hit["_id"],
                "title": hit["_source"]["title"],
                "summary": hit["_source"]["summary"],
                "metadata": hit["_source"]["metadata"],
                "score": hit["_score"],
                "highlights": self._get_highlights(hit)
            } for hit in hits
        ]
    
    def _get_highlights(self, hit: Dict) -> List[str]:
        # Extract relevant snippets from content
        content = hit["_source"]["content"]
        return self._extract_relevant_snippets(content)
```

#### 3. Vector Store Tests
```python
# tests/test_vector_store.py
import pytest
from src.vector_store.elasticsearch_store import ElasticsearchStore

@pytest.fixture
def es_store():
    # Create test store
    store = ElasticsearchStore(index_name="test-index")
    yield store
    # Cleanup
    store.es.indices.delete(index="test-index", ignore=[404])

def test_add_documents(es_store):
    # Test documents
    docs = [
        {
            "text": "This is a test document.",
            "metadata": {"source": "test1.txt"}
        },
        {
            "text": "This is another test document.",
            "metadata": {"source": "test2.txt"}
        }
    ]
    
    # Add documents
    es_store.add_documents(docs)
    
    # Verify documents were added
    result = es_store.es.count(index="test-index")
    assert result["count"] == 2

def test_similarity_search(es_store):
    # Add test documents
    docs = [
        {
            "text": "Artificial intelligence is transforming technology.",
            "metadata": {"source": "tech1.txt"}
        },
        {
            "text": "The weather is nice today.",
            "metadata": {"source": "weather.txt"}
        }
    ]
    es_store.add_documents(docs)
    
    # Test search
    results = es_store.similarity_search(
        "AI and technology",
        k=1
    )
    
    assert len(results) == 1
    assert "artificial intelligence" in results[0]["content"].lower()

def test_filtered_search(es_store):
    # Add test documents
    docs = [
        {
            "text": "Document 1",
            "metadata": {"type": "A"}
        },
        {
            "text": "Document 2",
            "metadata": {"type": "B"}
        }
    ]
    es_store.add_documents(docs)
    
    # Test filtered search
    results = es_store.similarity_search(
        "Document",
        filter={"metadata.type": "A"}
    )
    
    assert len(results) == 1
    assert results[0]["metadata"]["type"] == "A"
```

#### 4. Example Notebook: Vector Store
```python
# notebooks/02_vector_store.ipynb

# %% [markdown]
# # Vector Store Example
# This notebook demonstrates how to use the vector store component.

# %% [markdown]
# ## 1. Setup

# %%
from src.vector_store.elasticsearch_store import ElasticsearchStore
from src.document_processing.reader import DocumentReader
from src.document_processing.chunker import TextChunker

# Initialize components
reader = DocumentReader()
chunker = TextChunker(chunk_size=500, overlap=50)
store = ElasticsearchStore(index_name="demo-index")

# %% [markdown]
# ## 2. Prepare Sample Documents

# %%
# Create some sample documents
sample_docs = [
    """Artificial Intelligence (AI) is revolutionizing industries.
    Machine learning models can now perform complex tasks.
    Deep learning has enabled breakthroughs in computer vision.""",
    
    """Python is a popular programming language.
    It's widely used in data science and AI.
    The syntax is clean and readable.""",
    
    """Data science combines statistics and programming.
    It helps organizations make data-driven decisions.
    Python is commonly used in data analysis."""
]

# Process documents
processed_docs = []
for i, text in enumerate(sample_docs):
    chunks = chunker.create_chunks(text)
    for chunk in chunks:
        processed_docs.append({
            "text": chunk["text"],
            "metadata": {
                "doc_id": f"doc_{i}",
                "chunk_index": chunk["metadata"]["chunk_index"]
            }
        })

# Add to vector store
store.add_documents(processed_docs)

# %% [markdown]
# ## 3. Try Similarity Search

# %%
# Search for AI-related content
ai_results = store.similarity_search(
    "What can AI do?",
    k=2
)

print("AI-related results:")
for result in ai_results:
    print(f"\nScore: {result['score']:.3f}")
    print(f"Text: {result['text']}")
    print(f"Metadata: {result['metadata']}")

# Search for Python-related content
python_results = store.similarity_search(
    "Tell me about Python programming",
    k=2
)

print("\nPython-related results:")
for result in python_results:
    print(f"\nScore: {result['score']:.3f}")
    print(f"Text: {result['text']}")
    print(f"Metadata: {result['metadata']}")

# %% [markdown]
# ## 4. Try Filtered Search

# %%
# Search with document filter
filtered_results = store.similarity_search(
    "programming",
    k=1,
    filter={"metadata.doc_id": "doc_1"}
)

print("Filtered results (doc_1 only):")
for result in filtered_results:
    print(f"\nScore: {result['score']:.3f}")
    print(f"Text: {result['text']}")
    print(f"Metadata: {result['metadata']}")

# %% [markdown]
# ## 5. Cleanup

# %%
# Delete test index
store.es.indices.delete(index="demo-index")
```

#### 1. Basic Document Reader
```python
# document_reader.py
from typing import Dict, List
from pathlib import Path
import PyPDF2

class DocumentReader:
    """Simple document reader for MVP.
    Focus on PDF and TXT files initially.
    """
    def read_pdf(self, file_path: Path) -> str:
        """Read text from PDF file."""
        with open(file_path, 'rb') as file:
            reader = PyPDF2.PdfReader(file)
            text = ''
            for page in reader.pages:
                text += page.extract_text() + '\n'
            return text
    
    def read_txt(self, file_path: Path) -> str:
        """Read text from TXT file."""
        with open(file_path, 'r') as file:
            return file.read()
    
    def read_file(self, file_path: Path) -> Dict:
        """Read file and return text with metadata."""
        file_path = Path(file_path)
        
        if not file_path.exists():
            raise FileNotFoundError(f"File not found: {file_path}")
        
        # Start with just PDF and TXT
        if file_path.suffix.lower() == '.pdf':
            text = self.read_pdf(file_path)
        elif file_path.suffix.lower() == '.txt':
            text = self.read_txt(file_path)
        else:
            raise ValueError(f"Unsupported file type: {file_path.suffix}")
        
        return {
            "text": text,
            "metadata": {
                "source": str(file_path),
                "type": file_path.suffix.lower(),
                "created": file_path.stat().st_ctime
            }
        }

# Example usage:
if __name__ == "__main__":
    reader = DocumentReader()
    result = reader.read_file("sample.pdf")
    print(f"Read {len(result['text'])} characters")
```

#### 2. Basic Text Chunker
```python
# text_chunker.py
from typing import List, Dict

class TextChunker:
    """Simple text chunking for MVP.
    Start with basic character-based chunking.
    """
    def __init__(self, chunk_size: int = 1000, overlap: int = 100):
        self.chunk_size = chunk_size
        self.overlap = overlap
    
    def create_chunks(self, text: str) -> List[Dict]:
        """Split text into overlapping chunks."""
        chunks = []
        start = 0
        
        while start < len(text):
            # Find the end of the chunk
            end = start + self.chunk_size
            
            # If not at the end of text, try to break at a sentence
            if end < len(text):
                # Look for sentence endings (., !, ?) followed by space
                for i in range(end, max(start, end - 100), -1):
                    if text[i-1] in '.!?' and text[i:i+1].isspace():
                        end = i
                        break
            
            # Create chunk
            chunks.append({
                "text": text[start:end],
                "metadata": {
                    "chunk_index": len(chunks),
                    "start_char": start,
                    "end_char": end
                }
            })
            
            # Move start position, accounting for overlap
            start = end - self.overlap
        
        return chunks

# Example usage:
if __name__ == "__main__":
    chunker = TextChunker()
    text = "This is a sample text. It will be split into chunks. Each chunk will overlap slightly."
    chunks = chunker.create_chunks(text)
    print(f"Created {len(chunks)} chunks")
```

#### 3. Basic Unit Tests
```python
# test_document_processing.py
import pytest
from pathlib import Path
from document_reader import DocumentReader
from text_chunker import TextChunker

def test_document_reader():
    reader = DocumentReader()
    
    # Test PDF reading
    with pytest.raises(FileNotFoundError):
        reader.read_file("nonexistent.pdf")
    
    # Test unsupported format
    with pytest.raises(ValueError):
        reader.read_file("test.docx")
    
    # Create a test TXT file
    test_text = "This is a test document.\nIt has multiple lines."
    test_file = Path("test.txt")
    test_file.write_text(test_text)
    
    # Test TXT reading
    result = reader.read_file(test_file)
    assert result["text"] == test_text
    assert result["metadata"]["type"] == ".txt"
    
    # Cleanup
    test_file.unlink()

def test_text_chunker():
    chunker = TextChunker(chunk_size=10, overlap=2)
    text = "This is a test. Another test."
    
    chunks = chunker.create_chunks(text)
    
    # Test number of chunks
    assert len(chunks) > 0
    
    # Test chunk size
    for chunk in chunks:
        assert len(chunk["text"]) <= 10
    
    # Test overlap
    for i in range(len(chunks)-1):
        overlap_text = chunks[i]["text"][-2:]
        assert overlap_text in chunks[i+1]["text"]
```

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

### Search Implementation

#### 1. Document Search Service
```python
from typing import List, Dict
from elasticsearch import Elasticsearch
from sentence_transformers import SentenceTransformer

class DocumentSearchService:
    def __init__(self, model_name: str, es_host: str, index_name: str):
        self.model = SentenceTransformer(model_name)
        self.es = Elasticsearch(es_host)
        self.index_name = index_name
    
    def search_documents(self, query: str, filters: Dict = None, 
                        sort_by: str = 'relevance', page: int = 1, 
                        size: int = 10) -> Dict:
        # Generate query embedding
        query_vector = self.model.encode(query)
        
        # Build search query
        search_query = {
            "query": {
                "function_score": {
                    "query": {
                        "bool": {
                            "must": [
                                {
                                    "script_score": {
                                        "query": {"match_all": {}},
                                        "script": {
                                            "source": "cosineSimilarity(params.query_vector, 'document_vector') + 1.0",
                                            "params": {"query_vector": query_vector}
                                        }
                                    }
                                }
                            ]
                        }
                    }
                }
            },
            "size": size,
            "from": (page - 1) * size,
            "_source": ["title", "content", "metadata", "summary"]
        }
        
        # Add filters if provided
        if filters:
            for field, value in filters.items():
                search_query["query"]["function_score"]["query"]["bool"]["must"].append(
                    {"term": {field: value}}
                )
        
        # Add sorting
        if sort_by == 'date':
            search_query["sort"] = [{"metadata.created": {"order": "desc"}}]
        
        # Execute search
        response = self.es.search(
            index=self.index_name,
            body=search_query
        )
        
        # Format results
        hits = response["hits"]["hits"]
        return {
            "total": response["hits"]["total"]["value"],
            "documents": [
                {
                    "id": hit["_id"],
                    "title": hit["_source"]["title"],
                    "summary": hit["_source"]["summary"],
                    "metadata": hit["_source"]["metadata"],
                    "score": hit["_score"],
                    "highlights": self._get_highlights(hit)
                } for hit in hits
            ],
            "page": page,
            "total_pages": (response["hits"]["total"]["value"] + size - 1) // size
        }
    
    def _get_highlights(self, hit: Dict) -> List[str]:
        # Extract relevant snippets from content
        content = hit["_source"]["content"]
        return self._extract_relevant_snippets(content)
```

#### 2. Search API Endpoints
```python
from fastapi import APIRouter, Query, HTTPException
from typing import Optional, Dict

router = APIRouter()
search_service = DocumentSearchService(
    model_name="sentence-transformers/all-mpnet-base-v2",
    es_host="http://localhost:9200",
    index_name="documents"
)

@router.get("/search/documents")
async def search_documents(
    query: str,
    doc_type: Optional[str] = None,
    date_from: Optional[str] = None,
    date_to: Optional[str] = None,
    sort_by: str = Query("relevance", enum=["relevance", "date"]),
    page: int = Query(1, ge=1),
    size: int = Query(10, ge=1, le=100)
):
    try:
        # Build filters
        filters = {}
        if doc_type:
            filters["metadata.type"] = doc_type
        if date_from or date_to:
            filters["metadata.created"] = {
                "gte": date_from,
                "lte": date_to
            }
        
        # Execute search
        results = search_service.search_documents(
            query=query,
            filters=filters,
            sort_by=sort_by,
            page=page,
            size=size
        )
        
        return results
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
```

#### 3. Search UI Component
```python
# search_page.py
import streamlit as st
import requests
from datetime import datetime, timedelta

def render_search_page():
    st.title("📚 Document Search")
    
    # Create two tabs for different search modes
    tab1, tab2 = st.tabs(["💬 Q&A Chat", "🔍 Document Search"])
    
    with tab1:
        # Chat interface
        if "messages" not in st.session_state:
            st.session_state.messages = []

        # Display chat messages
        for message in st.session_state.messages:
            with st.chat_message(message["role"]):
                st.markdown(message["content"])
                
                # Show sources for assistant messages
                if message["role"] == "assistant" and "sources" in message:
                    with st.expander("📚 View Sources", expanded=True):
                        for idx, source in enumerate(message["sources"], 1):
                            st.markdown(f"**Source {idx}:** {source['source']}")
                            st.markdown(f"**Relevance Score:** {source['confidence_score']:.2f}")
                            with st.expander("View Snippet"):
                                st.markdown(f"> {source['snippet']}")
                            st.markdown(f"**Page:** {source['page_number']} | "
                                      f"**Type:** {source['document_type']} | "
                                      f"**Modified:** {source['last_modified']}")
                            
                            # Copy button for the snippet
                            if st.button(f"📋 Copy Snippet {idx}"):
                                pyperclip.copy(source['snippet'])
                                st.success("Copied to clipboard!")
        
        # Chat input
        if prompt := st.chat_input("Ask a question about your documents..."):
            st.session_state.messages.append({"role": "user", "content": prompt})
            
            with st.chat_message("assistant"):
                with st.spinner("Searching documents..."):
                    response = rag_chain.generate_answer(prompt)
                    
                    st.markdown(response["answer"])
                    
                    # Add response to chat history
                    st.session_state.messages.append({
                        "role": "assistant",
                        "content": response["answer"],
                        "sources": response["sources"]
                    })
    
    with tab2:
        # Document search interface
        query = st.text_input("Search documents...", key="doc_search")
    
    # Filters
    col1, col2, col3 = st.columns(3)
    with col1:
        doc_type = st.selectbox(
            "Document Type",
            [None, "PDF", "TXT", "MD"],
            format_func=lambda x: "All Types" if x is None else x
        )
    
    with col2:
        date_range = st.selectbox(
            "Date Range",
            ["All Time", "Last Week", "Last Month", "Last Year"],
        )
    
    with col3:
        sort_by = st.selectbox(
            "Sort By",
            ["relevance", "date"]
        )
    
    # Execute search
    if query:
        # Prepare date filters
        date_from = None
        if date_range != "All Time":
            today = datetime.now()
            if date_range == "Last Week":
                date_from = (today - timedelta(days=7)).isoformat()
            elif date_range == "Last Month":
                date_from = (today - timedelta(days=30)).isoformat()
            elif date_range == "Last Year":
                date_from = (today - timedelta(days=365)).isoformat()
        
        # Call search API
        response = requests.get(
            "http://localhost:8000/search/documents",
            params={
                "query": query,
                "doc_type": doc_type,
                "date_from": date_from,
                "sort_by": sort_by
            }
        )
        
        if response.status_code == 200:
            results = response.json()
            
            # Display results
            st.write(f"Found {results['total']} documents")
            
            for doc in results["documents"]:
                with st.expander(
                    f"{doc['title']} (Score: {doc['score']:.2f})"
                ):
                    st.write(f"**Summary:** {doc['summary']}")
                    
                    st.write("**Relevant Snippets:**")
                    for highlight in doc["highlights"]:
                        st.markdown(f"- {highlight}")
                    
                    st.write("**Metadata:**")
                    st.json(doc["metadata"])
        else:
            st.error("Failed to fetch search results")
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
    def __init__(self, model_name: str, es_host: str, index_name: str):
        self.llm = ChatOpenAI(model_name=model_name, temperature=0)
        self.es = Elasticsearch(es_host)
        self.index_name = index_name
        
        # Define RAG prompt template
        self.prompt = ChatPromptTemplate.from_messages([
            ("system", """
            You are a helpful AI assistant. Answer the question based ONLY on the provided context.
            Format your response in the following way:
            1. Give a clear, direct answer using information from the sources
            2. For each statement, cite the specific source and quote the relevant text
            3. If you're unsure about any part of the answer, explicitly state your uncertainty
            4. If you cannot find the answer in the context, say 'I don't have enough information to answer that.'
            
            Example format:
            According to [doc1.pdf], "[exact quote]", which indicates that...
            This is further supported by [doc2.txt] which states "[exact quote]"...
            
            Always maintain high precision over recall - it's better to say you're unsure than to make assumptions.
            """),
            ("human", "Context:\n{context}\n\nQuestion: {question}")
        ])
    
    def _get_relevant_chunks(self, question: str, k: int = 4) -> List[Dict]:
        """Get relevant document chunks for the question."""
        return self.es.search(
            index=self.index_name,
            body={
                "query": {
                    "match": {
                        "content": question
                    }
                },
                "size": k
            }
        )["hits"]["hits"]
    
    def _format_context(self, chunks: List[Dict]) -> str:
        """Format chunks into context string."""
        return "\n\n".join([
            f"[{chunk['_source']['metadata']['source']}]:\n{chunk['_source']['content']}"
            for chunk in chunks
        ])
    
    def generate_answer(self, question: str) -> Dict:
        """Generate answer with sources."""
        # Get relevant chunks
        chunks = self._get_relevant_chunks(question)
        
        # If no chunks found
        if not chunks:
            return {
                "answer": "I don't have enough information to answer that question.",
                "sources": []
            }
        
        # Format context
        context = self._format_context(chunks)
        
        # Generate answer
        chain = self.prompt | self.llm
        response = chain.invoke({
            "context": context,
            "question": question
        })
        
        return {
            "answer": response.content,
            "sources": [{
                "text": chunk["_source"]["content"],
                "metadata": chunk["_source"]["metadata"],
                "score": chunk["_score"]
            } for chunk in chunks]
        }
```

### Security Considerations for MVP

> Note: Full security implementation is out of scope for MVP. The following are basic precautions for development:

#### Local Development Security
- Use environment variables for sensitive values
- Don't commit API keys or credentials
- Basic input validation
- Error message sanitization

#### Data Handling
- Local file system storage only
- Basic file validation
- No sensitive data processing

#### Future Security Features (Phase 2)
- Authentication & authorization
- Encryption at rest and in transit
- Rate limiting
- Input sanitization
- XSS protection
- CSRF protection
- Secure headers
- Audit logging

## Testing Strategy

### Test Categories

#### 1. Must-Have Tests

##### Unit Tests
```python
# Document Processing
def test_document_reading():
    reader = DocumentReader()
    doc = reader.read_file("test.pdf")
    assert doc["text"]
    assert doc["metadata"]

# Vector Store
def test_vector_operations():
    store = ElasticsearchStore()
    # Test document addition
    store.add_documents([{"text": "test", "metadata": {}}])
    # Test basic search
    results = store.similarity_search("test")
    assert len(results) > 0

# RAG Chain
def test_answer_generation():
    chain = RAGChain()
    response = chain.generate_answer("test question")
    assert "answer" in response
    assert "sources" in response

# CLI
def test_cli_commands():
    result = runner.invoke(cli, ["process", "test.pdf"])
    assert result.exit_code == 0
```

##### Integration Tests
```python
# Core Pipeline
def test_document_to_vectors():
    # 1. Read document
    doc = reader.read_file("test.pdf")
    # 2. Create chunks
    chunks = chunker.create_chunks(doc["text"])
    # 3. Store vectors
    store.add_documents(chunks)
    assert store.count() > 0

# End-to-End Flow
def test_question_answering():
    # 1. Process document
    process_document("test.pdf")
    # 2. Ask question
    response = generate_answer("What is in the document?")
    assert response["answer"]
    assert len(response["sources"]) > 0

# Error Handling
def test_invalid_inputs():
    # Test invalid file
    with pytest.raises(InvalidFileError):
        reader.read_file("nonexistent.pdf")
    # Test empty query
    with pytest.raises(InvalidQueryError):
        store.similarity_search("")
```

### Test Coverage Goals

#### Must-Have
- Core functionality: 80%+ coverage
- Error handling for common cases
- CLI command validation
- Basic integration tests

#### Nice-to-Have
- Edge cases and rare errors
- UI component tests (if web app implemented)

#### Out of Scope
- Performance testing and benchmarks
  * Document processing speed
  * Search latency measurements
  * Response time thresholds
- Load testing
  * Concurrent operations
  * Resource usage monitoring
  * Scalability testing

### Running Tests
```bash
# Run core tests
python -m pytest tests/core/

# Run with coverage
python -m pytest --cov=src tests/core/
```

## Monitoring and Logging

### Development Logging
```python
# Enable debug logging
import logging
logging.basicConfig(
    level=logging.DEBUG,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)

# Log basic events
logger = logging.getLogger(__name__)
logger.info("Processing document: %s", doc_id)
logger.error("Failed to process: %s", error)
```

### Basic Metrics
- Document processing time
- Vector store query latency
- Memory usage
- Error rates
- Success/failure counts
