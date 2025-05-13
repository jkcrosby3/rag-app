# PRD and TDD: RAG

## Product Requirements

### Overview
A Retrieval Augmented Generation (RAG) system that processes various document formats from a local data directory and enables efficient information retrieval through both CLI and web interfaces, powered by Elasticsearch for vector storage.

### Document Source Requirements
- All documents must be stored in the `data/raw` directory
- Supported formats: PDF, TXT, MD, and other text-based files
- System will monitor the `data/raw` directory for new documents
- Processed documents will be tracked in `data/processed` with metadata
- Original documents remain unchanged in `data/raw`

### Core Features
1. Document Processing
   - Support for multiple file formats (PDF, TXT, MD)
   - Automatic text extraction and chunking
   - Vector embedding generation

2. Command Line Interface
   - Document ingestion commands
   - Search functionality
   - System status and management

3. Web Interface
   - Simple, intuitive UI
   - Document upload capability
   - Search interface with results display
   - Basic analytics dashboard

### User Requirements
- Easy document upload and processing
- Fast and accurate search results
- Clear result presentation
- System status visibility
- Scalable for growing document collections

## Technical Design

### Tech Stack
Backend:
- Python 3.11+
- FastAPI for web API
- Elasticsearch for vector storage
- LangChain for RAG pipeline
- PyPDF2 for PDF processing
- sentence-transformers for embeddings

Frontend:
- Streamlit for rapid UI development
- Streamlit-extras for enhanced components

Testing:
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

# Testing and Development
pytest>=7.4.0
pytest-asyncio>=0.21.0
pytest-cov>=4.1.0
black>=23.7.0
flake8>=6.0.0
```

### Architecture
```
[Documents] → [Document Processor] → [Vector Embeddings] → [Elasticsearch]
                      ↓                        ↓
              [FastAPI Backend] ←——→ [Streamlit Frontend]
```

### File Structure
```
rag/
├── src/
│   ├── backend/
│   │   ├── api/
│   │   ├── core/
│   │   └── utils/
│   ├── frontend/
│   └── cli/
├── data/
│   ├── raw/           # Source documents directory (PDF, TXT, MD)
│   ├── processed/     # Processed documents and metadata
│   └── index/         # Elasticsearch index data
├── tests/
├── data/
│   ├── raw/
│   └── processed/
├── config/
└── docs/
```

### Implementation Phases

Phase 1: Core Infrastructure
- Set up project structure with data directories
- Implement document watcher for data/raw
- Implement basic document processing
- Configure Elasticsearch
- Create basic CLI
- Set up document tracking system

Phase 2: Backend Development
- Develop FastAPI endpoints
- Implement vector search
- Add document management
- Create core utilities

Phase 3: Frontend Development
- Build Streamlit interface
- Implement search UI
- Add document upload
- Create results display

Phase 4: Testing & Optimization
- Write comprehensive tests
- Optimize performance
- Add monitoring
- Documentation

### Technical Considerations
1. Scalability
   - Chunking strategy for large documents
   - Batch processing for bulk uploads
   - Elasticsearch index optimization

2. Performance
   - Caching layer for frequent queries
   - Async processing for uploads
   - Efficient vector similarity search

### Testing Strategy
1. Unit Tests
   - Document processing functions
   - API endpoints
   - Search functionality
   - Vector operations

2. Integration Tests
   - End-to-end document processing
   - API-database interactions
   - Frontend-backend integration

### Detailed Tasks by Phase

Phase 1:
- [ ] Create project structure
- [ ] Set up virtual environment
- [ ] Create requirements.txt with initial dependencies
- [ ] Install and configure Elasticsearch
- [ ] Install project dependencies
- [ ] Implement document parser
- [ ] Create embedding generation
- [ ] Build basic CLI commands

Phase 2:
- [ ] Develop FastAPI router structure
- [ ] Implement document upload endpoint
- [ ] Create search endpoints
- [ ] Add document management API
- [ ] Implement error handling

Phase 3:
- [ ] Create Streamlit app structure
- [ ] Build upload interface
- [ ] Implement search UI
- [ ] Add results visualization
- [ ] Create system status page

Phase 4:
- [ ] Write unit tests
- [ ] Create integration tests
- [ ] Add performance monitoring
- [ ] Write documentation
- [ ] Optimize search performance

### Sample Integrations
1. Document Sources
   - Local file system
   - S3 buckets
   - Git repositories

2. Authentication
   - Basic auth
   - OAuth2
   - API keys

3. Monitoring
   - Prometheus metrics
   - Elasticsearch monitoring
   - Custom logging

### Next Steps
1. Review and finalize requirements
2. Set up development environment
3. Begin Phase 1 implementation
4. Regular progress reviews
