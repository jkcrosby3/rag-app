# RAG Application

A Retrieval Augmented Generation (RAG) system that enables users to efficiently search and query document collections through natural language conversations.

## Features

- Smart document processing with automatic processor selection
- Multiple PDF processing libraries support (PyMuPDF, pdfplumber, PyPDF2)
- Advanced text extraction with table and image handling
- Semantic chunking of documents
- Vector embeddings generation
- Vector database storage (FAISS for development, Elasticsearch for production)
- Semantic search and retrieval
- LLM integration with Claude for response generation
- Support for filtering by topic

## Project Structure

```
rag-app/
├── data/                     # Data directories
│   ├── chunked/              # Chunked documents
│   ├── embedded/             # Documents with embeddings
│   ├── processed/            # Processed documents
│   └── vector_db/            # FAISS vector database
├── docs/                     # Documentation
├── scripts/                  # Utility scripts
│   ├── process_documents.py  # Document processing script
│   ├── chunk_documents.py    # Document chunking script
│   ├── generate_embeddings.py# Embedding generation script
│   ├── build_vector_db.py    # Vector database building script
│   └── rag_demo.py           # Demo script for RAG system
├── src/                      # Source code
│   ├── document_processing/  # Document processing modules
│   ├── embeddings/           # Embedding generation modules
│   ├── llm/                  # LLM integration modules
│   ├── retrieval/            # Document retrieval modules
│   ├── vector_db/            # Vector database implementations
│   └── rag_system.py         # Complete RAG system
├── .env.template             # Template for environment variables
├── requirements.txt          # Python dependencies
└── README.md                 # Project documentation
```

## Setup

1. Create a virtual environment:
   ```
   python -m venv venv
   .\venv\Scripts\activate
   ```

2. Install document processing dependencies:
   ```
   pip install PyMuPDF PyPDF2 pdfplumber pdfminer.six pdfbox-python
   ```

3. Install core dependencies:
   ```
   pip install -r requirements.txt
   ```

3. Set up environment variables:
   ```
   cp .env.template .env
   ```
   Edit the `.env` file and add your API keys.

## Usage

### Document Processing Pipeline

1. Process documents with smart processor:
   ```
   python examples/document_processing_example.py
   ```
   
   The smart processor automatically handles all document processing needs. You don't need to worry about which specific processor to use - the system will automatically choose the best one based on your requirements.

   **Usage Examples**

   1. Basic text extraction:
   ```python
   from tools.smart_document_processor import SmartDocumentProcessor

   processor = SmartDocumentProcessor()
   result = processor.process_document(
       "example.pdf",
       requirements={
           'text': True,        # Extract text
           'tables': False,     # No tables needed
           'images': False,     # No images needed
           'metadata': False,   # No metadata needed
           'complexity': 'low'  # Simple document
       }
   )
   print(f"Extracted text length: {len(result['text'])}")
   ```

   2. Complex document with tables:
   ```python
   result = processor.process_document(
       "government_form.pdf",
       requirements={
           'text': True,        # Extract text
           'tables': True,      # Extract tables
           'images': False,     # No images
           'metadata': True,    # Extract metadata
           'complexity': 'medium'  # Medium complexity document
       }
   )
   print(f"Tables found: {len(result['tables'])}")
   ```

   3. Image-heavy document:
   ```python
   result = processor.process_document(
       "image_heavy.pdf",
       requirements={
           'text': True,        # Extract text
           'tables': False,     # No tables
           'images': True,      # Extract images
           'metadata': True,    # Extract metadata
           'complexity': 'high'  # Complex document with images
       }
   )
   print(f"Images found: {len(result['images'])}")
   ```

2. Chunk documents:
   ```
   python scripts/chunk_documents.py
   ```

3. Generate embeddings:
   ```
   python scripts/generate_embeddings.py
   ```

4. Build vector database:
   ```
   python scripts/build_vector_db.py
   ```

### RAG System

Run the RAG system with a query:
```
python src/rag_system.py "What were the key provisions of the Glass-Steagall Act?"
```

Options:
- `--vector-db`: Vector database backend to use (`faiss` or `elasticsearch`)
- `--top-k`: Number of documents to retrieve
- `--topics`: Comma-separated list of topics to filter by
- `--temperature`: Temperature for LLM generation

Example with filtering:
```
python src/rag_system.py "What was the SEC established to do?" --topics sec
```

## Environment Variables

- `ANTHROPIC_API_KEY`: API key for Anthropic's Claude
- `ELASTICSEARCH_API_KEY`: API key for Elasticsearch (production)
- `ELASTICSEARCH_CLOUD_ID`: Cloud ID for Elasticsearch (production)
- `LOG_LEVEL`: Logging level (INFO, DEBUG, etc.)

## License

[MIT License](LICENSE)
