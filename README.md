# RAG Application

A Retrieval Augmented Generation (RAG) system that enables users to efficiently search and query document collections through natural language conversations.

## Features

- Document processing and text extraction
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

2. Install dependencies:
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

1. Process documents:
   ```
   python scripts/process_documents.py
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
