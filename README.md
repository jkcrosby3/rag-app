# RAG Application

A Retrieval Augmented Generation (RAG) system that enables users to efficiently search and query document collections through natural language conversations.

## Features

- Smart document processing with automatic processor selection
- Multiple PDF processing libraries support (pdfplumber, PyPDF2)
- Advanced text extraction with table and image handling
- Semantic chunking of documents
- Vector embeddings generation
- Vector database storage (FAISS for development, Elasticsearch for production)
- Semantic search and retrieval
- LLM integration with support for multiple providers (Claude, OpenAI, etc.)
- Easy switching between different LLM providers using a factory pattern
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
```

## Available User Interfaces

The application provides several UI options to suit different needs:

1. **Unified Interface (Recommended)**
   - Combines query and document management in one interface
   - Streamlit-based with a modern, responsive design
   - Accessible at: http://localhost:8501

2. **Query Interface**
   - Focused interface just for querying the RAG system
   - Good for end-users who only need to ask questions
   - Streamlit-based
   - Accessible at: http://localhost:8501

3. **Document Management UI**
   - For managing and processing documents
   - Includes document upload, processing, and management
   - Gradio-based with a simple interface
   - Accessible at: http://localhost:7860

4. **Flask Web App**
   - Traditional web interface with more features
   - Good for advanced users and administration
   - Includes conversation history and document management
   - Accessible at: http://localhost:5000

## Installation and Startup

### Option 1: Using the Startup Script (Recommended)

#### Windows:
1. Double-click on `start_web_app.bat` or run it from the command line:
   ```
   .\start_web_app.bat
   ```
   - You'll be prompted to choose which UI to start
   - Press Enter to accept the default (Unified Interface)

#### Linux/MacOS:
1. Make the script executable (first time only):
   ```bash
   chmod +x start_web_app.sh
   ```
2. Run the script:
   ```bash
   ./start_web_app.sh
   ```
   - You'll be prompted to choose which UI to start
   - Press Enter to accept the default (Unified Interface)

The script will:
1. Create and activate a Python virtual environment
2. Install all required dependencies
3. Check and build the vector database if needed
4. Start the web application

### Option 2: Manual Installation

1. Create and activate a virtual environment:
   ```bash
   # Windows
   python -m venv venv
   .\venv\Scripts\activate
   
   # Linux/MacOS
   python -m venv venv
   source venv/bin/activate
   ```

2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   pip install -e .
   ```

3. Start the web application:
   ```bash
   python start_web_app.py
   ```
   Or for direct access:
   ```bash
   streamlit run app.py
   ```

3. Set up environment variables:
Create a `.env` file with:
```
ANTHROPIC_API_KEY=your_api_key
ELASTICSEARCH_API_KEY=your_es_key  # Only needed for production
ELASTICSEARCH_CLOUD_ID=your_cloud_id  # Only needed for production
```

## Usage

### Basic Query with Default LLM
```python
from src.rag_system import RAGSystem

# Uses default LLM (Claude)
rag = RAGSystem()
response = rag.process_query("What is the Great Depression?")
```

### Using a Specific LLM Provider
```python
from src.rag_system import RAGSystem

# Use OpenAI instead of Claude
rag = RAGSystem(
    llm_type="openai",
    llm_api_key="your_openai_api_key",
    llm_model_name="gpt-4"
)
response = rag.process_query("What is the Great Depression?")
```

### Getting Available LLM Providers
```python
from src.llm import get_available_llm_types

print(f"Available LLM providers: {get_available_llm_types()}")
```

```python
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

## LLM Providers

The RAG system supports multiple LLM providers through a factory pattern. The following providers are available:

1. **Claude (default)**
   - Provider ID: `claude`
   - Required Env Var: `ANTHROPIC_API_KEY`
   - Example: 
     ```python
     from src.llm import get_llm_client
     
     # Get Claude client
     client = get_llm_client(
         "claude",
         api_key="your_anthropic_api_key",
         model_name="claude-3-5-sonnet-20241022"
     )
     ```

2. **OpenAI**
   - Provider ID: `openai`
   - Required Env Var: `OPENAI_API_KEY`
   - Example:
     ```python
     client = get_llm_client(
         "openai",
         api_key="your_openai_api_key",
         model_name="gpt-4"
     )
     ```

### Adding a Custom LLM Provider

1. Create a new class that inherits from `BaseLLMClient`
2. Implement the required methods
3. Register your client using the `@register_llm_client` decorator

Example:
```python
from src.llm import BaseLLMClient, register_llm_client

@register_llm_client("my_llm")
class MyLLMClient(BaseLLMClient):
    def __init__(self, api_key: str, model_name: str, **kwargs):
        super().__init__(api_key, model_name, **kwargs)
        # Initialize your client

    def generate_response(self, query: str, retrieved_documents: List[Dict[str, Any]], 
                         system_prompt: Optional[str] = None, **kwargs) -> str:
        # Implement response generation
        pass

    def generate_response_direct(self, system_prompt: str, user_message: str, **kwargs) -> str:
        # Implement direct response generation
        pass
```

## Environment Variables

### LLM Configuration
- `ANTHROPIC_API_KEY`: API key for Anthropic's Claude
- `OPENAI_API_KEY`: API key for OpenAI models
- `DEFAULT_LLM`: Default LLM provider to use (e.g., 'claude', 'openai')

### Vector Database
- `ELASTICSEARCH_API_KEY`: API key for Elasticsearch (production)
- `ELASTICSEARCH_CLOUD_ID`: Cloud ID for Elasticsearch (production)

### Logging
- `LOG_LEVEL`: Logging level (INFO, DEBUG, etc.)

## License

[MIT License](LICENSE)
