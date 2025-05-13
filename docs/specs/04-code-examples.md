# RAG System Code Examples

> This document provides code examples that are used in both the notebooks and production code.
> For interactive examples, see the corresponding notebooks in the `/notebooks` directory.

## Document Processing

### Test Data Setup
```python
# Common test data setup used in notebooks and tests
from pathlib import Path
from reportlab.pdfgen import canvas

def create_test_files(base_dir: Path = Path("test_files")):
    """Create test files for examples and testing.
    Used in notebooks/01_document_processing.ipynb"""
    base_dir.mkdir(exist_ok=True)
    
    # Create a test PDF
    test_pdf = base_dir / "test.pdf"
    c = canvas.Canvas(str(test_pdf))
    c.drawString(100, 750, "This is a test PDF document.")
    c.save()
    
    # Create a test text file
    test_txt = base_dir / "test.txt"
    test_txt.write_text("This is a test text document.\nIt has multiple lines.")
    
    return {
        "pdf": test_pdf,
        "txt": test_txt
    }

### Document Reader
```python
from pathlib import Path
from typing import Dict, Any
import fitz  # PyMuPDF

class DocumentReader:
    def read_file(self, file_path: Path) -> Dict[str, Any]:
        """Read and extract text from a document."""
        if not file_path.exists():
            raise FileNotFoundError(f"File not found: {file_path}")
            
        text = ""
        metadata = {}
        
        if file_path.suffix.lower() == '.pdf':
            with fitz.open(file_path) as doc:
                metadata = {
                    "title": doc.metadata.get("title", ""),
                    "author": doc.metadata.get("author", ""),
                    "pages": len(doc)
                }
                text = "\n".join(page.get_text() for page in doc)
        elif file_path.suffix.lower() == '.txt':
            with open(file_path, 'r') as f:
                text = f.read()
                metadata = {
                    "title": file_path.stem,
                    "pages": 1
                }
                
        return {
            "text": text,
            "metadata": metadata,
            "source": str(file_path)
        }

class TextChunker:
    def __init__(self, chunk_size: int = 1000, chunk_overlap: int = 200):
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        
    def create_chunks(self, text: str) -> List[Dict[str, Any]]:
        """Split text into overlapping chunks."""
        chunks = []
        start = 0
        
        while start < len(text):
            end = start + self.chunk_size
            chunk_text = text[start:end]
            
            # Try to end at sentence boundary
            if end < len(text):
                last_period = chunk_text.rfind('.')
                if last_period != -1:
                    end = start + last_period + 1
                    chunk_text = text[start:end]
            
            chunks.append({
                "text": chunk_text,
                "start": start,
                "end": end
            })
            
            start = end - self.chunk_overlap
            
        return chunks
```

## Vector Store

### Elasticsearch Store
```python
from elasticsearch import Elasticsearch
from sentence_transformers import SentenceTransformer
import numpy as np

class ElasticsearchStore:
    def __init__(self, es_url: str, index_name: str = "documents"):
        self.es = Elasticsearch(es_url)
        self.index_name = index_name
        self.model = SentenceTransformer('all-MiniLM-L6-v2')
        
        # Create index if not exists
        if not self.es.indices.exists(index=index_name):
            self.es.indices.create(
                index=index_name,
                mappings={
                    "properties": {
                        "text": {"type": "text"},
                        "vector": {
                            "type": "dense_vector",
                            "dims": 384,
                            "index": True,
                            "similarity": "cosine"
                        },
                        "metadata": {"type": "object"}
                    }
                }
            )
    
    def add_documents(self, documents: List[Dict[str, Any]]):
        """Add document chunks to Elasticsearch."""
        for doc in documents:
            vector = self.model.encode(doc["text"])
            
            self.es.index(
                index=self.index_name,
                document={
                    "text": doc["text"],
                    "vector": vector,
                    "metadata": doc.get("metadata", {})
                }
            )
            
    def similarity_search(self, query: str, k: int = 3) -> List[Dict]:
        """Search for similar documents."""
        query_vector = self.model.encode(query)
        
        response = self.es.search(
            index=self.index_name,
            knn={
                "field": "vector",
                "query_vector": query_vector,
                "k": k,
                "num_candidates": 100
            },
            source=["text", "metadata"]
        )
        
        return [hit["_source"] for hit in response["hits"]["hits"]]
```

## RAG Chain

### Basic Implementation
```python
from typing import Dict, List
import openai

class RAGChain:
    def __init__(self, vector_store: ElasticsearchStore):
        self.vector_store = vector_store
        
    def generate_answer(self, question: str) -> Dict:
        """Generate answer using RAG approach."""
        # Get relevant context
        context_docs = self.vector_store.similarity_search(question)
        context = "\n\n".join(doc["text"] for doc in context_docs)
        
        # Generate answer
        prompt = f"""Answer the question based on the following context:

Context:
{context}

Question: {question}

Answer:"""

        response = openai.ChatCompletion.create(
            model="gpt-3.5-turbo",
            messages=[
                {"role": "system", "content": "You are a helpful assistant."},
                {"role": "user", "content": prompt}
            ],
            temperature=0
        )
        
        return {
            "answer": response.choices[0].message.content,
            "sources": context_docs
        }
```

## Command Line Interface

### CLI Implementation
```python
import argparse
from pathlib import Path
from typing import Optional

class RagCLI:
    def __init__(self):
        self.reader = DocumentReader()
        self.chunker = TextChunker()
        self.store = ElasticsearchStore("http://localhost:9200")
        self.chain = RAGChain(self.store)

    def process_file(self, file_path: str) -> None:
        """Process a document file."""
        doc = self.reader.read_file(Path(file_path))
        chunks = self.chunker.create_chunks(doc["text"])
        self.store.add_documents(chunks)
        print(f"Processed {file_path}")

    def ask_question(self, question: str) -> None:
        """Get answer for a question."""
        response = self.chain.generate_answer(question)
        print(f"\nAnswer: {response['answer']}\n")
        print("Sources:")
        for src in response['sources']:
            print(f"- {src['text'][:200]}...\n")

def main():
    parser = argparse.ArgumentParser(description="RAG System CLI")
    subparsers = parser.add_subparsers(dest="command")

    # Process command
    process_parser = subparsers.add_parser("process", help="Process a document")
    process_parser.add_argument("file", help="Path to document file")

    # Ask command
    ask_parser = subparsers.add_parser("ask", help="Ask a question")
    ask_parser.add_argument("question", help="Your question")

    args = parser.parse_args()
    cli = RagCLI()

    if args.command == "process":
        cli.process_file(args.file)
    elif args.command == "ask":
        cli.ask_question(args.question)

if __name__ == "__main__":
    main()
```

## Web Interface

### Streamlit App
```python
import streamlit as st
from pathlib import Path
import tempfile

class RagWebApp:
    def __init__(self):
        self.reader = DocumentReader()
        self.chunker = TextChunker()
        self.store = ElasticsearchStore("http://localhost:9200")
        self.chain = RAGChain(self.store)

    def process_uploaded_file(self, uploaded_file):
        with tempfile.NamedTemporaryFile(delete=False, suffix=Path(uploaded_file.name).suffix) as tmp:
            tmp.write(uploaded_file.getvalue())
            doc = self.reader.read_file(Path(tmp.name))
            chunks = self.chunker.create_chunks(doc["text"])
            self.store.add_documents(chunks)
        Path(tmp.name).unlink()

    def run(self):
        st.title("RAG System")

        # File upload
        uploaded_file = st.file_uploader("Upload Document", type=["pdf", "txt"])
        if uploaded_file:
            with st.spinner("Processing document..."):
                self.process_uploaded_file(uploaded_file)
            st.success("Document processed!")

        # Question answering
        question = st.text_input("Ask a question about your documents")
        if question:
            with st.spinner("Generating answer..."):
                response = self.chain.generate_answer(question)
            
            st.write("### Answer")
            st.write(response["answer"])
            
            st.write("### Sources")
            for src in response["sources"]:
                st.text(f"{src['text'][:200]}...")

if __name__ == "__main__":
    app = RagWebApp()
    app.run()
```

## Docker Setup

### Dockerfile
```dockerfile
FROM python:3.11-slim

WORKDIR /app

COPY requirements.txt .
RUN pip install -r requirements.txt

COPY src/ src/

EXPOSE 8501

CMD ["streamlit", "run", "src/app.py"]
```

### Docker Compose
```yaml
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
