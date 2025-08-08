import json
import os
from pathlib import Path
from src.document_processing.chunker import Chunker
from src.embeddings.generator import EmbeddingGenerator
from src.vector_db.faiss_db import FAISSVectorDB
from src.tools.metadata_validator import MetadataSchema

def process_classified_document():
    # Initialize components
    chunker = Chunker()
    embedding_generator = EmbeddingGenerator(model_name="all-MiniLM-L6-v2")
    vector_db = FAISSVectorDB()
    
    # Process the document
    doc_path = Path("data/documents/classified_strategic_plan.pdf")
    text = doc_path.read_text()
    
    # Create chunks with classification
    chunks = chunker.process_document({
        "text": text,
        "metadata": {
            "classification": "C",  # Set document-level classification
            "title": "Classified Strategic Plan",
            "source": str(doc_path),
            "topic": "strategic_plan"
        }
    })
    
    # Validate metadata
    for chunk in chunks:
        if not MetadataSchema.validate_metadata(chunk["metadata"]):
            raise ValueError(f"Invalid metadata for chunk: {chunk['metadata']}")
    
    # Generate embeddings
    for chunk in chunks:
        chunk["embedding"] = embedding_generator.generate_embedding(chunk["text"])
    
    # Save embedded chunks
    output_dir = Path("data/embedded")
    output_dir.mkdir(parents=True, exist_ok=True)
    
    for i, chunk in enumerate(chunks):
        output_path = output_dir / f"classified_strategic_plan_{i}.json"
        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(chunk, f, indent=2)
    
    # Add to vector database
    vector_db.add_documents(chunks)
    
    print(f"Processed {len(chunks)} chunks from classified document")
    print(f"Vector database now contains {vector_db.get_document_count()} documents")

if __name__ == "__main__":
    process_classified_document()
