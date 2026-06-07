"""
Document indexing script for the RAG system.

This script generates embeddings for document chunks and indexes them in the vector database.
"""
import os
import sys
import json
import logging
import argparse
from pathlib import Path
from typing import List, Dict, Any, Optional
import time

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)

# Add the src directory to the Python path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))

# Import RAG system components
try:
    from src.embeddings.generator import EmbeddingGenerator
    from src.vector_db.faiss_db import FAISSVectorDB
    from src.vector_db.elasticsearch_db import ElasticsearchVectorDB
except ImportError as e:
    logger.error(f"Error importing RAG system components: {str(e)}")
    sys.exit(1)


def load_document_chunks(input_dir: str) -> List[Dict[str, Any]]:
    """
    Load document chunks from a directory.
    
    Args:
        input_dir: Directory containing document chunks
        
    Returns:
        List of document chunks
    """
    input_dir = Path(input_dir)
    if not input_dir.exists():
        logger.error(f"Input directory not found: {input_dir}")
        return []
    
    # Find all JSON files
    json_files = list(input_dir.glob("*.json"))
    if not json_files:
        logger.warning(f"No JSON files found in {input_dir}")
        return []
    
    logger.info(f"Found {len(json_files)} JSON files in {input_dir}")
    
    # Load document chunks
    document_chunks = []
    for file_path in json_files:
        try:
            with open(file_path, 'r', encoding='utf-8') as file:
                document = json.load(file)
                document_chunks.append(document)
        except Exception as e:
            logger.error(f"Error loading document chunk {file_path}: {str(e)}")
    
    logger.info(f"Loaded {len(document_chunks)} document chunks")
    return document_chunks


def index_documents(
    document_chunks: List[Dict[str, Any]],
    embedding_generator: EmbeddingGenerator,
    vector_db,
    vector_db_type: str,
    vector_db_path: str,
    batch_size: int = 32
) -> None:
    """
    Generate embeddings for document chunks and index them in the vector database.
    
    Args:
        document_chunks: List of document chunks
        embedding_generator: Embedding generator
        vector_db: Vector database
        batch_size: Batch size for embedding generation
    """
    if not document_chunks:
        logger.warning("No document chunks to index")
        return
    
    logger.info(f"Indexing {len(document_chunks)} document chunks with batch size {batch_size}")
    
    # Process in batches
    total_batches = (len(document_chunks) + batch_size - 1) // batch_size
    start_time = time.time()
    
    for batch_idx in range(total_batches):
        batch_start = batch_idx * batch_size
        batch_end = min(batch_start + batch_size, len(document_chunks))
        batch = document_chunks[batch_start:batch_end]
        
        logger.info(f"Processing batch {batch_idx + 1}/{total_batches} ({batch_start}-{batch_end-1})")
        
        # Extract texts and metadata
        texts = [doc["text"] for doc in batch]
        metadata_list = [doc["metadata"] for doc in batch]
        
        # Generate embeddings
        embeddings = embedding_generator.generate_embeddings(texts)
        
        # Prepare documents for indexing in the format expected by the vector database
        documents_to_index = []
        for i, (text, embedding, metadata) in enumerate(zip(texts, embeddings, metadata_list)):
            documents_to_index.append({
                "text": text,
                "embedding": embedding,
                "metadata": metadata
            })
        
        # Index documents
        vector_db.add_documents(documents_to_index)
        
        # Log progress
        elapsed_time = time.time() - start_time
        docs_per_second = (batch_end) / elapsed_time if elapsed_time > 0 else 0
        logger.info(f"Indexed {batch_end}/{len(document_chunks)} documents "
                   f"({docs_per_second:.2f} docs/sec)")
    
    # Save the index
    if vector_db_type == "faiss":
        vector_db.save(vector_db_path)
    
    total_time = time.time() - start_time
    logger.info(f"Indexing completed in {total_time:.2f} seconds")


def main():
    parser = argparse.ArgumentParser(description="Index documents in the RAG system")
    parser.add_argument("--input-dir", type=str, required=True, help="Directory containing document chunks")
    parser.add_argument("--vector-db-type", type=str, default="faiss", choices=["faiss", "elasticsearch"], 
                        help="Vector database type")
    parser.add_argument("--vector-db-path", type=str, default="data/vector_db", 
                        help="Path to the vector database")
    parser.add_argument("--embedding-model", type=str, default="all-MiniLM-L6-v2", 
                        help="Embedding model to use")
    parser.add_argument("--batch-size", type=int, default=32, 
                        help="Batch size for embedding generation")
    # Note: The quantized model is handled internally by the embedding system
    
    args = parser.parse_args()
    
    # Load document chunks
    document_chunks = load_document_chunks(args.input_dir)
    if not document_chunks:
        logger.error("No document chunks to index")
        return
    
    # Initialize embedding generator
    embedding_generator = EmbeddingGenerator(
        model_name=args.embedding_model
    )
    
    # Initialize vector database
    if args.vector_db_type == "faiss":
        vector_db = FAISSVectorDB(
            index_path=args.vector_db_path,
            dimension=embedding_generator._get_embedding_dimension()
        )
    elif args.vector_db_type == "elasticsearch":
        vector_db = ElasticsearchVectorDB(
            index_name="rag_documents",
            embedding_size=embedding_generator._get_embedding_dimension()
        )
    else:
        logger.error(f"Unsupported vector database type: {args.vector_db_type}")
        return
    
    # Index documents
    index_documents(
        document_chunks=document_chunks,
        embedding_generator=embedding_generator,
        vector_db=vector_db,
        vector_db_type=args.vector_db_type,
        vector_db_path=args.vector_db_path,
        batch_size=args.batch_size
    )


if __name__ == "__main__":
    main()
