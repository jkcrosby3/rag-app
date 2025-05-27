"""
Script to build a vector database from embedded documents in the RAG system.

This script takes documents with embeddings and builds a vector database
for efficient similarity search and retrieval.
"""
import logging
import sys
import os
from pathlib import Path

# Add the project root to the Python path to fix import issues
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from src.vector_db.faiss_db import build_vector_db_from_embedded_docs


def main():
    """Build a vector database from embedded documents."""
    # Configure logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    
    logger = logging.getLogger(__name__)
    logger.info("Starting vector database build")
    
    # Define directories
    input_dir = project_root / "data" / "embedded"
    output_dir = project_root / "data" / "vector_db"
    
    # Ensure output directory exists
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Build vector database
    try:
        stats = build_vector_db_from_embedded_docs(
            input_dir=input_dir,
            output_dir=output_dir,
            index_type="Cosine"  # Use cosine similarity for semantic search
        )
        
        logger.info(f"Successfully built vector database with {stats['total_documents']} documents")
        
        # Print summary of documents by topic
        logger.info("Documents by topic:")
        for topic, count in stats["documents_by_topic"].items():
            logger.info(f"  - {topic}: {count} documents")
            
    except Exception as e:
        logger.error(f"Error building vector database: {str(e)}", exc_info=True)
        return 1
        
    return 0


if __name__ == "__main__":
    sys.exit(main())
