"""
Script to generate embeddings for chunked documents in the RAG system.

This script takes chunked documents and generates vector embeddings
for semantic search and retrieval.
"""
import logging
import sys
import os
from pathlib import Path

# Add the project root to the Python path to fix import issues
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from src.embeddings.generator import process_chunks_with_embeddings


def main():
    """Generate embeddings for chunked documents and save results."""
    # Configure logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    
    logger = logging.getLogger(__name__)
    logger.info("Starting embedding generation")
    
    # Define directories
    input_dir = project_root / "data" / "chunked"
    output_dir = project_root / "data" / "embedded"
    
    # Ensure output directory exists
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Process documents
    try:
        stats = process_chunks_with_embeddings(
            input_dir=input_dir,
            output_dir=output_dir,
            model_name="all-MiniLM-L6-v2"  # Lightweight model from sentence-transformers
        )
        
        logger.info(f"Successfully generated embeddings for {stats['total_chunks_processed']} chunks")
        
        # Print summary of processed chunks by topic
        logger.info("Chunks processed by topic:")
        for topic, count in stats["chunks_by_topic"].items():
            logger.info(f"  - {topic}: {count} chunks")
            
    except Exception as e:
        logger.error(f"Error generating embeddings: {str(e)}", exc_info=True)
        return 1
        
    return 0


if __name__ == "__main__":
    sys.exit(main())
