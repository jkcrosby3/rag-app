"""
Script to chunk processed documents for the RAG system.

This script takes processed documents and splits them into smaller,
semantically meaningful chunks for embedding and retrieval.
"""
import logging
import sys
import os
from pathlib import Path

# Add the project root to the Python path to fix import issues
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from src.document_processing.chunker import chunk_processed_documents


def main():
    """Chunk processed documents and save results."""
    # Configure logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    
    logger = logging.getLogger(__name__)
    logger.info("Starting document chunking")
    
    # Define directories
    input_dir = project_root / "data" / "processed"
    output_dir = project_root / "data" / "chunked"
    
    # Ensure output directory exists
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Process documents
    try:
        stats = chunk_processed_documents(
            input_dir=input_dir,
            output_dir=output_dir,
            chunk_size=500,  # Target size of each chunk in tokens
            chunk_overlap=50  # Number of tokens to overlap between chunks
        )
        
        logger.info(f"Successfully chunked {stats['total_documents']} documents into {stats['total_chunks']} chunks")
        
        # Print summary of chunked documents by topic
        logger.info("Chunks created by topic:")
        for topic, topic_stats in stats["documents_by_topic"].items():
            logger.info(f"  - {topic}: {topic_stats['document_count']} documents, {topic_stats['chunk_count']} chunks")
            
    except Exception as e:
        logger.error(f"Error chunking documents: {str(e)}", exc_info=True)
        return 1
        
    return 0


if __name__ == "__main__":
    sys.exit(main())
