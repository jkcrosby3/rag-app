"""
Script to process documents for the RAG system.

This script processes documents from the specified directory,
extracts text and metadata, and saves the results.
"""
import logging
import sys
import os
from pathlib import Path

# Add the project root to the Python path to fix import issues
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from src.document_processing.batch_processor import process_great_depression_documents


def main():
    """Process documents and save results."""
    # Configure logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    
    logger = logging.getLogger(__name__)
    logger.info("Starting document processing")
    
    # Define directories
    base_dir = project_root / "data" / "great_depression"
    output_dir = project_root / "data" / "processed"
    
    # Ensure output directory exists
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Process documents
    try:
        results = process_great_depression_documents(
            base_dir=base_dir,
            output_dir=output_dir
        )
        logger.info(f"Successfully processed {len(results)} documents")
        
        # Print summary of processed documents
        topics = {}
        for result in results:
            topic = result['metadata'].get('topic', 'unknown')
            if topic not in topics:
                topics[topic] = 0
            topics[topic] += 1
        
        logger.info("Documents processed by topic:")
        for topic, count in topics.items():
            logger.info(f"  - {topic}: {count} documents")
            
    except Exception as e:
        logger.error(f"Error processing documents: {str(e)}", exc_info=True)
        return 1
        
    return 0


if __name__ == "__main__":
    sys.exit(main())
