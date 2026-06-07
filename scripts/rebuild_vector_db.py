"""Script to rebuild the vector database with optimized settings.

This script rebuilds the vector database from embedded documents,
using the optimized IVF index for faster search.
"""
import logging
import sys
import os
import time
from pathlib import Path

# Add project root to Python path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.vector_db.faiss_db import build_vector_db_from_embedded_docs

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)

logger = logging.getLogger(__name__)

def main():
    """Rebuild the vector database with optimized settings."""
    start_time = time.time()
    
    # Default paths
    input_dir = Path("data/embedded")
    output_dir = Path("data/vector_db")
    
    # Build vector database with optimized settings
    logger.info(f"Rebuilding vector database from {input_dir} to {output_dir}")
    stats = build_vector_db_from_embedded_docs(
        input_dir=input_dir,
        output_dir=output_dir,
        index_type="Cosine"  # Cosine similarity is best for semantic search
    )
    
    # Log statistics
    build_time = time.time() - start_time
    logger.info(f"Vector database build complete in {build_time:.2f} seconds")
    logger.info(f"Added {stats['total_documents']} documents")
    
    for topic, count in stats["documents_by_topic"].items():
        logger.info(f"Topic '{topic}': {count} documents")

if __name__ == "__main__":
    main()
