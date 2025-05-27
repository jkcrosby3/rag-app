"""Script to preload models for faster RAG system initialization.

This script preloads the embedding model and LLM client to reduce
cold start times for the first query.
"""
import sys
import logging
import time
import argparse
from pathlib import Path

# Add project root to Python path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.preloader import preloader, preload_models

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)

logger = logging.getLogger(__name__)

def main():
    """Preload models and wait for completion."""
    # Parse arguments
    parser = argparse.ArgumentParser(description='Preload models for the RAG system')
    parser.add_argument('--quantized', action='store_true', help='Preload quantized embedding model')
    parser.add_argument('--quantization-type', type=str, choices=['int8', 'int4'], default='int8',
                        help='Type of quantization to use for the embedding model')
    args = parser.parse_args()
    
    start_time = time.time()
    
    # Start preloading
    if args.quantized:
        logger.info(f"Starting model preloading with {args.quantization_type} quantized embeddings...")
        preload_models(use_quantized_embeddings=True, quantization_type=args.quantization_type)
    else:
        logger.info("Starting model preloading...")
        preload_models()
    
    # Wait for preloading to complete
    logger.info("Waiting for preloading to complete...")
    if preloader.wait_for_preloading(timeout=60.0):
        preload_time = time.time() - start_time
        logger.info(f"Models preloaded successfully in {preload_time:.2f} seconds")
    else:
        logger.error("Timed out waiting for model preloading")
        return 1
    
    return 0

if __name__ == "__main__":
    sys.exit(main())
